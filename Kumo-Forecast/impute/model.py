# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Backbone-LF+ Model Architecture

Four novel modules (D-G) on top of Backbone-LF v2:
  D: ObservationAwareAttentionBias
  E: ForecastConsistencyLoss
  F: GapConditionedAuxReconHead
  G: MissingnessAdaptiveRevIN

CRS extension (cross-channel self-attention):
  CrossChannelAttention
  MissingnessResidualAdapter
  CRSConfig
  BackboneLFPlusCRS

Use build_model(args, n_channels, device) to get the right class based on args.use_crs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .backbone_lieflow import (
    TASKS,
    BackboneLF,
    LieEquivariantEncoderWrapper,
    Masking,
    NamespaceWithDefaults,
    RevIN,
    TimeseriesOutputs,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Base config and modules (non-CRS)
# =============================================================================

class PLUSConfig(NamespaceWithDefaults):
    """Configuration for Backbone-LF+ model with flags for all four modules."""

    use_obs_aware_attn: bool = True
    use_consistency_loss: bool = True
    use_aux_recon: bool = True
    use_adaptive_revin: bool = True
    obs_attn_max_gap: int = 32
    consistency_aug_rate: float = 0.12
    consistency_weight: float = 1.0
    aux_recon_weight: float = 0.5
    adaptive_revin_threshold: float = 0.5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _defaults = {
            "use_obs_aware_attn": True,
            "use_consistency_loss": True,
            "use_aux_recon": True,
            "use_adaptive_revin": True,
            "obs_attn_max_gap": 32,
            "consistency_aug_rate": 0.12,
            "consistency_weight": 1.0,
            "aux_recon_weight": 0.5,
            "adaptive_revin_threshold": 0.5,
        }
        for k, v in _defaults.items():
            if not hasattr(self, k):
                setattr(self, k, v)


class MissingnessAdaptiveRevIN(RevIN):
    """RevIN with empirical-Bayes shrinkage based on missing rate.

    Fix: shrink mean toward 0 and stdev toward the *global training stdev*
    (stored as a running EMA on the first call), not toward the arbitrary
    constant 1.0.  At high missing rates the per-sample stdev estimate is
    unreliable; blending toward the historical average is a better prior.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False, threshold: float = 0.5):
        super().__init__(num_features=num_features, eps=eps, affine=affine)
        self.threshold = threshold
        self._shrinkage = 0.0
        self._ema_alpha = 0.01
        # Use a real tensor sentinel (ones) so load_state_dict can copy the
        # trained EMA value into a fresh model without a None→Tensor shape error.
        # Ones are the correct uninitialised prior: unit variance is what you get
        # when no history is available.  _global_stdev_initialized tracks whether
        # the EMA has been seeded yet (False on a fresh model; True after the
        # first training batch or after loading a trained checkpoint).
        self.register_buffer(
            "_global_stdev", torch.ones(1, num_features, 1), persistent=True
        )
        self.register_buffer(
            "_global_stdev_initialized", torch.tensor(False), persistent=True
        )

    def set_missing_rate(self, missing_rate: float):
        if missing_rate <= self.threshold:
            self._shrinkage = 0.0
        else:
            span = max(1.0 - self.threshold, 1e-6)
            self._shrinkage = min((missing_rate - self.threshold) / span, 0.9)

    def _get_statistics(self, x, mask=None):
        super()._get_statistics(x, mask=mask)
        # Update global stdev EMA (detached — no gradient through the prior).
        # self.stdev has shape [B, C, 1] from parent RevIN.  Collapse the batch
        # dimension to [1, C, 1] before storing so the EMA is batch-size
        # independent and doesn't break when the last batch is smaller.
        # Update EMA only during training — freeze during val/test so evaluation
        # is deterministic and independent of batch order.
        if self.training:
            with torch.no_grad():
                batch_stdev = self.stdev.detach().mean(dim=0, keepdim=True)  # [1, C, 1]
                if not self._global_stdev_initialized.item():
                    # Seed the EMA from the first batch instead of blending from
                    # the ones prior — converges much faster on the real distribution.
                    self._global_stdev.copy_(batch_stdev)
                    self._global_stdev_initialized.fill_(True)
                else:
                    self._global_stdev.copy_(
                        (
                            (1.0 - self._ema_alpha) * self._global_stdev
                            + self._ema_alpha * batch_stdev
                        ).clamp(min=1e-2)  # prevent near-zero EMA stdev
                    )
        if self._shrinkage > 0.0:
            aa = self._shrinkage
            # Shrink mean toward 0 (fewer observations → regress to global mean).
            self.mean = self.mean * (1.0 - aa)
            # Shrink stdev toward the global historical stdev, not toward 1.0.
            # _global_stdev is [1, C, 1] and broadcasts over [B, C, 1].
            # When _global_stdev_initialized is False (fresh untrained model at
            # test time), fall back to unit variance as a safe prior.
            prior_stdev = (
                self._global_stdev.expand_as(self.stdev)
                if self._global_stdev_initialized.item()
                else torch.ones_like(self.stdev)
            )
            self.stdev = self.stdev * (1.0 - aa) + prior_stdev * aa


class ObservationAwareAttentionBias(nn.Module):
    """Learnable attention bias combining temporal proximity and observation confidence."""

    def __init__(self, n_heads: int, max_gap: int = 32):
        super().__init__()
        self.n_heads = n_heads
        self.max_gap = max_gap
        self.log_decay = nn.Parameter(torch.full((n_heads,), -2.0))
        self.obs_bonus = nn.Parameter(torch.full((n_heads,), 0.3))
        self.miss_penalty = nn.Parameter(torch.full((n_heads,), 0.3))

    def forward(self, n_patches, patch_obs_mask, gap_lengths):
        B = patch_obs_mask.shape[0]
        dev = patch_obs_mask.device

        pos = torch.arange(n_patches, device=dev).float()
        delta = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
        decay = self.log_decay.exp().clamp(max=5.0)
        temp_bias = -(decay.view(-1, 1, 1) * delta.unsqueeze(0))

        norm_gap = (gap_lengths.float() / self.max_gap).clamp(0.0, 1.0)
        obs_conf = (
            patch_obs_mask.float() * self.obs_bonus.mean()
            - (1.0 - patch_obs_mask.float()) * self.miss_penalty.mean() * norm_gap
        )

        conf_bias = obs_conf.unsqueeze(1).unsqueeze(1)
        bias = temp_bias.unsqueeze(0) + conf_bias
        # Clamp to prevent attention logit overflow after many gradient steps.
        # T5 uses fp32 for its own position bias; our additive bias must stay
        # in a range where softmax remains numerically stable.
        return bias.clamp(min=-10.0, max=10.0)


def _compute_gap_lengths(patch_obs_mask: torch.Tensor) -> torch.Tensor:
    """Vectorised computation of consecutive missing patch run lengths."""
    B, P = patch_obs_mask.shape
    device = patch_obs_mask.device
    obs = patch_obs_mask.bool()
    mis = ~obs

    obs_cumsum = obs.long().cumsum(dim=1)

    n_ids = P + 1
    flat_ids = obs_cumsum.view(B * P) + torch.arange(B, device=device).repeat_interleave(P) * n_ids
    mis_flat = mis.view(B * P).long()

    counts = torch.zeros(B * n_ids, dtype=torch.long, device=device)
    counts.scatter_add_(0, flat_ids, mis_flat)
    counts = counts.view(B, n_ids)

    run_len = counts.gather(dim=1, index=obs_cumsum)
    gap = run_len * mis.long()
    return gap.float()


class ForecastConsistencyLoss(nn.Module):
    """Consistency loss for augmented mask views."""

    def __init__(self, aug_rate: float = 0.12, weight: float = 1.0):
        super().__init__()
        self.aug_rate = aug_rate
        self.weight = weight

    def augment_mask(self, x_enc, input_mask):
        """Vectorised mask augmentation: drop aug_rate fraction of observed positions."""
        B, n_ch, S = x_enc.shape

        obs_count = input_mask.sum(dim=1, keepdim=True).clamp(min=1)
        n_drop = (obs_count * self.aug_rate).long().clamp(min=0)

        noise = torch.rand(B, S, device=x_enc.device)
        noise = torch.where(input_mask.bool(), noise, torch.ones_like(noise))

        sorted_idx = torch.argsort(noise, dim=1)
        rank = torch.arange(S, device=x_enc.device).unsqueeze(0)
        drop_mask = (rank < n_drop).long()

        drop_orig = torch.zeros(B, S, dtype=torch.long, device=x_enc.device)
        drop_orig.scatter_(1, sorted_idx, drop_mask)

        mask_aug = (input_mask.long() & ~drop_orig.bool()).float()
        x_aug = x_enc * mask_aug.unsqueeze(1)
        return x_aug, mask_aug

    def forward(self, pred_orig, pred_aug, drop_frac=0.12):
        diff = F.mse_loss(pred_orig, pred_aug.detach(), reduction="none")
        scale = 1.0 + drop_frac * 3.0
        return diff.mean() * scale  # weight applied externally in compute_crs_loss


class GapConditionedAuxReconHead(nn.Module):
    """Auxiliary reconstruction head for gap-conditioned self-supervised task."""

    def __init__(self, d_model: int, patch_len: int, hidden_dim: int | None = None, weight: float = 0.5):
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.weight = weight
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, patch_len),
        )

    def select_recon_patch(self, x_patched, patch_obs_mask):
        """Vectorised selection of reconstruction target patch."""
        B, n_ch, P, L = x_patched.shape
        device = x_patched.device

        noise = torch.rand(B, P, device=device)
        noise = torch.where(patch_obs_mask.bool(), noise, torch.full_like(noise, 1e9))
        pos_sel = noise.argmin(dim=1)

        idx_exp = pos_sel.view(B, 1, 1, 1).expand(B, n_ch, 1, L)
        target_patches = x_patched.gather(2, idx_exp).squeeze(2)

        masked_patched = x_patched.clone()
        masked_patch_obs = patch_obs_mask.clone()

        b_idx = torch.arange(B, device=device)
        pos_exp = pos_sel.view(B, 1, 1).expand(B, n_ch, L)
        masked_patched.scatter_(2, pos_exp.unsqueeze(2), torch.zeros_like(target_patches).unsqueeze(2))

        masked_patch_obs[b_idx, pos_sel] = 0.0

        return masked_patched, masked_patch_obs, target_patches, pos_sel

    def forward(self, enc_hidden, recon_pos, targets):
        B, n_ch, P, d = enc_hidden.shape

        pos_idx = recon_pos.view(B, 1, 1, 1).expand(B, n_ch, 1, d)
        h_sel = enc_hidden.gather(dim=2, index=pos_idx).squeeze(2)
        recon = self.head(h_sel)
        loss = F.mse_loss(recon, targets.detach())
        return loss  # weight applied externally in compute_crs_loss


@dataclass
class PLUSOutputs:
    """Output container for BackboneLFPlus forward pass."""

    forecast: torch.Tensor
    input_mask: torch.Tensor
    consistency_loss: torch.Tensor | None = None
    aux_recon_loss: torch.Tensor | None = None

    def total_loss(self, targets, consistency_weight=1.0, aux_recon_weight=0.5):
        loss = F.mse_loss(self.forecast, targets)
        if self.consistency_loss is not None:
            loss = loss + consistency_weight * self.consistency_loss
        if self.aux_recon_loss is not None:
            loss = loss + aux_recon_weight * self.aux_recon_loss
        return loss


class BackboneLFPlus(BackboneLF):
    """Backbone-LF+ with four novel modules for improved imputation."""

    def __init__(self, config, **kwargs):
        if not isinstance(config, PLUSConfig):
            if isinstance(config, dict):
                config = PLUSConfig(**config)
            else:
                cfg_dict = {k: v for k, v in vars(config).items() if not k.startswith("_")}
                config = PLUSConfig(**cfg_dict)

        super().__init__(config, **kwargs)

        if config.getattr("use_adaptive_revin", True):
            self.normalizer = MissingnessAdaptiveRevIN(
                num_features=config.n_channels,  # must match C so _global_stdev shape is [1,C,1]
                affine=config.getattr("revin_affine", False),
                threshold=config.getattr("adaptive_revin_threshold", 0.5),
            )

        self._use_obs_attn = config.getattr("use_obs_aware_attn", True)
        if self._use_obs_attn:
            n_heads = config.getattr("n_heads", config.getattr("lie_n_heads", 4))
            self.obs_attn_bias = ObservationAwareAttentionBias(
                n_heads=n_heads,
                max_gap=config.getattr("obs_attn_max_gap", 32),
            )

        self._use_consistency = config.getattr("use_consistency_loss", True)
        if self._use_consistency:
            self.consistency_loss_fn = ForecastConsistencyLoss(
                aug_rate=config.getattr("consistency_aug_rate", 0.12),
                weight=config.getattr("consistency_weight", 1.0),
            )

        self._use_aux_recon = config.getattr("use_aux_recon", True)
        if self._use_aux_recon:
            self.aux_recon_head = GapConditionedAuxReconHead(
                d_model=config.d_model,
                patch_len=config.patch_len,
                weight=config.getattr("aux_recon_weight", 0.5),
            )

        self.plus_config = config

    def _encode_plus(self, x_enc, input_mask, patch_mask=None, patch_times=None, return_patched=False):
        if patch_mask is None:
            patch_mask = torch.ones_like(input_mask)

        B, n_ch, _ = x_enc.shape

        x_patched = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_patched, mask=patch_mask)
        n_patches = enc_in.shape[2]

        enc_in = enc_in.reshape(B * n_ch, n_patches, self.config.d_model)

        if self.flow_bridge is not None:
            # Use max-over-patch (any observed → patch observed) for consistency
            # with the rest of the missingness-aware pipeline.
            pm_view = Masking.convert_seq_to_patch_view_max(patch_mask, self.patch_len)
            enc_in = self.flow_bridge(enc_in, pm_view, n_channels=n_ch, causal_mode=self.causal_flow)

        # max-over-patch: patch is observed if ANY timestep in it is observed.
        # The library's convert_seq_to_patch_view uses min (any missing → patch
        # missing), which marks ~83% of patches as missing for 20% point-missing
        # and 100% for 60% point-missing → all-masked T5 attention → NaN.
        _pm_raw = input_mask.reshape(input_mask.shape[0], -1, self.patch_len)
        pv_mask = _pm_raw.max(dim=-1).values
        attn_msk = pv_mask.repeat_interleave(n_ch, dim=0)

        if self._use_obs_attn:
            patch_obs = pv_mask
            gap_lens = _compute_gap_lengths(patch_obs)
            d_bias = self.obs_attn_bias(n_patches, patch_obs, gap_lens)
            d_bias = d_bias.repeat_interleave(n_ch, dim=0)
        else:
            d_bias = None

        outputs = self._run_encoder_with_bias(enc_in, attn_msk, patch_times, d_bias)
        enc_out = outputs.last_hidden_state.reshape(B, n_ch, n_patches, self.config.d_model)

        if return_patched:
            return enc_out, x_patched
        return enc_out

    def _run_encoder_with_bias(self, enc_in, attn_msk, patch_times, d_bias):
        encoder = self.encoder

        if d_bias is None:
            if isinstance(encoder, LieEquivariantEncoderWrapper):
                return encoder(enc_in, attention_mask=attn_msk, patch_times=patch_times)
            if self.config.getattr("transformer_type", "encoder_only") == "encoder_decoder":
                return encoder(inputs_embeds=enc_in, decoder_inputs_embeds=enc_in, attention_mask=attn_msk)
            return encoder(inputs_embeds=enc_in, attention_mask=attn_msk)

        raw_encoder = encoder.encoder if isinstance(encoder, LieEquivariantEncoderWrapper) else encoder
        hooks = []

        if hasattr(raw_encoder, "block"):
            for block in raw_encoder.block:
                try:
                    attn_module = block.layer[0].SelfAttention
                except (AttributeError, IndexError):
                    continue

                def make_hook(mod):
                    def hook(module, inputs, output):
                        if not isinstance(output, tuple) or len(output) < 2:
                            return output
                        pb = output[1]
                        if pb is not None:
                            # T5 returned a position bias — add our obs bias to it.
                            # d_bias: [B, n_heads, P, P]; pb: [1, n_heads, P, P]
                            # broadcast is safe as long as non-batch dims match.
                            if pb.shape[1:] == d_bias.shape[1:]:
                                return (output[0], pb + d_bias) + output[2:]
                            return output
                        # pb is None: this T5 block did not emit a position-bias
                        # tensor (output[1] doesn't exist).  We cannot inject
                        # d_bias into the attention scores in this case, so return
                        # output unchanged.  The obs-attn bias is still applied in
                        # the blocks that DO emit position bias (the first T5 block
                        # computes it; subsequent blocks receive it as pb ≠ None).
                        return output

                    return hook

                h = attn_module.register_forward_hook(make_hook(attn_module))
                hooks.append(h)

        try:
            if isinstance(encoder, LieEquivariantEncoderWrapper):
                out = encoder(enc_in, attention_mask=attn_msk, patch_times=patch_times)
            elif self.config.getattr("transformer_type", "encoder_only") == "encoder_decoder":
                out = encoder(inputs_embeds=enc_in, decoder_inputs_embeds=enc_in, attention_mask=attn_msk)
            else:
                out = encoder(inputs_embeds=enc_in, attention_mask=attn_msk)
        finally:
            for h in hooks:
                h.remove()

        return out

    def forward_plus(self, x_enc, input_mask=None, missing_rate=0.0, return_aux=False, **kwargs):
        if input_mask is None:
            input_mask = torch.ones_like(x_enc[:, 0, :])

        if isinstance(self.normalizer, MissingnessAdaptiveRevIN):
            self.normalizer.set_missing_rate(missing_rate)

        # Fix 3: generate the augmented mask NOW, before any normalisation or
        # mask-token injection, so the consistency loss compares two views of
        # raw observed data rather than two versions of an already-processed signal.
        aug_mask_for_consistency = None
        if return_aux and self._use_consistency:
            _, aug_mask_for_consistency = self.consistency_loss_fn.augment_mask(x_enc, input_mask)

        x_norm = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        # Snapshot main-view statistics before anything else can overwrite them.
        _revin_mean  = self.normalizer.mean.clone()   # [B, C, 1]
        _revin_stdev = self.normalizer.stdev.clone()  # [B, C, 1]
        x_norm = torch.nan_to_num(x_norm, nan=0, posinf=0, neginf=0)

        aux_recon_loss = None
        x_for_enc = x_norm
        mask_for_enc = input_mask

        if return_aux and self._use_aux_recon:
            x_patched = self.tokenizer(x=x_norm)
            _pm_aux = input_mask.reshape(input_mask.shape[0], -1, self.patch_len)
            pv_mask = _pm_aux.max(dim=-1).values

            (x_patched_masked, pv_mask_masked, target_patches, recon_pos) = self.aux_recon_head.select_recon_patch(
                x_patched, pv_mask
            )

            B2, n_ch2, P2, L2 = x_patched_masked.shape
            x_for_enc = x_patched_masked.reshape(B2, n_ch2, P2 * L2)
            mask_for_enc = pv_mask_masked.repeat_interleave(self.patch_len, dim=1)

        enc = self._encode_plus(x_for_enc, mask_for_enc, patch_times=kwargs.get("patch_times"))

        if return_aux and self._use_aux_recon:
            aux_recon_loss = self.aux_recon_head(enc, recon_pos, target_patches)

        forecast = self.normalizer(x=self.head(enc), mode="denorm")

        # Fix 3 (cont.): normalise the augmented view using the SAME statistics
        # computed for the main view (normalizer already stored them).  This is
        # correct because RevIN statistics are detached from the graph.
        consistency_loss = None
        if return_aux and self._use_consistency and aug_mask_for_consistency is not None:
            # Normalise augmented view with main-view statistics (already snapshotted
            # above).  Do NOT call mode="norm" here — that would overwrite self.mean /
            # self.stdev and trigger a second EMA update, causing denorm to use the
            # wrong scale for both pred_aug and any subsequent call.
            x_aug_raw = x_enc * aug_mask_for_consistency.unsqueeze(1)
            x_aug_norm = (x_aug_raw - _revin_mean) / (_revin_stdev + self.normalizer.eps)
            x_aug_norm = torch.nan_to_num(x_aug_norm, nan=0, posinf=0, neginf=0)
            with torch.no_grad():
                enc_aug = self._encode_plus(x_aug_norm, aug_mask_for_consistency,
                                            patch_times=kwargs.get("patch_times"))
                pred_aug = self.normalizer(x=self.head(enc_aug), mode="denorm")

            consistency_loss = self.consistency_loss_fn(
                forecast,
                pred_aug,
                drop_frac=self.plus_config.getattr("consistency_aug_rate", 0.12),
            )

        return PLUSOutputs(
            forecast=forecast,
            input_mask=input_mask,
            consistency_loss=consistency_loss,
            aux_recon_loss=aux_recon_loss,
        )

    def forecast(self, *, x_enc, input_mask=None, **kwargs):
        out = self.forward_plus(
            x_enc=x_enc,
            input_mask=input_mask,
            missing_rate=kwargs.get("missing_rate", 0.0),
            return_aux=self.training,
            **{k: v for k, v in kwargs.items() if k != "missing_rate"},
        )
        return TimeseriesOutputs(
            forecast=out.forecast,
            input_mask=out.input_mask,
            metadata={
                "consistency_loss": (out.consistency_loss.item() if out.consistency_loss is not None else None),
                "aux_recon_loss": (out.aux_recon_loss.item() if out.aux_recon_loss is not None else None),
            },
        )

    def forward(
        self,
        x_enc,
        input_mask=None,
        missing_rate=0.0,
        return_aux=False,
        channel_mask=None,       # accepted but ignored for non-CRS
        valid_channel_mask=None, # accepted but ignored for non-CRS
        **kwargs,
    ):
        """Forward wrapper. channel_mask and valid_channel_mask are accepted but
        ignored — they exist so the same training loop works for both
        BackboneLFPlus and BackboneLFPlusCRS without branching."""
        return self.forward_plus(
            x_enc=x_enc,
            input_mask=input_mask,
            missing_rate=missing_rate,
            return_aux=return_aux,
            **kwargs,
        )

    def compute_training_loss(self, x_enc, targets, input_mask, missing_rate=0.0):
        out = self.forward_plus(
            x_enc=x_enc,
            input_mask=input_mask,
            missing_rate=missing_rate,
            return_aux=True,
        )
        total = out.total_loss(
            targets,
            consistency_weight=self.plus_config.getattr("consistency_weight", 1.0),
            aux_recon_weight=self.plus_config.getattr("aux_recon_weight", 0.5),
        )
        loss_dict = {
            "pred": F.mse_loss(out.forecast, targets),
            "consistency": out.consistency_loss
            if out.consistency_loss is not None
            else torch.zeros(1, device=x_enc.device),
            "aux_recon": out.aux_recon_loss if out.aux_recon_loss is not None else torch.zeros(1, device=x_enc.device),
        }
        return total, loss_dict

    @property
    def trainable_params(self) -> dict:
        base = super().trainable_params

        def n(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        base["obs_attn_bias"] = n(self.obs_attn_bias) if self._use_obs_attn else 0
        base["consistency"] = n(self.consistency_loss_fn) if self._use_consistency else 0
        base["aux_recon_head"] = n(self.aux_recon_head) if self._use_aux_recon else 0
        base["total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return base


# =============================================================================
# CRS extension
# =============================================================================

class CrossChannelAttention(nn.Module):
    """Cross-channel attention applied independently at each patch position.

    Fix B: A learnable scalar residual gate (``residual_gate``) controls how
    strongly the cross-channel attention output is blended into the encoder
    representations.  It is initialised to -3, so sigmoid(-3) ≈ 0.047 — the
    cross-channel signal starts at ~5 % strength and grows only as the
    attention weights prove useful.  This prevents randomly-initialised
    attention from polluting the pretrained per-channel representations early
    in training.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1, n_channels: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Fix B: soft gate — sigmoid(-3) ≈ 0.047 at init, learned to grow if helpful.
        self.residual_gate = nn.Parameter(torch.tensor(-3.0))
        self.last_attention: torch.Tensor | None = None

        # Fix 4: learnable channel identity embeddings.
        # Without these, CRS must infer which channel is which purely from
        # content — unreliable after per-channel RevIN makes all channels
        # similarly scaled.  A dedicated embedding gives each channel a stable
        # learned identity independent of its current values.
        self._n_channels = n_channels
        self.channel_embed = nn.Embedding(n_channels, d_model)
        # Initialise to near-zero so the embedding starts as a small perturbation
        # and grows only as training finds it useful.
        nn.init.normal_(self.channel_embed.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        channel_observed: torch.Tensor | None = None,
        valid_channel_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        # x: [B, C, P, D]
        batch_size, n_channels, n_patches, d_model = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(batch_size * n_patches, n_channels, d_model)

        # Fix 4: add channel identity embeddings so CRS knows which channel is
        # which regardless of their current values (which look similar after RevIN).
        ch_ids = torch.arange(n_channels, device=x.device)
        # channel_embed may have been trained with a different n_channels cap;
        # clamp ids to avoid index-out-of-bounds when n_channels < _n_channels.
        ch_ids = ch_ids.clamp(max=self._n_channels - 1)
        ch_emb = self.channel_embed(ch_ids)  # [C, D]
        x_flat = x_flat + ch_emb.unsqueeze(0)  # broadcast over [B*P, C, D]

        key_padding_mask = None
        obs_flat = None
        if channel_observed is not None:
            if channel_observed.shape != (batch_size, n_channels, n_patches):
                raise ValueError(
                    "channel_observed must have shape "
                    f"{(batch_size, n_channels, n_patches)}, got {tuple(channel_observed.shape)}"
                )
            obs = channel_observed.to(device=x.device, dtype=torch.float32).clamp(0.0, 1.0)
            obs_flat = obs.permute(0, 2, 1).reshape(batch_size * n_patches, n_channels)
            key_padding_mask = obs_flat <= 0.0
            all_missing = key_padding_mask.all(dim=1)
            if torch.any(all_missing):
                key_padding_mask[all_missing, 0] = False

        attn_out, attn_weights = self.attn(
            x_flat,
            x_flat,
            x_flat,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        if return_attention and attn_weights is not None:
            self.last_attention = (
                attn_weights.reshape(batch_size, n_patches, self.n_heads, n_channels, n_channels)
                .permute(0, 2, 1, 3, 4)
                .detach()
            )
        else:
            self.last_attention = None

        # Fix B: gate the cross-channel residual so the module starts as a
        # near-identity and learns to contribute only when beneficial.
        gate = torch.sigmoid(self.residual_gate)
        out = self.norm(x_flat + gate * self.dropout(attn_out))
        out = out.reshape(batch_size, n_patches, n_channels, d_model).permute(0, 2, 1, 3)

        # Fix 2: restore original embeddings for padding channels so their
        # zero-filled representations don't pollute the residual stream.
        # valid_channel_mask [B, C]: 1=real channel, 0=padding.
        if valid_channel_mask is not None:
            gate_vcm = valid_channel_mask.to(dtype=x.dtype, device=x.device).view(
                batch_size, n_channels, 1, 1
            )
            out = x + (out - x) * gate_vcm

        return out


class MissingnessResidualAdapter(nn.Module):
    """Small mask-conditioned residual adapter with an exact no-op clean path."""

    def __init__(
        self,
        d_model: int,
        hidden_ratio: float = 0.25,
        dropout: float = 0.1,
        gate_mode: str = "any",
    ):
        super().__init__()
        if gate_mode not in {"any", "target", "target_or_aux"}:
            raise ValueError(f"Unsupported missingness adapter gate_mode: {gate_mode!r}")
        self.gate_mode = gate_mode
        hidden_dim = max(16, int(d_model * hidden_ratio))
        self.mask_proj = nn.Linear(2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(hidden_dim, d_model)
        self.gate_logit = nn.Parameter(torch.zeros(()))

        # Start as an identity mapping while still allowing gradients into up.*.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self,
        x: torch.Tensor,
        channel_observed: torch.Tensor,
        valid_channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: [B, C, P, D], channel_observed: [B, C, P] with 1=observed.
        batch_size, n_channels, n_patches, _ = x.shape
        observed = channel_observed.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)

        if valid_channel_mask is None:
            valid = torch.ones((batch_size, n_channels), dtype=x.dtype, device=x.device)
        else:
            valid = valid_channel_mask.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
            if valid.shape != (batch_size, n_channels):
                raise ValueError(
                    f"valid_channel_mask must have shape {(batch_size, n_channels)}, got {tuple(valid.shape)}"
                )

        valid_patch = valid.unsqueeze(-1)
        denom = valid_patch.sum(dim=1, keepdim=True).clamp(min=1.0)
        missing_fraction = ((1.0 - observed) * valid_patch).sum(dim=1, keepdim=True) / denom

        effective_missing = torch.where(valid_patch > 0.0, 1.0 - observed, torch.zeros_like(observed))
        mask_features = torch.stack(
            (
                effective_missing,
                missing_fraction.expand(-1, n_channels, -1),
            ),
            dim=-1,
        )

        hidden = self.norm(x + self.mask_proj(mask_features))
        residual = self.up(self.dropout(self.act(self.down(hidden))))
        if self.gate_mode == "target":
            gate_signal = effective_missing[:, :1, :]
        elif self.gate_mode == "target_or_aux":
            target_missing = effective_missing[:, :1, :]
            if n_channels > 1:
                aux_valid = valid[:, 1:].unsqueeze(-1)
                aux_denom = aux_valid.sum(dim=1, keepdim=True).clamp(min=1.0)
                aux_missing_fraction = ((1.0 - observed[:, 1:]) * aux_valid).sum(dim=1, keepdim=True) / aux_denom
            else:
                aux_missing_fraction = torch.zeros_like(target_missing)
            target_gate = torch.maximum(target_missing, aux_missing_fraction)
            gate_signal = torch.zeros_like(effective_missing)
            gate_signal[:, :1, :] = target_gate
        else:
            gate_signal = missing_fraction
        gate = gate_signal.unsqueeze(-1) * torch.sigmoid(self.gate_logit)
        return x + residual * gate * valid.view(batch_size, n_channels, 1, 1)


class CRSConfig(PLUSConfig):
    """PLUSConfig extended with cross-channel attention flags."""

    use_cross_channel_attn: bool = True
    cross_channel_n_heads: int = 4
    cross_channel_dropout: float = 0.1
    # Fix 3: enable by default — CRS now knows which channels are corrupted.
    use_missingness_residual_adapter: bool = True
    missingness_adapter_hidden_ratio: float = 0.25
    missingness_adapter_dropout: float = 0.1
    missingness_adapter_gate: str = "any"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        defaults = {
            "use_cross_channel_attn": True,
            "cross_channel_n_heads": 4,
            "cross_channel_dropout": 0.1,
            "use_missingness_residual_adapter": True,  # Fix 3
            "missingness_adapter_hidden_ratio": 0.25,
            "missingness_adapter_dropout": 0.1,
            "missingness_adapter_gate": "any",
        }
        for key, value in defaults.items():
            if not hasattr(self, key):
                setattr(self, key, value)


class BackboneLFPlusCRS(BackboneLFPlus):
    """Backbone-LF+ plus a per-patch cross-channel attention block."""

    def __init__(self, config, **kwargs):
        if not isinstance(config, CRSConfig):
            if isinstance(config, dict):
                config = CRSConfig(**config)
            else:
                cfg_dict = {key: value for key, value in vars(config).items() if not key.startswith("_")}
                config = CRSConfig(**cfg_dict)

        super().__init__(config, **kwargs)

        # Fix 5: learnable mask token replaces zero-fill for missing positions.
        # Shape [1, 1, 1] broadcasts to [B, C, L] at forward time.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1))

        self._use_cross_channel_attn = config.getattr("use_cross_channel_attn", True)
        self.last_channel_attention: torch.Tensor | None = None
        if self._use_cross_channel_attn:
            self.cross_channel_attn = CrossChannelAttention(
                d_model=config.d_model,
                n_heads=config.getattr("cross_channel_n_heads", 4),
                dropout=config.getattr("cross_channel_dropout", config.getattr("dropout", 0.1)),
                # Fix 4: pass n_channels so channel_embed is sized correctly.
                n_channels=config.getattr("n_channels", 64),
            )

        self._use_missingness_residual_adapter = config.getattr("use_missingness_residual_adapter", False)
        if self._use_missingness_residual_adapter:
            self.missingness_adapter = MissingnessResidualAdapter(
                d_model=config.d_model,
                hidden_ratio=config.getattr("missingness_adapter_hidden_ratio", 0.25),
                dropout=config.getattr("missingness_adapter_dropout", config.getattr("dropout", 0.1)),
                gate_mode=config.getattr("missingness_adapter_gate", "any"),
            )

        self.crs_config = config

    @staticmethod
    def _seq_to_patch_max(mask: torch.Tensor, patch_len: int) -> torch.Tensor:
        """Max-over-patch: patch is observed if ANY timestep in it is observed.

        Consistent with the max rule used in _encode_plus and model.py.
        The library's Masking.convert_seq_to_patch_view uses min (all must be
        observed), which at 20% point-missing marks ~83% of patches as missing
        and causes near-empty cross-channel attention key sets.
        """
        return mask.reshape(*mask.shape[:-1], -1, patch_len).max(dim=-1).values

    def _channel_patch_mask(
        self,
        channel_mask: torch.Tensor | None,
        input_mask: torch.Tensor,
        n_channels: int,
    ) -> torch.Tensor:
        if channel_mask is None:
            patch_mask = self._seq_to_patch_max(input_mask, self.patch_len)
            return patch_mask.unsqueeze(1).expand(-1, n_channels, -1)

        patch_mask = self._seq_to_patch_max(channel_mask, self.patch_len)
        if patch_mask.ndim == 2:
            return patch_mask.unsqueeze(1).expand(-1, n_channels, -1)
        if patch_mask.ndim != 3:
            raise ValueError(f"Expected channel_mask shape [B, C, L], got {tuple(channel_mask.shape)}")
        return patch_mask

    def _encode_plus(
        self,
        x_enc,
        input_mask,
        patch_mask=None,
        patch_times=None,
        return_patched=False,
        channel_mask: torch.Tensor | None = None,
        valid_channel_mask: torch.Tensor | None = None,
        return_channel_attention: bool = False,
        crs_channel_mask: torch.Tensor | None = None,
    ):
        """Encode x_enc and optionally apply cross-channel attention.

        Args:
            channel_mask: Per-channel observed mask ``[B, C, L]`` used for the
                T5 encoder's obs-attn-bias.  During aux-recon training this is
                the *reduced* mask (reconstruction patches zeroed out).
            crs_channel_mask: Per-channel observed mask used **only** for the
                cross-channel attention module.  Should always be the *original*
                (un-reduced) channel mask so that the cross-channel attention
                sees consistent missingness patterns between training and
                inference (Fix A — aux-recon train/inference mismatch).
                Defaults to ``channel_mask`` when not provided.
        """
        if channel_mask is not None and channel_mask.ndim == 3:
            B, C, L = channel_mask.shape
            per_ch_input_mask = channel_mask.reshape(B * C, L).clamp(0.0, 1.0)
            x_flat = x_enc.reshape(B * C, 1, L)
            emb_mask = patch_mask if patch_mask is not None else per_ch_input_mask
            result_flat = super()._encode_plus(
                x_flat,
                per_ch_input_mask,
                patch_mask=emb_mask,
                patch_times=patch_times,
                return_patched=return_patched,
            )
            if return_patched:
                enc_flat, x_patched_flat = result_flat
                n_patches = enc_flat.shape[2]
                enc_out = enc_flat.reshape(B, C, n_patches, enc_flat.shape[-1])
                x_patched = x_patched_flat.reshape(B, C, n_patches, x_patched_flat.shape[-1])
            else:
                enc_flat = result_flat
                n_patches = enc_flat.shape[2]
                enc_out = enc_flat.reshape(B, C, n_patches, enc_flat.shape[-1])
        else:
            result = super()._encode_plus(
                x_enc,
                input_mask,
                patch_mask=patch_mask,
                patch_times=patch_times,
                return_patched=return_patched,
            )
            if return_patched:
                enc_out, x_patched = result
            else:
                enc_out = result

        # Fix A: cross-channel attention always uses the *original* channel mask
        # (crs_channel_mask), not the aux-recon-reduced one (channel_mask).
        effective_crs_mask = crs_channel_mask if crs_channel_mask is not None else channel_mask

        self.last_channel_attention = None
        channel_observed = None
        if self._use_cross_channel_attn or self._use_missingness_residual_adapter:
            channel_observed = self._channel_patch_mask(effective_crs_mask, input_mask, enc_out.shape[1])

        if self._use_cross_channel_attn:
            enc_out = self.cross_channel_attn(
                enc_out,
                channel_observed=channel_observed,
                valid_channel_mask=valid_channel_mask,
                return_attention=return_channel_attention,
            )
            self.last_channel_attention = self.cross_channel_attn.last_attention

        if self._use_missingness_residual_adapter:
            enc_out = self.missingness_adapter(
                enc_out,
                channel_observed=channel_observed,
                valid_channel_mask=valid_channel_mask,
            )

        if return_patched:
            return enc_out, x_patched
        return enc_out

    def forward_plus(
        self,
        x_enc,
        input_mask=None,
        channel_mask: torch.Tensor | None = None,
        valid_channel_mask: torch.Tensor | None = None,
        missing_rate=0.0,
        return_aux=False,
        return_channel_attention: bool = False,
        **kwargs,
    ):
        if input_mask is None:
            input_mask = torch.ones_like(x_enc[:, 0, :])
        if channel_mask is None:
            channel_mask = input_mask.unsqueeze(1).expand(-1, x_enc.shape[1], -1)
        if valid_channel_mask is None:
            valid_channel_mask = torch.ones(
                (x_enc.shape[0], x_enc.shape[1]),
                dtype=x_enc.dtype,
                device=x_enc.device,
            )

        if isinstance(self.normalizer, MissingnessAdaptiveRevIN):
            self.normalizer.set_missing_rate(missing_rate)

        # Fix 5 (consistency): generate augmented mask NOW from raw x_enc,
        # BEFORE normalisation and mask-token injection.
        aug_mask_for_consistency = None
        if return_aux and self._use_consistency:
            _, aug_mask_for_consistency = self.consistency_loss_fn.augment_mask(x_enc, input_mask)

        # Fix C: pass a per-channel mask [B, C, L] to RevIN so that
        # zero-fill at missing positions in channel c is excluded from c's own
        # mean/std computation.
        if channel_mask is not None and channel_mask.ndim == 3:
            revin_mask = (channel_mask * valid_channel_mask.unsqueeze(-1)).to(dtype=input_mask.dtype)
        else:
            revin_mask = input_mask

        x_norm = self.normalizer(x=x_enc, mask=revin_mask, mode="norm")
        # Snapshot main-view statistics before anything else can overwrite them.
        _revin_mean  = self.normalizer.mean.clone()   # [B, C, 1]
        _revin_stdev = self.normalizer.stdev.clone()  # [B, C, 1]
        x_norm = torch.nan_to_num(x_norm, nan=0, posinf=0, neginf=0)
        x_norm = x_norm.clamp(min=-10.0, max=10.0)

        # Fix 5 (mask token): replace zero-fill at missing positions with a
        # learnable mask token.
        if channel_mask is not None and channel_mask.shape == x_norm.shape:
            missing = (1.0 - channel_mask.to(dtype=x_norm.dtype, device=x_norm.device))
            x_norm = x_norm + missing * self.mask_token

        aux_recon_loss = None
        x_for_enc = x_norm
        mask_for_enc = input_mask
        channel_mask_for_enc = channel_mask

        if return_aux and self._use_aux_recon:
            x_patched = self.tokenizer(x=x_norm)
            pv_mask = (
                input_mask
                .reshape(input_mask.shape[0], -1, self.patch_len)
                .max(dim=-1).values
            )

            x_patched_masked, pv_mask_masked, target_patches, recon_pos = self.aux_recon_head.select_recon_patch(
                x_patched, pv_mask
            )

            batch_size, n_channels, n_patches, patch_len = x_patched_masked.shape
            x_for_enc = x_patched_masked.reshape(batch_size, n_channels, n_patches * patch_len)
            mask_for_enc = pv_mask_masked.repeat_interleave(self.patch_len, dim=1)
            channel_mask_for_enc = channel_mask * mask_for_enc.unsqueeze(1)

        enc = self._encode_plus(
            x_for_enc,
            mask_for_enc,
            patch_times=kwargs.get("patch_times"),
            channel_mask=channel_mask_for_enc,
            valid_channel_mask=valid_channel_mask,
            return_channel_attention=return_channel_attention,
            crs_channel_mask=channel_mask,   # Fix A: always the original mask
        )

        if return_aux and self._use_aux_recon:
            aux_recon_loss = self.aux_recon_head(enc, recon_pos, target_patches)

        forecast = self.normalizer(x=self.head(enc), mode="denorm")

        consistency_loss = None
        if return_aux and self._use_consistency and aug_mask_for_consistency is not None:
            x_aug_raw = x_enc * aug_mask_for_consistency.unsqueeze(1)
            if channel_mask is not None and channel_mask.ndim == 3:
                aug_channel_mask = channel_mask * aug_mask_for_consistency.unsqueeze(1)
                revin_aug_mask = (aug_channel_mask * valid_channel_mask.unsqueeze(-1)).to(dtype=input_mask.dtype)
            else:
                aug_channel_mask = channel_mask
                revin_aug_mask = aug_mask_for_consistency
            # Reuse main-view statistics — do NOT call mode="norm" (would overwrite
            # self.mean/stdev and trigger a second EMA update this batch).
            x_aug_norm = (x_aug_raw - _revin_mean) / (_revin_stdev + self.normalizer.eps)
            x_aug_norm = torch.nan_to_num(x_aug_norm, nan=0, posinf=0, neginf=0)
            x_aug_norm = x_aug_norm.clamp(min=-10.0, max=10.0)
            if aug_channel_mask is not None and aug_channel_mask.shape == x_aug_norm.shape:
                aug_missing = (1.0 - aug_channel_mask.to(dtype=x_aug_norm.dtype, device=x_aug_norm.device))
                x_aug_norm = x_aug_norm + aug_missing * self.mask_token
            with torch.no_grad():
                enc_aug = self._encode_plus(
                    x_aug_norm,
                    aug_mask_for_consistency,
                    patch_times=kwargs.get("patch_times"),
                    channel_mask=aug_channel_mask,
                    valid_channel_mask=valid_channel_mask,
                )
                pred_aug = self.normalizer(x=self.head(enc_aug), mode="denorm")

            consistency_loss = self.consistency_loss_fn(
                forecast,
                pred_aug,
                drop_frac=self.plus_config.getattr("consistency_aug_rate", 0.12),
            )

        return PLUSOutputs(
            forecast=forecast,
            input_mask=input_mask,
            consistency_loss=consistency_loss,
            aux_recon_loss=aux_recon_loss,
        )

    def forward(
        self,
        x_enc,
        input_mask=None,
        channel_mask: torch.Tensor | None = None,
        valid_channel_mask: torch.Tensor | None = None,
        missing_rate=0.0,
        return_aux=False,
        return_channel_attention: bool = False,
        **kwargs,
    ):
        """DDP-friendly forward wrapper around forward_plus."""
        return self.forward_plus(
            x_enc=x_enc,
            input_mask=input_mask,
            channel_mask=channel_mask,
            valid_channel_mask=valid_channel_mask,
            missing_rate=missing_rate,
            return_aux=return_aux,
            return_channel_attention=return_channel_attention,
            **kwargs,
        )

    def forecast(self, *, x_enc, input_mask=None, **kwargs):
        out = self.forward_plus(
            x_enc=x_enc,
            input_mask=input_mask,
            channel_mask=kwargs.get("channel_mask"),
            valid_channel_mask=kwargs.get("valid_channel_mask"),
            missing_rate=kwargs.get("missing_rate", 0.0),
            return_aux=self.training,
            return_channel_attention=kwargs.get("return_channel_attention", False),
            **{
                key: value
                for key, value in kwargs.items()
                if key not in {"missing_rate", "channel_mask", "valid_channel_mask", "return_channel_attention"}
            },
        )
        metadata = {
            "consistency_loss": (out.consistency_loss.item() if out.consistency_loss is not None else None),
            "aux_recon_loss": (out.aux_recon_loss.item() if out.aux_recon_loss is not None else None),
        }
        if self.last_channel_attention is not None:
            metadata["channel_attention"] = self.last_channel_attention
        return TimeseriesOutputs(
            forecast=out.forecast,
            input_mask=out.input_mask,
            metadata=metadata,
        )

    def compute_training_loss(
        self,
        x_enc,
        targets,
        input_mask,
        channel_mask=None,
        valid_channel_mask=None,
        missing_rate=0.0,
    ):
        out = self.forward_plus(
            x_enc=x_enc,
            input_mask=input_mask,
            channel_mask=channel_mask,
            valid_channel_mask=valid_channel_mask,
            missing_rate=missing_rate,
            return_aux=True,
        )
        total = out.total_loss(
            targets,
            consistency_weight=self.plus_config.getattr("consistency_weight", 1.0),
            aux_recon_weight=self.plus_config.getattr("aux_recon_weight", 0.5),
        )
        loss_dict = {
            "pred": F.mse_loss(out.forecast, targets),
            "consistency": out.consistency_loss
            if out.consistency_loss is not None
            else torch.zeros(1, device=x_enc.device),
            "aux_recon": out.aux_recon_loss if out.aux_recon_loss is not None else torch.zeros(1, device=x_enc.device),
        }
        return total, loss_dict

    @property
    def trainable_params(self) -> dict:
        base = super().trainable_params
        if self._use_cross_channel_attn:
            base["cross_channel_attn"] = sum(
                param.numel() for param in self.cross_channel_attn.parameters() if param.requires_grad
            )
            base["total"] = sum(param.numel() for param in self.parameters() if param.requires_grad)
        if self._use_missingness_residual_adapter:
            base["missingness_adapter"] = sum(
                param.numel() for param in self.missingness_adapter.parameters() if param.requires_grad
            )
            base["total"] = sum(param.numel() for param in self.parameters() if param.requires_grad)
        return base


# =============================================================================
# Factory functions
# =============================================================================

def build_backbone_lfplus(args, n_channels: int, device: torch.device) -> BackboneLFPlus:
    """Factory: build BackboneLFPlus (non-CRS) from args."""
    ABLATION_CONFIGS = {
        "baseline": dict(use_lie_flow_embedding=False, use_flow_matching_latent=False, use_lie_equivariant_enc=False),
        "A": dict(use_lie_flow_embedding=True, use_flow_matching_latent=False, use_lie_equivariant_enc=False),
        "B": dict(use_lie_flow_embedding=False, use_flow_matching_latent=True, use_lie_equivariant_enc=False),
        "C": dict(use_lie_flow_embedding=False, use_flow_matching_latent=False, use_lie_equivariant_enc=True),
        "AB": dict(use_lie_flow_embedding=True, use_flow_matching_latent=True, use_lie_equivariant_enc=False),
        "AC": dict(use_lie_flow_embedding=True, use_flow_matching_latent=False, use_lie_equivariant_enc=True),
        "BC": dict(use_lie_flow_embedding=False, use_flow_matching_latent=True, use_lie_equivariant_enc=True),
        "full": dict(use_lie_flow_embedding=True, use_flow_matching_latent=True, use_lie_equivariant_enc=True),
    }

    _d_model = getattr(args, "d_model", 128)
    _d_ff = getattr(args, "d_ff", None) or (_d_model * 4)
    _feed_forward_proj = getattr(args, "feed_forward_proj", None)
    t5_config = {
        "d_model": _d_model,
        "d_ff": _d_ff,
        "num_heads": getattr(args, "n_heads", 4),
        "num_layers": getattr(args, "enc_layers", 3),
        "dropout_rate": getattr(args, "dropout", 0.1),
        "is_encoder_decoder": False,
        "vocab_size": 1,
    }
    if _feed_forward_proj:
        t5_config["feed_forward_proj"] = _feed_forward_proj
    else:
        t5_config["dense_act_fn"] = getattr(args, "dense_act_fn", "gelu_new")

    ablation_flags = ABLATION_CONFIGS[getattr(args, "ablation", "baseline")]

    config = PLUSConfig(
        task_name=TASKS.FORECASTING,
        forecast_horizon=args.pred_len,
        seq_len=args.seq_len,
        patch_len=args.patch_len,
        patch_stride_len=args.patch_len,
        d_model=getattr(args, "d_model", 128),
        transformer_backbone="google/flan-t5-small",
        transformer_type="encoder_only",
        t5_config=t5_config,
        n_channels=n_channels,
        freeze_embedder=False,
        freeze_encoder=False,
        freeze_head=False,
        lie_enc_mode=getattr(args, "lie_enc_mode", "adapter"),
        flow_solver=getattr(args, "flow_solver", "rk4"),
        flow_n_steps=getattr(args, "flow_n_steps", 6),
        lie_adapter_rank=getattr(args, "lie_adapter_rank", 16),
        lie_n_heads=getattr(args, "n_heads", 4),
        causal_flow=True,
        use_full_lie_bias=False,
        add_positional_embedding=True,
        value_embedding_bias=False,
        patch_dropout=getattr(args, "dropout", 0.1),
        head_dropout=getattr(args, "dropout", 0.1),
        orth_gain=1.41,
        revin_affine=False,
        mask_ratio=0.0,
        enable_gradient_checkpointing=False,
        randomly_initialize_backbone=True,
        use_obs_aware_attn=getattr(args, "use_obs_aware_attn", True),
        use_consistency_loss=getattr(args, "use_consistency_loss", True),
        use_aux_recon=getattr(args, "use_aux_recon", True),
        use_adaptive_revin=getattr(args, "use_adaptive_revin", True),
        obs_attn_max_gap=getattr(args, "obs_attn_max_gap", 32),
        consistency_aug_rate=getattr(args, "consistency_aug_rate", 0.12),
        consistency_weight=getattr(args, "consistency_weight", 1.0),
        aux_recon_weight=getattr(args, "aux_recon_weight", 0.5),
        adaptive_revin_threshold=getattr(args, "adaptive_revin_threshold", 0.5),
        **ablation_flags,
    )

    return BackboneLFPlus(config).to(device)


def build_backbone_lfplus_crs(args, n_channels: int, device: torch.device) -> BackboneLFPlusCRS:
    """Factory: build BackboneLFPlusCRS (with cross-channel attention) from args."""
    ablation_configs = {
        "baseline": dict(use_lie_flow_embedding=False, use_flow_matching_latent=False, use_lie_equivariant_enc=False),
        "A": dict(use_lie_flow_embedding=True, use_flow_matching_latent=False, use_lie_equivariant_enc=False),
        "B": dict(use_lie_flow_embedding=False, use_flow_matching_latent=True, use_lie_equivariant_enc=False),
        "C": dict(use_lie_flow_embedding=False, use_flow_matching_latent=False, use_lie_equivariant_enc=True),
        "AB": dict(use_lie_flow_embedding=True, use_flow_matching_latent=True, use_lie_equivariant_enc=False),
        "AC": dict(use_lie_flow_embedding=True, use_flow_matching_latent=False, use_lie_equivariant_enc=True),
        "BC": dict(use_lie_flow_embedding=False, use_flow_matching_latent=True, use_lie_equivariant_enc=True),
        "full": dict(use_lie_flow_embedding=True, use_flow_matching_latent=True, use_lie_equivariant_enc=True),
    }

    d_model = getattr(args, "d_model", 128)
    feed_forward_proj = getattr(args, "feed_forward_proj", None)
    t5_config = {
        "d_model": d_model,
        "d_kv": getattr(args, "d_kv", 64),
        "d_ff": getattr(args, "d_ff", None) or d_model * 4,
        "num_heads": getattr(args, "n_heads", 4),
        "num_layers": getattr(args, "enc_layers", 3),
        "dropout_rate": getattr(args, "dropout", 0.1),
        "is_encoder_decoder": False,
        "vocab_size": 1,
    }
    if feed_forward_proj:
        t5_config["feed_forward_proj"] = feed_forward_proj
    else:
        t5_config["dense_act_fn"] = getattr(args, "dense_act_fn", "gelu_new")

    ablation_flags = ablation_configs[getattr(args, "ablation", "baseline")]

    config = CRSConfig(
        task_name=TASKS.FORECASTING,
        forecast_horizon=args.pred_len,
        seq_len=args.seq_len,
        patch_len=args.patch_len,
        patch_stride_len=args.patch_len,
        d_model=d_model,
        transformer_backbone=getattr(args, "transformer_backbone", "google/flan-t5-small"),
        transformer_type="encoder_only",
        t5_config=t5_config,
        n_channels=n_channels,
        freeze_embedder=getattr(args, "freeze_embedder", False),
        freeze_encoder=getattr(args, "freeze_encoder", False),
        freeze_head=getattr(args, "freeze_head", False),
        lie_enc_mode=getattr(args, "lie_enc_mode", "adapter"),
        flow_solver=getattr(args, "flow_solver", "rk4"),
        flow_n_steps=getattr(args, "flow_n_steps", 6),
        lie_adapter_rank=getattr(args, "lie_adapter_rank", 16),
        lie_n_heads=getattr(args, "n_heads", 4),
        causal_flow=True,
        use_full_lie_bias=False,
        add_positional_embedding=True,
        value_embedding_bias=False,
        patch_dropout=getattr(args, "dropout", 0.1),
        head_dropout=getattr(args, "dropout", 0.1),
        orth_gain=1.41,
        revin_affine=False,
        mask_ratio=0.0,
        enable_gradient_checkpointing=False,
        randomly_initialize_backbone=getattr(args, "randomly_initialize_backbone", False),
        use_obs_aware_attn=getattr(args, "use_obs_aware_attn", True),
        use_consistency_loss=getattr(args, "use_consistency_loss", True),
        use_aux_recon=getattr(args, "use_aux_recon", True),
        use_adaptive_revin=getattr(args, "use_adaptive_revin", True),
        obs_attn_max_gap=getattr(args, "obs_attn_max_gap", 32),
        consistency_aug_rate=getattr(args, "consistency_aug_rate", 0.12),
        consistency_weight=getattr(args, "consistency_weight", 1.0),
        aux_recon_weight=getattr(args, "aux_recon_weight", 0.1),
        adaptive_revin_threshold=getattr(args, "adaptive_revin_threshold", 0.5),
        use_cross_channel_attn=getattr(args, "use_cross_channel_attn", True),
        cross_channel_n_heads=getattr(args, "cross_channel_n_heads", 4),
        cross_channel_dropout=getattr(args, "cross_channel_dropout", getattr(args, "dropout", 0.1)),
        use_missingness_residual_adapter=getattr(args, "use_missingness_residual_adapter", True),
        missingness_adapter_hidden_ratio=getattr(args, "missingness_adapter_hidden_ratio", 0.25),
        missingness_adapter_dropout=getattr(
            args,
            "missingness_adapter_dropout",
            getattr(args, "dropout", 0.1),
        ),
        missingness_adapter_gate=getattr(args, "missingness_adapter_gate", "any"),
        **ablation_flags,
    )

    return BackboneLFPlusCRS(config).to(device)


def build_model(args, n_channels: int, device: torch.device):
    """Unified factory: returns BackboneLFPlusCRS if args.use_crs else BackboneLFPlus."""
    if getattr(args, "use_crs", False):
        return build_backbone_lfplus_crs(args, n_channels, device)
    return build_backbone_lfplus(args, n_channels, device)


__all__ = [
    "BackboneLFPlus",
    "BackboneLFPlusCRS",
    "CRSConfig",
    "CrossChannelAttention",
    "ForecastConsistencyLoss",
    "GapConditionedAuxReconHead",
    "MissingnessAdaptiveRevIN",
    "MissingnessResidualAdapter",
    "ObservationAwareAttentionBias",
    "PLUSConfig",
    "PLUSOutputs",
    "_compute_gap_lengths",
    "build_backbone_lfplus",
    "build_backbone_lfplus_crs",
    "build_model",
]
