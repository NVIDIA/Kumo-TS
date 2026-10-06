# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference-time loader for Backbone-LF+ / Backbone-LF+ CRS checkpoints.

Single source of truth for turning a checkpoint folder (``best_model.pt`` +
``config.json`` / ``config_base.json``) back into an eval-mode model. Used by
the SDK's ``handle_missingness`` path.

Architecture is rebuilt from ``config.json`` and then corrected from the
checkpoint weights themselves (T5 ``d_ff`` / FFN type / heads / layers, and
which optional modules exist), so older configs with missing or stale fields
still load cleanly.

Checkpoints can be local (a ``.pt`` file or its folder) or on the Hugging Face
Hub as ``hf://<org>/<repo>[@revision][/subfolder]``; the released checkpoint
is ``hf://nvidia/Kumo-Forecast/kumo-forecast-1.2.0`` (``DEFAULT_IMPUTE_CKPT``). The config
file next to the weights may be named ``config.json`` or ``config_base.json``
(the trimmed, inference-only config published with the release).
"""

from __future__ import annotations

import json
import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .backbone_lieflow import TASKS
from .model import BackboneLFPlus, BackboneLFPlusCRS, CRSConfig, MissingnessAdaptiveRevIN, PLUSConfig

logger = logging.getLogger(__name__)


@contextmanager
def _quiet_head_warning():
    """The backbone warns that only its reconstruction head is pre-trained; here every weight (including the
    forecasting head) is loaded from the checkpoint right after the model is built, so the warning is misleading."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Only reconstruction head is pre-trained")
        yield


CHECKPOINT_FILENAMES = ("best_model.pt", "model.pt", "checkpoint.pt")
CONFIG_FILENAMES = ("config.json", "config_base.json")  # searched in this order
CONFIG_FILENAME = CONFIG_FILENAMES[0]
DATASET_METADATA_FILENAME = "dataset_metadata.json"

HF_PREFIX = "hf://"
# Files fetched from a Hub folder: the weights and the inference config. allow_patterns only
# filters, so a repo holding just best_model.pt + config_base.json downloads exactly those two.
HF_CHECKPOINT_FILES = ("best_model.pt", *CONFIG_FILENAMES)
DEFAULT_IMPUTE_REPO = "nvidia/Kumo-Forecast"
DEFAULT_IMPUTE_SUBFOLDER = "kumo-forecast-1.2.0"  # missingness-aware release (best_model.pt + config_base.json)
DEFAULT_IMPUTE_CKPT = f"{HF_PREFIX}{DEFAULT_IMPUTE_REPO}/{DEFAULT_IMPUTE_SUBFOLDER}"
# Values of the config "model" field that mean the cross-channel (CRS) architecture. The published
# kumo-forecast-1.2.0 config predates the Backbone* class names, so the legacy name stays accepted.
_CRS_MODEL_NAMES = ("BackboneLFPlusCRS", "MOMENTLFPlusCRS")

# Buffers added to MissingnessAdaptiveRevIN after some checkpoints were trained.
LEGACY_KEYS = frozenset({"normalizer._global_stdev", "normalizer._global_stdev_initialized"})

ABLATION_FLAGS: dict[str, dict[str, bool]] = {
    "baseline": dict(use_lie_flow_embedding=False, use_flow_matching_latent=False, use_lie_equivariant_enc=False),
    "A": dict(use_lie_flow_embedding=True, use_flow_matching_latent=False, use_lie_equivariant_enc=False),
    "B": dict(use_lie_flow_embedding=False, use_flow_matching_latent=True, use_lie_equivariant_enc=False),
    "C": dict(use_lie_flow_embedding=False, use_flow_matching_latent=False, use_lie_equivariant_enc=True),
    "AB": dict(use_lie_flow_embedding=True, use_flow_matching_latent=True, use_lie_equivariant_enc=False),
    "AC": dict(use_lie_flow_embedding=True, use_flow_matching_latent=False, use_lie_equivariant_enc=True),
    "BC": dict(use_lie_flow_embedding=False, use_flow_matching_latent=True, use_lie_equivariant_enc=True),
    "full": dict(use_lie_flow_embedding=True, use_flow_matching_latent=True, use_lie_equivariant_enc=True),
}


@dataclass
class ImputeModelInfo:
    """Metadata describing a loaded impute checkpoint."""

    checkpoint_path: Path
    config: dict[str, Any]
    seq_len: int
    pred_len: int
    patch_len: int
    trained_n_channels: int
    is_crs: bool
    channel_embed_cap: int | None = None
    target_col: str | None = None
    channels: list[str] | None = None
    epoch: int | None = None
    n_channels: int = 0  # channel count the model is currently adapted to
    trained_global_stdev: torch.Tensor | None = field(default=None, repr=False)
    # Real column name fed to each model channel slot during training (slot 0 = target),
    # or None when the checkpoint doesn't record it.
    channel_schema: list[str] | None = None
    # Per-source-CSV training standardizers: [{"name", "csv_path", "channels", "mean", "std"}].
    source_standardizers: list[dict[str, Any]] = field(default_factory=list)

    def find_standardizer(self, name: str | None = None) -> dict[str, Any]:
        """Return the saved training standardizer for source dataset ``name``.

        ``name`` matches the source CSV's file stem (e.g. ``"ETTh1"``), file name,
        or full path. With ``name=None`` the checkpoint must have exactly one source.
        """
        if not self.source_standardizers:
            raise ValueError("The impute checkpoint does not record any training standardizers")
        if name is None:
            if len(self.source_standardizers) == 1:
                return self.source_standardizers[0]
            raise ValueError(
                "The impute checkpoint was trained on several datasets; pass impute_source_dataset to pick one of: "
                f"{[s['name'] for s in self.source_standardizers]}"
            )
        for entry in self.source_standardizers:
            path = entry.get("csv_path") or ""
            if name in {entry["name"], Path(path).name, path}:
                return entry
        raise ValueError(
            f"impute_source_dataset={name!r} not found in the checkpoint; available: "
            f"{[s['name'] for s in self.source_standardizers]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Path / file helpers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HFReference:
    """Parsed ``hf://<org>/<repo>[@revision][/subfolder]`` checkpoint reference."""

    repo_id: str
    revision: str | None = None
    subfolder: str = ""

    def __str__(self) -> str:
        rev = f"@{self.revision}" if self.revision else ""
        sub = f"/{self.subfolder}" if self.subfolder else ""
        return f"{HF_PREFIX}{self.repo_id}{rev}{sub}"


def is_hf_reference(ckpt: str | Path) -> bool:
    return isinstance(ckpt, str) and ckpt.startswith(HF_PREFIX)


def parse_hf_reference(ref: str) -> HFReference:
    """Parse ``hf://org/repo[@revision][/sub/folder]``."""
    if not is_hf_reference(ref):
        raise ValueError(f"Not a Hugging Face reference (expected {HF_PREFIX}org/repo[@rev][/subfolder]): {ref!r}")
    parts = [p for p in ref[len(HF_PREFIX) :].split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Hugging Face reference needs <org>/<repo>: {ref!r}")
    org, repo = parts[0], parts[1]
    revision = None
    if "@" in repo:
        repo, revision = repo.split("@", 1)
        if not repo or not revision:
            raise ValueError(f"Invalid revision in Hugging Face reference: {ref!r}")
    if "@" in org or any("@" in p for p in parts[2:]):
        raise ValueError(f"'@revision' must follow the repo name: {ref!r}")
    return HFReference(repo_id=f"{org}/{repo}", revision=revision, subfolder="/".join(parts[2:]))


def _hf_error_message(ref: HFReference, error: Exception, local_files_only: bool) -> str:
    text = str(error)
    name = type(error).__name__
    if local_files_only or name == "LocalEntryNotFoundError":
        return (
            f"Impute checkpoint {ref} is not in the local Hugging Face cache and local_files_only=True. "
            "Download it once with network access (or unset local_files_only)."
        )
    if name in {"GatedRepoError", "RepositoryNotFoundError"} or "401" in text or "403" in text:
        return (
            f"Cannot access impute checkpoint {ref} ({name}). If the repository is private or gated, log in with "
            "`huggingface-cli login` or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN; also check the repo name."
        )
    if name == "RevisionNotFoundError":
        return f"Revision {ref.revision!r} not found for impute checkpoint {ref}."
    return f"Failed to download impute checkpoint {ref}: {error}"


def download_hf_checkpoint(
    ref: str | HFReference,
    *,
    local_files_only: bool = False,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Fetch only the inference files for ``ref`` into the Hugging Face cache; return the local folder."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover - huggingface_hub is a project dependency
        raise ImportError(
            "huggingface_hub is required for hf:// impute checkpoints: pip install huggingface_hub"
        ) from e

    ref = parse_hf_reference(ref) if isinstance(ref, str) else ref
    prefix = f"{ref.subfolder}/" if ref.subfolder else ""
    wanted = HF_CHECKPOINT_FILES
    try:
        snapshot = snapshot_download(
            repo_id=ref.repo_id,
            revision=ref.revision,
            allow_patterns=[prefix + name for name in wanted],
            local_files_only=local_files_only,
            token=token,
            cache_dir=cache_dir,
            library_name="kumo-ts",
        )
    except Exception as e:
        raise RuntimeError(_hf_error_message(ref, e, local_files_only)) from e

    folder = Path(snapshot) / ref.subfolder if ref.subfolder else Path(snapshot)
    if not (folder / "best_model.pt").is_file():
        raise FileNotFoundError(
            f"Impute checkpoint {ref} has no weights file (best_model.pt) in {ref.subfolder or 'the repo root'}"
        )
    if not any((folder / name).is_file() for name in CONFIG_FILENAMES):
        raise FileNotFoundError(
            f"Impute checkpoint {ref} has no config ({' or '.join(CONFIG_FILENAMES)}) next to the weights"
        )
    return folder


def resolve_checkpoint_path(
    ckpt: str | Path,
    *,
    local_files_only: bool = False,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Accept a ``.pt`` file, a folder containing one of ``CHECKPOINT_FILENAMES``, or an ``hf://`` reference."""
    if is_hf_reference(ckpt):
        ckpt = download_hf_checkpoint(ckpt, local_files_only=local_files_only, token=token, cache_dir=cache_dir)
    path = Path(ckpt).expanduser()
    if path.is_dir():
        for name in CHECKPOINT_FILENAMES:
            if (path / name).is_file():
                return path / name
        raise FileNotFoundError(f"No checkpoint file ({', '.join(CHECKPOINT_FILENAMES)}) found in directory {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Impute checkpoint not found: {path}")
    return path


def load_config_json(checkpoint_path: Path, config: str | Path | dict | None = None) -> dict[str, Any]:
    """Load the checkpoint config (``config.json``, else ``config_base.json``, next to the checkpoint)."""
    if isinstance(config, dict):
        return dict(config)
    if config is not None:
        config_path = Path(config)
    else:
        candidates = [checkpoint_path.parent / name for name in CONFIG_FILENAMES]
        config_path = next((c for c in candidates if c.is_file()), candidates[0])
    if not config_path.is_file():
        raise FileNotFoundError(
            f"config.json not found at {config_path} (config_base.json is also accepted). The impute checkpoint "
            "folder must contain the config written by the training pipeline next to the .pt file."
        )
    with open(config_path, encoding="utf-8") as fh:
        return json.load(fh)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # older checkpoints may pickle non-tensor objects
        logger.warning("weights_only load failed for %s; retrying with weights_only=False (trusted file)", path)
        return torch.load(path, map_location="cpu", weights_only=False)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture detection
# ─────────────────────────────────────────────────────────────────────────────


def _t5_prefix(state_dict: dict) -> str:
    # LieEquivariantEncoderWrapper (ablation C) nests the T5 stack under `encoder.encoder.`
    return "encoder.encoder." if any(k.startswith("encoder.encoder.block.") for k in state_dict) else "encoder."


def detect_t5_arch_from_state_dict(state_dict: dict, d_model: int) -> tuple[int, str | None, int | None, int | None]:
    """Infer ``(d_ff, feed_forward_proj, n_heads, enc_layers)`` from checkpoint weights.

    ``n_heads`` / ``enc_layers`` are ``None`` when they can't be inferred.
    """
    prefix = _t5_prefix(state_dict)
    wo_key = f"{prefix}block.0.layer.1.DenseReluDense.wo.weight"
    wi0_key = f"{prefix}block.0.layer.1.DenseReluDense.wi_0.weight"
    rel_bias_key = f"{prefix}block.0.layer.0.SelfAttention.relative_attention_bias.weight"

    if wo_key in state_dict:
        d_ff = int(state_dict[wo_key].shape[1])
        ffn_proj = "gated-gelu" if wi0_key in state_dict else None
    else:
        d_ff, ffn_proj = d_model * 4, None

    if "obs_attn_bias.log_decay" in state_dict:
        n_heads = int(state_dict["obs_attn_bias.log_decay"].shape[0])
    elif rel_bias_key in state_dict:
        n_heads = int(state_dict[rel_bias_key].shape[1])  # [num_buckets, n_heads]
    else:
        n_heads = None

    enc_layers = (
        sum(1 for k in state_dict if k.startswith(f"{prefix}block.") and k.endswith(".layer.0.layer_norm.weight"))
        or None
    )
    return d_ff, ffn_proj, n_heads, enc_layers


def detect_t5_d_kv(state_dict: dict, n_heads: int) -> int | None:
    """Infer the T5 per-head key/value size from the first query projection ``[n_heads * d_kv, d_model]``."""
    q_key = f"{_t5_prefix(state_dict)}block.0.layer.0.SelfAttention.q.weight"
    if q_key not in state_dict or n_heads <= 0:
        return None
    inner = int(state_dict[q_key].shape[0])
    return inner // n_heads if inner % n_heads == 0 else None


def _has_prefix(state_dict: dict, prefix: str) -> bool:
    return any(k.startswith(prefix) for k in state_dict)


def _trained_n_channels(config: dict, state_dict: dict) -> int:
    if "normalizer._global_stdev" in state_dict:
        return int(state_dict["normalizer._global_stdev"].shape[1])
    if "cross_channel_attn.channel_embed.weight" in state_dict:
        return int(state_dict["cross_channel_attn.channel_embed.weight"].shape[0])
    meta = config.get("meta") or {}
    return int(meta.get("n_channels") or config.get("max_channels") or 7)


def build_model_from_config(
    config: dict,
    n_channels: int,
    device: torch.device | str = "cpu",
    state_dict: dict | None = None,
) -> BackboneLFPlus:
    """Build an (un-loaded) BackboneLFPlus / BackboneLFPlusCRS matching ``config`` and ``state_dict``."""
    is_crs = bool(config.get("use_crs", False) or config.get("model") in _CRS_MODEL_NAMES)
    d_model = int(config.get("d_model", 128))

    if state_dict is not None:
        d_ff, ffp, det_heads, det_layers = detect_t5_arch_from_state_dict(state_dict, d_model)
        logger.info(
            "Detected arch from checkpoint: d_ff=%d ffp=%s n_heads=%s enc_layers=%s",
            d_ff,
            ffp or "relu",
            det_heads,
            det_layers,
        )
    else:
        d_ff = config.get("d_ff") or d_model * 4
        ffp = config.get("feed_forward_proj")
        det_heads = det_layers = None

    n_heads = det_heads or config.get("n_heads", 4)
    # The non-CRS train factory never passes d_kv (T5 default 64) even though
    # config.json records args.d_kv, so prefer the weights when available.
    d_kv = (detect_t5_d_kv(state_dict, n_heads) if state_dict is not None else None) or config.get("d_kv", 64)
    t5_config: dict[str, Any] = {
        "d_model": d_model,
        "d_kv": d_kv,
        "d_ff": d_ff,
        "num_heads": n_heads,
        "num_layers": det_layers or config.get("enc_layers", 3),
        "dropout_rate": config.get("dropout", 0.1),
        "is_encoder_decoder": False,
        "vocab_size": 1,
    }
    if ffp:
        t5_config["feed_forward_proj"] = ffp
    else:
        t5_config["dense_act_fn"] = config.get("dense_act_fn", "gelu_new")

    # Optional modules: trust the weights when we have them, else the config.
    def _flag(key: str, prefix: str, default: bool) -> bool:
        if state_dict is not None:
            return _has_prefix(state_dict, prefix)
        return bool(config.get(key, default))

    ablation = ABLATION_FLAGS.get(config.get("ablation", "baseline"), ABLATION_FLAGS["baseline"]).copy()
    if state_dict is not None:
        ablation["use_lie_flow_embedding"] = _has_prefix(state_dict, "patch_embedding.lie_flow.")
        ablation["use_flow_matching_latent"] = _has_prefix(state_dict, "flow_bridge.")
        ablation["use_lie_equivariant_enc"] = _has_prefix(state_dict, "encoder.encoder.")

    common = dict(
        task_name=TASKS.FORECASTING,
        forecast_horizon=config.get("pred_len", 24),
        seq_len=config.get("seq_len", 96),
        patch_len=config.get("patch_len", 8),
        patch_stride_len=config.get("patch_len", 8),
        d_model=d_model,
        transformer_backbone=config.get("transformer_backbone", "google/flan-t5-small"),
        transformer_type="encoder_only",
        t5_config=t5_config,
        n_channels=n_channels,
        freeze_embedder=False,
        freeze_encoder=False,
        freeze_head=False,
        lie_enc_mode=config.get("lie_enc_mode", "adapter"),
        flow_solver=config.get("flow_solver", "rk4"),
        flow_n_steps=config.get("flow_n_steps", 6),
        lie_adapter_rank=config.get("lie_adapter_rank", 16),
        lie_n_heads=n_heads,
        causal_flow=True,
        use_full_lie_bias=False,
        add_positional_embedding=True,
        value_embedding_bias=False,
        patch_dropout=config.get("dropout", 0.1),
        head_dropout=config.get("dropout", 0.1),
        orth_gain=1.41,
        revin_affine=False,
        mask_ratio=0.0,
        enable_gradient_checkpointing=False,
        randomly_initialize_backbone=True,  # weights come from the checkpoint; never download
        use_obs_aware_attn=_flag("use_obs_aware_attn", "obs_attn_bias.", True),
        use_consistency_loss=False,  # training-only, parameter-free
        use_aux_recon=_flag("use_aux_recon", "aux_recon_head.", True),
        use_adaptive_revin=config.get("use_adaptive_revin", True),
        obs_attn_max_gap=config.get("obs_attn_max_gap", 32),
        consistency_aug_rate=config.get("consistency_aug_rate", 0.12),
        consistency_weight=config.get("consistency_weight", 1.0),
        aux_recon_weight=config.get("aux_recon_weight", 0.5),
        adaptive_revin_threshold=config.get("adaptive_revin_threshold", 0.5),
        **ablation,
    )

    if is_crs:
        cfg = CRSConfig(
            **common,
            use_crs=True,
            use_cross_channel_attn=_flag("use_cross_channel_attn", "cross_channel_attn.", True),
            cross_channel_n_heads=config.get("cross_channel_n_heads", 4),
            cross_channel_dropout=config.get("cross_channel_dropout", config.get("dropout", 0.1)),
            use_missingness_residual_adapter=_flag("use_missingness_residual_adapter", "missingness_adapter.", True),
            missingness_adapter_hidden_ratio=config.get("missingness_adapter_hidden_ratio", 0.25),
            missingness_adapter_dropout=config.get("missingness_adapter_dropout", config.get("dropout", 0.1)),
            missingness_adapter_gate=config.get("missingness_adapter_gate", "any"),
        )
        with _quiet_head_warning():
            return BackboneLFPlusCRS(cfg).to(device)
    with _quiet_head_warning():
        return BackboneLFPlus(PLUSConfig(**common)).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Channel-count adaptation
# ─────────────────────────────────────────────────────────────────────────────


def set_channel_count(model: BackboneLFPlus, info: ImputeModelInfo, n_channels: int) -> None:
    """Adapt the model to ``n_channels`` input channels (in place).

    The backbone is channel-independent; the only channel-count-shaped state
    is the adaptive RevIN's learned ``_global_stdev`` prior ``[1, C_trained, 1]``
    (used when missing_rate > threshold). It is rebuilt from the trained values:
    channels that exist in training keep their prior, extra channels get the
    mean prior. The CRS channel embedding needs no change (ids are clamped).
    """
    if n_channels < 1:
        raise ValueError(f"n_channels must be >= 1, got {n_channels}")
    info.n_channels = n_channels
    normalizer = getattr(model, "normalizer", None)
    if not isinstance(normalizer, MissingnessAdaptiveRevIN) or info.trained_global_stdev is None:
        return

    trained = info.trained_global_stdev  # [1, C_trained, 1] on CPU
    c_trained = trained.shape[1]
    if n_channels <= c_trained:
        new = trained[:, :n_channels, :].clone()
    else:
        fill = trained.mean(dim=1, keepdim=True).expand(1, n_channels - c_trained, 1)
        new = torch.cat([trained, fill], dim=1)

    device = normalizer._global_stdev.device
    normalizer._global_stdev = new.to(device=device, dtype=normalizer._global_stdev.dtype)
    normalizer.num_features = n_channels


# ─────────────────────────────────────────────────────────────────────────────
# Training schema / standardizers (from config.json meta or dataset_metadata.json)
# ─────────────────────────────────────────────────────────────────────────────


def _dataset_metadata(config: dict, checkpoint_dir: Path) -> dict[str, Any]:
    meta = config.get("meta") or {}
    if isinstance(meta.get("dataset_metadata"), dict):
        return meta["dataset_metadata"]
    path = checkpoint_dir / DATASET_METADATA_FILENAME
    if path.is_file():
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    return {}


def _training_schema(
    config: dict, checkpoint_dir: Path, trained_n_channels: int
) -> tuple[list[str] | None, list[dict[str, Any]]]:
    """Recover the per-slot channel names and per-source standardizers used in training.

    Multi-CSV training pads every source to a name-aligned union
    (``source_union_channels``; slot ``i`` <-> union name ``i``). Single-CSV
    training records the real column names in ``meta.channels``.
    """
    meta = config.get("meta") or {}
    dm = _dataset_metadata(config, checkpoint_dir)

    schema = meta.get("source_union_channels") or dm.get("source_union_channels")
    if not schema:
        names = meta.get("channels") or dm.get("channels")
        # Multi-CSV "model channels" are placeholders (target, aux_1, ...), not real names.
        if names and not any(str(n).startswith("aux_") for n in names[1:]):
            schema = names
    if schema is not None and len(schema) != trained_n_channels:
        logger.warning(
            "Training channel schema has %d names but the checkpoint has %d channels; ignoring the schema",
            len(schema),
            trained_n_channels,
        )
        schema = None

    sources = []
    for entry in dm.get("source_datasets") or dm.get("datasets") or []:
        std = (entry or {}).get("standardizer")
        if not std:
            continue
        path = str(entry.get("csv_path") or "")
        sources.append(
            {
                "name": Path(path).stem if path else f"dataset_{len(sources)}",
                "csv_path": path,
                "channels": list(std.get("channels") or entry.get("channels") or []),
                "mean": [float(v) for v in std["mean"]],
                "std": [float(v) for v in std["std"]],
                "eps": float(std.get("eps", 1e-8)),
            }
        )
    return (list(schema) if schema else None), sources


# ─────────────────────────────────────────────────────────────────────────────
# Public loader
# ─────────────────────────────────────────────────────────────────────────────


def load_impute_model(
    ckpt: str | Path,
    device: torch.device | str = "cpu",
    n_channels: int | None = None,
    config: str | Path | dict | None = None,
    *,
    local_files_only: bool = False,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[BackboneLFPlus, ImputeModelInfo]:
    """Load an impute checkpoint for inference.

    Args:
        ckpt: ``.pt`` file, folder containing ``best_model.pt`` (or ``model.pt`` / ``checkpoint.pt``), or
            ``hf://<org>/<repo>[@revision][/subfolder]`` (e.g. ``DEFAULT_IMPUTE_CKPT``).
        device: Target device.
        n_channels: Number of input channels the caller will feed. ``None`` keeps the trained count.
        config: Optional config path or already-loaded dict (defaults to ``config.json`` /
            ``config_base.json`` next to ``ckpt``).
        local_files_only: For ``hf://`` references, use only the local Hugging Face cache.
        token: Hugging Face token for private/gated repos (defaults to the logged-in token / ``HF_TOKEN``).
        cache_dir: Optional Hugging Face cache directory.

    Returns:
        ``(model, info)`` with the model in eval mode on ``device``.

    Raises:
        FileNotFoundError: checkpoint or config.json missing.
        RuntimeError: checkpoint weights don't match the rebuilt architecture.
    """
    device = torch.device(device)
    checkpoint_path = resolve_checkpoint_path(ckpt, local_files_only=local_files_only, token=token, cache_dir=cache_dir)
    config_dict = load_config_json(checkpoint_path, config)

    checkpoint = _torch_load(checkpoint_path)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None

    trained_c = _trained_n_channels(config_dict, state_dict)
    model = build_model_from_config(config_dict, trained_c, device="cpu", state_dict=state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    bad_missing = [k for k in missing if k not in LEGACY_KEYS]
    bad_unexpected = [k for k in unexpected if k not in LEGACY_KEYS]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"Impute checkpoint does not match the rebuilt architecture — "
            f"missing: {bad_missing[:10]}, unexpected: {bad_unexpected[:10]}"
        )
    for key in sorted(set(missing) | set(unexpected)):
        logger.info("Legacy key handled: %s", key)

    is_crs = isinstance(model, BackboneLFPlusCRS)
    cap = None
    if is_crs and getattr(model, "_use_cross_channel_attn", False):
        cap = int(model.cross_channel_attn._n_channels)

    trained_stdev = None
    normalizer = getattr(model, "normalizer", None)
    if isinstance(normalizer, MissingnessAdaptiveRevIN):
        trained_stdev = normalizer._global_stdev.detach().cpu().clone()

    meta = config_dict.get("meta") or {}
    channel_schema, source_standardizers = _training_schema(config_dict, checkpoint_path.parent, trained_c)
    info = ImputeModelInfo(
        checkpoint_path=checkpoint_path,
        config=config_dict,
        seq_len=int(config_dict.get("seq_len", model.seq_len)),
        pred_len=int(config_dict.get("pred_len", 24)),
        patch_len=int(config_dict.get("patch_len", model.patch_len)),
        trained_n_channels=trained_c,
        is_crs=is_crs,
        channel_embed_cap=cap,
        target_col=config_dict.get("target_col"),
        channels=meta.get("channels"),
        epoch=epoch,
        n_channels=trained_c,
        trained_global_stdev=trained_stdev,
        channel_schema=channel_schema,
        source_standardizers=source_standardizers,
    )

    model.to(device).eval()
    if n_channels is not None and n_channels != trained_c:
        set_channel_count(model, info, n_channels)

    logger.info(
        "Loaded impute model %s (epoch %s): seq_len=%d pred_len=%d trained_channels=%d crs=%s",
        type(model).__name__,
        epoch,
        info.seq_len,
        info.pred_len,
        trained_c,
        is_crs,
    )
    return model, info


__all__ = [
    "DEFAULT_IMPUTE_CKPT",
    "HFReference",
    "ImputeModelInfo",
    "build_model_from_config",
    "detect_t5_arch_from_state_dict",
    "detect_t5_d_kv",
    "download_hf_checkpoint",
    "is_hf_reference",
    "load_impute_model",
    "parse_hf_reference",
    "resolve_checkpoint_path",
    "set_channel_count",
]
