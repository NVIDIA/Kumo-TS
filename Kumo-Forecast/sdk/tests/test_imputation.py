# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SDK's missingness-aware path (``handle_missingness=True``).

Most tests build tiny, randomly initialised Backbone-LF+ / Backbone-LF+ CRS
checkpoints with the same factory ``the training pipeline`` uses, so no
real weights are needed. The ``real_checkpoint`` tests run only when
``KUMO_IMPUTE_CKPT`` points at a trained checkpoint (file or folder), and
optionally ``KUMO_IMPUTE_CSV`` at a CSV (e.g. ETTh1.csv) for an accuracy check.
"""

from __future__ import annotations

import json
import logging
import os
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import numpy as np
import pandas as pd
import pytest
import torch
from impute.backbone_lieflow import LieEquivariantEncoderWrapper
from impute.inference import (
    DEFAULT_IMPUTE_CKPT,
    HFReference,
    download_hf_checkpoint,
    load_impute_model,
    parse_hf_reference,
    set_channel_count,
)
from impute.model import BackboneLFPlusCRS, MissingnessAdaptiveRevIN, build_model
from impute.preprocessing import CSVForecastMissingDataset
from sdk import forecasting, imputation

SEQ_LEN = 32
PRED_LEN = 8
PATCH_LEN = 8
TRAINED_CHANNELS = 3


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────


def _train_args(use_crs: bool) -> Namespace:
    """Mirror the subset of train.py args the model factory reads (tiny sizes)."""
    return Namespace(
        use_crs=use_crs,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        patch_len=PATCH_LEN,
        d_model=32,
        d_ff=64,
        d_kv=16,
        n_heads=2,
        enc_layers=1,
        dropout=0.0,
        dense_act_fn="gelu_new",
        ablation="baseline",
        target_col="target",
        use_obs_aware_attn=True,
        use_consistency_loss=True,
        use_aux_recon=True,
        use_adaptive_revin=True,
        use_cross_channel_attn=True,
        cross_channel_n_heads=2,
        use_missingness_residual_adapter=True,
        randomly_initialize_backbone=True,
    )


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("n_layers", [1, 3, 24])
def test_observation_bias_applies_once_before_every_attention(wrapped, n_layers):
    args = _train_args(use_crs=False)
    args.enc_layers = n_layers
    torch.manual_seed(7)
    model = build_model(args, n_channels=TRAINED_CHANNELS, device=torch.device("cpu")).eval()
    raw_encoder = model.encoder
    if wrapped:
        model.encoder = LieEquivariantEncoderWrapper(
            raw_encoder, d_model=32, n_heads=2, head_dim=16, adapter_rank=4
        ).eval()

    enc_in = torch.randn(2, 4, 32)
    mask = torch.tensor([[1, 0, 1, 1], [1, 1, 0, 1]])
    # A nonconstant bias changes the softmax, unlike a uniform offset.
    d_bias = torch.randn(2, 2, 4, 4, requires_grad=True)
    seen = []

    def capture_attention(module, inputs, output):
        seen.append(output[1])

    captures = [block.layer[0].SelfAttention.register_forward_hook(capture_attention) for block in raw_encoder.block]
    try:
        actual = model._run_encoder_with_bias(enc_in, mask, None, d_bias).last_hidden_state
    finally:
        for hook in captures:
            hook.remove()

    # Independent reference: T5 accepts a prepared additive mask, adds it to
    # relative bias before block 0, and shares that bias across all blocks.
    additive_mask = (1 - mask[:, None, None, :].to(enc_in.dtype)) * torch.finfo(enc_in.dtype).min
    expected = model.encoder(inputs_embeds=enc_in, attention_mask=additive_mask + d_bias).last_hidden_state
    torch.testing.assert_close(actual, expected)
    relative_bias = raw_encoder.block[0].layer[0].SelfAttention.compute_bias(4, 4)
    expected_bias = relative_bias + additive_mask + d_bias
    assert len(seen) == n_layers
    for bias in seen:
        torch.testing.assert_close(bias, expected_bias)
    assert all(not block.layer[0].SelfAttention._forward_pre_hooks for block in raw_encoder.block)

    # The injected bias must participate in attention even in a one-block model.
    actual.square().sum().backward()
    assert d_bias.grad is not None
    assert torch.isfinite(d_bias.grad).all()
    assert torch.count_nonzero(d_bias.grad) > 0


@pytest.mark.parametrize("bias", [None, torch.zeros(2, 2, 4, 4)])
def test_observation_bias_disabled_or_zero_preserves_t5_output(bias):
    args = _train_args(use_crs=False)
    args.enc_layers = 3
    model = build_model(args, n_channels=TRAINED_CHANNELS, device=torch.device("cpu")).eval()
    enc_in = torch.randn(2, 4, 32)
    mask = torch.tensor([[1, 0, 1, 1], [1, 1, 0, 1]])
    expected = model.encoder(inputs_embeds=enc_in, attention_mask=mask).last_hidden_state
    actual = model._run_encoder_with_bias(enc_in, mask, None, bias).last_hidden_state
    torch.testing.assert_close(actual, expected)


def _write_checkpoint(folder: Path, use_crs: bool) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    args = _train_args(use_crs)
    torch.manual_seed(0)
    model = build_model(args, n_channels=TRAINED_CHANNELS, device=torch.device("cpu"))
    # Give the adaptive-RevIN prior distinct, "trained" per-channel values.
    model.normalizer._global_stdev.copy_(torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1))
    model.normalizer._global_stdev_initialized.fill_(True)
    torch.save({"epoch": 1, "model_state_dict": model.state_dict()}, folder / "best_model.pt")
    config = vars(args).copy()
    config["meta"] = {"n_channels": TRAINED_CHANNELS, "channels": ["target", "f1", "f2"]}
    config["model"] = "BackboneLFPlusCRS" if use_crs else "BackboneLFPlus"
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return folder


@pytest.fixture(scope="module")
def ckpt_dirs(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("impute_ckpts")
    return {
        "plus": _write_checkpoint(root / "plus", use_crs=False),
        "crs": _write_checkpoint(root / "crs", use_crs=True),
    }


@pytest.fixture(params=["plus", "crs"])
def ckpt(request, ckpt_dirs) -> Path:
    return ckpt_dirs[request.param]


@pytest.fixture(autouse=True)
def _clear_caches():
    imputation.clear_impute_model_cache()
    yield
    imputation.clear_impute_model_cache()


def make_df(
    num_rows: int = 64,
    n_features: int = 2,
    missing_rate: float = 0.2,
    seed: int = 0,
    target: str = "target",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(num_rows, dtype=np.float64)
    data = {
        "timestamp": pd.date_range("2024-01-01", periods=num_rows, freq="h"),
        target: 10.0 + np.sin(t / 5.0) * 3.0,
    }
    for i in range(n_features):
        data[f"f{i + 1}"] = 100.0 * (i + 1) + np.cos(t / (3.0 + i)) * 5.0
    df = pd.DataFrame(data)
    value_cols = [c for c in df.columns if c != "timestamp"]
    mask = rng.random((num_rows, len(value_cols))) < missing_rate
    mask[-1, 0] = False  # keep at least one observed target value
    df[value_cols] = df[value_cols].mask(mask)
    return df


def cfg(ckpt_path: Path | None, **overrides) -> forecasting.ForecastingConfig:
    options = dict(
        seq_len=SEQ_LEN,
        forecast_horizon=PRED_LEN,
        handle_missingness=True,
        impute_ckpt=None if ckpt_path is None else str(ckpt_path),
        device="cpu",
    )
    options.update(overrides)
    return forecasting.ForecastingConfig(**options)


def _capture_forward_calls(ckpt_path: Path) -> list[dict]:
    """Hook the cached model and record every forward call's inputs."""
    model, _ = imputation._get_impute_model(ckpt_path, torch.device("cpu"))
    calls: list[dict] = []

    def hook(_module, args, kwargs):
        calls.append({"x": args[0].detach().clone(), **kwargs})

    model.register_forward_pre_hook(hook, with_kwargs=True)
    return calls


# ─────────────────────────────────────────────────────────────────────────────
# Window / mask construction (no model)
# ─────────────────────────────────────────────────────────────────────────────


def test_build_missingness_window_masks_and_observed_scaling():
    values = np.array(
        [
            [1.0, 10.0],
            [np.nan, 20.0],
            [3.0, np.nan],
            [5.0, 40.0],
        ]
    )
    window = imputation.build_missingness_window(values, seq_len=4)

    np.testing.assert_array_equal(window.channel_mask, [[1, 0, 1, 1], [1, 1, 0, 1]])
    np.testing.assert_array_equal(window.temporal_mask, [1, 1, 1, 1])
    np.testing.assert_array_equal(window.valid_channel_mask, [1, 1])
    assert window.missing_rate == pytest.approx(2 / 8)
    # Stats come from observed values only.
    np.testing.assert_allclose(window.scaler.mean, [3.0, 70.0 / 3.0])
    np.testing.assert_allclose(window.scaler.std, [np.std([1.0, 3.0, 5.0]), np.std([10.0, 20.0, 40.0])])
    # Gaps are zero-filled after scaling; no NaN reaches the model.
    assert np.isfinite(window.x).all()
    assert window.x[0, 1] == 0.0
    assert window.x[1, 2] == 0.0
    # Inverse transform maps scaled values back to original units.
    np.testing.assert_allclose(window.to_output(window.x)[:, 0], [1.0, 10.0])


def test_build_missingness_window_scaler_uses_full_history_not_the_window():
    values = np.column_stack([np.arange(10, dtype=float), np.full(10, np.nan)])
    window = imputation.build_missingness_window(values, seq_len=4)

    # Reference statistics come from every observed row passed in (like the
    # training-split scaler), not just the last seq_len rows.
    np.testing.assert_allclose(window.scaler.mean[0], np.mean(np.arange(10.0)))
    np.testing.assert_allclose(window.x[0], (np.arange(6.0, 10.0) - 4.5) / (np.std(np.arange(10.0)) + 1e-8))
    assert window.scaler.mean[1] == 0.0
    assert window.scaler.std[1] == 1.0
    assert window.channel_mask[1].sum() == 0
    assert window.missing_rate == pytest.approx(0.5)


def test_build_missingness_window_rejects_short_input():
    with pytest.raises(ValueError, match="at least 8 rows"):
        imputation.build_missingness_window(np.ones((4, 2)), seq_len=8)


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────


def test_load_impute_model_from_folder_and_file(ckpt):
    model, info = load_impute_model(ckpt)
    assert not model.training
    assert info.seq_len == SEQ_LEN
    assert info.pred_len == PRED_LEN
    assert info.trained_n_channels == TRAINED_CHANNELS
    assert info.epoch == 1
    assert info.is_crs == isinstance(model, BackboneLFPlusCRS)
    if info.is_crs:
        assert info.channel_embed_cap == TRAINED_CHANNELS

    _, info_from_file = load_impute_model(ckpt / "best_model.pt")
    assert info_from_file.checkpoint_path == ckpt / "best_model.pt"


def test_load_impute_model_detects_architecture_from_weights(ckpt):
    # A stale config must not break loading: architecture comes from the weights.
    config = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    config.update(d_ff=999, n_heads=7, enc_layers=5, d_kv=3)
    model, _ = load_impute_model(ckpt, config=config)
    t5 = model.encoder.config
    assert (t5.d_ff, t5.num_heads, t5.num_layers) == (64, 2, 1)


def test_load_impute_model_missing_config_raises(tmp_path, ckpt):
    (tmp_path / "best_model.pt").write_bytes((ckpt / "best_model.pt").read_bytes())
    with pytest.raises(FileNotFoundError, match=r"config\.json"):
        load_impute_model(tmp_path)


def test_load_impute_model_rejects_mismatched_weights(tmp_path, ckpt):
    state = torch.load(ckpt / "best_model.pt", map_location="cpu", weights_only=True)
    state["model_state_dict"]["unexpected.weight"] = torch.zeros(1)
    torch.save(state, tmp_path / "best_model.pt")
    (tmp_path / "config.json").write_text((ckpt / "config.json").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected"):
        load_impute_model(tmp_path)


@pytest.mark.parametrize(("n_channels", "expected"), [(2, [1.0, 2.0]), (3, [1.0, 2.0, 3.0]), (5, [1, 2, 3, 2, 2])])
def test_set_channel_count_resizes_trained_stdev_prior(ckpt, n_channels, expected):
    model, info = load_impute_model(ckpt)
    set_channel_count(model, info, n_channels)
    np.testing.assert_allclose(model.normalizer._global_stdev.flatten().numpy(), expected)
    # Re-adapting always starts from the trained values.
    set_channel_count(model, info, TRAINED_CHANNELS)
    np.testing.assert_allclose(model.normalizer._global_stdev.flatten().numpy(), [1.0, 2.0, 3.0])


# ─────────────────────────────────────────────────────────────────────────────
# perform_forecasting(handle_missingness=True)
# ─────────────────────────────────────────────────────────────────────────────


def test_handle_missingness_output_matches_default_format(ckpt):
    df = make_df()
    result = forecasting.perform_forecasting(df, config=cfg(ckpt))

    assert list(result.columns) == ["timestamp", "target_forecast"]
    assert len(result) == PRED_LEN
    assert result["timestamp"].iloc[0] == df["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
    assert pd.Series(result["timestamp"]).diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert np.isfinite(result["target_forecast"]).all()


def test_handle_missingness_return_all_channels(ckpt):
    result = forecasting.perform_forecasting(make_df(), config=cfg(ckpt, return_all_channels=True))
    assert list(result.columns) == ["timestamp", "target_forecast", "f1_forecast", "f2_forecast"]


def test_handle_missingness_forecasts_are_in_original_units(ckpt):
    # Features live around 100/200; the target around 10. Forecasts should follow each channel's scale.
    result = forecasting.perform_forecasting(make_df(missing_rate=0.1), config=cfg(ckpt, return_all_channels=True))
    assert result["f2_forecast"].mean() > result["f1_forecast"].mean() > result["target_forecast"].mean()


def test_handle_missingness_passes_masks_and_missing_rate(ckpt):
    df = make_df(missing_rate=0.3, seed=3)
    calls = _capture_forward_calls(ckpt)
    forecasting.perform_forecasting(df, config=cfg(ckpt))

    assert len(calls) == 1
    call = calls[0]
    window = df[["target", "f1", "f2"]].to_numpy()[-SEQ_LEN:]
    observed = ~np.isnan(window.T)

    assert torch.isfinite(call["x"]).all()
    np.testing.assert_array_equal(call["channel_mask"][0].numpy(), observed.astype(np.float32))
    np.testing.assert_array_equal(call["input_mask"][0].numpy(), observed.any(axis=0).astype(np.float32))
    np.testing.assert_array_equal(call["valid_channel_mask"][0].numpy(), np.ones(3, dtype=np.float32))
    assert call["missing_rate"] == pytest.approx(1.0 - observed.mean())
    # Missing positions are zero-filled (== channel mean after scaling).
    assert (call["x"][0].numpy()[~observed] == 0).all()


def test_handle_missingness_autoregressive_rollout(ckpt):
    horizon = 2 * PRED_LEN + 3
    df = make_df()
    calls = _capture_forward_calls(ckpt)
    result = forecasting.perform_forecasting(df, config=cfg(ckpt, forecast_horizon=horizon))

    assert len(result) == horizon
    assert len(calls) == 3
    # Rolled-in predictions are observed: the last PRED_LEN steps of the 2nd window are fully observed,
    # so its missing rate only counts the NaNs of the remaining (SEQ_LEN - PRED_LEN) real rows.
    assert calls[1]["channel_mask"][0, :, -PRED_LEN:].eq(1).all()
    real_rows = df[["target", "f1", "f2"]].to_numpy()[-(SEQ_LEN - PRED_LEN) :]
    assert calls[1]["missing_rate"] == pytest.approx(np.isnan(real_rows).sum() / (SEQ_LEN * 3))


@pytest.mark.parametrize(
    ("n_features", "alignment"), [(0, "name"), (1, "name"), (2, "name"), (0, "positional"), (5, "positional")]
)
def test_handle_missingness_any_channel_count_at_high_missing_rate(ckpt, n_features, alignment):
    # 70% missing engages the adaptive RevIN prior, which depends on the channel count.
    df = make_df(n_features=n_features, missing_rate=0.7, seed=7)
    config = cfg(ckpt, return_all_channels=True, impute_channel_alignment=alignment)
    result = forecasting.perform_forecasting(df, config=config)
    assert result.shape == (PRED_LEN, 2 + n_features)
    assert np.isfinite(result.drop(columns="timestamp").to_numpy()).all()


def test_handle_missingness_warns_past_channel_embedding_cap(ckpt_dirs, caplog):
    config = cfg(ckpt_dirs["crs"], impute_channel_alignment="positional")
    with caplog.at_level(logging.WARNING):
        forecasting.perform_forecasting(make_df(n_features=4), config=config)
    assert "share one identity embedding" in caplog.text


# ─────────────────────────────────────────────────────────────────────────────
# Channel alignment to the training schema (review P2)
# ─────────────────────────────────────────────────────────────────────────────


def test_name_alignment_preserves_channel_identity_for_feature_subsets(ckpt_dirs):
    """Checkpoint trained on [target, f1, f2]; input [target, f2] must keep f2 in f2's slot."""
    ckpt_path = ckpt_dirs["crs"]
    df = make_df(missing_rate=0.2, seed=5)[["timestamp", "target", "f2"]]
    calls = _capture_forward_calls(ckpt_path)
    result = forecasting.perform_forecasting(df, config=cfg(ckpt_path, return_all_channels=True))

    assert list(result.columns) == ["timestamp", "target_forecast", "f2_forecast"]
    call = calls[0]
    np.testing.assert_array_equal(call["valid_channel_mask"][0].numpy(), [1.0, 0.0, 1.0])
    assert call["channel_mask"][0, 1].eq(0).all()  # f1 slot is padding
    assert call["x"][0, 1].eq(0).all()
    observed = ~np.isnan(df[["target", "f2"]].to_numpy()[-SEQ_LEN:].T)
    np.testing.assert_array_equal(call["channel_mask"][0, [0, 2]].numpy(), observed.astype(np.float32))
    assert call["missing_rate"] == pytest.approx(1.0 - observed.mean())  # over valid channels only
    # Per-channel priors keep their trained slots: [1, 2, 3], not [1, 2].
    model, _ = imputation._get_impute_model(ckpt_path, torch.device("cpu"))
    np.testing.assert_allclose(model.normalizer._global_stdev.flatten().numpy(), [1.0, 2.0, 3.0])


def test_name_alignment_reorders_features_to_training_slots(ckpt_dirs):
    ckpt_path = ckpt_dirs["plus"]
    df = make_df(missing_rate=0.0)[["timestamp", "f2", "target", "f1"]]
    calls = _capture_forward_calls(ckpt_path)
    result = forecasting.perform_forecasting(df, config=cfg(ckpt_path, return_all_channels=True))

    # Output follows the input column order (target first), inputs follow the training slots.
    assert list(result.columns) == ["timestamp", "target_forecast", "f2_forecast", "f1_forecast"]
    scaler = imputation.ChannelScaler.fit(df[["target", "f1", "f2"]].to_numpy())
    expected = scaler.transform(df[["target", "f1", "f2"]].to_numpy()[-SEQ_LEN:]).T
    np.testing.assert_allclose(calls[0]["x"][0].numpy(), expected, rtol=1e-5, atol=1e-5)


def test_name_alignment_rejects_unknown_columns(ckpt_dirs):
    with pytest.raises(ValueError, match="not among the impute checkpoint's training channels"):
        forecasting.perform_forecasting(make_df(n_features=3), config=cfg(ckpt_dirs["crs"]))


def test_name_alignment_rejects_target_slot_conflict(ckpt_dirs):
    df = make_df()
    df["y"] = df["target"]
    with pytest.raises(ValueError, match="is the impute checkpoint's target channel"):
        forecasting.perform_forecasting(df, config=cfg(ckpt_dirs["crs"], target_column="y"))


def test_positional_alignment_is_opt_in_and_warns(ckpt_dirs, caplog):
    ckpt_path = ckpt_dirs["crs"]
    df = make_df()[["timestamp", "target", "f2"]]
    calls = _capture_forward_calls(ckpt_path)
    with caplog.at_level(logging.WARNING):
        forecasting.perform_forecasting(df, config=cfg(ckpt_path, impute_channel_alignment="positional"))

    assert "fed by position" in caplog.text
    np.testing.assert_array_equal(calls[0]["valid_channel_mask"][0].numpy(), [1.0, 1.0])
    model, _ = imputation._get_impute_model(ckpt_path, torch.device("cpu"))
    np.testing.assert_allclose(model.normalizer._global_stdev.flatten().numpy(), [1.0, 2.0])


def test_invalid_alignment_and_normalization_values(ckpt_dirs):
    with pytest.raises(ValueError, match="impute_channel_alignment"):
        forecasting.perform_forecasting(make_df(), config=cfg(ckpt_dirs["plus"], impute_channel_alignment="nope"))
    with pytest.raises(ValueError, match="impute_normalization"):
        forecasting.perform_forecasting(make_df(), config=cfg(ckpt_dirs["plus"], impute_normalization="window"))


# ─────────────────────────────────────────────────────────────────────────────
# Training-consistent normalization (review P1)
# ─────────────────────────────────────────────────────────────────────────────


class _RevINZeroForecast(torch.nn.Module):
    """Runs the real MissingnessAdaptiveRevIN and forecasts a normalized zero."""

    def __init__(self, stdev_prior: float, pred_len: int = 1):
        super().__init__()
        self.normalizer = MissingnessAdaptiveRevIN(num_features=1)
        self.normalizer._global_stdev.fill_(stdev_prior)
        self.normalizer._global_stdev_initialized.fill_(True)
        self.pred_len = pred_len

    def forward(self, x, input_mask=None, channel_mask=None, valid_channel_mask=None, missing_rate=0.0):
        self.normalizer.set_missing_rate(missing_rate)
        self.normalizer(x=x, mask=channel_mask, mode="norm")
        zeros = torch.zeros(x.shape[0], x.shape[1], self.pred_len)
        return SimpleNamespace(forecast=self.normalizer(x=zeros, mode="denorm"))


def test_adaptive_revin_priors_keep_training_meaning():
    """Review repro: train mean/std 100/10, input [110, 130] + 8 NaN, stdev prior 0.5.

    With the training-style fixed scaler the normalized-zero forecast maps to 108
    (mean shrinks toward the training mean). A per-window scaler would give 120.
    """
    model = _RevINZeroForecast(stdev_prior=0.5).eval()
    window = np.array([110.0, 130.0] + [np.nan] * 8).reshape(-1, 1)
    training_scaler = imputation.ChannelScaler(mean=np.array([100.0]), std=np.array([10.0]))

    fixed, rates = imputation.forecast_with_missingness(
        model, window, seq_len=10, pred_len=1, forecast_horizon=1, device=torch.device("cpu"), scaler=training_scaler
    )
    assert rates == [pytest.approx(0.8)]
    assert fixed[0, 0] == pytest.approx(108.0, abs=1e-3)

    per_window = imputation.ChannelScaler.fit(window)  # what the old implementation did
    recentered, _ = imputation.forecast_with_missingness(
        model, window, seq_len=10, pred_len=1, forecast_horizon=1, device=torch.device("cpu"), scaler=per_window
    )
    assert recentered[0, 0] == pytest.approx(120.0, abs=1e-3)


def test_history_normalization_uses_long_run_statistics():
    """The default scaler is fit on the whole input history, so it matches the training-style result."""
    model = _RevINZeroForecast(stdev_prior=0.5).eval()
    history = np.tile([90.0, 110.0], 200)  # mean 100, std 10
    values = np.concatenate([history, [110.0, 130.0], [np.nan] * 8]).reshape(-1, 1)
    scaler = imputation.ChannelScaler.fit(values)
    pred, _ = imputation.forecast_with_missingness(
        model, values, seq_len=10, pred_len=1, forecast_horizon=1, device=torch.device("cpu"), scaler=scaler
    )
    assert scaler.mean[0] == pytest.approx(100.0, abs=0.2)
    assert pred[0, 0] == pytest.approx(108.0, abs=0.3)


def _evaluator_setup(tmp_path, ckpt_dir: Path, missing_rate: float):
    """CSV + evaluator dataset + a checkpoint copy whose config records that dataset's standardizer."""
    n = 400
    t = np.arange(n, dtype=np.float64)
    rng = np.random.default_rng(1)
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "target": 20.0 + 4.0 * np.sin(t / 6.0) + rng.normal(0, 0.3, n),
            "f1": 100.0 + 7.0 * np.cos(t / 4.0),
            "f2": -5.0 + 2.0 * np.sin(t / 9.0),
        }
    )
    csv_path = tmp_path / "eval_source.csv"
    raw.to_csv(csv_path, index=False)
    ds = CSVForecastMissingDataset(
        csv_path=str(csv_path),
        data_split="test",
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        target_col="target",
        missing_pattern="point",
        missing_rate=missing_rate,
        random_seed=13,
    )
    ckpt_copy = tmp_path / "ckpt"
    ckpt_copy.mkdir()
    (ckpt_copy / "best_model.pt").write_bytes((ckpt_dir / "best_model.pt").read_bytes())
    config = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
    config["meta"]["dataset_metadata"] = {"datasets": [ds.metadata_summary]}
    (ckpt_copy / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return raw, ds, ckpt_copy


@pytest.mark.parametrize("missing_rate", [0.3, 0.7])
def test_checkpoint_normalization_matches_evaluator_preprocessing(tmp_path, ckpt, missing_rate):
    """SDK (impute_normalization='checkpoint') == test.py-style evaluation on the same window."""
    raw, ds, ckpt_copy = _evaluator_setup(tmp_path, ckpt, missing_rate)
    model, _ = load_impute_model(ckpt_copy)

    for item in range(min(3, len(ds))):
        hist, _fut, temporal_mask, channel_mask, valid = ds[item]
        with torch.no_grad():
            out = model(
                hist[None],
                input_mask=temporal_mask[None],
                channel_mask=channel_mask[None],
                valid_channel_mask=valid[None],
                missing_rate=float(1.0 - channel_mask.mean()),
            )
        expected = ds.standardizer.inverse_transform(out.forecast[0].numpy().T)  # [H, C] original units

        start = ds._starts[item]
        window = raw.iloc[start : start + SEQ_LEN].copy()
        mask = ds.observed_mask[start : start + SEQ_LEN].astype(bool)
        window[["target", "f1", "f2"]] = window[["target", "f1", "f2"]].mask(~mask)
        result = forecasting.perform_forecasting(
            window,
            config=cfg(ckpt_copy, impute_normalization="checkpoint", return_all_channels=True),
        )
        got = result[["target_forecast", "f1_forecast", "f2_forecast"]].to_numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-3)


def test_checkpoint_normalization_requires_saved_standardizer(ckpt_dirs):
    with pytest.raises(ValueError, match="does not record any training standardizers"):
        forecasting.perform_forecasting(make_df(), config=cfg(ckpt_dirs["plus"], impute_normalization="checkpoint"))


def test_checkpoint_normalization_source_selection(tmp_path, ckpt_dirs):
    _, _, ckpt_copy = _evaluator_setup(tmp_path, ckpt_dirs["plus"], 0.2)
    config = cfg(ckpt_copy, impute_normalization="checkpoint", impute_source_dataset="eval_source")
    assert len(forecasting.perform_forecasting(make_df(), config=config)) == PRED_LEN
    with pytest.raises(ValueError, match="not found in the checkpoint"):
        forecasting.perform_forecasting(
            make_df(), config=cfg(ckpt_copy, impute_normalization="checkpoint", impute_source_dataset="ETTh9")
        )


# ─────────────────────────────────────────────────────────────────────────────
# impute_history_rows / impute_normalization="provided"
# ─────────────────────────────────────────────────────────────────────────────


def test_history_rows_limits_the_reference_period(ckpt):
    """Limiting history to N rows == passing only the last N rows."""
    df = make_df(num_rows=200, missing_rate=0.2, seed=11)
    df.loc[df.index[:100], "target"] += 50.0  # an old regime that should not leak into the scaler
    limited = forecasting.perform_forecasting(df, config=cfg(ckpt, impute_history_rows=64))
    tail_only = forecasting.perform_forecasting(df.tail(64).reset_index(drop=True), config=cfg(ckpt))
    full = forecasting.perform_forecasting(df, config=cfg(ckpt))

    np.testing.assert_allclose(limited["target_forecast"], tail_only["target_forecast"], rtol=1e-6)
    assert not np.allclose(limited["target_forecast"], full["target_forecast"])


@pytest.mark.parametrize("rows", [SEQ_LEN - 1, 0, -5, 2.5, True])
def test_history_rows_validation(ckpt_dirs, rows):
    with pytest.raises(ValueError, match="impute_history_rows"):
        forecasting.perform_forecasting(make_df(), config=cfg(ckpt_dirs["plus"], impute_history_rows=rows))


def test_history_rows_ignored_outside_history_mode(ckpt_dirs, caplog):
    stats = imputation.fit_impute_scaler_stats(make_df())
    config = cfg(ckpt_dirs["plus"], impute_normalization="provided", impute_scaler_stats=stats, impute_history_rows=40)
    with caplog.at_level(logging.WARNING):
        forecasting.perform_forecasting(make_df(), config=config)
    assert "impute_history_rows is ignored" in caplog.text


def test_fit_impute_scaler_stats_format_and_observed_only():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
            "f1": [1.0, np.nan, 3.0, 5.0, np.nan],
            "target": [10.0, 20.0, np.nan, 40.0, 50.0],
            "label": ["a", "b", "c", "d", "e"],  # non-numeric: skipped
        }
    )
    stats = imputation.fit_impute_scaler_stats(df, target_column="target")

    assert list(stats) == ["mean", "std"]
    assert list(stats["mean"]) == ["target", "f1"]  # target first, like the forecasting paths
    assert stats["mean"]["target"] == pytest.approx(30.0)
    assert stats["std"]["target"] == pytest.approx(np.std([10.0, 20.0, 40.0, 50.0]))
    assert stats["mean"]["f1"] == pytest.approx(3.0)
    json.dumps(stats)  # plain, serializable floats
    from sdk import fit_impute_scaler_stats

    assert fit_impute_scaler_stats is imputation.fit_impute_scaler_stats


def test_provided_stats_round_trip_matches_history(ckpt):
    df = make_df(num_rows=128, missing_rate=0.3, seed=2)
    stats = imputation.fit_impute_scaler_stats(df, target_column="target")
    provided = forecasting.perform_forecasting(
        df, config=cfg(ckpt, impute_normalization="provided", impute_scaler_stats=stats, return_all_channels=True)
    )
    history = forecasting.perform_forecasting(df, config=cfg(ckpt, return_all_channels=True))
    pd.testing.assert_frame_equal(provided, history)


def test_provided_stats_reproduce_training_style_result():
    """Review repro through caller-supplied stats: mean/std 100/10 -> 108."""
    model = _RevINZeroForecast(stdev_prior=0.5).eval()
    scaler = imputation.ChannelScaler.from_stats(["x"], {"mean": {"x": 100.0}, "std": {"x": 10.0}})
    window = np.array([110.0, 130.0] + [np.nan] * 8).reshape(-1, 1)
    pred, _ = imputation.forecast_with_missingness(
        model, window, seq_len=10, pred_len=1, forecast_horizon=1, device=torch.device("cpu"), scaler=scaler
    )
    assert pred[0, 0] == pytest.approx(108.0, abs=1e-3)


def test_provided_stats_are_independent_of_request_history(ckpt_dirs):
    """With stored stats, sending more or less history does not change the forecast."""
    ckpt_path = ckpt_dirs["plus"]
    df = make_df(num_rows=200, seed=4)
    stats = imputation.fit_impute_scaler_stats(df.head(120))
    config = cfg(ckpt_path, impute_normalization="provided", impute_scaler_stats=stats)
    long_request = forecasting.perform_forecasting(df, config=config)
    short_request = forecasting.perform_forecasting(df.tail(SEQ_LEN).reset_index(drop=True), config=config)
    np.testing.assert_allclose(long_request["target_forecast"], short_request["target_forecast"], rtol=1e-6)


@pytest.mark.parametrize(
    ("stats", "match"),
    [
        (None, "requires impute_scaler_stats"),
        ({"mean": {"target": 1.0}}, "must be a mapping"),
        ({"mean": {"target": 1.0, "f1": 1.0}, "std": {"target": 1.0, "f1": 1.0}}, r"missing mean/std .*'f2'"),
        (
            {"mean": {"target": 1.0, "f1": 1.0, "f2": 1.0}, "std": {"target": 1.0, "f1": 0.0, "f2": 1.0}},
            r"bad std: \['f1'\]",
        ),
        (
            {"mean": {"target": float("nan"), "f1": 1.0, "f2": 1.0}, "std": {"target": 1.0, "f1": 1.0, "f2": 1.0}},
            r"bad mean: \['target'\]",
        ),
    ],
)
def test_provided_stats_validation(ckpt_dirs, stats, match):
    config = cfg(ckpt_dirs["plus"], impute_normalization="provided", impute_scaler_stats=stats)
    with pytest.raises(ValueError, match=match):
        forecasting.perform_forecasting(make_df(), config=config)


def test_provided_stats_warn_on_unknown_and_ignored(ckpt_dirs, caplog):
    stats = imputation.fit_impute_scaler_stats(make_df())
    stats["mean"]["legacy_col"] = 0.0
    stats["std"]["legacy_col"] = 1.0
    with caplog.at_level(logging.WARNING):
        forecasting.perform_forecasting(
            make_df(), config=cfg(ckpt_dirs["plus"], impute_normalization="provided", impute_scaler_stats=stats)
        )
        forecasting.perform_forecasting(make_df(), config=cfg(ckpt_dirs["plus"], impute_scaler_stats=stats))
    assert "not in the input (ignored): ['legacy_col']" in caplog.text
    assert "impute_scaler_stats is ignored unless impute_normalization='provided'" in caplog.text


def test_provided_stats_from_yaml(ckpt_dirs, tmp_path):
    config_path = tmp_path / "impute_provided.yaml"
    config_path.write_text(
        f"""
inference:
  seq_len: {SEQ_LEN}
  forecast_horizon: {PRED_LEN}
  device: cpu
  handle_missingness: true
  impute_ckpt: "{ckpt_dirs["plus"]}"
  impute_normalization: provided
  impute_scaler_stats:
    mean: {{target: 10.0, f1: 100.0, f2: 200.0}}
    std: {{target: 2.0, f1: 3.5, f2: 3.0}}
""",
        encoding="utf-8",
    )
    loaded = forecasting.load_forecasting_config(config_path)
    assert loaded.impute_scaler_stats["std"]["f1"] == 3.5
    assert len(forecasting.perform_forecasting(make_df(), config=config_path)) == PRED_LEN


def test_history_normalization_warns_on_short_history(ckpt_dirs, caplog):
    with caplog.at_level(logging.WARNING):
        forecasting.perform_forecasting(make_df(num_rows=SEQ_LEN + 4), config=cfg(ckpt_dirs["plus"]))
    assert "reference statistics come from little" in caplog.text


def test_handle_missingness_adopts_checkpoint_seq_len_when_default(ckpt):
    result = forecasting.perform_forecasting(make_df(), config=cfg(ckpt, seq_len=512))
    assert len(result) == PRED_LEN


def test_handle_missingness_rejects_seq_len_mismatch(ckpt):
    with pytest.raises(ValueError, match="does not match the impute checkpoint"):
        forecasting.perform_forecasting(make_df(), config=cfg(ckpt, seq_len=16))


def test_handle_missingness_rejects_too_few_rows(ckpt):
    with pytest.raises(ValueError, match="rows but seq_len requires"):
        forecasting.perform_forecasting(make_df(num_rows=SEQ_LEN - 1), config=cfg(ckpt))


def test_handle_missingness_rejects_all_missing_target(ckpt):
    df = make_df()
    df.loc[df.index[-SEQ_LEN:], "target"] = np.nan
    with pytest.raises(ValueError, match="no observed values"):
        forecasting.perform_forecasting(df, config=cfg(ckpt))


def test_handle_missingness_rejects_darr(ckpt_dirs):
    with pytest.raises(ValueError, match="DARR"):
        forecasting.perform_forecasting(make_df(), config=cfg(ckpt_dirs["plus"]), context_df=make_df())


def test_handle_missingness_rejects_interpretability(ckpt_dirs):
    with pytest.raises(ValueError, match="interpretability"):
        forecasting.perform_forecasting(make_df(), config=cfg(ckpt_dirs["plus"], interpretability=True))


def test_handle_missingness_save_preds(ckpt_dirs, tmp_path):
    out = tmp_path / "preds.csv"
    result = forecasting.perform_forecasting(make_df(), config=cfg(ckpt_dirs["plus"], save_preds=str(out)))
    saved = pd.read_csv(out)
    np.testing.assert_allclose(saved["target_forecast"], result["target_forecast"])


def test_handle_missingness_from_yaml(ckpt_dirs, tmp_path):
    config_path = tmp_path / "impute.yaml"
    config_path.write_text(
        f"""
inference:
  seq_len: {SEQ_LEN}
  forecast_horizon: {PRED_LEN}
  device: cpu
  handle_missingness: true
  impute_ckpt: "{ckpt_dirs["plus"]}"
""",
        encoding="utf-8",
    )
    loaded = forecasting.load_forecasting_config(config_path)
    assert loaded.handle_missingness is True
    assert loaded.impute_ckpt == str(ckpt_dirs["plus"])
    assert len(forecasting.perform_forecasting(make_df(), config=config_path)) == PRED_LEN


def test_handle_missingness_model_cache(ckpt_dirs):
    config = cfg(ckpt_dirs["plus"])
    forecasting.perform_forecasting(make_df(), config=config)
    first = next(iter(imputation._IMPUTE_MODEL_CACHE.values()))[0]
    forecasting.perform_forecasting(make_df(seed=1), config=config)
    assert len(imputation._IMPUTE_MODEL_CACHE) == 1
    assert next(iter(imputation._IMPUTE_MODEL_CACHE.values()))[0] is first

    forecasting.clear_model_cache()
    assert imputation._IMPUTE_MODEL_CACHE == {}


def test_handle_missingness_off_keeps_zero_fill_path(monkeypatch, caplog):
    """With the toggle off the impute path is never used and NULLs are zero-filled as before."""

    def fail(*args: object, **kwargs: object):
        raise AssertionError("impute path must not run when handle_missingness=False")

    class LastValueModel:
        def eval(self):
            return self

        def load_state_dict(self, state, strict=False):
            return

        def __call__(self, x_enc, input_mask):
            return type("Out", (), {"forecast": x_enc[:, :, -1:].repeat(1, 1, PRED_LEN)})()

    monkeypatch.setattr(imputation, "perform_missingness_aware_forecasting", fail)
    monkeypatch.setattr(forecasting, "build_model", lambda **kwargs: LastValueModel())
    monkeypatch.setattr(forecasting.torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        forecasting, "download_model_weights", lambda *, standardizer_pkl, ckpt, **kw: (standardizer_pkl, ckpt)
    )
    monkeypatch.setattr(
        forecasting.joblib,
        "load",
        lambda path: {"mean": np.array([0.0], dtype=np.float32), "std": np.array([1.0], dtype=np.float32)},
    )
    monkeypatch.setattr(forecasting, "control_randomness", lambda seed=0: None)
    forecasting.clear_model_cache()

    df = make_df(n_features=0, missing_rate=0.0)
    df.loc[df.index[-1], "target"] = np.nan
    config = forecasting.ForecastingConfig(
        seq_len=SEQ_LEN,
        forecast_horizon=PRED_LEN,
        model_horizon=PRED_LEN,
        ckpt="fake.pt",
        standardizer_pkl="fake.pkl",
        device="cpu",
    )
    with caplog.at_level(logging.WARNING):
        result = forecasting.perform_forecasting(df, config=config)
    forecasting.clear_model_cache()

    assert "filling with zeros" in caplog.text
    assert "handle_missingness=True" in caplog.text
    np.testing.assert_allclose(result["target_forecast"], 0.0)  # last value was zero-filled


# ─────────────────────────────────────────────────────────────────────────────
# Hugging Face Hub checkpoints (hub mocked — no network)
# ─────────────────────────────────────────────────────────────────────────────


class FakeHub:
    """Stand-in for huggingface_hub.snapshot_download serving a tiny checkpoint from <snapshot>/<subfolder>."""

    def __init__(
        self,
        root: Path,
        source_ckpt: Path,
        subfolder: str = "kumo-forecast-1.2.0",
        config_name: str = "config_base.json",
    ):
        self.root, self.source, self.subfolder, self.config_name = root, source_ckpt, subfolder, config_name
        self.calls: list[dict] = []
        self.error: Exception | None = None
        self.include_weights = True

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        folder = self.root / self.subfolder if self.subfolder else self.root
        folder.mkdir(parents=True, exist_ok=True)
        if self.include_weights:
            (folder / "best_model.pt").write_bytes((self.source / "best_model.pt").read_bytes())
        (folder / self.config_name).write_text(
            (self.source / "config.json").read_text(encoding="utf-8"), encoding="utf-8"
        )
        return str(self.root)


@pytest.fixture
def fake_hub(monkeypatch, tmp_path, ckpt_dirs) -> FakeHub:
    hub = FakeHub(tmp_path / "hf_snapshot", ckpt_dirs["crs"])
    monkeypatch.setattr(huggingface_hub, "snapshot_download", hub)
    return hub


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        (
            "hf://nvidia/Kumo-Forecast/kumo-forecast-1.2.0",
            HFReference("nvidia/Kumo-Forecast", None, "kumo-forecast-1.2.0"),
        ),
        ("hf://nvidia/Kumo-Forecast@v1.2/impute", HFReference("nvidia/Kumo-Forecast", "v1.2", "impute")),
        ("hf://org/repo", HFReference("org/repo", None, "")),
        ("hf://org/repo@abc123", HFReference("org/repo", "abc123", "")),
        ("hf://org/repo/a/b/", HFReference("org/repo", None, "a/b")),
    ],
)
def test_parse_hf_reference(ref, expected):
    assert parse_hf_reference(ref) == expected


@pytest.mark.parametrize("ref", ["hf://only-org", "hf://org/@rev", "hf://org/repo@", "hf://org@x/repo", "nvidia/Kumo"])
def test_parse_hf_reference_rejects_invalid(ref):
    with pytest.raises(ValueError, match=r"Hugging Face reference|revision|@revision"):
        parse_hf_reference(ref)


def test_default_impute_ckpt_is_released_location():
    assert DEFAULT_IMPUTE_CKPT == "hf://nvidia/Kumo-Forecast/kumo-forecast-1.2.0"


def test_download_hf_checkpoint_fetches_only_inference_files(fake_hub):
    folder = download_hf_checkpoint(
        "hf://nvidia/Kumo-Forecast@v1/kumo-forecast-1.2.0", local_files_only=True, token=False
    )

    call = fake_hub.calls[0]
    assert call["repo_id"] == "nvidia/Kumo-Forecast"
    assert call["revision"] == "v1"
    assert call["local_files_only"] is True
    assert call["token"] is False  # passed through (False = never send a token)
    patterns = call["allow_patterns"]
    assert sorted(patterns) == [
        "kumo-forecast-1.2.0/best_model.pt",
        "kumo-forecast-1.2.0/config.json",
        "kumo-forecast-1.2.0/config_base.json",
    ]
    assert folder == fake_hub.root / "kumo-forecast-1.2.0"


def test_config_base_json_is_accepted_locally(tmp_path, ckpt_dirs):
    folder = tmp_path / "release"
    folder.mkdir()
    (folder / "best_model.pt").write_bytes((ckpt_dirs["plus"] / "best_model.pt").read_bytes())
    (folder / "config_base.json").write_text(
        (ckpt_dirs["plus"] / "config.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    _, info = load_impute_model(folder)
    assert info.seq_len == SEQ_LEN


def test_legacy_model_name_in_config_still_loads_crs(tmp_path, ckpt_dirs):
    """The published kumo-forecast-1.2.0 config says "MOMENTLFPlusCRS"; it must map to BackboneLFPlusCRS."""
    folder = tmp_path / "legacy"
    folder.mkdir()
    (folder / "best_model.pt").write_bytes((ckpt_dirs["crs"] / "best_model.pt").read_bytes())
    config = json.loads((ckpt_dirs["crs"] / "config.json").read_text(encoding="utf-8"))
    config["model"] = "MOMENTLFPlusCRS"
    config.pop("use_crs", None)
    (folder / "config_base.json").write_text(json.dumps(config), encoding="utf-8")
    model, info = load_impute_model(folder)
    assert isinstance(model, BackboneLFPlusCRS)
    assert info.is_crs


def test_perform_forecasting_with_hf_reference_downloads_once(fake_hub):
    config = cfg("hf://nvidia/Kumo-Forecast/kumo-forecast-1.2.0")
    first = forecasting.perform_forecasting(make_df(), config=config)
    second = forecasting.perform_forecasting(make_df(seed=3), config=config)
    assert len(first) == len(second) == PRED_LEN
    assert len(fake_hub.calls) == 1  # resolved path + model are cached for the process


def test_perform_forecasting_defaults_to_released_checkpoint(fake_hub):
    forecasting.perform_forecasting(make_df(), config=cfg(None))
    call = fake_hub.calls[0]
    assert call["repo_id"] == "nvidia/Kumo-Forecast"
    assert call["revision"] is None
    assert all(p.startswith("kumo-forecast-1.2.0/") for p in call["allow_patterns"])


def test_local_files_only_is_passed_to_the_hub(fake_hub):
    forecasting.perform_forecasting(
        make_df(), config=cfg("hf://nvidia/Kumo-Forecast/kumo-forecast-1.2.0", local_files_only=True)
    )
    assert fake_hub.calls[0]["local_files_only"] is True


@pytest.mark.parametrize(
    ("error_name", "local_only", "match"),
    [
        ("RepositoryNotFoundError", False, "huggingface-cli login"),
        ("GatedRepoError", False, "HF_TOKEN"),
        ("RevisionNotFoundError", False, "Revision 'v9' not found"),
        ("LocalEntryNotFoundError", True, "local Hugging Face cache"),
        ("ConnectionError", False, "Failed to download impute checkpoint"),
    ],
)
def test_hf_download_errors_are_actionable(fake_hub, error_name, local_only, match):
    fake_hub.error = type(error_name, (Exception,), {})("boom")
    with pytest.raises(RuntimeError, match=match):
        download_hf_checkpoint("hf://nvidia/Kumo-Forecast@v9/kumo-forecast-1.2.0", local_files_only=local_only)


def test_hf_checkpoint_without_weights_raises(fake_hub):
    fake_hub.include_weights = False
    with pytest.raises(FileNotFoundError, match="no weights file"):
        download_hf_checkpoint("hf://nvidia/Kumo-Forecast/kumo-forecast-1.2.0")


# ─────────────────────────────────────────────────────────────────────────────
# Real checkpoint (opt-in)
# ─────────────────────────────────────────────────────────────────────────────

REAL_CKPT = os.environ.get("KUMO_IMPUTE_CKPT")
REAL_CSV = os.environ.get("KUMO_IMPUTE_CSV")
REAL_EVAL_WINDOWS = int(os.environ.get("KUMO_IMPUTE_EVAL_WINDOWS", "20"))
real_checkpoint = pytest.mark.skipif(not REAL_CKPT, reason="set KUMO_IMPUTE_CKPT to a trained impute checkpoint")


@pytest.fixture(scope="module")
def real_model():
    """Load the (large) real checkpoint once per module and reuse it across tests."""
    imputation.clear_impute_model_cache()
    model, info = imputation._get_impute_model(REAL_CKPT, forecasting.DEVICE)
    snapshot = dict(imputation._IMPUTE_MODEL_CACHE)
    return model, info, snapshot


@pytest.fixture
def real_cached(real_model):
    # The autouse fixture clears the cache per test; put the loaded model back.
    model, info, snapshot = real_model
    imputation._IMPUTE_MODEL_CACHE.update(snapshot)
    return model, info


def _real_frame(info) -> tuple[pd.DataFrame, str]:
    if REAL_CSV:
        df = pd.read_csv(REAL_CSV)
        ts_col = "timestamp" if "timestamp" in df.columns else "date"
        target = info.target_col if info.target_col in df.columns else df.columns[-1]
        return df.rename(columns={ts_col: "timestamp"}), target
    n = info.seq_len * 3 + info.pred_len * REAL_EVAL_WINDOWS
    t = np.arange(n, dtype=np.float64)
    data = {"timestamp": pd.date_range("2024-01-01", periods=n, freq="h"), "target": np.sin(t / 12.0)}
    for i in range(max(0, info.trained_n_channels - 1)):
        data[f"f{i}"] = np.cos(t / (7.0 + i))
    return pd.DataFrame(data), "target"


def _inject_missingness(block: pd.DataFrame, value_cols: list[str], target: str, pattern: str, rate: float, rng):
    mask = np.zeros((len(block), len(value_cols)), dtype=bool)
    if pattern == "point":
        mask = rng.random(mask.shape) < rate
    else:
        width = int(rate * len(block))
        if width:
            start = rng.integers(0, len(block) - width + 1)
            mask[start : start + width, :] = True
    mask[-1, value_cols.index(target)] = False  # keep the target observed at the forecast origin
    return block[value_cols].mask(mask)


@real_checkpoint
def test_real_checkpoint_loads_cleanly(real_cached):
    model, info = real_cached  # loading already raised on any missing/unexpected key
    assert not model.training
    assert info.seq_len > 0
    assert info.pred_len > 0
    print(
        f"\nreal ckpt: {type(model).__name__} seq_len={info.seq_len} pred_len={info.pred_len} "
        f"trained_channels={info.trained_n_channels} epoch={info.epoch}"
    )


@real_checkpoint
# "checkpoint" normalization needs training scalers saved in the checkpoint config; the released
# kumo-forecast-1.2.0 config has none, so it is covered by the unit tests (test_checkpoint_normalization_*) only.
@pytest.mark.parametrize("normalization", ["history", "provided"])
@pytest.mark.parametrize("pattern", ["point", "block"])
@pytest.mark.parametrize("rate", [0.0, 0.1, 0.3, 0.5, 0.7])
def test_real_checkpoint_forecasts_under_missingness(real_cached, normalization, pattern, rate):
    """Evaluate the SDK path on the last REAL_EVAL_WINDOWS non-overlapping windows.

    Each call gets the full history up to the forecast origin (so "history"
    normalization sees long-run statistics); missingness is injected into the
    last seq_len rows only. Metrics are in the target's train-split
    standardized units (first 1 - val_ratio - test_ratio of the rows),
    comparable in spirit to test.py / test_release.py.
    """
    _, info = real_cached
    df, target = _real_frame(info)
    value_cols = [c for c in df.columns if c != "timestamp"]
    horizon, seq_len = info.pred_len, info.seq_len

    train_end = int(len(df) * (1.0 - info.config.get("val_ratio", 0.1) - info.config.get("test_ratio", 0.2)))
    train_target = df[target].to_numpy()[: max(train_end, seq_len)]
    sd = float(np.nanstd(train_target)) or 1.0
    # "provided": reference stats fit once on the training portion, like a deployment would store them.
    stats = (
        imputation.fit_impute_scaler_stats(df.iloc[: max(train_end, seq_len)], target_column=target)
        if normalization == "provided"
        else None
    )

    config = forecasting.ForecastingConfig(
        target_column=target,
        seq_len=seq_len,
        forecast_horizon=horizon,
        handle_missingness=True,
        impute_ckpt=REAL_CKPT,
        impute_normalization=normalization,
        impute_source_dataset=Path(REAL_CSV).stem if normalization == "checkpoint" else None,
        impute_scaler_stats=stats,
        # Synthetic frames use placeholder column names, so they can only be fed by position.
        impute_channel_alignment="name" if REAL_CSV else "positional",
    )
    rng = np.random.default_rng(13)
    sq_err, abs_err = [], []
    for w in range(REAL_EVAL_WINDOWS):
        origin = len(df) - horizon - w * horizon  # index of the first forecast row
        if origin - seq_len < 0:
            break
        history = df.iloc[:origin].copy()
        window_idx = history.index[-seq_len:]
        history.loc[window_idx, value_cols] = _inject_missingness(
            history.loc[window_idx], value_cols, target, pattern, rate, rng
        )
        future = df[target].to_numpy()[origin : origin + horizon]

        pred = forecasting.perform_forecasting(history, config=config)[f"{target}_forecast"].to_numpy()
        assert np.isfinite(pred).all()
        err = (pred - future) / sd
        sq_err.append(err**2)
        abs_err.append(np.abs(err))

    mse, mae = float(np.mean(sq_err)), float(np.mean(abs_err))
    print(
        f"\nreal ckpt | {normalization:10s} {pattern:5s} rate={rate:.1f} | windows={len(sq_err)} | "
        f"MSE={mse:.4f} MAE={mae:.4f}"
    )


# ── Column-name handling on the real ETT CSV ────────────────────────────────

real_csv = pytest.mark.skipif(
    not (REAL_CKPT and REAL_CSV), reason="set KUMO_IMPUTE_CKPT and KUMO_IMPUTE_CSV (an ETT CSV) for name tests"
)


def _real_request(info, rate: float = 0.3, seed: int = 21) -> tuple[pd.DataFrame, str]:
    """Full history up to the last forecast origin, with point missingness in the last seq_len rows."""
    df, target = _real_frame(info)
    history = df.iloc[: len(df) - info.pred_len].copy()
    value_cols = [c for c in history.columns if c != "timestamp"]
    window_idx = history.index[-info.seq_len :]
    history.loc[window_idx, value_cols] = _inject_missingness(
        history.loc[window_idx], value_cols, target, "point", rate, np.random.default_rng(seed)
    )
    return history, target


def _real_config(info, target: str, **overrides) -> forecasting.ForecastingConfig:
    options = dict(
        target_column=target,
        seq_len=info.seq_len,
        forecast_horizon=info.pred_len,
        handle_missingness=True,
        impute_ckpt=REAL_CKPT,
        return_all_channels=True,
    )
    options.update(overrides)
    return forecasting.ForecastingConfig(**options)


def _schema_or_skip(info) -> list[str]:
    if not info.channel_schema:
        pytest.skip("the real checkpoint does not record its training channel names")
    return info.channel_schema


@real_csv
def test_real_checkpoint_name_alignment_is_column_order_invariant(real_cached):
    """With name alignment, shuffling the input columns must not change any forecast."""
    _, info = real_cached
    _schema_or_skip(info)
    history, target = _real_request(info)
    features = [c for c in history.columns if c not in ("timestamp", target)]
    shuffled = history[["timestamp", *reversed(features), target]]

    base = forecasting.perform_forecasting(history, config=_real_config(info, target))
    other = forecasting.perform_forecasting(shuffled, config=_real_config(info, target))
    for col in base.columns.drop("timestamp"):
        np.testing.assert_allclose(other[col], base[col], rtol=1e-5, atol=1e-5)


@real_csv
def test_real_checkpoint_name_alignment_feature_subset(real_cached):
    """Dropping features pads their training slots; kept features stay in their own slots."""
    model, info = real_cached
    schema = _schema_or_skip(info)
    history, target = _real_request(info)
    features = [c for c in history.columns if c not in ("timestamp", target)]
    keep = features[::2]
    subset = history[["timestamp", target, *keep]]

    calls: list[dict] = []
    handle = model.register_forward_pre_hook(lambda _m, _a, kw: calls.append(dict(kw)), with_kwargs=True)
    try:
        result = forecasting.perform_forecasting(subset, config=_real_config(info, target))
    finally:
        handle.remove()

    assert list(result.columns) == ["timestamp", f"{target}_forecast", *(f"{c}_forecast" for c in keep)]
    assert np.isfinite(result.drop(columns="timestamp").to_numpy()).all()
    expected_valid = np.zeros(len(schema), dtype=np.float32)
    expected_valid[[0, *(schema.index(c) for c in keep)]] = 1.0
    np.testing.assert_array_equal(calls[0]["valid_channel_mask"][0].cpu().numpy(), expected_valid)


@real_csv
def test_real_checkpoint_positional_equals_name_in_training_order(real_cached):
    """Renamed columns fed positionally in the training slot order == the named ETT columns."""
    _, info = real_cached
    schema = _schema_or_skip(info)
    history, target = _real_request(info)
    if set(schema[1:]) - set(history.columns):
        pytest.skip("KUMO_IMPUTE_CSV does not contain every training channel")

    named = forecasting.perform_forecasting(history, config=_real_config(info, target))
    renamed = history[["timestamp", target, *schema[1:]]].rename(
        columns={c: f"x{i}" for i, c in enumerate(schema[1:], start=1)}
    )
    positional = forecasting.perform_forecasting(
        renamed, config=_real_config(info, target, impute_channel_alignment="positional")
    )
    np.testing.assert_allclose(positional[f"{target}_forecast"], named[f"{target}_forecast"], rtol=1e-5, atol=1e-5)
    for i, c in enumerate(schema[1:], start=1):
        np.testing.assert_allclose(positional[f"x{i}_forecast"], named[f"{c}_forecast"], rtol=1e-5, atol=1e-5)


@real_csv
def test_real_checkpoint_positional_in_csv_order_vs_name(real_cached):
    """Anonymous columns in the CSV's own order (not the training order): shows what name alignment prevents."""
    _, info = real_cached
    schema = _schema_or_skip(info)
    history, target = _real_request(info)
    features = [c for c in history.columns if c not in ("timestamp", target)]
    if features == schema[1:]:
        pytest.skip("CSV column order already matches the training order")

    named = forecasting.perform_forecasting(history, config=_real_config(info, target))
    anonymous = history[["timestamp", target, *features]].rename(
        columns={c: f"x{i}" for i, c in enumerate(features, start=1)}
    )
    positional = forecasting.perform_forecasting(
        anonymous, config=_real_config(info, target, impute_channel_alignment="positional")
    )
    delta = np.abs(positional[f"{target}_forecast"].to_numpy() - named[f"{target}_forecast"].to_numpy())
    print(
        f"\nreal ckpt | CSV-order positional vs name alignment | target |diff| "
        f"mean={delta.mean():.4f} max={delta.max():.4f}"
    )
    assert np.isfinite(delta).all()


@real_csv
def test_real_checkpoint_unknown_column_names_require_positional(real_cached):
    _, info = real_cached
    _schema_or_skip(info)
    history, target = _real_request(info)
    features = [c for c in history.columns if c not in ("timestamp", target)]
    anonymous = history.rename(columns={c: f"x{i}" for i, c in enumerate(features, start=1)})
    with pytest.raises(ValueError, match="not among the impute checkpoint's training channels"):
        forecasting.perform_forecasting(anonymous, config=_real_config(info, target))
    result = forecasting.perform_forecasting(
        anonymous, config=_real_config(info, target, impute_channel_alignment="positional")
    )
    assert np.isfinite(result.drop(columns="timestamp").to_numpy()).all()


# ── Real Hugging Face download (opt-in) ─────────────────────────────────────

REAL_HF = os.environ.get("KUMO_IMPUTE_HF")  # e.g. hf://nvidia/Kumo-Forecast/kumo-forecast-1.2.0


@pytest.mark.skipif(
    not REAL_HF, reason="set KUMO_IMPUTE_HF (e.g. hf://nvidia/Kumo-Forecast/kumo-forecast-1.2.0) to test the Hub"
)
def test_real_hf_checkpoint_downloads_and_forecasts():
    imputation.clear_impute_model_cache()
    model, info = imputation._get_impute_model(REAL_HF, forecasting.DEVICE)
    print(
        f"\nreal hf: {REAL_HF} -> {info.checkpoint_path} | {type(model).__name__} "
        f"seq_len={info.seq_len} pred_len={info.pred_len} channels={info.channel_schema}"
    )
    if REAL_CSV:
        df = pd.read_csv(REAL_CSV)
        df = df.rename(columns={"date": "timestamp"}) if "date" in df.columns else df
        target, alignment = (info.target_col or df.columns[-1]), "name"
    else:
        n = info.seq_len * 4
        t = np.arange(n, dtype=np.float64)
        df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="h"), "y": np.sin(t / 12.0)})
        target, alignment = "y", "positional"
    df.loc[df.index[-10:-2], target] = np.nan
    config = forecasting.ForecastingConfig(
        target_column=target,
        forecast_horizon=info.pred_len,
        handle_missingness=True,
        impute_ckpt=REAL_HF,
        impute_channel_alignment=alignment,
    )
    result = forecasting.perform_forecasting(df, config=config)
    assert len(result) == info.pred_len
    assert np.isfinite(result[f"{target}_forecast"]).all()


# ─────────────────────────────────────────────────────────────────────────────
# Real checkpoint on generated, non-ETT data (opt-in via KUMO_IMPUTE_CKPT)
#
# Frames are built on the fly with random column names, levels, scales, start dates and frequencies,
# but a known internal structure: target = level + scale * (seasonality + slow trend + AR(1) noise);
# every feature = a lagged, rescaled copy of the target's signal + its own phase-shifted cycle + noise.
# Because the structure is known, the tests check that forecasts are *sensible*, not just finite.
# ─────────────────────────────────────────────────────────────────────────────

SYN_WINDOWS = int(os.environ.get("KUMO_IMPUTE_SYN_WINDOWS", "10"))
# name -> (pandas freq, season length in steps, number of feature columns)
SYN_SPECS = {
    "hourly_daily_cycle_4feat": ("h", 24, 4),
    "15min_short_cycle_2feat": ("15min", 16, 2),
    "daily_weekly_cycle_6feat": ("D", 7, 6),
    "univariate_hourly": ("h", 24, 0),
    "wide_12_channels": ("h", 12, 11),
}


def _random_name(rng, taken: set[str]) -> str:
    stems = ["sensor", "meter", "load", "flow", "temp", "kpi", "rate", "signal", "node", "unit"]
    while True:
        name = f"{rng.choice(stems)}_{''.join(rng.choice(list('abcdefghjkmnpqrstuvwxyz0123456789'), 5))}"
        if name not in taken:
            taken.add(name)
            return name


def _synthetic_frame(spec: str, n_rows: int, seed: int = 0) -> tuple[pd.DataFrame, str, int]:
    """-> (frame with a 'timestamp' column + random-named numeric columns, target column name, season length)."""
    freq, period, n_feat = SYN_SPECS[spec]
    rng = np.random.default_rng(seed)
    t = np.arange(n_rows, dtype=np.float64)

    def ar1(sigma: float) -> np.ndarray:
        e = rng.normal(0, sigma, n_rows)
        out = np.empty(n_rows)
        out[0] = e[0]
        for i in range(1, n_rows):
            out[i] = 0.6 * out[i - 1] + e[i]
        return out

    base = np.sin(2 * np.pi * t / period) + 0.5 * np.sin(4 * np.pi * t / period + 0.7) + 0.002 * t + ar1(0.12)
    taken: set[str] = set()
    target = _random_name(rng, taken)
    cols = {target: rng.uniform(-50, 500) + rng.uniform(0.5, 40) * base}
    for k in range(n_feat):
        lag = int(rng.integers(1, period))
        own = np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
        signal = 0.7 * np.roll(base, lag) + 0.3 * own + ar1(0.1)
        cols[_random_name(rng, taken)] = rng.uniform(-100, 100) + rng.uniform(0.1, 20) * signal
    order = list(cols)
    rng.shuffle(order)  # the target is not necessarily the first column
    start = pd.Timestamp("2015-01-01") + pd.Timedelta(days=int(rng.integers(0, 3000)))
    df = pd.DataFrame({"timestamp": pd.date_range(start, periods=n_rows, freq=freq), **{c: cols[c] for c in order}})
    return df, target, period


def _syn_config(info, target: str, **overrides) -> forecasting.ForecastingConfig:
    options = dict(
        target_column=target,
        seq_len=info.seq_len,
        forecast_horizon=info.pred_len,
        handle_missingness=True,
        impute_ckpt=REAL_CKPT,
        impute_channel_alignment="positional",  # random names cannot match the training schema
    )
    options.update(overrides)
    return forecasting.ForecastingConfig(**options)


def _syn_eval(info, df, target, period, *, rate=0.0, pattern="point", windows=SYN_WINDOWS, seed=5):
    """MAE (in target-std units) of the model and three baselines over the last `windows` origins."""
    h, L = info.pred_len, info.seq_len
    value_cols = [c for c in df.columns if c != "timestamp"]
    sd = float(df[target].std()) or 1.0
    config = _syn_config(info, target)
    rng = np.random.default_rng(seed)
    errs = {"model": [], "window_mean": [], "last_value": [], "seasonal_naive": []}
    flat = []
    for w in range(windows):
        origin = len(df) - h - w * h
        history = df.iloc[:origin].copy()
        idx = history.index[-L:]
        if rate > 0:
            history.loc[idx, value_cols] = _inject_missingness(history.loc[idx], value_cols, target, pattern, rate, rng)
        future = df[target].to_numpy()[origin : origin + h]
        pred = forecasting.perform_forecasting(history, config=config)[f"{target}_forecast"].to_numpy()
        assert np.isfinite(pred).all()
        clean = df[target].to_numpy()[:origin]
        errs["model"].append(np.abs(pred - future))
        flat.append(np.std(pred) / (np.std(future) + 1e-12))
        errs["window_mean"].append(np.abs(clean[-L:].mean() - future))
        errs["last_value"].append(np.abs(clean[-1] - future))
        errs["seasonal_naive"].append(np.abs(np.resize(clean[-period:], h) - future))
    out = {k: float(np.mean(v)) / sd for k, v in errs.items()}
    out["flatness"] = float(np.median(flat))  # std(forecast)/std(actual) over the horizon; ~0 = flat line
    return out


@real_checkpoint
@pytest.mark.parametrize("spec", list(SYN_SPECS))
def test_real_synthetic_forecasts_are_sensible(real_cached, spec):
    """Release gate on clean structured data with any names/frequency/width: finite forecasts, no worse than
    1.2 x the last-value (persistence) forecast; baselines are printed alongside for reference."""
    _, info = real_cached
    df, target, period = _synthetic_frame(
        spec, info.seq_len * 6 + info.pred_len * SYN_WINDOWS, seed=101 + list(SYN_SPECS).index(spec)
    )
    m = _syn_eval(info, df, target, period)
    print(
        f"\nreal ckpt synthetic | {spec:26s} | channels={df.shape[1] - 1:2d} | MAE model={m['model']:.3f} "
        f"window_mean={m['window_mean']:.3f} last_value={m['last_value']:.3f} seasonal_naive={m['seasonal_naive']:.3f} "
        f"flatness={m['flatness']:.2f}"
    )
    assert m["model"] <= 1.2 * m["last_value"], m


@real_checkpoint
@pytest.mark.parametrize("pattern", ["point", "block"])
def test_real_synthetic_missingness_degrades_gracefully(real_cached, pattern):
    """Gaps in the input window may cost accuracy, but the model must stay well-behaved and better than flat."""
    _, info = real_cached
    df, target, period = _synthetic_frame(
        "hourly_daily_cycle_4feat", info.seq_len * 6 + info.pred_len * SYN_WINDOWS, seed=11
    )
    results = {rate: _syn_eval(info, df, target, period, rate=rate, pattern=pattern) for rate in (0.0, 0.3, 0.5)}
    for rate, m in results.items():
        print(
            f"\nreal ckpt synthetic | missing {pattern:5s} rate={rate:.1f} | MAE model={m['model']:.3f} "
            f"window_mean={m['window_mean']:.3f} seasonal_naive={m['seasonal_naive']:.3f}"
        )
    assert results[0.3]["model"] <= 2.0 * results[0.0]["model"] + 0.05, results
    assert results[0.5]["model"] <= 1.2 * results[0.5]["last_value"], results


@real_checkpoint
def test_real_synthetic_affine_equivariance(real_cached):
    """Rescaling/shifting every column (units change) must rescale/shift the forecast identically."""
    _, info = real_cached
    df, target, _ = _synthetic_frame("hourly_daily_cycle_4feat", info.seq_len * 4, seed=3)
    value_cols = [c for c in df.columns if c != "timestamp"]
    rng = np.random.default_rng(0)
    df.loc[df.index[-info.seq_len :], value_cols] = _inject_missingness(
        df.iloc[-info.seq_len :], value_cols, target, "point", 0.2, rng
    )
    a = {c: rng.uniform(0.01, 1000) for c in value_cols}
    b = {c: rng.uniform(-1e4, 1e4) for c in value_cols}
    moved = df.copy()
    for c in value_cols:
        moved[c] = a[c] * df[c] + b[c]
    config = _syn_config(info, target)
    f = forecasting.perform_forecasting(df, config=config)[f"{target}_forecast"].to_numpy()
    g = forecasting.perform_forecasting(moved, config=config)[f"{target}_forecast"].to_numpy()
    scale = a[target] * float(df[target].std())
    np.testing.assert_allclose(g, a[target] * f + b[target], rtol=1e-4, atol=1e-3 * scale)


@real_checkpoint
def test_real_synthetic_random_names_need_positional(real_cached):
    """Unknown names are rejected under name alignment; positional works and outputs keep the caller's names."""
    _, info = real_cached
    df, target, _ = _synthetic_frame("daily_weekly_cycle_6feat", info.seq_len * 3, seed=8)
    if info.channel_schema:
        with pytest.raises(ValueError, match="not among the impute checkpoint's training channels"):
            forecasting.perform_forecasting(df, config=_syn_config(info, target, impute_channel_alignment="name"))
    out = forecasting.perform_forecasting(df, config=_syn_config(info, target, return_all_channels=True))
    value_cols = [c for c in df.columns if c != "timestamp"]
    assert list(out.columns) == ["timestamp", f"{target}_forecast"] + [
        f"{c}_forecast" for c in value_cols if c != target
    ]
    assert np.isfinite(out.drop(columns="timestamp").to_numpy()).all()
    assert out["timestamp"].iloc[0] == df["timestamp"].iloc[-1] + pd.Timedelta(days=1)


@real_checkpoint
def test_real_synthetic_long_horizon_rollout_is_consistent(real_cached):
    """A 3 x pred_len forecast must start with exactly the single-block forecast (same scaler, same first pass)."""
    _, info = real_cached
    df, target, _ = _synthetic_frame("15min_short_cycle_2feat", info.seq_len * 4, seed=4)
    short = forecasting.perform_forecasting(df, config=_syn_config(info, target))[f"{target}_forecast"].to_numpy()
    long = forecasting.perform_forecasting(df, config=_syn_config(info, target, forecast_horizon=3 * info.pred_len))
    long = long[f"{target}_forecast"].to_numpy()
    assert len(long) == 3 * info.pred_len and np.isfinite(long).all()
    np.testing.assert_allclose(long[: info.pred_len], short, rtol=1e-5, atol=1e-6 * float(df[target].std()))
