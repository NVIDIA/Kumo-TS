# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Missingness-aware forecasting path for the Kumo-Forecast SDK.

Used by :func:`sdk.forecasting.perform_forecasting` when
``ForecastingConfig.handle_missingness=True``. Instead of zero-filling NULLs
and running the base forecaster, it runs the ``impute`` model
(Backbone-LF+ / Backbone-LF+ CRS), which reads the real missingness pattern
through per-channel / per-timestep masks.

The input is prepared the way ``impute`` training/evaluation
prepares it (``CSVForecastMissingDataset`` / ``MultiCSVForecastMissingDataset``):

1. **Channel layout** — channels are aligned *by name* to the checkpoint's
   training schema (slot 0 = target). Training channels absent from the input
   are fed as padding (zeros, ``channel_mask=0``, ``valid_channel_mask=0``);
   outputs are mapped back to the input column names. Unknown columns are
   rejected unless ``impute_channel_alignment="positional"``.
2. **Normalization** — one fixed per-channel scaler, fit on *observed* values
   only, like the training-split ``ChannelObservedScaler``:
   ``impute_normalization="history"`` (default) fits it on the rows passed in
   (optionally only the last ``impute_history_rows``); ``"checkpoint"`` reuses
   the standardizer saved with the checkpoint; ``"provided"`` uses caller-supplied
   ``impute_scaler_stats`` (see :func:`fit_impute_scaler_stats`).
   The scaler is never re-fit per window, so the adaptive RevIN priors (mean
   shrinkage toward 0, stdev shrinkage toward the saved prior) keep their
   training meaning.
3. **Masks** are built from the real NaNs before anything is filled; gaps are
   zero-filled in scaled space (== the channel's reference mean).
4. ``missing_rate`` over the valid channels is passed to the model.

Horizons longer than the checkpoint's ``pred_len`` are produced
autoregressively: each prediction block is appended as observed data and the
window slides forward (the scaler stays fixed).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from pathlib import Path

    from sdk.forecasting import ForecastingConfig

logger = logging.getLogger(__name__)

# SDK default for ForecastingConfig.seq_len. When the caller leaves it at the
# default we adopt the impute checkpoint's seq_len instead of failing.
_DEFAULT_SDK_SEQ_LEN = 512
MAX_FORECAST_HORIZON = 512
_LOW_TARGET_OBSERVED_FRACTION = 0.1
# With "history" normalization, fewer rows than this multiple of seq_len
# means the reference statistics are close to window statistics.
_SHORT_HISTORY_FACTOR = 2
_SCALER_EPS = 1e-8  # ChannelObservedScaler.eps

NORMALIZATION_MODES = ("history", "checkpoint", "provided")
ALIGNMENT_MODES = ("name", "positional")

_IMPUTE_MODEL_CACHE: dict[tuple[str, float, str], tuple[Any, Any]] = {}
# hf:// reference -> resolved local checkpoint file, so cached requests don't hit the Hub again.
_HF_RESOLVED: dict[str, Path] = {}
# Cached models carry per-call state (channel-count adaptation, RevIN statistics),
# so inference on a shared model is serialized.
_IMPUTE_INFERENCE_LOCK = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Model cache
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_impute_ckpt(ckpt: str | Path, local_files_only: bool = False) -> Path:
    """Local checkpoint file for ``ckpt``; ``hf://`` references are downloaded once per process."""
    from impute.inference import is_hf_reference, resolve_checkpoint_path

    if is_hf_reference(ckpt):
        if ckpt not in _HF_RESOLVED:
            logger.info("Resolving impute checkpoint %s from the Hugging Face Hub", ckpt)
            _HF_RESOLVED[ckpt] = resolve_checkpoint_path(ckpt, local_files_only=local_files_only)
        return _HF_RESOLVED[ckpt]
    return resolve_checkpoint_path(ckpt)


def _get_impute_model(ckpt: str | Path, device: torch.device, local_files_only: bool = False) -> tuple[Any, Any]:
    """Load (or reuse) an impute model + info for ``ckpt`` on ``device``."""
    from impute.inference import load_impute_model

    ckpt_path = _resolve_impute_ckpt(ckpt, local_files_only=local_files_only)
    try:
        mtime = ckpt_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    key = (str(ckpt_path.resolve()), mtime, str(device))
    if key in _IMPUTE_MODEL_CACHE:
        logger.info("Using cached impute model for %s", ckpt_path)
        return _IMPUTE_MODEL_CACHE[key]

    model, info = load_impute_model(ckpt_path, device=device)
    _IMPUTE_MODEL_CACHE[key] = (model, info)
    return model, info


def clear_impute_model_cache() -> None:
    """Drop all cached impute models (and resolved hf:// paths)."""
    _IMPUTE_MODEL_CACHE.clear()
    _HF_RESOLVED.clear()
    _WARNED_POSITIONAL.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ChannelScaler:
    """Fixed per-channel standardizer (same maths as ``ChannelObservedScaler``)."""

    mean: np.ndarray  # [C]
    std: np.ndarray  # [C] (>0)
    eps: float = _SCALER_EPS

    @classmethod
    def fit(cls, values: np.ndarray) -> ChannelScaler:
        """Fit on the observed (finite) values of ``values`` ``[T, C]``."""
        values = np.asarray(values, dtype=np.float64)
        observed = np.isfinite(values)
        n_obs = observed.sum(axis=0)
        safe = np.where(observed, values, 0.0)
        mean = np.divide(safe.sum(axis=0), n_obs, out=np.zeros(values.shape[1]), where=n_obs > 0)
        centered = np.where(observed, values - mean, 0.0)
        var = np.divide((centered**2).sum(axis=0), n_obs, out=np.ones(values.shape[1]), where=n_obs > 0)
        std = np.sqrt(var)
        return cls(mean=mean, std=np.where(std < 1e-8, 1.0, std))

    @classmethod
    def from_checkpoint(cls, columns: list[str], standardizer: dict[str, Any]) -> ChannelScaler:
        """Map a saved training standardizer onto ``columns`` (target first) by name."""
        names = list(standardizer["channels"])
        mean_all = np.asarray(standardizer["mean"], dtype=np.float64)
        std_all = np.asarray(standardizer["std"], dtype=np.float64)
        idx = [0]  # the input target uses the source dataset's target statistics
        missing = []
        for col in columns[1:]:
            if col in names[1:]:
                idx.append(names.index(col))
            else:
                missing.append(col)
        if missing:
            raise ValueError(
                f"impute_normalization='checkpoint': columns {missing} have no saved statistics in source dataset "
                f"{standardizer.get('name')!r} (channels: {names}). Drop them or use impute_normalization='history'."
            )
        std = std_all[idx]
        return cls(
            mean=mean_all[idx],
            std=np.where(std < 1e-8, 1.0, std),
            eps=float(standardizer.get("eps", _SCALER_EPS)),
        )

    @classmethod
    def from_stats(cls, columns: list[str], stats: dict[str, Any]) -> ChannelScaler:
        """Build from caller-supplied ``{"mean": {col: m}, "std": {col: s}}`` (see ``fit_impute_scaler_stats``)."""
        if (
            not isinstance(stats, dict)
            or not isinstance(stats.get("mean"), dict)
            or not isinstance(stats.get("std"), dict)
        ):
            raise ValueError('impute_scaler_stats must be a mapping {"mean": {column: value}, "std": {column: value}}')
        means, stds = stats["mean"], stats["std"]
        missing = [c for c in columns if c not in means or c not in stds]
        if missing:
            raise ValueError(f"impute_scaler_stats is missing mean/std for columns {missing}")
        mean = np.asarray([float(means[c]) for c in columns], dtype=np.float64)
        std = np.asarray([float(stds[c]) for c in columns], dtype=np.float64)
        bad_mean = [c for c, m in zip(columns, mean, strict=True) if not np.isfinite(m)]
        bad_std = [c for c, v in zip(columns, std, strict=True) if not (np.isfinite(v) and v > 0)]
        if bad_mean or bad_std:
            raise ValueError(
                f"impute_scaler_stats needs finite means and finite, positive stds; "
                f"bad mean: {bad_mean}, bad std: {bad_std}"
            )
        extra = sorted((set(means) | set(stds)) - set(columns))
        if extra:
            logger.warning("impute_scaler_stats has entries for columns not in the input (ignored): %s", extra)
        return cls(mean=mean, std=std)

    def to_stats(self, columns: list[str]) -> dict[str, dict[str, float]]:
        """Serialize as ``{"mean": {col: m}, "std": {col: s}}`` (JSON/YAML friendly)."""
        return {
            "mean": {c: float(m) for c, m in zip(columns, self.mean, strict=True)},
            "std": {c: float(v) for c, v in zip(columns, self.std, strict=True)},
        }

    def transform(self, values: np.ndarray) -> np.ndarray:
        """``[..., C]`` original units -> scaled."""
        return (values - self.mean) / (self.std + self.eps)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """``[C, H]`` scaled -> original units."""
        return values * (self.std[:, None] + self.eps) + self.mean[:, None]


# ─────────────────────────────────────────────────────────────────────────────
# Channel layout (input columns -> model channel slots)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ChannelLayout:
    """Where each input column (target first) sits among the model's channel slots."""

    n_slots: int
    slots: np.ndarray  # [C_in] slot index per input column

    @classmethod
    def identity(cls, n_columns: int) -> ChannelLayout:
        return cls(n_slots=n_columns, slots=np.arange(n_columns))

    @property
    def valid(self) -> np.ndarray:
        valid = np.zeros(self.n_slots, dtype=np.float32)
        valid[self.slots] = 1.0
        return valid


_WARNED_POSITIONAL: set[tuple[str, ...]] = set()  # log the positional-alignment warning once per schema


def resolve_channel_layout(columns: list[str], info: Any, alignment: str = "name") -> ChannelLayout:
    """Align input columns to the checkpoint's training channel schema.

    ``alignment="name"``: the target goes to slot 0 and every feature to the
    slot its name had in training; training channels absent from the input
    become padding. Unknown feature names raise ``ValueError``.
    ``alignment="positional"``: channels are fed in input order (identities
    and per-channel priors are assigned by position).
    """
    if alignment not in ALIGNMENT_MODES:
        raise ValueError(f"impute_channel_alignment must be one of {ALIGNMENT_MODES}, got {alignment!r}")

    schema = getattr(info, "channel_schema", None)
    if alignment == "positional":
        if schema is not None and tuple(schema) not in _WARNED_POSITIONAL:
            _WARNED_POSITIONAL.add(tuple(schema))
            logger.warning(
                "impute_channel_alignment='positional': channels are fed by position, so channel identities "
                "and per-channel priors follow input order rather than the training schema %s",
                schema,
            )
        return ChannelLayout.identity(len(columns))

    if schema is None:
        logger.warning(
            "The impute checkpoint does not record its training channel names; aligning channels by position"
        )
        return ChannelLayout.identity(len(columns))

    slots = [0]
    unknown: list[str] = []
    for col in columns[1:]:
        if col not in schema:
            unknown.append(col)
            continue
        slot = schema.index(col)
        if slot == 0:
            raise ValueError(
                f"Column {col!r} is the impute checkpoint's target channel; set target_column={col!r} "
                "or drop the column."
            )
        slots.append(slot)
    if unknown:
        raise ValueError(
            f"Columns {unknown} are not among the impute checkpoint's training channels {schema[1:]}. "
            "Rename or drop them, or set impute_channel_alignment='positional' to feed channels by position "
            "(channel identities and per-channel priors are then not preserved)."
        )

    if columns[0] != schema[0]:
        logger.info("Target column %r is fed to the checkpoint's target slot (trained as %r)", columns[0], schema[0])
    absent = [name for i, name in enumerate(schema) if i not in set(slots)]
    if absent:
        logger.info("Training channels absent from the input (masked as padding): %s", absent)
    return ChannelLayout(n_slots=len(schema), slots=np.asarray(slots))


# ─────────────────────────────────────────────────────────────────────────────
# Windowing / masks (pure numpy — no model needed)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MissingnessWindow:
    """One model-ready input window built from data that may contain NaNs."""

    x: np.ndarray  # [S, L] scaled, gaps and padding zero-filled
    channel_mask: np.ndarray  # [S, L] 1 = observed
    temporal_mask: np.ndarray  # [L]    1 = any channel observed at this step
    valid_channel_mask: np.ndarray  # [S]    1 = slot fed by an input column, 0 = padding
    missing_rate: float  # over valid slots only (as in training / test.py)
    scaler: ChannelScaler
    layout: ChannelLayout

    def to_output(self, pred: np.ndarray) -> np.ndarray:
        """Map a model forecast ``[S, H]`` (scaled) to input columns ``[C_in, H]`` in original units."""
        return self.scaler.inverse_transform(pred[self.layout.slots])

    def column_observed_fraction(self) -> np.ndarray:
        """Observed fraction per input column ``[C_in]``."""
        return self.channel_mask[self.layout.slots].mean(axis=1)


def build_missingness_window(
    values: np.ndarray,
    seq_len: int,
    scaler: ChannelScaler | None = None,
    layout: ChannelLayout | None = None,
) -> MissingnessWindow:
    """Build masks and a scaled window from the last ``seq_len`` rows.

    Args:
        values: ``[T, C_in]`` float array (target first); NaN marks a missing value.
        seq_len: Window length.
        scaler: Fixed reference scaler. ``None`` fits one on all of ``values``.
        layout: Input-column -> model-slot mapping. ``None`` feeds columns in order.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"values must be 2-D [T, C], got shape {values.shape}")
    if values.shape[0] < seq_len:
        raise ValueError(f"Need at least {seq_len} rows, got {values.shape[0]}")
    scaler = scaler if scaler is not None else ChannelScaler.fit(values)
    layout = layout if layout is not None else ChannelLayout.identity(values.shape[1])

    window = values[-seq_len:]  # [L, C_in]
    observed = np.isfinite(window)
    scaled = np.where(observed, scaler.transform(np.where(observed, window, 0.0)), 0.0)

    x = np.zeros((layout.n_slots, seq_len), dtype=np.float32)
    channel_mask = np.zeros((layout.n_slots, seq_len), dtype=np.float32)
    x[layout.slots] = scaled.T
    channel_mask[layout.slots] = observed.T
    valid = layout.valid
    n_valid = max(float(valid.sum()), 1.0)
    return MissingnessWindow(
        x=x,
        channel_mask=channel_mask,
        temporal_mask=channel_mask.any(axis=0).astype(np.float32),
        valid_channel_mask=valid,
        missing_rate=float(1.0 - channel_mask.sum() / (n_valid * seq_len)),
        scaler=scaler,
        layout=layout,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Input preparation
# ─────────────────────────────────────────────────────────────────────────────


def _prepare_frame(df: pd.DataFrame, cfg: ForecastingConfig) -> tuple[pd.DataFrame, list[str]]:
    """Validate the input and return ``(working_df, columns_to_process)``. NaNs are kept."""
    if cfg.timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column '{cfg.timestamp_column}' not found in DataFrame")
    if df[cfg.timestamp_column].isnull().any():
        raise ValueError(f"Timestamp column '{cfg.timestamp_column}' contains NULL values")

    working_df = df.copy()
    try:
        if not pd.api.types.is_datetime64_any_dtype(working_df[cfg.timestamp_column]):
            working_df[cfg.timestamp_column] = pd.to_datetime(working_df[cfg.timestamp_column])
    except Exception as e:
        raise ValueError(f"Cannot parse timestamp column '{cfg.timestamp_column}' as datetime: {e}") from e

    if cfg.target_column not in df.columns:
        raise ValueError(f"Target column '{cfg.target_column}' not found in DataFrame")
    if not pd.api.types.is_numeric_dtype(working_df[cfg.target_column]):
        raise ValueError(f"Target column '{cfg.target_column}' must contain numeric values")

    numeric_columns = working_df.select_dtypes(include=[np.number]).columns.tolist()
    if cfg.target_column in numeric_columns:
        numeric_columns.remove(cfg.target_column)
    columns_to_process = [cfg.target_column] + numeric_columns

    if cfg.return_all_channels:
        colliding = [c for c in columns_to_process if f"{c}_forecast" == cfg.timestamp_column]
        if colliding:
            raise ValueError(
                f"timestamp_column {cfg.timestamp_column!r} collides with the forecast column "
                f"emitted for input column {colliding[0]!r}; rename one of them"
            )
    return working_df, columns_to_process


def _numeric_columns(df: pd.DataFrame, target_column: str) -> list[str]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError(f"Target column '{target_column}' must contain numeric values")
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [target_column] + [c for c in numeric if c != target_column]


def fit_impute_scaler_stats(df: pd.DataFrame, target_column: str = "target") -> dict[str, dict[str, float]]:
    """Fit reference statistics for ``impute_normalization="provided"``.

    Uses observed (non-NULL) values only — the same maths as the training-split
    ``ChannelObservedScaler`` — over the target and every other numeric column.
    Fit once on a clean reference period, store the result (it is plain
    JSON/YAML), and pass it back as ``impute_scaler_stats`` on every call.
    """
    columns = _numeric_columns(df, target_column)
    return ChannelScaler.fit(df[columns].to_numpy(dtype=np.float64)).to_stats(columns)


def _resolve_seq_len(cfg: ForecastingConfig, ckpt_seq_len: int) -> int:
    if cfg.seq_len == ckpt_seq_len:
        return ckpt_seq_len
    if cfg.seq_len == _DEFAULT_SDK_SEQ_LEN:
        logger.info(
            "handle_missingness: using the impute checkpoint's seq_len=%d (config seq_len left at the default %d)",
            ckpt_seq_len,
            _DEFAULT_SDK_SEQ_LEN,
        )
        return ckpt_seq_len
    raise ValueError(
        f"seq_len={cfg.seq_len} does not match the impute checkpoint's seq_len={ckpt_seq_len}. "
        f"Set seq_len={ckpt_seq_len} (or leave it at the default) when handle_missingness=True."
    )


def _build_scaler(
    values: np.ndarray, columns: list[str], info: Any, cfg: ForecastingConfig, seq_len: int
) -> ChannelScaler:
    normalization = cfg.impute_normalization
    if normalization not in NORMALIZATION_MODES:
        raise ValueError(f"impute_normalization must be one of {NORMALIZATION_MODES}, got {normalization!r}")
    if cfg.impute_scaler_stats is not None and normalization != "provided":
        logger.warning("impute_scaler_stats is ignored unless impute_normalization='provided'")
    if cfg.impute_history_rows is not None and normalization != "history":
        logger.warning("impute_history_rows is ignored unless impute_normalization='history'")

    if normalization == "checkpoint":
        standardizer = info.find_standardizer(cfg.impute_source_dataset)
        logger.info("Using the checkpoint's training standardizer for source dataset %r", standardizer["name"])
        return ChannelScaler.from_checkpoint(columns, standardizer)
    if normalization == "provided":
        if cfg.impute_scaler_stats is None:
            raise ValueError(
                "impute_normalization='provided' requires impute_scaler_stats (e.g. from sdk.fit_impute_scaler_stats)"
            )
        return ChannelScaler.from_stats(columns, cfg.impute_scaler_stats)

    rows = values
    if cfg.impute_history_rows is not None:
        if cfg.impute_history_rows < seq_len:
            raise ValueError(f"impute_history_rows={cfg.impute_history_rows} must be >= seq_len={seq_len}")
        rows = values[-cfg.impute_history_rows :]
    if rows.shape[0] < _SHORT_HISTORY_FACTOR * seq_len:
        logger.warning(
            "impute_normalization='history' with only %d rows (seq_len=%d): reference statistics come from little "
            "more than the input window. Pass a longer history, raise impute_history_rows, or use "
            "impute_normalization='checkpoint' / 'provided'.",
            rows.shape[0],
            seq_len,
        )
    return ChannelScaler.fit(rows)


def _warn_on_window_quality(window: MissingnessWindow, columns: list[str], info: Any) -> None:
    observed_frac = window.column_observed_fraction()
    if observed_frac[0] == 0:
        raise ValueError(
            f"Target column '{columns[0]}' has no observed values in the last {window.x.shape[1]} rows; "
            "nothing to forecast from."
        )
    if observed_frac[0] < _LOW_TARGET_OBSERVED_FRACTION:
        logger.warning(
            "Target column '%s' is only %.1f%% observed in the input window; forecasts may be unreliable",
            columns[0],
            100 * observed_frac[0],
        )
    empty = [col for col, frac in zip(columns[1:], observed_frac[1:], strict=True) if frac == 0]
    if empty:
        logger.warning("Feature columns fully missing in the input window (treated as unobserved): %s", empty)
    cap = getattr(info, "channel_embed_cap", None)
    if cap is not None and window.layout.n_slots > cap:
        logger.warning(
            "Input has %d channels but the impute checkpoint's channel embedding covers %d; "
            "channels beyond %d share one identity embedding",
            window.layout.n_slots,
            cap,
            cap,
        )


def _future_timestamps(times: pd.Series, horizon: int) -> pd.DatetimeIndex:
    time_diffs = times.diff().dropna()
    if len(time_diffs) > 0:
        modes = time_diffs.mode()
        inferred_freq = modes[0] if len(modes) > 0 else time_diffs.median()
    else:
        inferred_freq = pd.Timedelta(hours=1)
    return pd.date_range(start=times.iloc[-1] + inferred_freq, periods=horizon, freq=inferred_freq)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def _predict_block(model: Any, window: MissingnessWindow, device: torch.device) -> np.ndarray:
    """Run one forward pass; returns ``[C_in, pred_len]`` in original units."""
    x = torch.from_numpy(window.x).unsqueeze(0).to(device)
    channel_mask = torch.from_numpy(window.channel_mask).unsqueeze(0).to(device)
    temporal_mask = torch.from_numpy(window.temporal_mask).unsqueeze(0).to(device)
    valid_channel_mask = torch.from_numpy(window.valid_channel_mask).unsqueeze(0).to(device)

    out = model(
        x,
        input_mask=temporal_mask,
        channel_mask=channel_mask,
        valid_channel_mask=valid_channel_mask,
        missing_rate=window.missing_rate,
    )
    pred = out.forecast[0].detach().float().cpu().numpy()  # [S, pred_len]
    if not np.all(np.isfinite(pred[window.layout.slots])):
        raise RuntimeError("Impute model produced non-finite forecasts; check the input window and checkpoint")
    return window.to_output(pred)


def forecast_with_missingness(
    model: Any,
    values: np.ndarray,
    seq_len: int,
    pred_len: int,
    forecast_horizon: int,
    device: torch.device,
    scaler: ChannelScaler | None = None,
    layout: ChannelLayout | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Forecast ``forecast_horizon`` steps from ``values`` ``[T, C_in]`` (NaN = missing).

    ``scaler`` defaults to one fit on all of ``values`` and stays fixed across
    autoregressive steps. Returns ``(forecast [C_in, forecast_horizon], missing_rates_per_step)``.
    """
    values = np.asarray(values, dtype=np.float64)
    scaler = scaler if scaler is not None else ChannelScaler.fit(values)
    history = values[-seq_len:]
    blocks: list[np.ndarray] = []
    rates: list[float] = []
    produced = 0
    while produced < forecast_horizon:
        window = build_missingness_window(history, seq_len, scaler=scaler, layout=layout)
        rates.append(window.missing_rate)
        block = _predict_block(model, window, device)  # [C_in, pred_len]
        blocks.append(block)
        produced += block.shape[1]
        # Predictions become fully observed history for the next step.
        history = np.concatenate([history, block.T], axis=0)[-seq_len:]
    return np.concatenate(blocks, axis=1)[:, :forecast_horizon], rates


def perform_missingness_aware_forecasting(
    df: pd.DataFrame,
    cfg: ForecastingConfig,
    device: torch.device,
) -> pd.DataFrame:
    """Missingness-aware counterpart of ``perform_forecasting`` (same output format)."""
    if cfg.forecast_horizon <= 0:
        raise ValueError(f"forecast_horizon must be positive, got {cfg.forecast_horizon}")
    if cfg.forecast_horizon > MAX_FORECAST_HORIZON:
        raise ValueError(f"forecast_horizon must be <= {MAX_FORECAST_HORIZON}, got {cfg.forecast_horizon}")
    if cfg.impute_normalization not in NORMALIZATION_MODES:
        raise ValueError(f"impute_normalization must be one of {NORMALIZATION_MODES}, got {cfg.impute_normalization!r}")
    history_rows = cfg.impute_history_rows
    if history_rows is not None and (
        isinstance(history_rows, bool) or not isinstance(history_rows, (int, np.integer)) or history_rows < 1
    ):
        raise ValueError(f"impute_history_rows must be a positive integer or None, got {history_rows!r}")
    if cfg.impute_channel_alignment not in ALIGNMENT_MODES:
        raise ValueError(
            f"impute_channel_alignment must be one of {ALIGNMENT_MODES}, got {cfg.impute_channel_alignment!r}"
        )
    if df is None or df.empty:
        raise ValueError("Input DataFrame is required and cannot be empty")

    working_df, columns = _prepare_frame(df, cfg)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    from impute.inference import DEFAULT_IMPUTE_CKPT

    ckpt = cfg.impute_ckpt or DEFAULT_IMPUTE_CKPT
    model, info = _get_impute_model(ckpt, device, local_files_only=cfg.local_files_only)
    seq_len = _resolve_seq_len(cfg, info.seq_len)
    if len(working_df) < seq_len:
        raise ValueError(f"DataFrame has {len(working_df)} rows but seq_len requires at least {seq_len} rows")

    from impute.inference import set_channel_count

    values = working_df[columns].to_numpy(dtype=np.float64)  # NaN preserved
    layout = resolve_channel_layout(columns, info, cfg.impute_channel_alignment)
    scaler = _build_scaler(values, columns, info, cfg, seq_len)
    first_window = build_missingness_window(values, seq_len, scaler=scaler, layout=layout)
    _warn_on_window_quality(first_window, columns, info)

    logger.info(
        "handle_missingness: %s | input channels=%d model channels=%d seq_len=%d pred_len=%d | "
        "normalization=%s alignment=%s | input window missing_rate=%.3f (%d values)",
        type(model).__name__,
        len(columns),
        layout.n_slots,
        seq_len,
        info.pred_len,
        cfg.impute_normalization,
        cfg.impute_channel_alignment,
        first_window.missing_rate,
        int(np.isnan(values[-seq_len:]).sum()),
    )
    if cfg.forecast_horizon > info.pred_len:
        logger.info(
            "forecast_horizon=%d > impute pred_len=%d: autoregressive rollout (%d steps)",
            cfg.forecast_horizon,
            info.pred_len,
            -(-cfg.forecast_horizon // info.pred_len),
        )

    with _IMPUTE_INFERENCE_LOCK:
        set_channel_count(model, info, layout.n_slots)
        forecast, _ = forecast_with_missingness(
            model, values, seq_len, info.pred_len, cfg.forecast_horizon, device, scaler=scaler, layout=layout
        )  # [C_in, H]

    forecast_timestamps = _future_timestamps(working_df[cfg.timestamp_column], cfg.forecast_horizon)
    output_channels = columns if cfg.return_all_channels else [cfg.target_column]
    cols: dict[str, Any] = {cfg.timestamp_column: forecast_timestamps}
    for ch_idx, name in enumerate(output_channels):
        cols[f"{name}_forecast"] = forecast[ch_idx].tolist()
    result_df = pd.DataFrame(cols)

    if cfg.save_preds:
        result_df.to_csv(cfg.save_preds, index=False)
        logger.info("Saved predictions to %s", cfg.save_preds)

    logger.info("Added columns: %s", ", ".join(f"{c}_forecast" for c in output_channels))
    return result_df


__all__ = [
    "ALIGNMENT_MODES",
    "NORMALIZATION_MODES",
    "ChannelLayout",
    "ChannelScaler",
    "MissingnessWindow",
    "build_missingness_window",
    "clear_impute_model_cache",
    "fit_impute_scaler_stats",
    "forecast_with_missingness",
    "perform_missingness_aware_forecasting",
    "resolve_channel_layout",
]
