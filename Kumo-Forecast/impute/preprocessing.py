# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data preprocessing and loading pipeline for time series imputation.

Supports multiple missingness patterns (point, block, column dropout)
and standard train/val/test splits with observed-value-only standardization.
"""

import os
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def get_point_missing_mask(data: np.ndarray, rate: float, rng=None) -> np.ndarray:
    """MCAR point missingness. Returns mask: 1=observed, 0=missing.

    Args:
        rng: Optional np.random.RandomState for reproducible per-sample generation.
             Defaults to the global np.random state.
    """
    if rate <= 0:
        return np.ones_like(data)
    if rate >= 1:
        return np.zeros_like(data)
    _rng = rng if rng is not None else np.random
    mask = _rng.binomial(1, 1.0 - rate, size=data.shape).astype(np.int32)
    return mask


def get_col_dropout_mask(data: np.ndarray, rate: float, rng=None) -> np.ndarray:
    """Column (variate) dropout: each variable independently dropped with prob=rate.

    Args:
        rng: Optional np.random.RandomState for reproducible per-sample generation.
    """
    _rng = rng if rng is not None else np.random
    mask = np.ones_like(data)
    n_cols = data.shape[1]
    n_drop = int(round(n_cols * rate))
    drop_idx = _rng.choice(n_cols, size=n_drop, replace=False)
    mask[:, drop_idx] = 0
    return mask.astype(np.int32)


def get_block_missing_mask(
    data: np.ndarray,
    rate: float,
    block_size: Tuple[int, int] = (24, 1),
    min_block: int = 24,
    max_block: int = 96,
    rng=None,
) -> np.ndarray:
    """Block missingness: one contiguous block per channel.

    Block length is drawn uniformly from:
        [lo, hi] = [clamp(rate*rows * 0.5, min_block, max_block),
                    clamp(rate*rows * 1.5, min_block, max_block)]

    This keeps the expected gap proportional to `rate` while honouring
    min_block / max_block, so 0.2 / 0.4 / 0.6 represent distinct
    missingness severities.  The upper bound is also clamped to (rows - 8)
    so at least one patch worth of timesteps remains observed per channel,
    preventing NaN in RevIN.

    Args:
        block_size: Kept for API compatibility; unused.
        min_block:  Hard lower bound on block length (default 24).
        max_block:  Hard upper bound on block length (default 96).
        rng: Optional np.random.RandomState for per-sample reproducibility.
    """
    if rate <= 0:
        return np.ones_like(data)
    if rate >= 1:
        return np.zeros_like(data)
    _rng = rng if rng is not None else np.random
    mask = np.ones_like(data)
    rows, cols = data.shape

    # Rate-derived target, ±50%, then clamped to [min_block, max_block] and
    # additionally capped at (rows - 8) to guarantee at least one observed patch.
    _safe_max = min(max_block, rows - 8)
    target = max(1, int(rows * rate))
    lo = int(np.clip(target * 0.5, min_block, _safe_max))
    hi = int(np.clip(target * 1.5, min_block, _safe_max))
    lo = min(lo, hi)

    for ch in range(cols):
        bh = int(_rng.randint(lo, hi + 1))
        r = _rng.randint(0, max(1, rows - bh + 1))
        mask[r : r + bh, ch] = 0

    return mask.astype(np.int32)


def make_mask(
    data: np.ndarray,
    pattern: str,
    rate: float,
    block_height: int = 24,
    block_width: int = 1,
    min_block: int = 24,
    max_block: int = 96,
    protect_target_channel: bool = False,
    rng=None,
) -> np.ndarray:
    """Dispatch to the right mask generator.

    Args:
        block_height: Fallback fixed block height (used only when min_block is
            explicitly set to 0, disabling random draw).
        block_width:  Kept for API compatibility; blocks are always 1-channel-wide.
        min_block:    Minimum random block length for block pattern (default 24).
        max_block:    Maximum random block length for block pattern (default 96).
        protect_target_channel: If True, channel 0 (the forecast target) is
            always fully observed in the returned mask regardless of pattern or
            rate.  All other channels are masked normally.
        rng: Optional np.random.RandomState for per-sample reproducibility.
    """
    if pattern == "point":
        mask = get_point_missing_mask(data, rate, rng=rng)
    elif pattern == "col":
        mask = get_col_dropout_mask(data, rate, rng=rng)
    elif pattern == "block":
        mask = get_block_missing_mask(
            data,
            rate,
            block_size=(block_height, block_width),
            min_block=min_block,
            max_block=max_block,
            rng=rng,
        )
    else:
        raise ValueError(f"Unknown pattern: {pattern!r}. Choose 'point','block','col'.")

    if protect_target_channel and data.ndim == 2:
        # data shape: [T, C] — force channel 0 fully observed
        mask[:, 0] = 1
    return mask


def _load_legacy_hdf5(path: str) -> np.ndarray:
    """Load legacy HDF5 files using h5py."""
    import h5py

    with h5py.File(path, "r") as f:

        def _find_data(node):
            if isinstance(node, h5py.Dataset):
                return node[:]
            for key in node.keys():
                result = _find_data(node[key])
                if result is not None:
                    return result
            return None

        for top_key in list(f.keys()):
            group = f[top_key]
            if "block0_values" in group:
                arr = group["block0_values"][:]
                return arr.T.astype(np.float32)
            if isinstance(group, h5py.Dataset):
                return np.array(group[:], dtype=np.float32)

        arrays = []

        def _collect(node):
            if isinstance(node, h5py.Dataset):
                d = node[:]
                if d.ndim >= 1 and np.issubdtype(d.dtype, np.number):
                    arrays.append(d.reshape(-1) if d.ndim == 1 else d)
            elif isinstance(node, h5py.Group):
                for k in node.keys():
                    _collect(node[k])

        _collect(f)
        if arrays:
            arr = np.stack(arrays, axis=-1) if arrays[0].ndim == 1 else np.concatenate(arrays, axis=-1)
            return arr.astype(np.float32)

    raise ValueError(f"Could not extract numeric data from HDF5 file: {path}")


def load_raw_data(dataset: str, data_root: str = "./data") -> np.ndarray:
    """Load raw time series data as (T, N) float32 array."""
    ds = dataset.lower()

    if ds in ("etth1", "etth2"):
        fname = "ETTh1.csv" if "1" in ds else "ETTh2.csv"
        df = pd.read_csv(os.path.join(data_root, "ETT", fname))
        return df.iloc[:, 1:].values.astype(np.float32)

    if ds in ("ettm1", "ettm2"):
        fname = "ETTm1.csv" if "1" in ds else "ETTm2.csv"
        df = pd.read_csv(os.path.join(data_root, "ETT", fname))
        return df.iloc[:, 1:].values.astype(np.float32)

    if ds == "weather":
        df = pd.read_csv(os.path.join(data_root, "weather", "weather.csv"))
        return df.iloc[:, 1:].values.astype(np.float32)

    if ds in ("electricity", "elec"):
        df = pd.read_csv(os.path.join(data_root, "Electricity", "electricity.csv"))
        df = df.iloc[:, 1:]
        return df.values.astype(np.float32)

    if ds in ("pems", "pems-bay", "pemsbay"):
        return _load_legacy_hdf5(os.path.join(data_root, "pems_bay", "pems_bay.h5"))

    if ds in ("metr", "metr-la", "metrla"):
        return _load_legacy_hdf5(os.path.join(data_root, "metr_la", "metr_la.h5"))

    if ds in ("beijingair", "beijing_air", "air"):
        df = pd.read_excel(os.path.join(data_root, "BeijingAirQuality", "BeijingAirQuality.xlsx"))
        data = df.to_numpy().astype(np.float32)
        data = np.nan_to_num(data, nan=0.0)
        return data

    raise ValueError(f"Unknown dataset: {dataset!r}")


def sliding_windows(data: np.ndarray, mask: np.ndarray, seq_len: int, pred_len: int):
    """Slide window over (T, N) data."""
    T = len(data)
    W = T - seq_len - pred_len + 1
    assert W > 0, f"Time series too short: T={T}, seq_len={seq_len}, pred_len={pred_len}"
    X, Y, M = [], [], []
    for i in range(W):
        X.append(data[i : i + seq_len])
        Y.append(data[i + seq_len : i + seq_len + pred_len])
        M.append(mask[i : i + seq_len + pred_len])
    return (np.stack(X).astype(np.float32), np.stack(Y).astype(np.float32), np.stack(M).astype(np.float32))


def sliding_windows_xy(data: np.ndarray, seq_len: int, pred_len: int):
    """Slide window over (T, N) data without a mask.

    Used by DynamicMissingTSDataset which generates masks per-sample
    at __getitem__ time so each training sample sees a freshly drawn mask.
    """
    T = len(data)
    W = T - seq_len - pred_len + 1
    assert W > 0, f"Time series too short: T={T}, seq_len={seq_len}, pred_len={pred_len}"
    X, Y = [], []
    for i in range(W):
        X.append(data[i : i + seq_len])
        Y.append(data[i + seq_len : i + seq_len + pred_len])
    return np.stack(X).astype(np.float32), np.stack(Y).astype(np.float32)


class DynamicMissingTSDataset(Dataset):
    """Per-sample mask generation: each __getitem__ draws a fresh random mask.

    Fix 6: the original MissingTSDataset generates ONE global mask for the
    entire dataset at load time.  All training windows therefore see the same
    fixed missing positions (shifted by window stride), biasing the model to
    memorise which timesteps are typically missing rather than learning general
    missingness robustness.

    This class fixes that by seeding with ``base_seed + idx`` so every sample
    gets an independently drawn mask that still varies deterministically across
    epochs (reproducibility is preserved for a fixed seed).
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        pattern: str,
        rate: float,
        seed: int = 123,
        protect_target_channel: bool = False,
        block_height: int = 24,
        block_width: int = 1,
        min_block: int = 24,
        max_block: int = 96,
        rate_min: float | None = None,
    ):
        """
        X: (W, seq_len, N)  — scaled, unmasked context windows
        Y: (W, pred_len, N) — scaled, clean target windows

        rate_min: If set, the missing rate is sampled from Uniform(rate_min, rate)
            independently per sample, making the model robust to any missingness
            level within that range.  rate becomes the upper bound.
            Defaults to None (fixed rate, backward compatible).
        """
        self.X = X
        self.Y = Y
        self.pattern = pattern
        self.rate = rate
        self.rate_min = rate_min
        self.seed = seed
        self.protect_target_channel = protect_target_channel
        self.block_height = block_height
        self.block_width = block_width
        self.min_block = min_block
        self.max_block = max_block
        self._epoch = 0  # updated each epoch via set_epoch() for train split

    def set_epoch(self, epoch: int):
        """Call at the start of each training epoch to vary masks across epochs.

        Val/test datasets should NOT call this — their masks stay fixed so
        early-stopping and evaluation are deterministic.
        """
        self._epoch = epoch

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_np = self.X[idx]  # [seq_len, N]
        y_np = self.Y[idx]  # [pred_len, N]

        # Seed from (base_seed, epoch, idx) so masks vary across epochs during
        # training while remaining deterministic for a given (epoch, idx) pair.
        rng = np.random.RandomState(self.seed + self._epoch * 100003 + idx)
        # Sample rate from Uniform(rate_min, rate) if range is specified.
        if self.rate_min is not None and self.rate_min < self.rate:
            sample_rate = float(rng.uniform(self.rate_min, self.rate))
        else:
            sample_rate = self.rate
        mask = make_mask(
            x_np,
            self.pattern,
            sample_rate,
            block_height=self.block_height,
            block_width=self.block_width,
            min_block=self.min_block,
            max_block=self.max_block,
            protect_target_channel=self.protect_target_channel,
            rng=rng,
        )  # [seq_len, N]

        x = torch.from_numpy(x_np.copy()).float()  # [seq_len, N]
        y = torch.from_numpy(y_np.copy()).float()  # [pred_len, N]
        m = torch.from_numpy(mask).float()  # [seq_len, N]

        m_ch = m.permute(1, 0)  # [N, seq_len]
        x = x.permute(1, 0) * m_ch  # [N, seq_len]  — zero-fill missing
        y = y.permute(1, 0)  # [N, pred_len]
        input_mask = m.any(dim=-1).float()  # [seq_len]

        return x, y, input_mask, m_ch


def patch_array(arr: np.ndarray, patch_len: int) -> np.ndarray:
    """(W, T, N) → (W, P, N, L) where P = T // patch_len, L = patch_len."""
    W, T, N = arr.shape
    P = T // patch_len
    return arr[:, : P * patch_len, :].reshape(W, P, patch_len, N).transpose(0, 1, 3, 2)


def chronological_split(n: int, val_ratio: float = 0.2, test_ratio: float = 0.2):
    """Chronological train/val/test split."""
    train_end = int(n * (1 - val_ratio - test_ratio))
    val_end = int(n * (1 - test_ratio))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


class ObservedScaler:
    """Standardise using only observed (mask=1) values from the training set."""

    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, x: np.ndarray, mask: np.ndarray) -> "ObservedScaler":
        obs = x[mask.astype(bool)]
        self.mean = float(obs.mean())
        self.std = float(obs.std()) + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


class MissingTSDataset(Dataset):
    """Yields (x_enc, targets, input_mask)."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, mask_seq: np.ndarray, seq_len: int):
        """
        X: (W, seq_len, N)
        Y: (W, pred_len, N)
        mask_seq: (W, seq_len, N)
        """
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.mask = torch.from_numpy(mask_seq).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].permute(1, 0).clone()
        y = self.Y[idx].permute(1, 0)
        m = self.mask[idx]

        m_ch = m.permute(1, 0)
        x = x * m_ch

        input_mask = m.any(dim=-1).float()
        return x, y, input_mask, m_ch


def collate_fn(batch):
    x, y, imask, mmask = zip(*batch, strict=False)
    return (torch.stack(x), torch.stack(y), torch.stack(imask), torch.stack(mmask))


def load_crib_dataset(
    dataset: str,
    data_root: str,
    seq_len: int,
    pred_len: int,
    patch_len: int,
    missing_pattern: str,
    missing_rate: float,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 123,
    block_height: int = 24,
    block_width: int = 1,
    min_block: int = 24,
    max_block: int = 96,
    protect_target_channel: bool = True,
    missing_rate_min: float | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, ObservedScaler, dict]:
    """Full CRIB-compatible data pipeline for Backbone-LF.

    Fix 6: Each training sample now draws a fresh random mask via
    DynamicMissingTSDataset (seeded with base_seed + sample_idx).  This
    removes the fixed-seed memorisation bias of the original single-mask
    approach while preserving determinism across runs.

    Fix 7: protect_target_channel now defaults to True — channel 0 (the
    forecast target) is always fully observed, matching the realistic
    deployment scenario where target history is always available.

    Args:
        protect_target_channel: If True, channel 0 (the forecast target) is
            always fully observed — missingness is applied only to auxiliary
            channels 1-N.  Defaults to True (deployment-realistic).
    """
    np.random.seed(seed)
    random.seed(seed)

    data = load_raw_data(dataset, data_root)
    data = np.nan_to_num(data, nan=0.0).astype(np.float32)
    T, N = data.shape

    # Slide windows on clean data — masks are generated per-sample at __getitem__ time.
    X_all, Y_all = sliding_windows_xy(data, seq_len, pred_len)
    W = len(X_all)

    tr_sl, va_sl, te_sl = chronological_split(W, val_ratio, test_ratio)

    # Fit scaler on training data using a single representative mask.
    # (This is only used for normalisation — dynamic per-sample masks are
    # applied inside DynamicMissingTSDataset.)
    scaler_mask = make_mask(
        data,
        missing_pattern,
        missing_rate,
        block_height=block_height,
        block_width=block_width,
        min_block=min_block,
        max_block=max_block,
        protect_target_channel=protect_target_channel,
    )
    _, _, M_scaler = sliding_windows(data, scaler_mask, seq_len, pred_len)
    M_scaler_seq = M_scaler[:, :seq_len, :]
    scaler = ObservedScaler().fit(X_all[tr_sl], M_scaler_seq[tr_sl])

    def scale(arr):
        return scaler.transform(arr)

    X_tr = scale(X_all[tr_sl])
    Y_tr = scale(Y_all[tr_sl])
    X_va = scale(X_all[va_sl])
    Y_va = scale(Y_all[va_sl])
    X_te = scale(X_all[te_sl])
    Y_te = scale(Y_all[te_sl])

    _ds_kwargs = dict(
        protect_target_channel=protect_target_channel,
        block_height=block_height,
        block_width=block_width,
        min_block=min_block,
        max_block=max_block,
    )
    # Different base seeds per split so train/val/test masks never coincide.
    # rate_min applied to train only — val/test use fixed missing_rate for
    # deterministic evaluation at the declared upper bound.
    tr_ds = DynamicMissingTSDataset(
        X_tr, Y_tr, missing_pattern, missing_rate, seed=seed, rate_min=missing_rate_min, **_ds_kwargs
    )
    va_ds = DynamicMissingTSDataset(X_va, Y_va, missing_pattern, missing_rate, seed=seed + 10_000, **_ds_kwargs)
    te_ds = DynamicMissingTSDataset(X_te, Y_te, missing_pattern, missing_rate, seed=seed + 20_000, **_ds_kwargs)

    kw = dict(
        batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available()
    )
    tr_loader = DataLoader(tr_ds, shuffle=True, **kw)
    va_loader = DataLoader(va_ds, shuffle=False, **kw)
    te_loader = DataLoader(te_ds, shuffle=False, **kw)

    meta = dict(
        n_channels=N,
        seq_len=seq_len,
        pred_len=pred_len,
        patch_len=patch_len,
        dataset=dataset,
        missing_pattern=missing_pattern,
        missing_rate=missing_rate,
        missing_rate_min=missing_rate_min,
        min_block=min_block,
        max_block=max_block,
        protect_target_channel=protect_target_channel,
        train_size=len(tr_ds),
        val_size=len(va_ds),
        test_size=len(te_ds),
    )
    return tr_loader, va_loader, te_loader, scaler, meta


@dataclass
class ChannelObservedScaler:
    """Per-channel standardizer fit only on observed training values."""

    mean: np.ndarray
    std: np.ndarray
    channels: list[str]
    eps: float = 1e-8

    @classmethod
    def fit(cls, values: np.ndarray, observed_mask: np.ndarray, channels: list[str]) -> "ChannelObservedScaler":
        values = np.asarray(values, dtype=np.float32)
        observed_mask = np.asarray(observed_mask, dtype=bool)
        means = np.zeros(values.shape[1], dtype=np.float32)
        stds = np.ones(values.shape[1], dtype=np.float32)

        for channel_idx in range(values.shape[1]):
            obs = values[:, channel_idx][observed_mask[:, channel_idx]]
            if obs.size == 0:
                obs = values[:, channel_idx]
            means[channel_idx] = float(obs.mean())
            std = float(obs.std())
            stds[channel_idx] = 1.0 if std < 1e-8 else std

        return cls(mean=means, std=stds, channels=list(channels))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / (self.std + self.eps)).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * (self.std + self.eps) + self.mean

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "channels": list(self.channels),
            "eps": self.eps,
        }


def _resolve_timestamp_col(df: pd.DataFrame, requested: str = "timestamp") -> str:
    if requested in df.columns:
        return requested
    for candidate in ("timestamp", "date", "ds", "time"):
        if candidate in df.columns:
            return candidate
    return str(df.columns[0])


def _read_numeric_csv(
    csv_path: str,
    target_col: str | None,
    timestamp_col: str = "timestamp",
    max_channels: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    timestamp_col = _resolve_timestamp_col(df, timestamp_col)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.sort_values(timestamp_col)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError(f"No numeric columns found in {csv_path}")

    if target_col is not None:
        if target_col not in numeric_cols:
            raise ValueError(f"target column '{target_col}' not found among numeric columns in {csv_path}")
        numeric_cols = [target_col] + [col for col in numeric_cols if col != target_col]

    if max_channels is not None and len(numeric_cols) > max_channels:
        numeric_cols = numeric_cols[:max_channels]

    values = df[[timestamp_col] + numeric_cols].dropna()[numeric_cols].to_numpy(dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return values, numeric_cols


def _split_bounds(total: int, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    train_end = int(round(total * (1.0 - val_ratio - test_ratio)))
    val_end = int(round(total * (1.0 - test_ratio)))
    return train_end, val_end


class CSVForecastMissingDataset(Dataset):
    """CSV forecasting windows with synthetic missingness masks.

    Returns `(history, future, temporal_mask, channel_mask, valid_channel_mask)`.
    History and future are shaped `[C, L]` and `[C, H]`; `channel_mask` is
    `[C, L]` with 1 for observed input values.
    """

    def __init__(
        self,
        csv_path: str,
        data_split: str,
        seq_len: int,
        pred_len: int,
        target_col: str | None = None,
        missing_pattern: str = "point",
        missing_rate: float = 0.2,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        random_seed: int = 13,
        standardize: bool = True,
        stride: int | None = None,
        timestamp_col: str = "timestamp",
        max_channels: int | None = None,
        protect_target_channel: bool = False,
        rate_min: float | None = None,
        min_block: int = 24,
        max_block: int = 96,
    ) -> None:
        super().__init__()
        if data_split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {data_split}")

        self.csv_path = str(csv_path)
        self.split = data_split
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.stride = int(stride or pred_len)
        self.missing_pattern = missing_pattern
        self.missing_rate = float(missing_rate)
        self.rate_min = rate_min
        self.protect_target_channel = protect_target_channel
        self.min_block = min_block
        self.max_block = max_block
        self._random_seed = random_seed
        self._epoch = 0  # updated via set_epoch() each training epoch

        raw_values, channels = _read_numeric_csv(
            csv_path=self.csv_path,
            target_col=target_col,
            timestamp_col=timestamp_col,
            max_channels=max_channels,
        )
        if raw_values.shape[0] < self.seq_len + self.pred_len:
            raise ValueError(f"Series too short for {self.seq_len}+{self.pred_len}: {csv_path}")

        self.channels = channels
        train_end, val_end = _split_bounds(raw_values.shape[0], val_ratio, test_ratio)
        if train_end < self.seq_len + self.pred_len:
            raise ValueError(f"Training split too short in {csv_path}")

        # Generate train mask first (always with split_offset=0) so the scaler
        # is always fit on the same observed values regardless of which split
        # this instance is loading.  Each split then generates its own mask for
        # the actual missingness pattern used during that split's windows.
        rng_state = np.random.get_state()
        try:
            np.random.seed(random_seed)  # train offset = 0
            train_mask = make_mask(
                raw_values,
                missing_pattern,
                missing_rate,
                min_block=min_block,
                max_block=max_block,
                protect_target_channel=protect_target_channel,
            )
            split_offset = {"train": 0, "val": 10_000, "test": 20_000}[data_split]
            if split_offset == 0:
                full_mask = train_mask
            else:
                np.random.seed(random_seed + split_offset)
                full_mask = make_mask(
                    raw_values,
                    missing_pattern,
                    missing_rate,
                    min_block=min_block,
                    max_block=max_block,
                    protect_target_channel=protect_target_channel,
                )
        finally:
            np.random.set_state(rng_state)

        if standardize:
            # Fit scaler on train-split observed values only (consistent across splits).
            self.standardizer = ChannelObservedScaler.fit(
                raw_values[:train_end],
                train_mask[:train_end],  # always train mask, not split mask
                channels=self.channels,
            )
            values = self.standardizer.transform(raw_values)
        else:
            self.standardizer = None
            values = raw_values.astype(np.float32)

        self.values = values
        # Train split uses dynamic per-sample masks (generated in __getitem__)
        # so the model sees a fresh mask every sample — same as DynamicMissingTSDataset.
        # Val/test use a fixed static mask for deterministic evaluation.
        if data_split == "train" and rate_min is not None:
            self.observed_mask = None  # dynamic: generated per sample in __getitem__
        else:
            self.observed_mask = full_mask.astype(np.float32)
        self.realized_missing_rate = float(1.0 - full_mask.mean())

        if data_split == "train":
            start, stop = 0, train_end
        elif data_split == "val":
            start, stop = train_end, val_end
        else:
            start, stop = val_end, raw_values.shape[0]

        self._starts = list(range(start, stop - (self.seq_len + self.pred_len) + 1, self.stride))
        if data_split == "train":
            rng = np.random.RandomState(random_seed)
            rng.shuffle(self._starts)

    def __len__(self) -> int:
        return len(self._starts)

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each training epoch so __getitem__ generates fresh masks."""
        self._epoch = epoch

    def __getitem__(self, idx):
        start = self._starts[idx]
        hist_end = start + self.seq_len
        fut_end = hist_end + self.pred_len

        hist_raw = self.values[start:hist_end]  # [L, C]
        fut = self.values[hist_end:fut_end].T.copy()  # [C, H]

        if self.observed_mask is None:
            # Dynamic per-sample mask (train split with rate_min).
            # Seed from (base_seed, epoch, idx) for deterministic-but-varied masks.
            rng = np.random.RandomState(self._random_seed + self._epoch * 100003 + idx)
            rate = self.missing_rate
            if self.rate_min is not None and self.rate_min < rate:
                rate = float(rng.uniform(self.rate_min, rate))
            window_mask = make_mask(
                hist_raw,
                self.missing_pattern,
                rate,
                min_block=self.min_block,
                max_block=self.max_block,
                protect_target_channel=self.protect_target_channel,
                rng=rng,
            )  # [L, C]
            channel_mask = window_mask.T.astype(np.float32)  # [C, L]
        else:
            channel_mask = self.observed_mask[start:hist_end].T.copy()  # [C, L]

        hist = hist_raw.T.copy() * channel_mask  # [C, L]
        temporal_mask = (channel_mask.sum(axis=0) > 0).astype(np.float32)
        valid_channel_mask = np.ones((hist.shape[0],), dtype=np.float32)

        return (
            torch.from_numpy(hist).float(),
            torch.from_numpy(fut).float(),
            torch.from_numpy(temporal_mask).float(),
            torch.from_numpy(channel_mask).float(),
            torch.from_numpy(valid_channel_mask).float(),
        )

    @property
    def metadata_summary(self) -> dict:
        return {
            "csv_path": self.csv_path,
            "channels": self.channels,
            "requested_missing_rate": self.missing_rate,
            "realized_missing_rate": round(self.realized_missing_rate, 4),
            "standardizer": self.standardizer.to_dict() if self.standardizer is not None else None,
            "windows": len(self),
        }


class MultiCSVForecastMissingDataset(Dataset):
    """Combine multiple CSV splits, padding each sample to a common channel count."""

    def __init__(
        self,
        csv_paths: list[str],
        data_split: str,
        seq_len: int,
        pred_len: int,
        target_col: str | None = None,
        missing_pattern: str = "point",
        missing_rate: float = 0.2,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        random_seed: int = 13,
        standardize: bool = True,
        stride: int | None = None,
        timestamp_col: str = "timestamp",
        max_channels: int | None = None,
        protect_target_channel: bool = False,
        rate_min: float | None = None,
    ) -> None:
        super().__init__()
        if not csv_paths:
            raise ValueError("csv_paths must not be empty")

        self.datasets: list[CSVForecastMissingDataset] = []
        self.dataset_names: list[str] = []
        self.index: list[tuple[int, int]] = []
        self.max_channels = 0
        self.split = data_split
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)

        for csv_path in csv_paths:
            ds = CSVForecastMissingDataset(
                csv_path=csv_path,
                data_split=data_split,
                seq_len=seq_len,
                pred_len=pred_len,
                target_col=target_col,
                missing_pattern=missing_pattern,
                missing_rate=missing_rate,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                random_seed=random_seed,
                standardize=standardize,
                stride=stride,
                timestamp_col=timestamp_col,
                max_channels=max_channels,
                protect_target_channel=protect_target_channel,
                rate_min=rate_min,
            )
            if len(ds) == 0:
                continue
            self.datasets.append(ds)
            self.dataset_names.append(str(csv_path))
            self.max_channels = max(self.max_channels, len(ds.channels))

        if not self.datasets:
            raise ValueError("No non-empty CSV datasets available")

        union_columns = {channel for ds in self.datasets for channel in ds.channels}
        if target_col and target_col in union_columns:
            self.channels = [target_col] + sorted(channel for channel in union_columns if channel != target_col)
        else:
            self.channels = sorted(union_columns)
        # Padded tensors must cover every column in the union, not just the
        # largest individual CSV.  A={target,a} + B={target,b} → union has 3
        # columns but max_channels-per-file is 2; using 2 would cause an
        # out-of-bounds write when mapping column b to union index 2.
        self.max_channels = len(self.channels)
        target_name = target_col if target_col is not None else "target"
        self.model_channels = [target_name] + [f"aux_{idx}" for idx in range(1, self.max_channels)]

        # Pre-build per-dataset column→union-index maps for name-based alignment.
        self._channel_maps: list[list[int]] = []
        for ds in self.datasets:
            self._channel_maps.append([self.channels.index(col) for col in ds.channels])

        for ds_idx, ds in enumerate(self.datasets):
            for item_idx in range(len(ds)):
                self.index.append((ds_idx, item_idx))

        if data_split == "train":
            rng = np.random.RandomState(random_seed)
            rng.shuffle(self.index)

    def set_epoch(self, epoch: int) -> None:
        """Propagate epoch to all sub-datasets so they generate fresh masks."""
        for ds in self.datasets:
            ds.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx):
        ds_idx, item_idx = self.index[idx]
        hist, fut, temporal_mask, channel_mask, valid_channel_mask = self.datasets[ds_idx][item_idx]
        col_map = self._channel_maps[ds_idx]  # local col_idx → union col_idx

        hist_padded = torch.zeros((self.max_channels, hist.shape[1]), dtype=torch.float32)
        fut_padded = torch.zeros((self.max_channels, fut.shape[1]), dtype=torch.float32)
        channel_mask_padded = torch.zeros((self.max_channels, channel_mask.shape[1]), dtype=torch.float32)
        valid_padded = torch.zeros((self.max_channels,), dtype=torch.float32)

        # Align by column name (col_map maps local index → union index), not
        # by position.  CSVs with different column orderings are handled correctly.
        for local_i, union_i in enumerate(col_map):
            hist_padded[union_i] = hist[local_i]
            fut_padded[union_i] = fut[local_i]
            channel_mask_padded[union_i] = channel_mask[local_i]
            valid_padded[union_i] = valid_channel_mask[local_i]

        return hist_padded, fut_padded, temporal_mask, channel_mask_padded, valid_padded

    @property
    def standardizers(self) -> list[dict]:
        return [ds.metadata_summary for ds in self.datasets]

    @property
    def metadata_summary(self) -> dict:
        return {
            "channels": self.model_channels,
            "source_union_channels": self.channels,
            "source_datasets": self.standardizers,
            "windows": len(self),
        }


def load_csv_forecast_datasets(
    csv_paths: list[str],
    seq_len: int,
    pred_len: int,
    target_col: str | None,
    missing_pattern: str,
    missing_rate: float,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 123,
    stride: int | None = None,
    timestamp_col: str = "timestamp",
    max_channels: int | None = None,
    protect_target_channel: bool = False,
    missing_rate_min: float | None = None,
):
    """Build train/val/test datasets for one or more target-plus-feature CSVs."""
    dataset_cls = MultiCSVForecastMissingDataset if len(csv_paths) > 1 else CSVForecastMissingDataset
    common_kwargs = dict(
        seq_len=seq_len,
        pred_len=pred_len,
        target_col=target_col,
        missing_pattern=missing_pattern,
        missing_rate=missing_rate,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=seed,
        standardize=True,
        stride=stride,
        timestamp_col=timestamp_col,
        max_channels=max_channels,
        protect_target_channel=protect_target_channel,
    )
    # rate_min applied to train split only; val/test always use a fixed rate
    # for deterministic evaluation.
    train_kwargs = {**common_kwargs, "rate_min": missing_rate_min}
    if len(csv_paths) > 1:
        train_ds = dataset_cls(csv_paths=csv_paths, data_split="train", **train_kwargs)
        val_ds = dataset_cls(csv_paths=csv_paths, data_split="val", **common_kwargs)
        test_ds = dataset_cls(csv_paths=csv_paths, data_split="test", **common_kwargs)
    else:
        train_ds = dataset_cls(csv_path=csv_paths[0], data_split="train", **train_kwargs)
        val_ds = dataset_cls(csv_path=csv_paths[0], data_split="val", **common_kwargs)
        test_ds = dataset_cls(csv_path=csv_paths[0], data_split="test", **common_kwargs)

    if hasattr(train_ds, "model_channels"):
        dataset_metadata = train_ds.metadata_summary
    else:
        dataset_metadata = {"datasets": [train_ds.metadata_summary]}

    meta = {
        "n_channels": train_ds.max_channels if hasattr(train_ds, "max_channels") else len(train_ds.channels),
        "channels": train_ds.model_channels if hasattr(train_ds, "model_channels") else train_ds.channels,
        "source_union_channels": train_ds.channels if hasattr(train_ds, "model_channels") else None,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "missing_pattern": missing_pattern,
        "missing_rate": missing_rate,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "csv_paths": list(csv_paths),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "dataset_metadata": dataset_metadata,
    }
    return train_ds, val_ds, test_ds, meta
