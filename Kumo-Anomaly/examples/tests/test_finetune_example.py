# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples import finetune_example


def write_training_csv(path, rows: int = 500) -> None:
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="h")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "sensor_1": np.linspace(0.0, 1.0, rows, dtype=np.float32),
            "sensor_2": np.linspace(1.0, 2.0, rows, dtype=np.float32),
            "is_anomaly": np.zeros(rows, dtype=np.int64),
        }
    )
    df.to_csv(path, index=False)


def make_args(csv_path, *, val_ratio: float = 0.3, val_csv: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        csv=str(csv_path),
        val_csv=val_csv,
        timestamp_col="timestamp",
        label_col="is_anomaly",
        drop_cols=None,
        val_ratio=val_ratio,
    )


def test_parse_args_defaults_to_validation_ratio_that_supports_sample_dataset(monkeypatch, tmp_path):
    csv_path = tmp_path / "train.csv"
    write_training_csv(csv_path)
    monkeypatch.setattr("sys.argv", ["finetune_example.py", "--csv", str(csv_path)])

    args = finetune_example.parse_args()

    assert args.val_ratio == 0.3
    assert args.window_length is None


def test_default_validation_split_creates_windows_for_500_row_dataset(tmp_path):
    csv_path = tmp_path / "train.csv"
    write_training_csv(csv_path, rows=500)

    train_raw, val_raw, columns = finetune_example.split_arrays(make_args(csv_path))

    assert columns == ["sensor_1", "sensor_2"]
    assert len(train_raw) == 350
    assert len(val_raw) == 150

    val_dataset = finetune_example.MaskedWindowDataset(torch.tensor(val_raw), window_length=100, stride=1, split=4)
    assert len(val_dataset) == 51


def test_old_small_validation_split_fails_clearly_for_default_window_length(tmp_path):
    csv_path = tmp_path / "train.csv"
    write_training_csv(csv_path, rows=500)

    _, val_raw, _ = finetune_example.split_arrays(make_args(csv_path, val_ratio=0.1))

    assert len(val_raw) == 50
    with pytest.raises(ValueError, match="Need at least 100 rows"):
        finetune_example.MaskedWindowDataset(torch.tensor(val_raw), window_length=100, stride=1, split=4)


def test_feature_adapter_pads_to_target_dim_and_scales_train_stats():
    data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
    adapter = finetune_example.FeatureAdapter(target_dim=4, scale_factor=20.0, seed=42)

    adapter.fit(data)
    transformed = adapter.transform(data)

    assert transformed.shape == (3, 4)
    assert torch.allclose(transformed[:, 0], torch.tensor([0.0, 10.0, 20.0]))
    assert torch.allclose(transformed[:, 1], torch.tensor([0.0, 10.0, 20.0]))
    assert torch.all(transformed[:, 2:] == 0.0)
