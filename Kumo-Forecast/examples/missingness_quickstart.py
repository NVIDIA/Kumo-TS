# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Missingness-aware forecasting quickstart.

Forecasts a multivariate series that has gaps (NaN) with ``handle_missingness=True``: the released
missingness-aware checkpoint (``hf://nvidia/Kumo-Forecast/kumo-forecast-1.2.0``, ~1.3 GB, downloaded
once into the Hugging Face cache) reads the gaps through observation masks instead of zero-filling them.

    cd Kumo-Forecast
    uv run python examples/missingness_quickstart.py                       # synthetic ETT-style data
    uv run python examples/missingness_quickstart.py --csv /path/to/ETTh1.csv --target OT

Columns named like the checkpoint's training channels (OT, HUFL, HULL, LUFL, LULL, MUFL, MULL) are aligned
by name; any other column names need ``impute_channel_alignment="positional"`` (used automatically below).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sdk import ForecastingConfig, perform_forecasting

TRAINING_CHANNELS = {"OT", "HUFL", "HULL", "LUFL", "LULL", "MUFL", "MULL"}


def synthetic_frame(rows: int = 1000, seed: int = 0) -> pd.DataFrame:
    """ETT-like hourly frame: OT plus six load features with a daily cycle and noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(rows, dtype=np.float64)
    day = np.sin(2 * np.pi * t / 24)
    data = {"timestamp": pd.date_range("2024-01-01", periods=rows, freq="h")}
    for i, name in enumerate(["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]):
        data[name] = 5 + 2 * np.roll(day, i) + 0.2 * rng.standard_normal(rows)
    data["OT"] = 20 + 3 * day + 0.3 * rng.standard_normal(rows)
    return pd.DataFrame(data)


def add_gaps(df: pd.DataFrame, rate: float, seed: int = 1) -> pd.DataFrame:
    """Blank `rate` of the values at random, plus one block gap in a feature; keep the last target value."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    values = [c for c in out.columns if c != "timestamp"]
    out[values] = out[values].mask(rng.random((len(out), len(values))) < rate)
    out.loc[out.index[-40:-25], values[0]] = np.nan
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=None, help="CSV with a timestamp/date column and numeric columns")
    parser.add_argument("--target", default="OT")
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--missing-rate", type=float, default=0.15)
    parser.add_argument(
        "--ckpt", default=None, help="hf://... reference or local folder (default: released checkpoint)"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv).rename(columns={"date": "timestamp"}) if args.csv else synthetic_frame()
    df = df.tail(2000).reset_index(drop=True)
    gappy = add_gaps(df, args.missing_rate)
    gappy.loc[gappy.index[-1], args.target] = df[args.target].iloc[-1]
    features = set(gappy.columns) - {"timestamp", args.target}
    alignment = "name" if features <= TRAINING_CHANNELS else "positional"
    print(
        f"{len(gappy)} rows, {int(gappy.drop(columns='timestamp').isna().sum().sum())} missing cells, alignment={alignment}"
    )

    config = ForecastingConfig(
        target_column=args.target,
        forecast_horizon=args.horizon,  # longer than the native 24 steps -> autoregressive rollout
        handle_missingness=True,
        impute_ckpt=args.ckpt,  # None = hf://nvidia/Kumo-Forecast/kumo-forecast-1.2.0
        impute_channel_alignment=alignment,
        return_all_channels=True,
    )
    forecast = perform_forecasting(gappy, config=config)
    print(forecast.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
