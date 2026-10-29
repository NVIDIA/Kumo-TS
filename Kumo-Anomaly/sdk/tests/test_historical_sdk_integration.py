# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for historical attribution against the current SDK contracts."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdk import anomaly_analysis, historical_attribution


@pytest.mark.parametrize("embedding_mode", ["shared_mean_norm_matched_zero_pad", "positional"])
def test_historical_attribution_preserves_padding_and_replay_settings(monkeypatch, embedding_mode):
    """Historical replay keeps inference options and excludes padded rows from the trace."""
    frame = pd.DataFrame({"signal": np.arange(5, dtype=float)}, index=[10, 20, 30, 40, 50])
    native = {
        "target": np.column_stack((np.arange(6, dtype=float), np.zeros(6))),
        "recon": np.zeros((6, 2)),
        "residual": np.array([0.1, 0.2, 0.8, 0.3, 0.1, 1000.0]),
        "valid_feature_mask": np.array([True, False]),
    }
    inference = Mock(return_value=native)
    monkeypatch.setattr(anomaly_analysis, "_resolve_model_paths", lambda *_: ("resolved.pt", "resolved.yaml"))
    monkeypatch.setattr(anomaly_analysis, "get_model_target_dim", lambda *_: 2)
    monkeypatch.setattr(anomaly_analysis, "inference_ad_tesseract2_mp", inference)
    detector = Mock()
    detector.detect_anomalies.return_value = [False, False, True, False, False, True]
    detector.segments = [(0, 4)]
    detector.segment_bounds = {0: (np.zeros(5), np.full(5, 0.5))}
    detector.percentile_filter = 80.0
    monkeypatch.setattr(anomaly_analysis, "SCSThresholdStrategy", lambda: Mock(scs_thresholder=detector))

    def attribute(input_frame, results, flags, scores, request, replay, model, config, **kwargs):
        pd.testing.assert_frame_equal(input_frame, frame)
        assert results is native
        assert flags.tolist() == [False, False, True, False, False]
        np.testing.assert_array_equal(scores, native["residual"][:5])
        assert (model, config) == ("resolved.pt", "resolved.yaml")
        assert kwargs["threshold_trace"] == {
            "strategy": "scs",
            "sdk_detected": True,
            "segment": [0, 4],
            "lower": 0.0,
            "upper": 0.5,
            "percentile_filter": 80.0,
            "percentile_threshold": pytest.approx(np.percentile(native["residual"][:5], 80)),
            "rule": "outside segment bounds AND score above global percentile",
        }
        replay(input_frame.copy())
        return {"status": "ok"}

    monkeypatch.setattr(historical_attribution, "run_sdk_attribution", attribute)
    output = anomaly_analysis.perform_anomaly_analysis_with_diffusion(
        frame,
        threshold_strategy="scs",
        nsample=7,
        feature_embedding_mode=embedding_mode,
        historical_attribution=historical_attribution.HistoricalAttributionRequest(),
    )

    assert output.index.equals(frame.index)
    assert output.attrs["historical_attribution"] == {"status": "ok"}
    threshold_scores, threshold_target = detector.detect_anomalies.call_args.args
    np.testing.assert_array_equal(threshold_scores, native["residual"][:5])
    np.testing.assert_array_equal(threshold_target, native["target"][:5, :1])
    assert inference.call_count == 2
    for call in inference.call_args_list:
        assert call.kwargs["model_path"] == "resolved.pt"
        assert call.kwargs["config_path"] == "resolved.yaml"
        assert call.kwargs["nsample"] == 7
        assert call.kwargs["preprocess_model_dir"] is None
        assert call.kwargs["feature_embedding_mode"] == embedding_mode


def test_historical_attribution_accepts_saved_schema_with_extra_serving_columns(monkeypatch):
    """A saved direct feature schema remains valid when serving has unused columns."""
    frame = pd.DataFrame({"ignored": np.arange(5), "b": np.arange(5), "a": np.arange(5)})
    native = {
        "target": np.zeros((5, 2)),
        "recon": np.zeros((5, 2)),
        "residual": np.zeros(5),
        "feature_names": ["a", "b"],
    }
    monkeypatch.setattr(anomaly_analysis, "_resolve_model_paths", lambda *_: ("resolved.pt", "resolved.yaml"))
    monkeypatch.setattr(anomaly_analysis, "get_model_target_dim", lambda *_: 2)
    monkeypatch.setattr(anomaly_analysis, "inference_ad_tesseract2_mp", lambda **_: native)
    detector = Mock()
    detector.detect_anomalies.return_value = np.zeros(5, dtype=bool)
    monkeypatch.setattr(anomaly_analysis, "MACSThresholdStrategy", lambda: Mock(macs_thresholder=detector))
    adapter = Mock(return_value={"status": "unavailable", "reason": "no_sdk_events"})
    monkeypatch.setattr(historical_attribution, "run_sdk_attribution", adapter)

    output = anomaly_analysis.perform_anomaly_analysis_with_diffusion(
        frame,
        threshold_strategy="macs",
        historical_attribution=historical_attribution.HistoricalAttributionRequest(),
    )

    adapter.assert_called_once()
    pd.testing.assert_frame_equal(adapter.call_args.args[0], frame)
    assert adapter.call_args.args[1]["feature_names"] == ["a", "b"]
    assert output.attrs["historical_attribution"]["reason"] == "no_sdk_events"
