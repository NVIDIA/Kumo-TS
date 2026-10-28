# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for anomaly-detection PDF reporting."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdk import reporting
from sdk.reporting import (
    _aggregate_explanation_contributions,
    _create_contribution_graph_page,
    _create_mae_note_page,
    _resolve_diagnosis_rows,
    _resolve_explanation_rows,
    _resolve_x_axis,
    generate_anomaly_detection_report,
    infer_ground_truth_column,
)


def _report_frame() -> pd.DataFrame:
    """Create a compact result frame with timestamps, predictions, and labels."""
    samples = 12
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=samples, freq="min"),
            "temperature": np.linspace(20.0, 24.0, samples),
            "pressure": np.sin(np.linspace(0.0, 2.0 * np.pi, samples)),
            "GT": [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            "Anomaly": [False, False, False, True, False, False, False, True, False, False, False, False],
            "MAE": np.linspace(0.05, 0.7, samples),
        }
    )


def test_generate_report_with_detected_and_ground_truth_anomalies(tmp_path: Path) -> None:
    """A labeled result should produce a non-empty multi-page PDF."""
    destination = tmp_path / "anomaly_report.pdf"

    result = generate_anomaly_detection_report(_report_frame(), destination, max_features_per_page=1)

    assert result == destination
    assert destination.read_bytes().startswith(b"%PDF")
    assert destination.stat().st_size > 5_000


def test_generate_report_without_ground_truth(tmp_path: Path) -> None:
    """Ground truth should be optional."""
    destination = tmp_path / "unlabeled_report.pdf"
    frame = _report_frame().drop(columns="GT")

    generate_anomaly_detection_report(frame, destination)

    assert destination.exists()


def test_generate_report_rejects_missing_ad_columns(tmp_path: Path) -> None:
    """The report requires the standard anomaly flag and score columns."""
    with pytest.raises(ValueError, match="must contain"):
        generate_anomaly_detection_report(
            _report_frame().drop(columns="MAE"),
            tmp_path / "invalid.pdf",
        )


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("Anomaly", [-1, 1] * 6),
        ("GT", np.linspace(0.0, 1.0, 12)),
    ],
)
def test_generate_report_rejects_non_binary_masks(tmp_path: Path, column: str, values) -> None:
    """Detected and ground-truth masks must contain true binary values."""
    frame = _report_frame()
    frame[column] = values

    with pytest.raises(ValueError, match="must contain only binary values"):
        generate_anomaly_detection_report(frame, tmp_path / "non_binary.pdf")


def test_generate_report_preserves_non_string_feature_labels(tmp_path: Path) -> None:
    """Explicit integer feature labels should match the original DataFrame columns."""
    frame = _report_frame().rename(columns={"temperature": 10, "pressure": 20})
    destination = tmp_path / "integer_features.pdf"

    generate_anomaly_detection_report(frame, destination, feature_columns=[10, 20])

    assert destination.read_bytes().startswith(b"%PDF")


def test_infers_lowercase_anomaly_as_ground_truth() -> None:
    """A conventional lowercase anomaly label should be treated as report metadata."""
    frame = pd.DataFrame({"signal": [1.0, 2.0], "anomaly": [0, 1]})

    assert infer_ground_truth_column(frame) == "anomaly"


def test_resolve_x_axis_identifies_row_numbers_without_timestamp() -> None:
    """Reports without timestamps should identify their zero-based row-number axis."""
    values, label, is_datetime = _resolve_x_axis(pd.DataFrame({"signal": [1.0, 2.0]}), None)

    assert values.tolist() == [0, 1]
    assert label == "Row number (no timestamp column)"
    assert is_datetime is False


def test_resolve_x_axis_preserves_timestamp_values() -> None:
    """A valid timestamp column should remain the report's datetime axis."""
    frame = _report_frame()

    values, label, is_datetime = _resolve_x_axis(frame, "timestamp")

    assert values.equals(frame["timestamp"])
    assert label == "timestamp"
    assert is_datetime is True


def test_mae_note_page_explains_row_number_time_steps() -> None:
    """The PDF should explain row-number time steps when timestamps are unavailable."""
    figure = _create_mae_note_page(page_number=3, uses_row_numbers=True)

    assert any(
        " ".join(text.get_text().split())
        == "No usable timestamp column was provided, so time steps in this report are zero-based DataFrame row numbers."
        for text in figure.axes[0].texts
    )


def test_mae_note_page_omits_row_number_note_for_timestamped_data() -> None:
    """Timestamped reports should not show the row-number fallback note."""
    figure = _create_mae_note_page(page_number=3, uses_row_numbers=False)

    assert not any("zero-based DataFrame row numbers" in text.get_text() for text in figure.axes[0].texts)


def test_overview_page_explains_anomaly_score_direction() -> None:
    """The first report page should state what lower and higher MAE mean."""
    figure = reporting._create_overview_page(
        np.arange(3),
        np.array([0.1, 0.2, 0.3]),
        np.array([False, False, True]),
        None,
        title="Anomaly report",
        x_label="Row number",
        is_datetime=False,
    )

    page_text = [text.get_text() for axis in figure.axes for text in axis.texts]
    assert "How to read the anomaly score" in page_text
    assert reporting.SCORE_DIRECTION_NOTE in page_text


def test_diagnosis_note_page_explains_metrics_and_likely_issue_values() -> None:
    """Question 2 reports should explain every end-user metric and issue label."""
    figure = _create_mae_note_page(page_number=3, include_diagnosis=True)

    page_text = "\n".join(text.get_text() for text in figure.axes[0].texts)
    assert "Likely issue" in page_text
    assert "Repair impact" in page_text
    assert "Diagnostic confidence" in page_text
    assert "Explanation consistency" in page_text
    for issue in ["Level shift", "Trend change", "Transient spike", "Variance burst", "No supported repair"]:
        assert issue in page_text


@pytest.mark.parametrize("diagnosis", [False, True])
@pytest.mark.parametrize("row_numbers", [False, True])
def test_historical_guide_preserves_existing_notes_and_fits_page(diagnosis, row_numbers):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.text import Text

    settings = dict(page_number=5, include_diagnosis=diagnosis, uses_row_numbers=row_numbers)
    original = _create_mae_note_page(**settings)
    updated = _create_mae_note_page(**settings, include_historical=True)
    old_texts = [text.get_text() for text in original.axes[0].texts]
    new_texts = [text.get_text() for text in updated.axes[0].texts]
    assert [" ".join(text.split()) for text in old_texts[1:]] == [" ".join(text.split()) for text in new_texts[1:]]
    assert new_texts[0] == "How to interpret this report"
    text = " ".join(t.get_text() for t in updated.findobj(Text))
    for phrase in [
        "Top historical times",
        "Aligned timelines",
        "whole interval",
        "not a fault probability",
        "Matching numbers",
        "Takeaway:",
    ]:
        assert phrase in text
    canvas = FigureCanvasAgg(updated)
    canvas.draw()
    renderer = canvas.get_renderer()
    historical_header = next(t for t in updated.texts if t.get_text() == "Historical attribution")
    for text in updated.axes[0].texts:
        assert text.get_window_extent(renderer).y0 > historical_header.get_window_extent(renderer).y1
        if text.get_text().startswith("MAE means"):
            assert text.get_window_extent(renderer).x1 < updated.axes[0].transAxes.transform((0.49, 0))[0]
    for text in updated.findobj(Text):
        if text.get_text() and text.get_visible():
            box = text.get_window_extent(renderer)
            assert box.x0 >= 0 and box.x1 <= updated.bbox.width
            assert box.y0 >= 0 and box.y1 <= updated.bbox.height


def test_generate_report_includes_explainability_metrics(tmp_path: Path) -> None:
    """Explainability output should add anomaly-detail metrics to the PDF."""
    frame = _report_frame()
    frame["TopContributors"] = ["[]"] * len(frame)
    frame["ContributionShares"] = ["[]"] * len(frame)
    frame["ExplanationCoverage"] = 0.0
    frame.loc[3, ["TopContributors", "ContributionShares", "ExplanationCoverage"]] = [
        '["temperature","pressure"]',
        "[0.75,0.25]",
        1.0,
    ]
    frame.loc[7, ["TopContributors", "ContributionShares", "ExplanationCoverage"]] = [
        '["pressure"]',
        "[0.6]",
        0.6,
    ]
    destination = tmp_path / "explainable_report.pdf"

    generate_anomaly_detection_report(frame, destination)

    rows = _resolve_explanation_rows(
        frame,
        frame["Anomaly"].to_numpy(dtype=bool),
        frame["timestamp"],
        is_datetime=True,
    )
    assert rows == [
        ("2026-01-01\n00:03:00", "temperature\npressure", "75.0%\n25.0%", "100.0%"),
        ("2026-01-01\n00:07:00", "pressure", "60.0%", "60.0%"),
    ]
    assert destination.read_bytes().startswith(b"%PDF")
    csv_destination = tmp_path / "explainable_report_explanations.csv"
    exported = pd.read_csv(csv_destination)
    assert list(exported.columns) == [
        "Anomalous timestamp",
        "Top contributors",
        "Contribution shares",
        "Explanation coverage",
    ]
    assert len(exported) == 2
    assert exported.loc[0, "Anomalous timestamp"] == "2026-01-01 00:03:00"
    assert exported.loc[0, "Top contributors"] == '["temperature","pressure"]'
    assert exported.loc[0, "Contribution shares"] == "[0.75,0.25]"
    assert exported.loc[0, "Explanation coverage"] == 1.0


def test_report_adds_simplified_reconstruction_diagnosis_without_raw_scores(tmp_path: Path) -> None:
    frame = _report_frame()
    frame["TopContributors"] = ["[]"] * len(frame)
    frame["ContributionShares"] = ["[]"] * len(frame)
    frame["ExplanationCoverage"] = 0.0
    frame["LikelyReconstructionIssue"] = ""
    frame["RepairImpact"] = 0.0
    frame["DiagnosticConfidence"] = ""
    frame["ExplanationConsistency"] = ""
    frame.loc[3, ["TopContributors", "ContributionShares", "ExplanationCoverage"]] = [
        '["temperature","pressure"]',
        "[0.75,0.25]",
        1.0,
    ]
    frame.loc[7, ["TopContributors", "ContributionShares", "ExplanationCoverage"]] = [
        '["pressure"]',
        "[0.6]",
        0.6,
    ]
    frame.loc[3, ["LikelyReconstructionIssue", "RepairImpact", "DiagnosticConfidence", "ExplanationConsistency"]] = [
        "Level shift",
        0.63,
        "High",
        "Not evaluated",
    ]
    frame.loc[7, ["LikelyReconstructionIssue", "RepairImpact", "DiagnosticConfidence", "ExplanationConsistency"]] = [
        "Transient spike",
        0.31,
        "Moderate",
        "75%",
    ]

    destination = tmp_path / "diagnosis_report.pdf"
    generate_anomaly_detection_report(frame, destination)

    assert _resolve_diagnosis_rows(frame, frame["Anomaly"].to_numpy(dtype=bool)) == [
        ("Level shift", "63.0%", "High", "Not evaluated"),
        ("Transient spike", "31.0%", "Moderate", "75%"),
    ]
    exported = pd.read_csv(tmp_path / "diagnosis_report_explanations.csv")
    assert list(exported.columns[-4:]) == [
        "Likely reconstruction issue",
        "Repair impact",
        "Diagnostic confidence",
        "Explanation consistency",
    ]
    assert not any(column.startswith(("effect_", "score_after_")) for column in exported.columns)
    assert destination.read_bytes().startswith(b"%PDF")


def test_generate_report_without_explanations_keeps_metrics_optional() -> None:
    """Reports without explanation columns should preserve the existing path."""
    frame = _report_frame()

    rows = _resolve_explanation_rows(
        frame,
        frame["Anomaly"].to_numpy(dtype=bool),
        frame["timestamp"],
        is_datetime=True,
    )

    assert rows is None


def test_contribution_graph_visualizes_shares_and_coverage() -> None:
    """The graph should label feature shares and retain timestamp-only row labels."""
    figure = _create_contribution_graph_page(
        [("2026-01-01\n00:03:00", "temperature\npressure", "75.0%\n15.0%", "90.0%")],
        page_number=3,
        graph_page_number=1,
        graph_page_count=1,
    )

    axis = figure.axes[0]
    assert len(axis.patches) == 3  # Two contributor segments plus uncovered error.
    assert axis.get_title() == ""
    assert axis.get_yticklabels()[0].get_text() == "2026-01-01 00:03:00"
    assert any(text.get_text() == "temperature\n75%" for text in axis.texts)
    assert any(text.get_text() == "pressure\n15%" for text in axis.texts)
    assert any(text.get_text() == "90.0% covered" for text in axis.texts)


def test_aggregate_explanations_weights_features_by_anomaly_score() -> None:
    """Overall contributors should reflect their contribution to total anomaly MAE."""
    frame = pd.DataFrame(
        {
            "Anomaly": [True, True],
            "MAE": [1.0, 3.0],
            "TopContributors": ['["temperature","pressure"]', '["pressure","temperature"]'],
            "ContributionShares": ["[0.5,0.5]", "[0.75,0.25]"],
            "ExplanationCoverage": [1.0, 1.0],
        }
    )

    contributions, coverage = _aggregate_explanation_contributions(
        frame,
        frame["Anomaly"].to_numpy(dtype=bool),
        score_column="MAE",
        top_k=2,
    )

    assert [feature for feature, _ in contributions] == ["pressure", "temperature"]
    assert [share for _, share in contributions] == pytest.approx([0.6875, 0.3125])
    assert coverage == pytest.approx(1.0)


def test_large_explanation_report_uses_one_consolidated_page(monkeypatch, tmp_path: Path) -> None:
    """Large anomaly sets should switch from per-anomaly pages to one summary page."""
    sample_count = 80
    frame = pd.DataFrame(
        {
            "signal": np.linspace(0.0, 1.0, sample_count),
            "Anomaly": [True] * sample_count,
            "MAE": np.linspace(0.1, 1.0, sample_count),
            "TopContributors": ['["signal"]'] * sample_count,
            "ContributionShares": ["[1.0]"] * sample_count,
            "ExplanationCoverage": [1.0] * sample_count,
        }
    )
    consolidated = Mock(wraps=reporting._create_consolidated_contribution_page)
    detailed = Mock(wraps=reporting._create_contribution_graph_page)
    monkeypatch.setattr(reporting, "_create_consolidated_contribution_page", consolidated)
    monkeypatch.setattr(reporting, "_create_contribution_graph_page", detailed)

    generate_anomaly_detection_report(frame, tmp_path / "consolidated.pdf", max_report_pages=10)

    consolidated.assert_called_once()
    detailed.assert_not_called()
    assert consolidated.call_args.kwargs["max_report_pages"] == 10
    assert consolidated.call_args.kwargs["anomaly_count"] == sample_count


@pytest.mark.parametrize("matched_control", [True, False], ids=["aligned_control", "retrieved_donor"])
def test_historical_appendix_preserves_original_overview(tmp_path, monkeypatch, matched_control):
    """Run replacement and write the real graphic appendix page alongside the original SDK report."""
    import json

    import matplotlib.dates as mdates
    from matplotlib.text import Text
    from sdk import historical_attribution as historical_reporting

    count, event = 100, 80
    positions = np.arange(count)
    features = ["temperature", "pressure"]
    normal = pd.DataFrame({"temperature": 20 + np.sin(positions / 7), "pressure": 100 + np.sin(positions / 5)})
    frame = normal.copy()
    frame.loc[60:event, "pressure"] += np.linspace(0, 6, event - 59)
    timestamps = pd.date_range("2026-01-01", periods=count, freq="min")
    values = frame[features].to_numpy()
    visible = (positions // 10) % 2 != (event // 10) % 2

    def infer(observations):
        # Deterministic test callback; masked truth cannot enter this interpolation.
        observed = np.where(visible[:, None], observations, np.nan)
        return pd.DataFrame(observed).interpolate(limit_direction="both").to_numpy()

    before = infer(values)
    frame["timestamp"] = timestamps
    frame["Anomaly"] = positions == event
    frame["MAE"] = np.abs(values - before).mean(axis=1)
    for column, value in {
        "TopContributors": '["pressure"]',
        "ContributionShares": "[1.0]",
        "ExplanationCoverage": 1.0,
        "LikelyReconstructionIssue": "Level shift",
        "RepairImpact": 0.5,
        "DiagnosticConfidence": "Moderate",
        "ExplanationConsistency": "Not evaluated",
    }.items():
        frame[column] = pd.Series({event: value})
    source = {"id": "normal_test_fixture", "normal_evidence": "declared normal test data"}
    resolver = (
        historical_reporting.AlignedNormalReference(normal, source, alignment_verified=True)
        if matched_control
        else historical_reporting.NearestNormalWindowRetriever(
            normal, [(0, count)], [(0, 30)], source, same_source=False
        )
    )
    request = historical_reporting.HistoricalAttributionRequest(
        resolver=resolver, target="pressure", case_name="Historical attribution test fixture", units={"pressure": "kPa"}
    )
    result = historical_reporting.analyze_single_window(
        values,
        features,
        frame["Anomaly"],
        frame["MAE"],
        before,
        infer,
        request,
        visible=lambda _event: visible,
        time_labels=timestamps,
    )
    assert result["status"] == "ok" and len(result["points"]) == 20
    appendix = historical_reporting.report_data(result)
    appendix["report_context"]["inference_note"] = (
        "Deterministic masked-interpolation test callback; no pretrained-model inference."
    )
    (tmp_path / "historical_attribution_test_report.historical.json").write_text(
        json.dumps(appendix, indent=2), encoding="utf-8"
    )

    saved_pages, page_counts = [], []
    original_savefig = reporting.PdfPages.savefig

    def capture_savefig(pdf, figure, *args: object, **kwargs: object):
        # Observe the real writer rather than mocking away appendix generation.
        original_savefig(pdf, figure, *args, **kwargs)
        saved_pages.append(figure)
        page_counts.append(pdf.get_pagecount())

    monkeypatch.setattr(reporting.PdfPages, "savefig", capture_savefig)
    original_path, attributed_path = tmp_path / "original.pdf", tmp_path / "historical_attribution_test_report.pdf"
    settings = {"feature_columns": features, "timestamp_column": "timestamp", "feature_units": {"pressure": "kPa"}}
    generate_anomaly_detection_report(frame, original_path, **settings)
    original_pages = saved_pages.copy()
    generate_anomaly_detection_report(frame, attributed_path, historical_attribution=appendix, **settings)
    attributed_pages = saved_pages[len(original_pages) :]

    def page_text(page):
        return [text.get_text() for text in page.findobj(Text)]

    assert len(original_pages) == 4
    assert len(attributed_pages) == page_counts[-1] == len(original_pages) + 2
    assert [page_text(page) for page in original_pages[:-1]] == [page_text(page) for page in attributed_pages[:-3]]
    guide_text = page_text(attributed_pages[-1])
    for text in page_text(original_pages[-1]):
        if not text.startswith("Page "):
            assert " ".join(text.split()) in [" ".join(value.split()) for value in guide_text]
    assert "Top historical times" in guide_text and "Aligned timelines" in guide_text
    assert "How to interpret this report" in guide_text
    contribution_text = " ".join(page_text(attributed_pages[2]))
    assert "Feature contributions" in contribution_text and "Reconstruction diagnosis" in contribution_text

    historical_page, timeline_page = attributed_pages[-3:-1]
    historical_text = " ".join(" ".join(page_text(historical_page)).split())
    assert "Historical attribution" in historical_text and "Top-5 historical timestamps" in historical_text
    assert "2026-01-01 01:20" in historical_text and "Historical attribution test fixture" in historical_text
    bars = historical_page.axes[0]
    signal, response = timeline_page.axes
    assert response.get_xlabel().startswith("Time")
    assert "kPa" in signal.get_ylabel()
    start, stop = result["window"]["start"], result["window"]["stop"]
    expected = np.abs(np.asarray(result["curves"]["after"])[start:stop] - before[start:stop, 1])
    assert expected.sum() > 0
    np.testing.assert_allclose(signal.lines[0].get_ydata(), values[:, 1])
    np.testing.assert_allclose(signal.lines[0].get_xdata(), mdates.date2num(timestamps.to_pydatetime()))
    indices = np.argsort(-expected, kind="stable")[:5]
    expected_shares = 100 * expected[indices] / expected.sum()
    np.testing.assert_allclose([bar.get_width() for bar in bars.patches], expected_shares)
    for index, share in zip(indices, expected_shares, strict=True):
        assert timestamps[start + index].strftime("%H:%M") in historical_text
        assert f"{share:.1f}%" in historical_text
    saved = json.loads((tmp_path / "historical_attribution_test_report.historical.json").read_text())
    assert saved == json.loads(json.dumps(appendix))
    assert saved["historical_result"] == json.loads(json.dumps(result))
    assert (result["metrics"]["CAI"] is not None) == matched_control
    assert result["metrics"]["ExplanationStability"]["reason"] == "single_donor"
    for forbidden in ["g=1", "L=20", "gap=", "Why this target", "Top 5 sensors", "What the five metrics mean"]:
        assert forbidden not in historical_text
    assert attributed_path.read_bytes().startswith(b"%PDF")


@pytest.mark.parametrize("historical_status", [None, "ok", "unavailable"])
@pytest.mark.parametrize("feature_count", [1, 17])
def test_historical_appendix_counts_toward_total_page_budget(tmp_path, monkeypatch, historical_status, feature_count):
    """Reserve the actual two-page available appendix before deciding whether to consolidate."""
    from matplotlib.text import Text
    from sdk.historical_attribution import (
        AlignedNormalReference,
        HistoricalAttributionRequest,
        analyze_single_window,
    )

    count = 100
    columns = [f"signal_{i}" for i in range(feature_count)]
    features = pd.DataFrame(np.arange(count * feature_count).reshape(count, feature_count) / 10, columns=columns)
    flags = np.arange(count) < 49
    frame = features.assign(
        Anomaly=flags,
        MAE=np.linspace(0.1, 1, count),
        TopContributors=json.dumps(columns[:1]),
        ContributionShares="[1.0]",
        ExplanationCoverage=1.0,
    )
    historical = None
    if historical_status == "unavailable":
        historical = dict(status="unavailable", reason="no_acceptable_replacement", case_name="Test fixture")
    elif historical_status == "ok":
        values = features.to_numpy()

        def infer(data):
            return np.roll(data, 1, axis=0) * 0.5

        request = HistoricalAttributionRequest(
            resolver=AlignedNormalReference(features * 0.7, {"id": "normal"}, True),
            event_position=40,
            target=columns[0],
            validation_repeats=False,
        )
        historical = analyze_single_window(
            values,
            columns,
            flags,
            frame.MAE,
            infer(values),
            infer,
            request,
            visible=lambda p: np.arange(count) % 2 != p % 2,
        )
        assert historical["status"] == "ok"
    pages = []
    savefig = reporting.PdfPages.savefig

    def capture(pdf, figure, *args: object, **kwargs: object):
        savefig(pdf, figure, *args, **kwargs)
        pages.append(figure)
        assert pdf.get_pagecount() <= 10

    monkeypatch.setattr(reporting.PdfPages, "savefig", capture)
    destination = tmp_path / "budget.pdf"
    generate_anomaly_detection_report(
        frame,
        destination,
        feature_columns=columns,
        max_report_pages=10,
        historical_attribution=historical,
    )
    expected = {1: {None: 10, "ok": 6, "unavailable": 5}, 17: {None: 8, "ok": 10, "unavailable": 9}}
    assert len(pages) == expected[feature_count][historical_status]
    texts = [" ".join(t.get_text() for t in page.findobj(Text)) for page in pages]
    assert "Kumo-Anomaly Report" in texts[0]
    for number, text in enumerate(texts, 1):
        assert f"Page {number}" in text
    assert "MAE" in texts[-1]
    if historical_status is not None:
        assert "Historical attribution" in texts[-2]
    if historical_status == "ok":
        assert "Top-5 historical" in texts[-3]
        assert "Measured signal and historical response" in texts[-2]
        assert "How to interpret this report" in texts[-1]
        assert "What the five metrics mean" not in texts[-1]
    if historical_status == "unavailable":
        assert "Not available: no_acceptable_replacement" in texts[-2]


@pytest.mark.parametrize("historical_status,explanations", [("ok", True), ("unavailable", True)])
def test_page_budget_rejects_impossible_historical_appendix(tmp_path, historical_status, explanations):
    frame = _report_frame()
    if explanations:
        frame = frame.assign(TopContributors='["pressure"]', ContributionShares="[1.0]", ExplanationCoverage=1.0)
    historical = dict(status=historical_status, case_name="Test fixture", reason="no_acceptable_replacement")
    destination = tmp_path / "too_small.pdf"
    with pytest.raises(ValueError, match="cannot accommodate the required"):
        generate_anomaly_detection_report(frame, destination, historical_attribution=historical, max_report_pages=4)
    assert not destination.exists()
    assert not (tmp_path / "too_small_explanations.csv").exists()


def test_unavailable_historical_appendix_fits_four_page_minimum_without_explanations(tmp_path, monkeypatch):
    pages = []
    savefig = reporting.PdfPages.savefig

    def capture(pdf, figure, *args: object, **kwargs: object):
        savefig(pdf, figure, *args, **kwargs)
        pages.append(pdf.get_pagecount())

    monkeypatch.setattr(reporting.PdfPages, "savefig", capture)
    generate_anomaly_detection_report(
        _report_frame(),
        tmp_path / "minimal.pdf",
        max_report_pages=4,
        historical_attribution=dict(status="unavailable", reason="no_sdk_anomaly"),
    )
    assert pages == [1, 2, 3, 4]
