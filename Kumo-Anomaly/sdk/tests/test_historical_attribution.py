# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset-independent historical replacement, metrics and PDF invariants."""

import json
from dataclasses import replace
from itertools import pairwise

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pytest
import torch
from sdk.historical_attribution import (
    AlignedNormalReference,
    CosineDistance,
    CosineWindowRetriever,
    EuclideanDistance,
    HistoricalAttributionMetrics,
    HistoricalAttributionRequest,
    LinearBlendReplacement,
    MetricsConfig,
    MultiDonorReplacement,
    NearestNormalWindowRetriever,
    NormalDonor,
    NormalizedMAEDistance,
    ReferenceMatch,
    ReplacementCondition,
    ReplacementContext,
    alarm_label,
    analyze_single_window,
    blended_replacement,
    historical_axis,
    historical_summary,
    report_data,
    run_sdk_attribution,
    timestamp_labels,
)


def fixture(count=60, event=40):
    x = np.arange(count * 3, dtype=float).reshape(count, 3) / 10
    flags = np.zeros(count, bool)
    flags[event] = True
    normal = pd.DataFrame(x * 0.7, columns=["a", "b", "c"])
    resolver = AlignedNormalReference(normal, {"id": "verified_fixture"}, True)
    req = HistoricalAttributionRequest(resolver=resolver, target="b", validation_repeats=True)

    def visible(p):
        return np.arange(count) % 2 != p % 2

    def infer(a):
        return np.roll(a, 1, axis=0) * 0.5

    return x, flags, req, visible, infer


def test_single_intervention_twenty_responses_and_real_alarm():
    x, flags, req, visible, infer = fixture()
    h = analyze_single_window(x, ["a", "b", "c"], flags, np.ones(60), infer(x), infer, req, visible=visible)
    assert h["status"] == "ok" and len(h["points"]) == 20
    assert h["window"]["positions"] == list(range(20, 40))
    assert h["validation"]["self_input_equal"]
    assert h["validation"]["repeat_max_abs_difference"] == 0
    assert h["validation"]["self_max_abs_difference"] == 0
    assert h["validation"]["alarm_observation_unchanged"]
    assert h["metrics"]["HistoricalContributionShare"]["status"] == "not_applicable"
    assert h["metrics"]["CAI"] is not None
    for r in h["points"]:
        assert r["historical_reconstruction_change_score"] == abs(r["after"] - r["before"])
    req.event_position = 41
    with pytest.raises(ValueError, match="actual SDK anomaly"):
        analyze_single_window(x, ["a", "b", "c"], flags, np.ones(60), infer(x), infer, req, visible=visible)


def test_blending_never_touches_alarm_or_outside_block():
    x, _, _, _, _ = fixture()
    segment = x[20:40] + 7
    out, w = blended_replacement(x, segment, 20, 40, [1], 2)
    np.testing.assert_array_equal(out[:20], x[:20])
    np.testing.assert_array_equal(out[40:], x[40:])
    np.testing.assert_array_equal(out[:, [0, 2]], x[:, [0, 2]])
    np.testing.assert_allclose(w[:3], [1 / 3, 2 / 3, 1])
    identity, _ = blended_replacement(x, x[20:40], 20, 40, [0, 1, 2], 2)
    np.testing.assert_array_equal(identity, x)


def test_no_donor_does_not_infer_or_manufacture_scores():
    x, flags, req, visible, infer = fixture()

    class Reject:
        def resolve(self, *args: object):
            return ReferenceMatch("no_acceptable_replacement", reason="test_rejection")

    req.resolver = Reject()
    req.validation_repeats = False

    def forbidden(x):
        raise AssertionError("Rejected donor must not trigger replacement inference")

    h = analyze_single_window(x, ["a", "b", "c"], flags, np.ones(60), infer(x), forbidden, req, visible=visible)
    assert h["status"] == "no_acceptable_replacement" and "points" not in h


def test_pdf_includes_historical_attribution_multi_donor(tmp_path, monkeypatch):
    """Write one complete SDK PDF with the two attribution pages using the existing fixture."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.text import Text
    from sdk import reporting

    x, flags, request, visible, infer = fixture(count=100, event=80)
    columns, timestamps = ["a", "b", "c"], pd.date_range("2026-01-01", periods=len(x), freq="min")
    request = replace(request, case_name="Historical attribution test fixture", units={"b": "test units"})
    donor_count = 5
    normal = request.resolver.frame.to_numpy()
    donors = [
        NormalDonor(normal[60:80] + 2 * i, {"id": f"normal_{i}", "start": 60, "stop": 80}, normal)
        for i in range(donor_count)
    ]

    class Resolve:
        def resolve(self, *args: object):
            return ReferenceMatch("accepted", donors=donors)

    request = replace(request, resolver=Resolve(), replacement_strategy=MultiDonorReplacement())
    before = infer(x)
    scores = np.abs(x - before).mean(axis=1)
    result = analyze_single_window(
        x, columns, flags, scores, before, infer, request, visible=visible, time_labels=timestamps
    )
    assert result["status"] == "ok" and len(result["points"]) == 20
    appendix = report_data(result)
    assert appendix["summary"]["condition_count"] == donor_count
    assert len(appendix["summary"]["alarm_metrics_by_condition"]) == donor_count
    appendix["report_context"]["inference_note"] = "Deterministic test callback; no pretrained-model inference."
    frame = pd.DataFrame(x, columns=columns).assign(timestamp=timestamps, Anomaly=flags, MAE=scores)
    for column, value in {
        "TopContributors": '["b"]',
        "ContributionShares": "[1.0]",
        "ExplanationCoverage": 1.0,
        "LikelyReconstructionIssue": "Not evaluated",
        "RepairImpact": 0.0,
        "DiagnosticConfidence": "Not evaluated",
        "ExplanationConsistency": "Not evaluated",
    }.items():
        frame[column] = pd.Series({80: value})

    pages, page_counts = [], []
    savefig = reporting.PdfPages.savefig

    def capture_page(pdf, figure, *args: object, **kwargs: object):
        savefig(pdf, figure, *args, **kwargs)
        pages.append(figure)
        page_counts.append(pdf.get_pagecount())

    monkeypatch.setattr(reporting.PdfPages, "savefig", capture_page)
    destination = tmp_path / "historical_attribution_test_report.pdf"
    reporting.generate_anomaly_detection_report(
        frame,
        destination,
        feature_columns=columns,
        timestamp_column="timestamp",
        historical_attribution=appendix,
        feature_units=request.units,
    )
    (tmp_path / "historical_attribution_test_report.historical.json").write_text(
        json.dumps(appendix, indent=2), encoding="utf-8"
    )
    assert len(pages) == page_counts[-1] == 6
    assert list(tmp_path.glob("*.pdf")) == [destination]
    assert destination.read_bytes().startswith(b"%PDF")
    assert (tmp_path / "historical_attribution_test_report_explanations.csv").is_file()

    def page_text(page):
        return " ".join(" ".join(t.get_text() for t in page.findobj(Text)).split())

    assert "Kumo-Anomaly Report" in page_text(pages[0])
    assert "Feature contributions" in page_text(pages[2])
    assert "Reconstruction diagnosis" in page_text(pages[2])
    historical_page, timeline_page = pages[-3:-1]
    historical_text = page_text(historical_page)
    assert "How to interpret this report" in page_text(pages[-1])
    assert "Top historical times" in page_text(pages[-1])
    assert "A large share is not a fault probability" in page_text(pages[-1])
    assert "Historical attribution" in historical_text and "Top-5 historical timestamps" in historical_text
    assert "2026-01-01 01:20" in historical_text
    bars = historical_page.axes[0]
    signal, response = timeline_page.axes
    assert bars.get_xlabel() == response.get_ylabel() == "Historical response share (%)"
    assert response.get_xlabel().startswith("Time")
    assert signal.get_ylabel() == "b\n(test units)"
    assert (
        "Summary We replace the selected historical block with reference data and compare the target reconstructions "
        "before and after. Each timestamp\u2019s percentage is its absolute reconstruction change divided by the "
        "total absolute change across the entire historical interval."
    ) in historical_text
    assert (
        "The upper plot shows measured values in the sensor\u2019s original unit; the lower plot shows relative "
        "reconstruction-change shares (%), not input measurements."
    ) in page_text(timeline_page)
    start, stop = result["window"]["start"], result["window"]["stop"]
    expected = np.abs(np.asarray(result["curves"]["after"])[start:stop] - before[start:stop, 1])
    np.testing.assert_allclose(signal.lines[0].get_ydata(), x[:, 1])
    np.testing.assert_allclose(signal.lines[0].get_xdata(), mdates.date2num(timestamps.to_pydatetime()))
    np.testing.assert_allclose(
        response.collections[1].get_offsets()[:, 0], mdates.date2num(timestamps[start:stop].to_pydatetime())
    )
    indices = np.argsort(-expected, kind="stable")[:5]
    shares = 100 * expected[indices] / expected.sum()
    np.testing.assert_allclose([bar.get_width() for bar in bars.patches], shares)
    assert [tick.get_text() for tick in bars.get_yticklabels()] == [
        f"{rank + 1}. {timestamps[start + i].strftime('%H:%M')}" for rank, i in enumerate(indices)
    ]
    for share in shares:
        assert f"{share:.1f}%" in historical_text
    saved = json.loads((tmp_path / "historical_attribution_test_report.historical.json").read_text())
    assert saved == json.loads(json.dumps(appendix))  # Rendering never changes any saved result.
    assert saved["historical_result"] == json.loads(json.dumps(result))
    assert all(
        saved["summary"]["alarm_metrics_by_condition"][c["condition"]["id"]] == c["metrics"]
        for c in result["conditions"]
    )
    stability = result["metrics"]["ExplanationStability"]
    assert result["condition_count"] == stability["distinct_donor_count"] == donor_count
    assert stability["status"] == stability["alarm_effect"]["candidates"][0]["status"] == "ok"
    assert stability["checked_axes"] == ["donor"] and stability["unverified_axes"] == ["seed", "mask", "boundary"]
    for forbidden in ["g=1", "L=20", "gap=", "Top 5 sensors", "What the five metrics mean", "SDK Top-1 contributor"]:
        assert forbidden not in historical_text
    assert not any("\u4e00" <= char <= "\u9fff" for char in historical_text)
    for page in [historical_page, timeline_page]:
        FigureCanvasAgg(page).draw()
    historical_page.savefig(tmp_path / "historical_attribution_test_report_historical.png", dpi=120)
    timeline_page.savefig(tmp_path / "historical_attribution_test_report_timelines.png", dpi=120)


@pytest.mark.parametrize(
    "effects,expected_status,expected_reason,expected_agreement",
    [
        pytest.param([0, 0], "warning", "all_near_zero", 1.0, id="zero"),
        pytest.param(
            [MetricsConfig().numerical_tolerance / 2, -MetricsConfig().numerical_tolerance / 2],
            "warning",
            "all_near_zero",
            1.0,
            id="below_tolerance",
        ),
        pytest.param(
            [MetricsConfig().numerical_tolerance, -MetricsConfig().numerical_tolerance],
            "warning",
            "all_near_zero",
            1.0,
            id="at_tolerance",
        ),
        pytest.param(
            [
                np.nextafter(MetricsConfig().numerical_tolerance, np.inf),
                -np.nextafter(MetricsConfig().numerical_tolerance, np.inf),
            ],
            "ok",
            None,
            0.5,
            id="just_above_tolerance",
        ),
        pytest.param([1, 2], "ok", None, 1.0, id="positive_effects"),
        pytest.param([-1, -2], "ok", None, 1.0, id="negative_effects"),
        pytest.param([1, None], "warning", "insufficient_repeats", 1.0, id="one_valid_effect"),
        pytest.param([1, 2, None], "warning", "missing_replicates", 1.0, id="missing_effect"),
        pytest.param([None, 1, 2], "warning", "missing_replicates", None, id="missing_reference_effect"),
        pytest.param([None, None], "warning", "no_valid_repeats", None, id="no_valid_effects"),
    ],
)
def test_pdf_preserves_computed_stability_boundary_states(
    effects, expected_status, expected_reason, expected_agreement
):
    """Check computed metric states through JSON and the existing PDF renderer."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.text import Text
    from sdk.historical_attribution import create_historical_pages

    x, flags, request, visible, infer = fixture()
    result = analyze_single_window(
        x, ["a", "b", "c"], flags, np.ones(len(x)), infer(x), infer, request, visible=visible
    )
    appendix = report_data(result)
    alarm = HistoricalAttributionMetrics().compute_stability(
        {f"donor-{i}": {"fixed_block": effect} for i, effect in enumerate(effects)},
        axis="donor",
        expected_candidates=["fixed_block"],
    )
    candidate = alarm["candidates"][0]
    assert candidate["status"] == expected_status and candidate["reason"] == expected_reason
    assert candidate["sign_agreement"] == expected_agreement
    primary = appendix["summary"]["alarm_metrics_by_condition"][appendix["summary"]["primary_condition_id"]]
    primary["ExplanationStability"] = dict(
        status=alarm["status"],
        reason=alarm["reason"],
        alarm_effect=alarm,
        checked_axes=["donor"],
        unverified_axes=["seed", "mask", "boundary"],
        effect_metric="supplied paired effect",
        output_response_ranking={"rank_comparisons": []},
    )
    appendix["report_context"]["distinct_donor_count"] = len(effects)
    appendix["report_context"]["inference_note"] = "Metric-rendering fixture; supplied paired effects."
    appendix = json.loads(json.dumps(appendix, allow_nan=False))
    saved = json.dumps(appendix, sort_keys=True, allow_nan=False)
    pages = list(create_historical_pages(appendix))
    assert len(pages) == 2
    for page in pages:
        FigureCanvasAgg(page).draw()
    assert json.dumps(appendix, sort_keys=True, allow_nan=False) == saved
    metric = appendix["summary"]["alarm_metrics_by_condition"][appendix["summary"]["primary_condition_id"]]
    assert metric["ExplanationStability"]["alarm_effect"]["candidates"][0] == candidate
    assert "What the five metrics mean" not in " ".join(t.get_text() for t in pages[0].findobj(Text))


@pytest.mark.parametrize("count", [100, 1000])
@pytest.mark.parametrize("length", [1, 3, 20, 37])
@pytest.mark.parametrize("response", ["zero", "small", "equal", "varying"])
def test_historical_graph_adapts_interval_ranking_and_summary(length, response, count):
    """Exercise display edge cases with supplied saved responses, not new inference settings."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.text import Text
    from sdk.historical_attribution import create_historical_pages

    x, flags, request, visible, infer = fixture(count=count, event=80)
    request = replace(request, length=length, blend_points=min(request.blend_points, length // 2))
    normal = request.resolver.frame.iloc[:30].to_numpy()
    request.resolver.source.update(
        normal_evidence="verified normal fixture",
        calibration_intervals=[[0, 30]],
        calibration_scale=(np.quantile(normal, 0.95, axis=0) - np.quantile(normal, 0.05, axis=0)).tolist(),
    )
    result = analyze_single_window(
        x, ["a", "b", "c"], flags, np.ones(len(x)), infer(x), infer, request, visible=visible
    )
    selected = result["conditions"][0]
    magnitudes = (
        np.zeros(length)
        if response == "zero"
        else np.full(length, 1e-12)
        if response == "small"
        else np.ones(length)
        if response == "equal"
        else np.arange(1, length + 1, dtype=float)
    )
    for row, magnitude in zip(selected["points"], magnitudes, strict=True):
        row.update(signed_change=float(magnitude), historical_reconstruction_change_score=float(magnitude))
        selected["curves"]["after"][row["position"]] = selected["curves"]["before"][row["position"]] + float(magnitude)
    appendix = report_data(result)
    saved = json.dumps(appendix, sort_keys=True)
    pages = list(create_historical_pages(appendix, start_page=7))
    assert len(pages) == 2
    page = pages[0]
    FigureCanvasAgg(page).draw()
    text = " ".join(" ".join(t.get_text() for t in page.findobj(Text)).split())
    bars = page.axes[0]
    signal, response_axis = pages[1].axes
    assert bars.get_xlabel() == response_axis.get_ylabel() == "Historical response share (%)"
    assert signal.get_ylabel() == f"{appendix['case']['target']}\n({appendix['case']['target_unit']})"
    timeline_canvas = FigureCanvasAgg(pages[1])
    timeline_canvas.draw()
    assert signal.get_position().width >= 0.8 and signal.get_position().height >= 0.2
    assert response_axis.get_position().width >= 0.8 and response_axis.get_position().height >= 0.2
    assert signal.get_xlim() == response_axis.get_xlim()
    assert len(signal.lines[0].get_xdata()) == count
    timeline_text = " ".join(t.get_text() for t in pages[1].findobj(Text))
    assert "Page 8" in timeline_text and "previous page" in timeline_text
    heading = next(t for t in pages[1].texts if t.get_text() == "Measured signal and historical response")
    alarm = next(t for t in signal.texts if t.get_text() == "Alarm")
    renderer = timeline_canvas.get_renderer()
    assert heading.get_window_extent(renderer).y0 > alarm.get_window_extent(renderer).y1 + 2
    summary_text = next(t for t in page.texts if t.get_text().startswith("We replace the selected historical block"))
    summary_canvas = page.canvas
    summary_renderer = summary_canvas.get_renderer()
    footer = next(t for t in page.texts if t.get_text() == "Page 7")
    assert summary_text.get_window_extent(summary_renderer).y0 > footer.get_window_extent(summary_renderer).y1 + 5
    unit_note = next(t for t in pages[1].texts if t.get_text().startswith("The upper plot shows measured values"))
    link_note = next(t for t in pages[1].texts if t.get_text().startswith("Numbers link the same times"))
    timeline_footer = next(t for t in pages[1].texts if t.get_text() == "Page 8")
    box = unit_note.get_window_extent(renderer)
    assert 0 <= box.x0 < box.x1 <= pages[1].bbox.width
    assert box.y0 > timeline_footer.get_window_extent(renderer).y1 + 5
    assert box.y1 < link_note.get_window_extent(renderer).y0 - 3
    prose = historical_summary(appendix)
    assert "Page 7" in text and "Observation index" in timeline_text
    if response == "zero":
        assert "No observed model response" in text and "Top-" not in text
        assert not bars.patches
        assert "No historical model response was observed in this interval" in text
        assert not response_axis.collections
        assert "percentages are undefined" in prose["result"]
    else:
        order = np.argsort(-magnitudes, kind="stable")[: min(5, length)]
        np.testing.assert_allclose(
            [bar.get_width() for bar in bars.patches], 100 * magnitudes[order] / magnitudes.sum()
        )
        assert [label.get_text() for label in bars.get_yticklabels()] == [
            f"{rank + 1}. {80 - length + i}" for rank, i in enumerate(order)
        ]
        assert f"Top-{min(5, length)} historical observation indices" in text
        assert f"Percentages use all {length} historical points" in text
        maximum_share = 100 * magnitudes.max() / magnitudes.sum()
        if length > 1 and response in {"small", "equal"}:
            assert f"{length} times share the maximum response ({maximum_share:.1f}% each)" in prose["result"]
        else:
            assert (
                f"Maximum relative response: {80 - length + order[0]} ({maximum_share:.1f}% of the interval total)"
                in prose["result"]
            )
        np.testing.assert_allclose(response_axis.collections[1].get_offsets()[:, 0], np.arange(80 - length, 80))
        np.testing.assert_allclose(
            response_axis.collections[1].get_offsets()[:, 1], 100 * magnitudes / magnitudes.sum()
        )
        assert len(signal.collections[0].get_offsets()) == min(5, length)
    assert (
        "not confirmed fault sources" in prose["takeaway"]
        if response != "zero"
        else "does not rule out a real fault" in prose["takeaway"]
    )
    assert "Summary" in text and "matching numbers" in text
    assert "Darker points" not in text
    assert json.dumps(appendix, sort_keys=True) == saved


@pytest.mark.parametrize("scale", [None, 0.0, -1.0])
def test_relative_response_timeline_does_not_depend_on_color_calibration(scale):
    from sdk.historical_attribution import create_historical_pages

    x, flags, request, visible, infer = fixture()
    request.resolver.source.update(normal_evidence="normal fixture", calibration_intervals=[[0, 10]])
    if scale is not None:
        request.resolver.source["calibration_scale"] = [scale] * 3
    result = analyze_single_window(
        x, ["a", "b", "c"], flags, np.ones(len(x)), infer(x), infer, request, visible=visible
    )
    pages = list(create_historical_pages(result))
    page = pages[1]
    expected = np.array([p["historical_reconstruction_change_score"] for p in result["points"]])
    np.testing.assert_allclose(page.axes[1].collections[1].get_offsets()[:, 1], 100 * expected / expected.sum())
    assert len(page.axes) == 2  # Two aligned timelines; no color spectrum.


@pytest.mark.parametrize(
    "onset,expected",
    [
        ("2026-01-01 00:30", "before"),
        ("2026-01-01 01:05", "within"),
        ("2026-01-01 01:21", "after"),
        (None, None),
        ("invalid", None),
    ],
)
def test_summary_uses_only_recorded_onset_and_saved_values(onset, expected):
    x, flags, request, visible, infer = fixture(count=100, event=80)
    times = pd.date_range("2026-01-01", periods=100, freq="min")
    result = analyze_single_window(
        x,
        ["a", "b", "c"],
        flags,
        np.ones(100),
        infer(x),
        infer,
        request,
        visible=visible,
        time_labels=times,
        threshold_trace={"labelled_onset": onset},
    )
    saved = json.dumps(result, sort_keys=True)
    summary = historical_summary(result)
    assert "SDK alarm at 2026-01-01 01:20" in summary["overview"]
    assert "for b" in summary["overview"]
    assert "One earlier block of 1 selected signal(s)" in summary["overview"]
    if expected is None:
        assert "Labelled anomaly onset" not in summary["takeaway"]
    else:
        assert f"is {expected} the reviewed interval" in summary["takeaway"]
    assert "not confirmed fault sources" in summary["takeaway"]
    assert json.dumps(result, sort_keys=True) == saved
    assert historical_summary(report_data(result)) == summary


def test_aligned_reference_requires_identity_time_feature_validation():
    x, flags, req, visible, infer = fixture()
    req.resolver.alignment_verified = False
    h = analyze_single_window(x, ["a", "b", "c"], flags, np.ones(60), infer(x), infer, req, visible=visible)
    assert h["reason"] == "baseline_alignment_not_verified"
    req.resolver.alignment_verified = True
    req.resolver.frame.columns = ["b", "a", "c"]
    with pytest.raises(ValueError, match="feature order"):
        analyze_single_window(x, ["a", "b", "c"], flags, np.ones(60), infer(x), infer, req, visible=visible)


def test_cosine_ranking_does_not_override_difference_gate():
    bank = pd.DataFrame({"a": np.sin(np.arange(200) / 9), "b": np.cos(np.arange(200) / 13)})
    r = CosineWindowRetriever(bank, [(0, 200)], [(0, 30)], {"normal_evidence": "fixture"}, same_source=False)
    values = bank.to_numpy() + 1000
    result = r.resolve(values, list(bank.columns), 120, 140, np.ones(200, bool), 140)
    assert result.status == "no_acceptable_replacement"
    assert result.source["accepted_count"] == 0 and len(result.candidates) > 0
    assert all(not c["accepted"] for c in result.candidates)


def test_nearest_donor_minimizes_scaled_mae_without_quality_gate():
    bank = pd.DataFrame({"a": np.sin(np.arange(200) / 9), "b": np.cos(np.arange(200) / 13)})
    values = bank.to_numpy() + 1000
    resolver = NearestNormalWindowRetriever(
        bank, [(0, 200)], [(0, 30)], {"normal_evidence": "fixture"}, same_source=False
    )
    result = resolver.resolve(values, list(bank.columns), 120, 140, np.ones(200, bool), 140)
    scale = np.quantile(bank.iloc[:30], 0.95, axis=0) - np.quantile(bank.iloc[:30], 0.05, axis=0)
    distances = [
        (float(np.mean(np.abs(values[112:120] - bank.iloc[ds - 8 : ds].to_numpy()) / scale)), ds)
        for ds in range(38, 181)
    ]
    expected_distance, expected_start = min(distances)
    assert result.status == "accepted"
    assert result.source["start"] == expected_start
    assert result.source["selected_distance"] == pytest.approx(expected_distance)
    assert result.source["acceptance_threshold"] is None
    assert result.source["accepted_count"] == result.source["bank_count"] == len(distances)
    assert expected_distance > 0.05
    np.testing.assert_array_equal(result.values, bank.iloc[expected_start : expected_start + 20].to_numpy())
    assert [x["normalized_mae"] for x in result.candidates] == sorted(x["normalized_mae"] for x in result.candidates)
    # Changing one feature's unit must not change the selected donor or scaled distance.
    conversion = np.array([1000.0, 0.01])
    converted = NearestNormalWindowRetriever(
        bank * conversion, [(0, 200)], [(0, 30)], {"normal_evidence": "fixture"}, same_source=False
    ).resolve(values * conversion, list(bank.columns), 120, 140, np.ones(200, bool), 140)
    assert converted.source["start"] == expected_start
    assert converted.source["selected_distance"] == pytest.approx(expected_distance)


def test_nearest_retrieval_preserves_normal_exclusions_and_visible_context():
    bank = pd.DataFrame({"a": np.sin(np.arange(300) / 9), "b": np.cos(np.arange(300) / 13)})
    settings = dict(
        normal_intervals=[(0, 30), (60, 180)], calibration_intervals=[(0, 30)], source={"normal_evidence": "fixture"}
    )
    original = bank.to_numpy().copy()
    visible = np.arange(300) % 2 == 1
    result = NearestNormalWindowRetriever(bank, **settings).resolve(
        original, list(bank.columns), 220, 240, visible, 240
    )
    assert result.status == "accepted"
    assert min(result.source["donor_context_positions"]) >= 60
    assert result.source["stop"] <= 180
    hidden_changed = original.copy()
    hidden_changed[~visible] = 1e30
    hidden_changed[220:] = np.nan
    other = bank.copy()
    other.iloc[200:] = np.nan
    repeated = NearestNormalWindowRetriever(other, **settings).resolve(
        hidden_changed, list(bank.columns), 220, 240, visible, 240
    )
    assert repeated.source["start"] == result.source["start"]
    assert repeated.source["selected_distance"] == result.source["selected_distance"]
    assert repeated.source["query_hash"] == result.source["query_hash"]
    np.testing.assert_array_equal(repeated.values, result.values)
    # No distance gate does not authorize using abnormal / excluded history.
    empty = NearestNormalWindowRetriever(bank, [(0, 30)], [(0, 30)], {"normal_evidence": "fixture"})
    assert empty.resolve(original, list(bank.columns), 220, 240, visible, 240).reason == "no_eligible_normal_windows"


def test_nearest_distance_can_rank_a_zero_centered_query():
    bank = pd.DataFrame({"a": np.sin(np.arange(200) / 9), "b": np.cos(np.arange(200) / 13)})
    values = np.tile(np.median(bank.iloc[:30], axis=0), (200, 1))
    resolver = NearestNormalWindowRetriever(
        bank, [(0, 200)], [(0, 30)], {"normal_evidence": "fixture"}, same_source=False
    )
    result = resolver.resolve(values, list(bank.columns), 120, 140, np.ones(200, bool), 140)
    assert result.status == "accepted"
    assert result.candidates[0]["cosine_distance"] is None
    assert np.isfinite(result.source["selected_distance"])


def test_dataset_event_is_explicit_and_does_not_fabricate_sdk_alarm():
    x, flags, req, visible, infer = fixture()
    flags[:] = False
    req.event_position = 40
    req.event_source = "dataset_label"
    labels = np.zeros(len(x))
    labels[20:45] = 1
    h = analyze_single_window(
        x, ["a", "b", "c"], flags, np.ones(len(x)), infer(x), infer, req, visible=visible, event_labels=labels
    )
    assert h["status"] == "ok"
    assert h["event"]["source"] == "dataset_label"
    assert h["event"]["sdk_detected"] is False
    assert h["event"]["position"] == 40
    assert h["validation"]["outside_window_unchanged"]
    req.event_source = "sdk"
    h = analyze_single_window(
        x, ["a", "b", "c"], flags, np.ones(len(x)), infer(x), infer, req, visible=visible, event_labels=labels
    )
    assert h["reason"] == "no_sdk_anomaly"


@pytest.mark.parametrize("labels", [None, [1], [2] * 60, [0] * 60])
def test_dataset_event_requires_valid_positive_aligned_labels(labels):
    x, flags, req, visible, infer = fixture()
    req.event_source = "dataset_label"
    req.event_position = 40
    with pytest.raises(ValueError, match="dataset"):
        analyze_single_window(
            x, ["a", "b", "c"], flags, np.ones(len(x)), infer(x), infer, req, visible=visible, event_labels=labels
        )


def case(timestamps=None):
    data = dict(
        case=dict(steps=5, alarm_step=4), summary=dict(primary=dict(points=[dict(case_step=i) for i in [1, 2, 3]]))
    )
    if timestamps is not None:
        data["case"]["timestamps"] = timestamps
    return data


def test_timestamp_labels_and_spacing_are_preserved():
    data = case(["2026-01-01 21:00", "2026-01-01 22:00", "2026-01-02 00:00", "2026-01-02 02:00", "2026-01-02 03:00"])
    info = historical_axis(data)
    assert info["is_datetime"]
    assert np.diff(info["x"]) * 24 == pytest.approx([1, 2])
    assert alarm_label(data) == "2026-01-02 02:00"
    assert info["labels"][3] == "2026-01-02 00:00"
    assert mdates.num2date(info["alarm_x"]).hour == 2


def test_timestamp_free_case_uses_zero_based_rows():
    info = historical_axis(case())
    assert not info["is_datetime"]
    assert info["x"].tolist() == [0, 1, 2]
    assert info["alarm_label"] == "row 3"
    assert info["axis_label"] == "Row number (no timestamp column)"


@pytest.mark.parametrize(
    "values",
    [[1, 2, 3], ["2026-01-01"], ["2026-01-01", None, "2026-01-03"], ["2026-01-02", "2026-01-01", "2026-01-03"]],
)
def test_incomplete_or_invented_time_axis_is_rejected(values):
    with pytest.raises((ValueError, TypeError)):
        timestamp_labels(values, expected_length=3)


def test_date_ticks_preserve_original_timezone():
    from matplotlib.figure import Figure
    from sdk.historical_attribution import style_time_axis

    data = case(
        [
            "2026-01-01 21:00-07:00",
            "2026-01-01 22:00-07:00",
            "2026-01-02 00:00-07:00",
            "2026-01-02 02:00-07:00",
            "2026-01-02 03:00-07:00",
        ]
    )
    info = historical_axis(data)
    axis = Figure().add_subplot()
    style_time_axis(axis, info)
    labels = axis.xaxis.get_major_formatter().format_ticks(info["x"])
    assert any(label.endswith("21:00") for label in labels)
    assert info["labels"][1] == "2026-01-01 21:00 -0700"


def test_short_irregular_axis_ticks_show_original_seconds():
    from matplotlib.figure import Figure
    from sdk.historical_attribution import style_time_axis

    data = case(
        [
            "2020-03-09 10:44:33",
            "2020-03-09 10:44:34",
            "2020-03-09 10:44:36",
            "2020-03-09 10:44:37",
            "2020-03-09 10:44:38",
        ]
    )
    info = historical_axis(data)
    axis = Figure().add_subplot()
    axis.plot(info["x"], [1, 2, 3])
    style_time_axis(axis, info)
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["10:44:33", "10:44:34", "10:44:36", "10:44:37"]
    np.testing.assert_allclose(np.diff(axis.get_xticks()) * 86400, [1, 2, 1], atol=1e-6)


@pytest.mark.parametrize("start,spacing", [(0.123, 0.001), (1e9, 0.125)], ids=["fractional_scale", "large_offset"])
def test_compact_numeric_ticks_preserve_values_and_distinct_labels(start, spacing):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from sdk.historical_attribution import style_time_axis

    values = start + np.arange(20) * spacing
    axis = Figure(figsize=(11, 8.5)).add_axes([0.10, 0.468, 0.82, 0.27])
    axis.plot(values, np.arange(20))
    style_time_axis(
        axis,
        dict(x=values, alarm_x=start + 20 * spacing, is_datetime=False, time_kind="numeric", axis_label="time"),
    )
    canvas = FigureCanvasAgg(axis.figure)
    canvas.draw()
    labels = axis.get_xticklabels()
    text = [label.get_text() for label in labels]
    assert len(text) > 6 and len(set(text)) == len(text)
    assert all("." not in value or len(value.split(".")[1]) <= 2 for value in text)
    boxes = sorted((label.get_window_extent(canvas.get_renderer()) for label in labels), key=lambda box: box.x0)
    assert all(right.x0 - left.x1 >= 4 for left, right in pairwise(boxes))
    np.testing.assert_array_equal(axis.lines[0].get_xdata(), values)
    assert axis.xaxis.get_offset_text().get_text()


def test_time_tick_density_adapts_to_width_and_font_without_losing_coordinates():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from sdk.historical_attribution import style_time_axis

    values = 30 + np.arange(20) * 0.5 + (np.arange(20) % 2) * 0.125
    figure = Figure(figsize=(11, 8.5))
    axis = figure.add_axes([0.10, 0.468, 0.82, 0.27])
    axis.plot(values, np.arange(20))
    style_time_axis(axis, dict(x=values, alarm_x=40, is_datetime=False, time_kind="numeric", axis_label="time"))
    counts, selected_ticks = [], []
    for width, fontsize in [(11, 8), (2.5, 8), (11, 28)]:
        figure.set_size_inches(width, 8.5)
        axis.tick_params(labelsize=fontsize)
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        ticks = axis.get_xticks()
        assert ticks[0] == values[0] and ticks[-1] == 40
        assert np.isin(ticks, np.r_[values, 40]).all()
        labels = axis.get_xticklabels()
        boxes = sorted((label.get_window_extent(canvas.get_renderer()) for label in labels), key=lambda box: box.x0)
        assert all(right.x0 - left.x1 >= 4 for left, right in pairwise(boxes))
        counts.append(len(labels))
        selected_ticks.append(ticks)
    assert counts[0] > 6 and counts[0] > counts[1]
    assert not np.array_equal(selected_ticks[0], selected_ticks[2])
    np.testing.assert_array_equal(axis.lines[0].get_xdata(), values)


def test_automatic_target_is_native_scaled_top1_not_largest_physical_unit(tmp_path, monkeypatch):
    checkpoint = tmp_path / "test_config.pt"
    torch.save({"config": {"dataset": {"window_length": 100, "split": 10, "scale_factor": 1.0}}}, checkpoint)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    frame = pd.DataFrame({"large_unit": np.linspace(0, 1000, 100), "small_unit": np.linspace(0, 1, 100)})
    target = frame.to_numpy() / np.array([1000, 1])
    recon = target.copy()
    recon[80] -= [0.2, 0.5]
    flags = np.zeros(100, bool)
    flags[80] = True
    request = HistoricalAttributionRequest(event_position=80, target=None, validation_repeats=False)

    def forbidden(_):
        raise AssertionError("Unavailable donor must not trigger new inference")

    result = run_sdk_attribution(
        frame, dict(target=target, recon=recon), flags, np.ones(100), request, forbidden, checkpoint, checkpoint
    )
    assert 0.2 * 1000 > 0.5
    assert result["event"]["target"] == "small_unit"
    assert result["event"]["target_selection"] == "largest_original_SDK_scaled_residual"
    assert result["event"]["source"] == "sdk" and result["event"]["sdk_detected"]


def test_five_metrics_keep_definitions_and_missing_controls():
    metric = HistoricalAttributionMetrics()
    effect = metric.compute_effects(10, 7, 9, normal_truth=5, normal_before=4, normal_after=4.5)["points"][0]
    assert effect["HI"] == pytest.approx(200 / 3)
    assert effect["TRI_F"] == 2 and effect["CAI"] == 1.5
    missing = metric.compute_effects(10, 7, 9)["points"][0]
    assert missing["CAI"] is None
    assert metric.compute_effects(10, 10, 9)["points"][0]["HI"] is None
    group = dict(
        kind="historical_blocks",
        case_id="case",
        run_id=1,
        target="x",
        unit="kPa",
        replacement_source="normal",
        forecast_time=0,
        scope="target",
    )
    records = [dict(group, candidate_id=key, CAI=value) for key, value in zip("abc", [3, 1, -2], strict=True)]
    shares = metric.compute_contribution_share(records, group=group, expected_candidates=list("abc"))
    assert [row["share_pct"] for row in shares["rows"]] == [75, 25, 0]
    incomplete = metric.compute_contribution_share(records[:1], group=group, expected_candidates=list("abc"))
    assert incomplete["status"] == "unavailable"
    repeats = {"first": dict(a=3, b=1, c=-2), "second": dict(a=4, b=2, c=-1)}
    stable = metric.compute_stability(repeats, axis="donor", expected_candidates=list("abc"))
    assert all(row["sign_agreement"] == 1 for row in stable["candidates"])
    one = metric.compute_stability({"first": repeats["first"]}, axis="donor", expected_candidates=list("abc"))
    assert one["reason"] == "insufficient_repeats"


def test_only_selected_outputs_enter_time_shares():
    x, flags, req, visible, infer = fixture()
    h = analyze_single_window(x, ["a", "b", "c"], flags, np.ones(60), infer(x), infer, req, visible=visible)
    condition = h["conditions"][0]
    for i, point in enumerate(condition["points"], 1):
        point["signed_change"] = i * (-1 if i % 2 else 1)
        point["historical_reconstruction_change_score"] = i
    condition["curves"]["after"] = [1e8] * 60  # The denominator is the selected outputs only.
    primary = report_data(h)["summary"]["primary"]
    assert sum(p["share_pct"] for p in primary["points"]) == pytest.approx(100)
    assert primary["time_rankings"][0]["share_pct"] == pytest.approx(20 / 210 * 100)
    assert primary["points"][0]["signed_change"] == -1
    for point in condition["points"]:
        point["signed_change"] = point["historical_reconstruction_change_score"] = 0
    assert all(p["share_pct"] is None for p in report_data(h)["summary"]["primary"]["points"])


def strategy_bank(similarity=None, **settings):
    bank = pd.DataFrame(
        {"feed": np.sin(np.arange(200) / 9), "pressure": np.cos(np.arange(200) / 13)},
        index=pd.date_range("2026-01-01", periods=200, freq="s"),
    )
    resolver = NearestNormalWindowRetriever(
        bank,
        [(0, 200)],
        [(0, 30)],
        {"run": "normal", "normal_evidence": "test"},
        same_source=False,
        similarity=similarity,
        **settings,
    )
    return bank, resolver


@pytest.mark.parametrize("similarity", [NormalizedMAEDistance(), CosineDistance(), EuclideanDistance()])
def test_injected_distances_use_calibrated_context_and_save_provenance(similarity):
    bank, resolver = strategy_bank(similarity, top_n=3)
    values = bank.to_numpy() + 0.1
    match = resolver.resolve(values, list(bank.columns), 120, 140, np.ones(200, bool), 140)
    location = np.median(bank.iloc[:30], axis=0)
    scale = np.quantile(bank.iloc[:30], 0.95, axis=0) - np.quantile(bank.iloc[:30], 0.05, axis=0)
    query = (values[112:120] - location) / scale
    expected = [
        (similarity.compare(query, (bank.iloc[ds - 8 : ds].to_numpy() - location) / scale), ds) for ds in range(38, 181)
    ]
    best_score = min(score for score, _ in expected)
    assert match.source["selected_score"] == pytest.approx(best_score)
    assert match.source["similarity_method"] == similarity.name
    assert match.source["score_direction"] == "minimize"
    assert match.source["scale_definition"].endswith("normal Q95-Q05 span")
    assert match.source["acceptance_threshold"] is None  # Injecting a method adds no gate.
    assert len(match.donors) == 3
    for rank, donor in enumerate(match.donors, 1):
        assert donor.accepted and donor.source["accepted"]
        assert donor.source["rank"] == rank
        assert donor.source["start_time"] == str(bank.index[donor.source["start"]])
        np.testing.assert_array_equal(donor.values, bank.iloc[donor.source["start"] : donor.source["stop"]].to_numpy())


def test_custom_maximize_similarity_and_independent_acceptance():
    class Similarity:
        name = "negative_L2"
        direction = "maximize"

        def compare(self, query, candidate):
            return -float(np.linalg.norm(query - candidate))

    bank, distance = strategy_bank(EuclideanDistance())
    _, similarity = strategy_bank(Similarity())
    values = bank.to_numpy() + 0.2
    args = values, list(bank.columns), 120, 140, np.ones(200, bool), 140
    low, high = distance.resolve(*args), similarity.resolve(*args)
    assert low.source["start"] == high.source["start"]
    assert low.source["selected_score"] == -high.source["selected_score"]
    assert high.source["selected_distance"] is None
    assert [d.source["score"] for d in high.donors] == sorted([d.source["score"] for d in high.donors], reverse=True)
    # Normal-range acceptance is unchanged when the ranking method changes.
    distance.max_difference = similarity.max_difference = 0.001
    low, high = distance.resolve(*args), similarity.resolve(*args)
    assert low.status == high.status == "no_acceptable_replacement"
    assert low.donors == high.donors == []
    similarity.similarity.direction = "ambiguous"
    with pytest.raises(ValueError, match="direction"):
        similarity.resolve(*args)


def test_undefined_comparisons_are_not_zero_distance_donors():
    class Undefined:
        name = "undefined"
        direction = "minimize"

        def compare(self, query, candidate):
            return np.nan

    bank, resolver = strategy_bank(Undefined())
    result = resolver.resolve(bank.to_numpy(), list(bank.columns), 120, 140, np.ones(200, bool), 140)
    assert result.status == "no_acceptable_replacement" and not result.donors
    assert result.reason == "no_defined_similarity_scores"
    assert result.source["legal_window_count"] == result.source["undefined_similarity_count"] > 0


def analyze_fixture(request, callback=None):
    x, flags, _, visible, infer = fixture()
    return analyze_single_window(
        x,
        ["a", "b", "c"],
        flags,
        np.ones(60),
        infer(x),
        callback or infer,
        request,
        visible=visible,
    )


def test_default_strategy_matches_legacy_numerics_and_output_shape():
    x, _, req, _, infer = fixture()
    implicit = analyze_fixture(req)
    explicit = analyze_fixture(replace(req, replacement_strategy=LinearBlendReplacement()))
    # Frozen legacy arithmetic, independently of the new strategy wrapper.
    expected = x.copy()
    weights = np.array([1 / 3, 2 / 3] + [1.0] * 16 + [2 / 3, 1 / 3])
    expected[20:40, 1] = (1 - weights) * x[20:40, 1] + weights * req.resolver.frame.iloc[20:40, 1]
    np.testing.assert_array_equal(implicit["replacement_input"], expected)
    np.testing.assert_array_equal(implicit["reconstruction"], infer(expected))
    assert implicit == explicit
    assert implicit["condition_count"] == 1
    assert implicit["primary_condition_id"] == implicit["conditions"][0]["condition"]["id"]
    for key in ["curves", "points", "top_points", "metrics", "alarm_comparison"]:
        assert implicit[key] == implicit["conditions"][0][key]
    json.dumps(implicit, allow_nan=False)


@pytest.mark.parametrize("donor_count", [1, 2, 5, 8])
def test_multi_donor_independent_inputs_inferences_controls_and_metrics(donor_count):
    x, _, req, _, infer = fixture()
    normal = req.resolver.frame.to_numpy()
    donors = [
        NormalDonor(normal[20:40] + 2 * i, {"run": f"normal_{i}", "start": 20, "stop": 40}, control=normal)
        for i in range(donor_count)
    ]
    donors.append(NormalDonor(normal[0:20] + 100, {"run": "rejected"}, accepted=False))

    class Resolve:
        def resolve(self, *args: object):
            return ReferenceMatch("accepted", donors=donors)

    seen = []

    def record(a):
        seen.append(a.copy())
        # Inference callbacks cannot accidentally mutate persisted conditions.
        a[:] += 0
        return infer(a)

    req = replace(
        req, resolver=Resolve(), scope="all", validation_repeats=False, replacement_strategy=MultiDonorReplacement()
    )
    result = analyze_fixture(req, record)
    assert result["condition_count"] == result["distinct_donor_count"] == donor_count
    assert len(seen) == 3 * donor_count  # fault, control, replaced control for EACH donor
    for index, condition in enumerate(result["conditions"]):
        expected, _ = blended_replacement(x, donors[index].values, 20, 40, [0, 1, 2], 2)
        np.testing.assert_array_equal(condition["replacement_input"], expected)
        np.testing.assert_array_equal(seen[index * 3], expected)
        np.testing.assert_array_equal(condition["reconstruction"], infer(expected))
        assert condition["condition"]["donor"]["source"] == donors[index].source
        assert condition["replaced_features"] == ["a", "b", "c"]
        assert condition["metrics"]["CAI"] is not None
        stability = condition["metrics"]["ExplanationStability"]
        assert stability["axis"] == "donor"
        assert stability["checked_axes"] == (["donor"] if donor_count > 1 else [])
        assert stability["unverified_axes"] == ["seed", "mask", "boundary"]
        assert stability["distinct_donor_count"] == donor_count
        if donor_count > 1:
            assert len(stability["alarm_effect"]["replicates"]) == donor_count
            assert stability["output_response_ranking"]["rank_comparisons"]
        else:
            assert stability["status"] == "unavailable" and stability["reason"] == "single_donor"
        assert condition["metrics"]["HistoricalContributionShare"]["reason"] == "single_intervention"
    np.testing.assert_array_equal(x, fixture()[0])  # No accumulation or donor averaging.
    json.dumps(result, allow_nan=False)
    report = json.loads(json.dumps(report_data(result), allow_nan=False))
    assert report["summary"]["condition_count"] == donor_count
    assert report["report_context"]["distinct_donor_count"] == donor_count
    assert "primary_seed" not in report["summary"] and "alarm_metrics_by_seed" not in report["summary"]
    for condition in result["conditions"]:
        identifier = condition["condition"]["id"]
        assert report["summary"]["alarm_metrics_by_condition"][identifier] == condition["metrics"]
        assert report["summary"]["conditions"][identifier]["donor_source"] == condition["condition"]["donor"]["source"]


@pytest.mark.parametrize("donor_count", [1, 2, 5, 7, np.int64(5)])
def test_retrieval_top_n_is_not_multi_donor_inference(donor_count):
    bank, resolver = strategy_bank(top_n=donor_count)
    x = bank.to_numpy() + 0.1
    flags = np.zeros(200, bool)
    flags[140] = True

    def visible(p):
        return np.arange(200) != p

    calls = []

    def infer(a):
        calls.append(a.copy())
        return a * 0.5

    req = HistoricalAttributionRequest(resolver=resolver, target="pressure", validation_repeats=False)
    one = analyze_single_window(x, list(bank.columns), flags, np.ones(200), x * 0.5, infer, req, visible=visible)
    assert one["condition_count"] == len(calls) == 1
    assert len(one["reference"]["candidates"]) == donor_count
    assert one["reference"]["source"]["requested_donor_count"] == donor_count
    assert one["reference"]["source"]["returned_donor_count"] == donor_count
    req.replacement_strategy = MultiDonorReplacement()
    many = analyze_single_window(x, list(bank.columns), flags, np.ones(200), x * 0.5, infer, req, visible=visible)
    assert many["condition_count"] == many["distinct_donor_count"] == donor_count
    assert len(calls) == 1 + donor_count
    assert all(r["metrics"]["CAI"] is None for r in many["conditions"])
    assert many["metrics"]["ExplanationStability"]["effect_metric"] == "TRI_F"
    json.dumps(many, allow_nan=False)


@pytest.mark.parametrize("available", [0, 2])
def test_multi_donor_uses_only_available_normal_windows(available):
    bank, resolver = strategy_bank(top_n=5)
    # Eight context rows plus twenty donor rows leave exactly two legal starts.
    intervals = [(0, 30)] + ([(40, 69)] if available else [])
    resolver = replace(resolver, normal_intervals=intervals)
    values = bank.iloc[:100].to_numpy() + 0.1
    flags = np.arange(len(values)) == 80
    calls = []

    def infer(a):
        calls.append(a.copy())
        return a * 0.5

    request = HistoricalAttributionRequest(
        resolver=resolver,
        scope="all",
        replacement_strategy=MultiDonorReplacement(),
        validation_repeats=False,
    )
    result = analyze_single_window(
        values,
        list(bank.columns),
        flags,
        np.ones(len(values)),
        values * 0.5,
        infer,
        request,
        visible=lambda p: np.arange(len(values)) != p,
    )
    assert len(calls) == available
    assert result["reference"]["source"]["requested_donor_count"] == 5
    assert result["reference"]["source"]["returned_donor_count"] == available
    if available:
        assert result["condition_count"] == result["distinct_donor_count"] == available
    else:
        assert result["status"] == "no_acceptable_replacement" and "conditions" not in result


@pytest.mark.parametrize("count", [0, -1, 1.5, True, "5"])
def test_donor_count_requires_a_positive_integer(count):
    bank, resolver = strategy_bank(top_n=count)
    with pytest.raises(ValueError, match="positive integer"):
        resolver.resolve(bank.to_numpy(), list(bank.columns), 120, 140, np.ones(len(bank), bool), 140)


@pytest.mark.parametrize("offset", [0, 1])
def test_multiple_conditions_per_donor_preserve_every_result_and_self_check(offset):
    class TwoRules:
        def generate(self, context):
            conditions = []
            for edge in [0, 2]:
                condition = LinearBlendReplacement(blend_points=edge).generate(context)[0]
                if edge == 2:
                    ix = np.ix_(range(context.start, context.stop), context.channels)
                    condition.values[ix] += offset
                    if condition.control_values is not None:
                        condition.control_values[ix] += offset
                    condition.parameters["offset"] = offset
                conditions.append(replace(condition, condition_id=f"blend_{edge}"))
            return conditions

    x, _, request, _, _ = fixture()
    normal = request.resolver.frame.to_numpy()
    donors = [NormalDonor(normal[20:40] + i, {"id": f"normal_{i}"}, normal) for i in range(5)]

    class Resolve:
        def resolve(self, *args: object):
            return ReferenceMatch("accepted", donors=donors)

    result = analyze_fixture(
        replace(request, resolver=Resolve(), replacement_strategy=MultiDonorReplacement(TwoRules()))
    )
    assert result["condition_count"] == 10 and result["distinct_donor_count"] == 5
    assert len(result["validation"]["self_conditions"]) == 2
    assert result["validation"]["self_input_equal"] == (offset == 0)
    assert result["validation"]["self_max_abs_difference"] == pytest.approx(offset * 0.5)
    for index, condition in enumerate(result["conditions"]):
        edge = condition["condition"]["parameters"]["blend_points"]
        expected, _ = blended_replacement(x, donors[index // 2].values, 20, 40, [1], edge)
        if edge == 2:
            expected[20:40, 1] += offset
        np.testing.assert_array_equal(condition["replacement_input"], expected)
        assert condition["metrics"]["ExplanationStability"]["reason"] == "nonmatched_replacement_rules"
    selected = dict(result, primary_condition_id=result["conditions"][-1]["condition"]["id"])
    report = json.loads(json.dumps(report_data(selected), allow_nan=False))
    summary = report["summary"]
    assert summary["condition_count"] == len(summary["conditions"]) == len(summary["alarm_metrics_by_condition"]) == 10
    assert report["report_context"]["distinct_donor_count"] == 5
    assert summary["primary"]["condition_id"] == summary["primary_condition_id"]
    assert summary["alarm_metrics_by_condition"][summary["primary_condition_id"]] == result["conditions"][-1]["metrics"]
    with pytest.raises(ValueError, match="Primary condition ID"):
        report_data(dict(result, primary_condition_id="missing"))


def test_report_renderer_accepts_previously_saved_seed_schema():
    from matplotlib.text import Text
    from sdk.historical_attribution import create_historical_pages

    _, _, request, _, _ = fixture()
    report = report_data(analyze_fixture(request))
    summary = report["summary"]
    effect = summary.pop("alarm_metrics_by_condition")[summary.pop("primary_condition_id")]
    summary.update(primary_seed=23, alarm_metrics_by_seed={"23": effect})
    page = list(create_historical_pages(json.loads(json.dumps(report))))[-1]
    text = " ".join(t.get_text() for t in page.findobj(Text))
    assert "Historical attribution" in text
    assert report["summary"]["alarm_metrics_by_seed"]["23"] == effect


def test_multi_donor_rule_cannot_switch_to_another_admitted_donor():
    x, _, request, _, _ = fixture()
    donors = [NormalDonor(x[20:40] + i, {"id": f"normal_{i}"}) for i in range(2)]

    class Resolve:
        def resolve(self, *args: object):
            return ReferenceMatch("accepted", donors=donors)

    class WrongDonorRule:
        def generate(self, context):
            other = ReferenceMatch("accepted", donors=donors[1:])
            return LinearBlendReplacement().generate(replace(context, reference=other))

    request = replace(
        request,
        resolver=Resolve(),
        replacement_strategy=MultiDonorReplacement(WrongDonorRule()),
        validation_repeats=False,
    )
    with pytest.raises(ValueError, match="admitted donor"):
        analyze_fixture(request, lambda _: pytest.fail("A mismatched donor must fail before inference"))


def test_custom_replacement_needs_no_main_flow_method_branch():
    class Direct:
        def generate(self, ctx):
            donor = NormalDonor(ctx.reference.values, ctx.reference.source, ctx.reference.control)
            values = ctx.values.copy()
            values[np.ix_(range(ctx.start, ctx.stop), ctx.channels)] = donor.values[:, ctx.channels]
            return [ReplacementCondition("direct", values, "custom_direct", {}, donor)]

    _, _, req, _, _ = fixture()
    result = analyze_fixture(replace(req, signals=["c", "a"], replacement_strategy=Direct()))
    assert result["replaced_features"] == ["a", "c"]  # Model order is preserved.
    assert result["condition"]["strategy"] == "custom_direct"
    assert result["metrics"]["CAI"] is None
    assert result["normal_control"]["reason"] == "strategy_did_not_supply_matched_control"
    assert result["changes"]["weights"] is None
    report = report_data(result)
    assert report["report_context"]["control_unavailable_reason"] == "matched-control replacement missing"
    assert "An aligned normal control exists" in report["report_context"]["control_note"]


@pytest.mark.parametrize("signals", [[], ["a", "a"], ["missing"], "a"])
def test_invalid_signal_lists_are_rejected(signals):
    _, _, req, _, _ = fixture()
    with pytest.raises(ValueError, match="signals"):
        analyze_fixture(replace(req, signals=signals))


def test_explicit_signal_list_and_all_are_unambiguous():
    _, _, req, _, _ = fixture()
    with pytest.raises(ValueError, match="signals"):
        analyze_fixture(replace(req, scope="signals"))
    with pytest.raises(ValueError, match="signals"):
        analyze_fixture(replace(req, scope="all", signals=["a"]))
    result = analyze_fixture(replace(req, scope="signals", signals=["c", "a"]))
    assert result["changes"]["changed_features"] == ["a", "c"]


def test_hidden_truth_cannot_change_donor_or_visible_replacement():
    bank, resolver = strategy_bank(top_n=2)
    values = bank.to_numpy() + 0.3
    visible = np.arange(200) % 2 == 1
    times = tuple(map(str, bank.index))
    match = resolver.resolve(values, list(bank.columns), 120, 140, visible, 140)
    ctx = ReplacementContext(values, tuple(bank.columns), visible, 140, 120, 140, (0, 1), times, match)
    variants = MultiDonorReplacement().generate(ctx)
    other = values.copy()
    other[~visible] = 1e20
    # Disallowed visible future values are not query/endpoint inputs either.
    other[140:] = -1e20
    changed_match = resolver.resolve(other, list(bank.columns), 120, 140, visible, 140)
    changed_variants = MultiDonorReplacement().generate(replace(ctx, values=other, reference=changed_match))
    assert match.source["query_hash"] == changed_match.source["query_hash"]
    for original, changed in zip(variants, changed_variants, strict=True):
        assert original.donor.source == changed.donor.source
        np.testing.assert_array_equal(
            original.values[120:140][visible[120:140]],
            changed.values[120:140][visible[120:140]],
        )
    # Raw hidden placeholders differ; this is NOT a native SDK leakage claim.


@pytest.mark.parametrize("kind", ["outside", "wrong_channel", "unadmitted"])
def test_invalid_custom_conditions_fail_before_any_inference(kind):
    class Invalid:
        def generate(self, ctx):
            donor = NormalDonor(ctx.reference.values, ctx.reference.source, ctx.reference.control)
            values = ctx.values.copy()
            if kind == "outside":
                values[ctx.event] += 10
            elif kind == "wrong_channel":
                values[ctx.start, 0] += 10
            else:
                donor = replace(donor, source={"id": "unadmitted"})
            return [ReplacementCondition("bad", values, "bad_rule", {}, donor)]

    _, _, req, _, _ = fixture()

    def forbidden(_):
        raise AssertionError("Invalid condition must fail before inference")

    req = replace(req, replacement_strategy=Invalid(), validation_repeats=False)
    with pytest.raises(ValueError, match="Replacement"):
        analyze_fixture(req, forbidden)


def test_no_eligible_donor_and_empty_strategy_do_not_force_inference():
    class Empty:
        def generate(self, context):
            return []

    _, _, req, _, _ = fixture()
    req = replace(req, replacement_strategy=Empty(), validation_repeats=False)

    def forbidden(_):
        raise AssertionError("No condition must mean no replacement inference")

    result = analyze_fixture(req, forbidden)
    assert result["reason"] == "strategy_produced_no_conditions"


def test_multi_donor_self_reference_is_exact_and_repeatable():
    x, _, req, _, infer = fixture()
    req.resolver = AlignedNormalReference(pd.DataFrame(x, columns=["a", "b", "c"]), {"id": "self"}, True)
    req.replacement_strategy = MultiDonorReplacement()
    first, second = analyze_fixture(req), analyze_fixture(req)
    assert first == second
    np.testing.assert_array_equal(first["replacement_input"], x)
    np.testing.assert_array_equal(first["reconstruction"], infer(x))
    assert first["validation"]["self_max_abs_difference"] == 0


def test_disabled_attribution_does_not_call_new_pipeline(monkeypatch):
    from sdk import anomaly_analysis as sdk
    from sdk import historical_attribution as historical

    def forbidden(*args: object, **kwargs: object):
        raise AssertionError("Disabled attribution must not enter strategy/reconstruction pipeline")

    frame = pd.DataFrame({"signal": np.arange(8, dtype=float)})
    native = dict(
        target=frame.to_numpy(),
        recon=frame.to_numpy() * 0.5,
        residual=np.arange(8, dtype=float),
        residual_l2=np.arange(8, dtype=float),
        target_dim=1,
    )
    calls = []
    monkeypatch.setattr(sdk, "_resolve_model_paths", lambda *_args: ("model.pth", "config.yaml"))
    monkeypatch.setattr(sdk, "get_model_target_dim", lambda *_args: 1)

    def inference(**kwargs):
        calls.append(kwargs)
        return native

    class Threshold:
        @property
        def macs_thresholder(self):
            return self

        def detect_anomalies(self, scores, target):
            return np.asarray(scores) > 3

    monkeypatch.setattr(sdk, "inference_ad_tesseract2_mp", inference)
    monkeypatch.setattr(sdk, "MACSThresholdStrategy", Threshold)
    monkeypatch.setattr(historical, "run_sdk_attribution", forbidden)
    implicit = sdk.perform_anomaly_analysis_with_diffusion(frame, threshold_strategy="macs")
    explicit = sdk.perform_anomaly_analysis_with_diffusion(
        frame, threshold_strategy="macs", historical_attribution=None
    )
    pd.testing.assert_frame_equal(implicit, explicit)
    assert "historical_attribution" not in implicit.attrs
    assert len(calls) == 2


def test_sdk_adapter_forwards_signals_and_multi_donor_strategy(tmp_path, monkeypatch):
    checkpoint = tmp_path / "config.pt"
    torch.save({"config": {"dataset": {"window_length": 100, "split": 10, "scale_factor": 1.0}}}, checkpoint)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    columns = ["input", "target", "actuator"]
    x = np.arange(300, dtype=float).reshape(100, 3) / 10
    frame = pd.DataFrame(x, columns=columns)
    normal = x * 0.8
    donors = [
        NormalDonor(normal[60:80], {"id": "first", "start": 60, "stop": 80}, normal),
        NormalDonor(normal[40:60], {"id": "second", "start": 40, "stop": 60}, normal),
    ]

    class Resolve:
        def resolve(self, *args: object):
            return ReferenceMatch("accepted", donors=donors)

    calls = []

    def native(a):
        scaled = (a - a.min(axis=0)) / np.ptp(a, axis=0)
        return dict(target=scaled, recon=scaled * 0.8)

    def infer(data):
        assert list(data.columns) == columns
        calls.append(data.to_numpy().copy())
        return native(data.to_numpy())

    flags = np.zeros(100, bool)
    flags[80] = True
    request = HistoricalAttributionRequest(
        resolver=Resolve(),
        event_position=80,
        target="target",
        signals=["actuator", "input"],
        replacement_strategy=MultiDonorReplacement(),
        validation_repeats=False,
    )
    result = run_sdk_attribution(frame, native(x), flags, np.ones(100), request, infer, checkpoint, checkpoint)
    assert result["condition_count"] == 2 and len(calls) == 6
    assert result["event"]["target"] == "target"
    assert result["replaced_features"] == ["input", "actuator"]
    assert result["window"]["visible_count"] == 10
    assert len(result["native_settings"]["preprocessing_audit"]) == 7
    for condition in result["conditions"]:
        assert condition["validation"]["outside_window_unchanged"]
        assert condition["metrics"]["ExplanationStability"]["axis"] == "donor"
        np.testing.assert_array_equal(np.asarray(condition["replacement_input"])[:, 1], x[:, 1])


def test_identical_multidonor_inputs_have_identical_outputs():
    x, _, req, _, _ = fixture()

    class Resolve:
        def resolve(self, *args: object):
            return ReferenceMatch(
                "accepted",
                donors=[
                    NormalDonor(x[20:40].copy(), {"id": "one"}),
                    NormalDonor(x[20:40].copy(), {"id": "two"}),
                ],
            )

    req = replace(req, resolver=Resolve(), replacement_strategy=MultiDonorReplacement(), validation_repeats=False)
    result = analyze_fixture(req)
    one, two = result["conditions"]
    assert one["replacement_hash"] == two["replacement_hash"]
    assert one["reconstruction"] == two["reconstruction"]
    assert one["metrics"]["delta_F"] == two["metrics"]["delta_F"] == 0


def test_only_rejected_payloads_do_not_trigger_multidonor_inference():
    x, _, req, _, _ = fixture()

    class Resolve:
        def resolve(self, *args: object):
            return ReferenceMatch("accepted", donors=[NormalDonor(x[20:40], {"accepted": False})])

    def forbidden(_):
        raise AssertionError("Rejected donor payload must not be inferred")

    req = replace(req, resolver=Resolve(), replacement_strategy=MultiDonorReplacement(), validation_repeats=False)
    result = analyze_fixture(req, forbidden)
    assert result["status"] == "no_acceptable_replacement"
    assert "conditions" not in result


@pytest.mark.parametrize("multiple", [False, True])
def test_rm_object_can_be_swapped_for_single_or_multiple_donors(multiple):
    class DirectDonorRM:
        def generate(self, context):
            return [
                replace(condition, strategy="direct_donor")
                for condition in LinearBlendReplacement(blend_points=0).generate(context)
            ]

    x, _, req, _, infer = fixture()
    normal = req.resolver.frame.to_numpy()
    donors = [
        NormalDonor(normal[20:40], {"id": "first"}, normal),
        NormalDonor(normal[10:30] + 2, {"id": "second"}, normal),
    ]

    class Resolve:
        def resolve(self, *args: object):
            return ReferenceMatch("accepted", donors=donors)

    rm = DirectDonorRM()
    strategy = MultiDonorReplacement(rule=rm) if multiple else rm
    request = replace(
        req,
        resolver=Resolve(),
        replacement_strategy=strategy,
        signals=["c", "a"],
        validation_repeats=False,
    )
    result = analyze_fixture(request)
    assert result["condition_count"] == (2 if multiple else 1)
    for index, condition in enumerate(result["conditions"]):
        variant = np.asarray(condition["replacement_input"])
        np.testing.assert_array_equal(variant[20:40][:, [0, 2]], donors[index].values[:, [0, 2]])
        np.testing.assert_array_equal(variant[:, 1], x[:, 1])
        np.testing.assert_array_equal(condition["reconstruction"], infer(variant))
        assert condition["metrics"]["CAI"] is not None
        if multiple:
            assert condition["condition"]["parameters"]["rule"] == "direct_donor"
            assert condition["condition"]["parameters"]["donor_mode"] == "separate"
        else:
            assert condition["condition"]["strategy"] == "direct_donor"
    # The custom RM changes the edge arithmetic; it is not only a donor-count switch.
    blended = analyze_fixture(replace(request, replacement_strategy=LinearBlendReplacement()))
    assert result["replacement_input"][20][0] != blended["replacement_input"][20][0]


def _mock_public_sdk(tmp_path, monkeypatch, features, *, flags=None, errors=None, dataset=None, cuda=False):
    """Exercise the public SDK and real attribution adapter without model downloads."""
    from sdk import anomaly_analysis as sdk

    count, dim = features.shape
    dataset = dict(dataset or {})
    factor = dataset.get("scale_factor", 1.0)
    checkpoint = tmp_path / "config.pt"
    torch.save({"config": {"model": {"target_dim": dim}, "dataset": dataset}}, checkpoint)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(sdk, "_resolve_model_paths", lambda *_: (str(checkpoint), str(checkpoint)))
    monkeypatch.setattr(sdk, "get_model_target_dim", lambda *_: dim)
    if flags is None:
        flags = np.arange(count) == 80
    errors = np.asarray(errors if errors is not None else np.full(dim, 0.2))
    calls = []

    def native(**kwargs):
        data = kwargs["data"]
        assert list(data.columns) == list(features.columns)
        calls.append(data.copy())
        values = data.to_numpy(dtype=float)
        span = np.ptp(values, axis=0)
        span = np.where(span == 0, 1.0, span)
        target = (values - values.min(axis=0)) / span * factor
        return dict(
            target=target,
            recon=target - errors,
            residual=np.full(count, float(errors.mean())),
            residual_l2=np.full(count, float(np.linalg.norm(errors))),
            target_dim=dim,
        )

    class Threshold:
        @property
        def macs_thresholder(self):
            return self

        def detect_anomalies(self, scores, target):
            return np.asarray(flags, bool).copy()

    monkeypatch.setattr(sdk, "inference_ad_tesseract2_mp", native)
    monkeypatch.setattr(sdk, "MACSThresholdStrategy", Threshold)
    return sdk, calls, native


@pytest.mark.parametrize("event_position", [-1, 100, 100_000, 79])
def test_automatic_target_rejects_invalid_event_before_indexing(tmp_path, monkeypatch, event_position):
    features = pd.DataFrame(np.arange(200).reshape(100, 2) / 10, columns=["a", "b"])
    sdk, calls, _ = _mock_public_sdk(tmp_path, monkeypatch, features)
    request = HistoricalAttributionRequest(event_position=event_position, target=None, validation_repeats=False)
    with pytest.raises(ValueError, match="Requested event must be an actual SDK anomaly"):
        sdk.perform_anomaly_analysis_with_diffusion(features, threshold_strategy="macs", historical_attribution=request)
    assert len(calls) == 1


@pytest.mark.parametrize("time_kind", ["mixed_offset_column", "dateutil_named_index", "dateutil_fixed_index"])
@pytest.mark.parametrize("include_historical", [False, True])
def test_public_reports_preserve_instants_for_nonportable_timezones(
    tmp_path, monkeypatch, time_kind, include_historical
):
    from dateutil import tz
    from sdk import reporting

    expected_times = pd.date_range("2026-11-01 08:30", periods=100, freq="min", tz="UTC")
    features = pd.DataFrame(np.arange(300).reshape(100, 3) / 10, columns=["a", "b", "c"])
    if time_kind == "dateutil_named_index":
        features.index = expected_times.tz_convert(tz.gettz("America/Los_Angeles"))
    elif time_kind == "dateutil_fixed_index":
        features.index = expected_times.tz_convert(tz.tzoffset("IST", 19_800))
    frame = features.copy()
    if time_kind == "mixed_offset_column":
        frame["timestamp"] = [value.isoformat() for value in expected_times.tz_convert("America/Los_Angeles")]
    sdk, calls, _ = _mock_public_sdk(tmp_path, monkeypatch, features)
    request = (
        HistoricalAttributionRequest(
            resolver=AlignedNormalReference(features * 0.7, {"id": "normal"}, True),
            event_position=80,
            target="b",
            validation_repeats=False,
        )
        if include_historical
        else None
    )
    pages = []
    savefig = reporting.PdfPages.savefig

    def capture_page(pdf, figure, *args: object, **kwargs: object):
        savefig(pdf, figure, *args, **kwargs)
        pages.append(figure)

    monkeypatch.setattr(reporting.PdfPages, "savefig", capture_page)
    destination = tmp_path / "timezone-report.pdf"
    output = sdk.perform_anomaly_analysis_with_diffusion(
        frame,
        threshold_strategy="macs",
        historical_attribution=request,
        sdk_config=sdk.ADDiffusionConfig(report_path=destination),
    )
    assert destination.read_bytes().startswith(b"%PDF")
    axis = reporting.resolve_time_axis(frame, "timestamp" if "timestamp" in frame else None)
    assert axis["timezone"] == "UTC"
    assert axis["label"].endswith(" (UTC)")
    pd.testing.assert_index_equal(pd.DatetimeIndex(pd.to_datetime(axis["values"], utc=True)), expected_times)
    expected_x = mdates.date2num(expected_times.to_pydatetime())
    overview = next(axis for axis in pages[0].axes if axis.get_ylabel() == "MAE")
    np.testing.assert_allclose(mdates.date2num(overview.lines[0].get_xdata()), expected_x)
    if include_historical:
        result = json.loads(destination.with_suffix(".historical.json").read_text())
        assert result == output.attrs["historical_attribution"]
        assert result["time_axis"] == axis
        assert result["event"]["time"] == str(expected_times[80])
        assert historical_axis(report_data(result))["alarm_x"] == pytest.approx(expected_x[80])
        signal, response = pages[-2].axes
        np.testing.assert_allclose(signal.lines[0].get_xdata(), expected_x)
        np.testing.assert_allclose(response.collections[1].get_offsets()[:, 0], expected_x[60:80])
        assert response.get_xlabel().endswith(" (UTC)")
    else:
        assert "historical_attribution" not in output.attrs
        assert len(calls) == 1


@pytest.mark.parametrize(
    "time_kind", ["datetime_column", "datetime_index", "numeric_time", "row_position", "invalid_time"]
)
def test_public_sdk_pdf_preserves_time_axis_identity(tmp_path, monkeypatch, time_kind):
    """The real PDF writer and JSON must use the same axis as the public SDK."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.text import Text
    from sdk import reporting

    features = pd.DataFrame(np.arange(300).reshape(100, 3) / 10, columns=["a", "b", "c"])
    times = pd.date_range("2026-10-08 12:00", periods=100, freq="min", tz="Asia/Taipei")
    features.index = times if time_kind == "datetime_index" else pd.Index(1000 + np.arange(100) * 5)
    sdk, calls, _ = _mock_public_sdk(tmp_path, monkeypatch, features, errors=[0.1, 0.5, 0.2])
    frame = features.copy()
    if time_kind == "datetime_column":
        frame["time"] = times
    elif time_kind == "numeric_time":
        frame["time"] = np.arange(100) * 0.5 + (np.arange(100) % 2) * 0.125
    elif time_kind == "invalid_time":
        frame["time"] = "unavailable"
    request = HistoricalAttributionRequest(
        resolver=AlignedNormalReference(features * 0.7, {"id": "normal"}, True),
        target=None,
        event_position=80,
        validation_repeats=False,
    )
    pages = []
    savefig = reporting.PdfPages.savefig

    def capture_page(pdf, figure, *args: object, **kwargs: object):
        savefig(pdf, figure, *args, **kwargs)
        pages.append(figure)

    monkeypatch.setattr(reporting.PdfPages, "savefig", capture_page)
    destination = tmp_path / "historical_attribution_test_report.pdf"
    output = sdk.perform_anomaly_analysis_with_diffusion(
        frame,
        threshold_strategy="macs",
        historical_attribution=request,
        sdk_config=sdk.ADDiffusionConfig(explain=True, report_path=destination),
    )
    result = json.loads(destination.with_suffix(".historical.json").read_text())
    assert result == output.attrs["historical_attribution"]
    appendix = json.loads(json.dumps(report_data(result)))
    assert destination.read_bytes().startswith(b"%PDF")
    assert all(list(call.columns) == list(features.columns) for call in calls)
    _, label, is_datetime = reporting._resolve_x_axis(frame, "time" if "time" in frame else None)
    expected = (
        mdates.date2num(times.to_pydatetime())
        if is_datetime
        else frame["time"].to_numpy()
        if time_kind == "numeric_time"
        else np.arange(len(frame))
    )
    overview_axis = next(axis for axis in pages[0].axes if axis.get_ylabel() == "MAE")
    overview_line = overview_axis.lines[0]
    overview_x = mdates.date2num(overview_line.get_xdata()) if is_datetime else overview_line.get_xdata()
    np.testing.assert_allclose(overview_x, expected)
    info = historical_axis(appendix)
    signal, response = pages[-2].axes
    np.testing.assert_allclose(signal.lines[0].get_xdata(), expected)
    np.testing.assert_allclose(response.collections[1].get_offsets()[:, 0], expected[60:80])
    assert info["alarm_x"] == pytest.approx(expected[80])
    assert appendix["case"]["alarm_position"] == result["event"]["position"] == 80
    assert appendix["case"]["time_axis"] == result["time_axis"]
    assert [p["position"] for p in appendix["summary"]["primary"]["points"]] == list(range(60, 80))
    expected_kind = "datetime" if is_datetime else "numeric" if time_kind == "numeric_time" else "row_position"
    assert response.get_xlabel() == (
        "Observation index" if expected_kind == "row_position" else overview_axis.get_xlabel()
    )
    assert result["time_axis"]["kind"] == expected_kind
    assert result["time_axis"]["position_base"] == 0
    text = " ".join(" ".join(t.get_text() for page in pages[-3:-1] for t in page.findobj(Text)).split())
    assert "SDK Top-1 contributor at this alarm" in text
    main_text = " ".join(
        " ".join(t.get_text() for page in [*pages[:-3], pages[-1]] for t in page.findobj(Text)).split()
    )
    assert ("observation index 80" if expected_kind == "row_position" else alarm_label(appendix)) in text
    for point in appendix["summary"]["primary"]["time_rankings"][:5]:
        point_label = (
            pd.Timestamp(times[point["position"]]).strftime("%H:%M")
            if is_datetime
            else info["labels"][point["case_step"]]
        )
        assert point_label in text
    if is_datetime:
        assert result["event"]["time"] == str(times[80])
        assert "2026-10-08 13:20 +0800" in text
        assert "2026-10-08 13:20:00" in main_text
    elif expected_kind == "numeric":
        assert result["event"]["time"] == "40.0" and "time 40.0" in text
        assert "timestamps" not in appendix["case"]
        assert "40.0" in main_text
    else:
        assert result["event"]["time"] == "80" and "SDK alarm: observation index 80" in text
        assert "case step 81" not in text
        assert label.startswith("Row number")
        assert "zero-based DataFrame row numbers" in main_text
    # Check the rendered text geometry, including irregular numeric time values.
    canvas = FigureCanvasAgg(pages[-2])
    canvas.draw()
    renderer = canvas.get_renderer()
    tick_labels = [tick for tick in response.get_xticklabels() if tick.get_visible() and tick.get_text()]
    boxes = sorted((tick.get_window_extent(renderer) for tick in tick_labels), key=lambda box: box.x0)
    assert all(right.x0 - left.x1 >= 4 for left, right in pairwise(boxes))
    if not is_datetime:
        assert len(tick_labels) >= 3
        assert all("." not in tick.get_text() or len(tick.get_text().split(".")[1]) <= 2 for tick in tick_labels)
    pages[-3].savefig(tmp_path / "historical_attribution_test_report_historical.png", dpi=120)
    pages[-2].savefig(tmp_path / "historical_attribution_test_report_timelines.png", dpi=120)


@pytest.mark.parametrize("donor_count", [1, 5, 8])
def test_public_sdk_retrieves_and_analyzes_requested_donor_count(tmp_path, monkeypatch, donor_count):
    """The public SDK uses real retrieval/replacement/metrics; only model inference is mocked."""
    bank, resolver = strategy_bank(top_n=donor_count)
    features = bank.iloc[:100].copy() + 0.1
    sdk, calls, _ = _mock_public_sdk(tmp_path, monkeypatch, features, errors=[0.1, 0.5])
    request = HistoricalAttributionRequest(
        resolver=resolver,
        scope="all",
        replacement_strategy=MultiDonorReplacement(),
        validation_repeats=False,
    )
    output = sdk.perform_anomaly_analysis_with_diffusion(
        features, threshold_strategy="macs", historical_attribution=request
    )
    result = output.attrs["historical_attribution"]
    assert result["condition_count"] == result["distinct_donor_count"] == donor_count
    assert len(calls) == 1 + donor_count  # Original SDK call plus each independent donor condition.
    assert result["event"]["target_index"] == 1
    for index, condition in enumerate(result["conditions"]):
        source = condition["condition"]["donor"]["source"]
        donor = bank.iloc[source["start"] : source["stop"]].to_numpy()
        np.testing.assert_array_equal(condition["condition"]["donor"]["values"], donor)
        expected, _ = blended_replacement(features.to_numpy(), donor, 60, 80, [0, 1], request.blend_points)
        np.testing.assert_array_equal(condition["replacement_input"], expected)
        np.testing.assert_array_equal(calls[index + 1], expected)
        assert list(calls[index + 1].columns) == list(features.columns)
        assert condition["metrics"]["CAI"] is None  # Retrieval does not establish a paired normal control.
        assert condition["metrics"]["ExplanationStability"]["distinct_donor_count"] == donor_count
    report = report_data(result)
    assert len(report["summary"]["alarm_metrics_by_condition"]) == donor_count
    assert report["summary"]["condition_count"] == donor_count


@pytest.mark.parametrize(
    "columns,target_index",
    [
        ([0, 1, 2], 0),
        ([0, "0", 2], 0),
        ([0, "0", 2], 1),
        (["feed", "pressure", "actuator"], 1),
        (["1", "0", "2"], 1),
        ([10, "10", -3], 1),
        (["流量", "壓力", "閥位"], 1),
        ([("line", 0), "('line', 0)", "pressure"], 0),
        ([("line", 0), "('line', 0)", "pressure"], 1),
    ],
)
@pytest.mark.parametrize("scope", ["target", "all", "signals"])
@pytest.mark.parametrize("unchanged_feature", [None, 0], ids=["all_donor_features_differ", "first_feature_matches"])
def test_public_sdk_preserves_integer_and_mixed_target_keys(
    tmp_path, monkeypatch, columns, target_index, *, scope, unchanged_feature
):
    features = pd.DataFrame(np.arange(300).reshape(100, 3) / 10, columns=columns)
    errors = np.full(3, 0.1)
    errors[target_index] = 0.5
    sdk, calls, _ = _mock_public_sdk(tmp_path, monkeypatch, features, errors=errors)
    units = {column: f"unit-{i}" for i, column in enumerate(columns)}
    normal = features * 0.7
    if unchanged_feature is not None:
        normal.iloc[:, unchanged_feature] = features.iloc[:, unchanged_feature]
    request = HistoricalAttributionRequest(
        resolver=AlignedNormalReference(normal, {"id": "normal"}, True),
        target=None,
        scope=scope,
        signals=[columns[2], columns[0]] if scope == "signals" else None,
        validation_repeats=False,
        units=units,
    )
    output = sdk.perform_anomaly_analysis_with_diffusion(
        features, threshold_strategy="macs", historical_attribution=request
    )
    result = output.attrs["historical_attribution"]
    assert result["status"] == "ok"
    assert result["event"]["target_index"] == target_index
    assert result["event"]["target"] == str(columns[target_index])
    assert result["event"]["target_selection"] == "largest_original_SDK_scaled_residual"
    assert result["unit"] == units[columns[target_index]]
    np.testing.assert_array_equal(result["curves"]["observed"], features.iloc[:, target_index])
    replaced = [target_index] if scope == "target" else [0, 1, 2] if scope == "all" else [0, 2]
    changed = [i for i in replaced if i != unchanged_feature]
    unchanged = [i for i in range(3) if i not in changed]
    np.testing.assert_array_equal(calls[1].iloc[:, unchanged], features.iloc[:, unchanged])
    assert list(calls[1].columns) == columns
    restored = json.loads(json.dumps(result, allow_nan=False))
    summaries = restored["changes"]["per_feature"]
    assert len(summaries) == len(replaced)
    assert list(summaries) == [str(i) for i in replaced]
    assert restored["changes"]["changed_feature_indices"] == changed
    assert restored["changes"]["changed_features"] == [str(columns[i]) for i in changed]
    if not changed:
        assert result["condition"]["input_hash"] == result["input_hash"]
        np.testing.assert_array_equal(result["curves"]["before"], result["curves"]["after"])
    start, stop = result["window"]["start"], result["window"]["stop"]
    visible = result["condition"]["visibility"]
    for i in replaced:
        summary = summaries[str(i)]
        assert summary["feature_index"] == i
        assert summary["display_label"] == str(columns[i])
        for name, values in [("before", features), ("after", calls[1])]:
            np.testing.assert_allclose(summary[name]["mean"], values.iloc[start:stop, i].mean())
            np.testing.assert_allclose(summary[name]["std"], values.iloc[start:stop, i].to_numpy().std())
            for seam, observed, expected in [
                ("left_seam", visible[start - 1], values.iloc[start, i] - values.iloc[start - 1, i]),
                ("right_seam", visible[stop], values.iloc[stop, i] - values.iloc[stop - 1, i]),
            ]:
                if observed:
                    np.testing.assert_allclose(summary[name][seam], expected)
                else:
                    assert summary[name][seam] is None


@pytest.mark.parametrize("target", [None, 0, "0"])
def test_direct_analysis_preserves_target_index_with_mixed_labels(target):
    x, flags, request, visible, infer = fixture()
    columns = [0, "0", 2]
    request = replace(
        request,
        target=target,
        resolver=AlignedNormalReference(pd.DataFrame(x * 0.7, columns=columns), {"id": "normal"}, True),
    )
    result = analyze_single_window(x, columns, flags, np.ones(60), infer(x), infer, request, visible=visible)
    selected = int(np.argmax(np.abs(x[40] - infer(x)[40]))) if target is None else columns.index(target)
    assert result["event"]["target_index"] == selected
    np.testing.assert_array_equal(result["curves"]["observed"], x[:, selected])
    np.testing.assert_array_equal(result["curves"]["before"], infer(x)[:, selected])
    np.testing.assert_array_equal(result["curves"]["after"], np.asarray(result["reconstruction"])[:, selected])


@pytest.mark.parametrize("constant", [False, True], ids=["nonconstant", "constant"])
@pytest.mark.parametrize("factor", [None, 7.0], ids=["omitted_factor", "explicit_factor"])
def test_public_cpu_adapter_restores_original_units(tmp_path, monkeypatch, constant, factor):
    values = np.full((100, 2), [11.0, 71.0]) if constant else np.arange(200).reshape(100, 2) / 10
    features = pd.DataFrame(values, columns=["input", "target"])
    dataset = {} if factor is None else {"scale_factor": factor}
    sdk, _, _ = _mock_public_sdk(tmp_path, monkeypatch, features, dataset=dataset)
    request = HistoricalAttributionRequest(
        resolver=AlignedNormalReference(features * 0.7, {"id": "normal"}, True),
        target="target",
        validation_repeats=False,
    )
    result = sdk.perform_anomaly_analysis_with_diffusion(
        features, threshold_strategy="macs", historical_attribution=request
    ).attrs["historical_attribution"]
    effective_factor = 1.0 if factor is None else factor
    assert result["status"] == "ok"
    assert result["native_settings"]["scale_factor"] == effective_factor
    for condition in result["conditions"]:
        replacement = np.asarray(condition["replacement_input"])
        span = np.ptp(replacement, axis=0)
        span = np.where(span == 0, 1.0, span)
        np.testing.assert_allclose(condition["reconstruction"], replacement - 0.2 / effective_factor * span)
    expected_span = np.ptp(values, axis=0)
    expected_span = np.where(expected_span == 0, 1.0, expected_span)
    np.testing.assert_allclose(result["curves"]["before"], values[:, 1] - 0.2 / effective_factor * expected_span[1])


@pytest.mark.parametrize("alarm_position", [None, 50, 80], ids=["no_alarm", "unrelated_alarm", "event_alarm"])
def test_public_sdk_dataset_label_preserves_explicit_target(tmp_path, monkeypatch, alarm_position):
    features = pd.DataFrame(np.arange(300).reshape(100, 3) / 10, columns=["a", "b", "c"])
    features.index = pd.Index(np.arange(100) * 3 + 10)
    flags = np.zeros(100, bool)
    if alarm_position is not None:
        flags[alarm_position] = True
    sdk, calls, _ = _mock_public_sdk(tmp_path, monkeypatch, features, flags=flags, errors=[0.1, 0.2, 0.8])
    labels = np.arange(100) == 80
    frame = features.assign(GT=labels, timestamp=pd.date_range("2026-01-01", periods=100, freq="min"))
    request = HistoricalAttributionRequest(
        resolver=AlignedNormalReference(features * 0.7, {"id": "normal"}, True),
        target="b",
        event_position=80,
        event_source="dataset_label",
        validation_repeats=False,
    )
    result = sdk.perform_anomaly_analysis_with_diffusion(
        frame,
        threshold_strategy="macs",
        historical_attribution=request,
        sdk_config=sdk.ADDiffusionConfig(ground_truth_column="GT", timestamp_column="timestamp"),
    ).attrs["historical_attribution"]
    assert result["status"] == "ok"
    assert result["event"]["source"] == "dataset_label"
    assert result["event"]["position"] == 80
    assert result["event"]["target_index"] == 1
    assert result["event"]["target"] == "b"
    assert result["event"]["target_selection"] == "explicit_target"
    assert result["event"]["sdk_detected"] == bool(flags[80])
    assert result["event"]["threshold_trace"]["sdk_detected"] == bool(flags[80])
    assert result["event"]["time"] == str(frame.timestamp.iloc[80])
    assert all(list(call.columns) == ["a", "b", "c"] and call.index.equals(features.index) for call in calls)
    assert all("GT" not in call and "timestamp" not in call for call in calls)
    from matplotlib.text import Text
    from sdk.historical_attribution import create_historical_pages

    appendix = report_data(result)
    # Even misleading legacy rank metadata cannot override an explicit target.
    appendix["alarm_contributors"] = {"target_rank": 1}
    page = next(create_historical_pages(appendix))
    text = " ".join(t.get_text() for t in page.findobj(Text))
    assert "Dataset-labelled analysis point" in text and "Target: b" in text
    assert "SDK Top-1 contributor" not in text and "SDK alarm:" not in text


@pytest.mark.parametrize("alarm_position", [None, 50, 80], ids=["no_alarm", "unrelated_alarm", "event_alarm"])
def test_public_sdk_dataset_label_requires_target_regardless_of_detector_flags(tmp_path, monkeypatch, alarm_position):
    features = pd.DataFrame(np.arange(300).reshape(100, 3) / 10, columns=["a", "b", "c"])
    flags = np.zeros(100, bool)
    if alarm_position is not None:
        flags[alarm_position] = True
    sdk, calls, _ = _mock_public_sdk(tmp_path, monkeypatch, features, flags=flags, errors=[0.1, 0.2, 0.8])
    frame = features.assign(GT=np.arange(100) == 80)
    request = HistoricalAttributionRequest(
        resolver=AlignedNormalReference(features * 0.7, {"id": "normal"}, True),
        event_source="dataset_label",
        event_position=80,
        target=None,
        validation_repeats=False,
    )
    with pytest.raises(ValueError, match="A dataset-labelled event requires an explicit position and target"):
        sdk.perform_anomaly_analysis_with_diffusion(
            frame,
            threshold_strategy="macs",
            historical_attribution=request,
            sdk_config=sdk.ADDiffusionConfig(ground_truth_column="GT"),
        )
    assert len(calls) == 1  # Original SDK inference only; no historical replacement replay.
    assert request.target is None


@pytest.mark.parametrize("invalid", ["missing", "nan", "nonbinary", "text", "misaligned"])
def test_public_sdk_rejects_invalid_dataset_event_labels(tmp_path, monkeypatch, invalid):
    features = pd.DataFrame(np.arange(300).reshape(100, 3) / 10, columns=["a", "b", "c"])
    sdk, calls, _ = _mock_public_sdk(tmp_path, monkeypatch, features)
    frame = features.copy()
    if invalid != "missing":
        labels = pd.Series((np.arange(100) == 80).astype(float))
        if invalid == "nan":
            labels.iloc[0] = np.nan
        if invalid == "nonbinary":
            labels.iloc[0] = 2
        if invalid == "text":
            labels = labels.astype(str)
        if invalid == "misaligned":
            labels.index = labels.index + 1
        frame["GT"] = labels
    request = HistoricalAttributionRequest(event_position=80, target="b", event_source="dataset_label")
    with pytest.raises(ValueError, match="dataset event labels"):
        sdk.perform_anomaly_analysis_with_diffusion(frame, threshold_strategy="macs", historical_attribution=request)
    assert not calls


def test_sdk_adapter_rejects_reordered_label_index_before_inference(tmp_path, monkeypatch):
    features = pd.DataFrame(np.arange(300).reshape(100, 3) / 10, columns=["a", "b", "c"])
    _, calls, native = _mock_public_sdk(tmp_path, monkeypatch, features)
    labels = pd.Series(np.arange(100) == 80, index=features.index[::-1])
    request = HistoricalAttributionRequest(event_position=80, target="b", event_source="dataset_label")
    with pytest.raises(ValueError, match="row index and order"):
        run_sdk_attribution(
            features,
            native(data=features),
            np.zeros(100, bool),
            np.ones(100),
            request,
            lambda _: pytest.fail("must not infer"),
            tmp_path / "config.pt",
            tmp_path / "config.pt",
            event_labels=labels,
        )
    assert len(calls) == 1


@pytest.mark.parametrize("scenario", ["calibration_overlap", "donor_overlap"])
@pytest.mark.parametrize("cuda", [False, True], ids=["cpu", "cuda"])
def test_public_sdk_uses_effective_200_point_window_for_same_source_exclusion(tmp_path, monkeypatch, scenario, cuda):
    count, event = (220, 190) if scenario == "calibration_overlap" else (500, 390)
    positions = np.arange(count)
    frame = pd.DataFrame({"a": np.sin(positions / 9), "b": np.cos(positions / 13)})
    normal_intervals = [(0, count)] if scenario == "calibration_overlap" else [(0, 30), (230, 300)]
    resolver = NearestNormalWindowRetriever(
        frame, normal_intervals, [(0, 30)], {"normal_evidence": "normal fixture"}, same_source=True
    )
    sdk, calls, _ = _mock_public_sdk(
        tmp_path,
        monkeypatch,
        frame,
        flags=positions == event,
        dataset={"window_length": 200, "split": 4},
        cuda=cuda,
    )
    request = HistoricalAttributionRequest(
        resolver=resolver, event_position=event, target="a", validation_repeats=False
    )
    result = sdk.perform_anomaly_analysis_with_diffusion(
        frame, threshold_strategy="macs", historical_attribution=request
    ).attrs["historical_attribution"]
    assert result["native_settings"]["window_length"] == 200
    assert resolver.model_window_length == 100  # The caller's resolver was not mutated.
    expected = (
        "calibration_overlaps_evaluation_or_future"
        if scenario == "calibration_overlap"
        else "no_eligible_normal_windows"
    )
    assert result["reason"] == expected
    assert result["status"] in {"unavailable", "no_acceptable_replacement"}
    assert len(calls) == 1


@pytest.mark.parametrize("constant", [False, True], ids=["varying_input", "constant_input"])
def test_sdk_adapter_preserves_saved_scaling_schema_and_window_settings(tmp_path, constant):
    from sdk.feature_adapter import FeatureAdapter, transform_dataframe_from_metadata

    columns = ["temperature", "pressure"]
    adapter = FeatureAdapter(target_dim=4, scale_factor=7.0, seed=42)
    adapter.fit(np.array([[10.0, 100.0], [20.0, 120.0]], dtype=np.float32))
    metadata = adapter.metadata()
    metadata.update(columns=columns, window_length=200, split=10)
    checkpoint = tmp_path / "trained.pt"
    torch.save(
        {
            "config": {"model": {"target_dim": 4}, "dataset": {"scale_factor": 999.0}},
            "args": {"window_length": 100, "split": 4},
            "preprocessing": metadata,
        },
        checkpoint,
    )
    values = np.tile([100.0, 1000.0], (220, 1))
    if not constant:
        values += np.arange(220)[:, None] / 10
    features = pd.DataFrame(values, columns=columns)
    serving = features.assign(extra=42).loc[:, ["extra", "pressure", "temperature"]]
    calls = []

    def infer(frame):
        calls.append(frame.copy())
        target = transform_dataframe_from_metadata(frame, metadata).numpy()
        return {"target": target, "recon": target - [0.1, 0.3, 100.0, 100.0], "feature_names": columns}

    native = infer(serving)
    request = HistoricalAttributionRequest(
        resolver=AlignedNormalReference(features * 0.7, {"id": "normal"}, True), validation_repeats=False
    )
    result = run_sdk_attribution(
        serving, native, np.arange(220) == 190, np.ones(220), request, infer, checkpoint, checkpoint
    )
    assert result["status"] == "ok"
    assert result["feature_order"] == columns
    assert result["event"]["target"] == "pressure"
    assert result["native_settings"]["window_length"] == 200
    assert result["native_settings"]["split"] == 10
    assert result["native_settings"]["scale_factor"] == 7.0
    for call in calls[1:]:
        assert list(call.columns) == columns
    np.testing.assert_allclose(result["curves"]["before"], values[:, 1] - 0.3 / 7 * 20, rtol=1e-6)
    for condition in result["conditions"]:
        replacement = np.asarray(condition["replacement_input"])
        np.testing.assert_allclose(
            condition["reconstruction"], replacement - np.array([0.1, 0.3]) / 7 * [10, 20], rtol=1e-6
        )
    for audit in result["native_settings"]["preprocessing_audit"]:
        assert audit["scaling_source"] == "checkpoint_min_max"
        assert audit["external_min"] == [10, 100]
        assert audit["external_span"] == [10, 20]


def test_sdk_adapter_honors_legacy_checkpoint_window_overrides(tmp_path, monkeypatch):
    frame = pd.DataFrame({"a": np.sin(np.arange(220) / 9), "b": np.cos(np.arange(220) / 13)})
    sdk, calls, _ = _mock_public_sdk(tmp_path, monkeypatch, frame, flags=np.arange(220) == 190)
    checkpoint_path = tmp_path / "config.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["args"] = {"window_length": 200, "split": 8}
    torch.save(checkpoint, checkpoint_path)
    request = HistoricalAttributionRequest(
        resolver=NearestNormalWindowRetriever(
            frame, [(0, 220)], [(0, 30)], {"normal_evidence": "normal fixture"}, same_source=True
        ),
        target="a",
        validation_repeats=False,
    )
    result = sdk.perform_anomaly_analysis_with_diffusion(
        frame, threshold_strategy="macs", historical_attribution=request
    ).attrs["historical_attribution"]
    assert result["native_settings"]["window_length"] == 200
    assert result["native_settings"]["split"] == 8
    assert result["reason"] == "calibration_overlaps_evaluation_or_future"
    assert len(calls) == 1


def test_sdk_adapter_rejects_saved_pca_before_attribution_inference(tmp_path):
    from sdk.feature_adapter import FeatureAdapter

    adapter = FeatureAdapter(target_dim=2, scale_factor=1.0, seed=42)
    adapter.fit(np.arange(30, dtype=np.float32).reshape(10, 3))
    checkpoint = tmp_path / "pca.pt"
    torch.save({"config": {"model": {"target_dim": 2}}, "preprocessing": adapter.metadata()}, checkpoint)
    x, flags, request, _, _ = fixture()
    with pytest.raises(ValueError, match="direct feature mapping; no PCA attribution"):
        run_sdk_attribution(
            pd.DataFrame(x, columns=["a", "b", "c"]),
            {},
            flags,
            np.ones(60),
            request,
            lambda _: pytest.fail("PCA must not run attribution inference"),
            checkpoint,
            checkpoint,
        )


@pytest.mark.parametrize("distinct_source", [None, "run", "time"])
def test_donor_stability_deduplicates_reordered_nested_provenance(distinct_source):
    x, _, request, _, _ = fixture()
    normal = request.resolver.frame.to_numpy()
    first = {"id": "normal", "start": 20, "stop": 40, "metadata": {"unit": "kPa", "verified": True}}
    second = {"metadata": {"verified": True, "unit": "kPa"}, "stop": 40, "start": 20, "id": "normal"}
    if distinct_source == "run":
        second["id"] = "different run"
    if distinct_source == "time":
        second.update(start=40, stop=60)
    donors = [NormalDonor(normal[20:40].copy(), source, normal) for source in [first, second]]

    class Resolve:
        def resolve(self, *args: object):
            return ReferenceMatch("accepted", donors=donors)

    result = analyze_fixture(
        replace(
            request,
            resolver=Resolve(),
            replacement_strategy=MultiDonorReplacement(),
            validation_repeats=False,
        )
    )
    assert result["condition_count"] == 2  # Both requested conditions are saved, but repeats are deduplicated.
    stability = result["metrics"]["ExplanationStability"]
    count = 1 if distinct_source is None else 2
    assert stability["distinct_donor_count"] == count
    assert stability["alarm_effect"]["candidates"][0]["valid_repeats"] == count
    assert stability["status"] == ("warning" if count == 1 else "ok")
    assert report_data(result)["report_context"]["distinct_donor_count"] == count
    if count == 1:
        assert stability["reason"] == "insufficient_repeats"
