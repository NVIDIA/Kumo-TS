# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset-independent historical reconstruction attribution.

Callers supply observations, feature names/order, timestamps, units and a normal
reference resolver. This module provides retrieval, replacement, five metrics,
the existing SDK adapter and a two-page PDF appendix. No dataset is loaded or
selected here. Visible future context and native normalization remain unchanged.
"""

import hashlib
import json
from collections.abc import Hashable
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from dataclasses import field as dataclass_field
from itertools import combinations
from textwrap import fill
from typing import Callable, Protocol

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.textpath import TextToPath
from matplotlib.ticker import FuncFormatter, Locator

from .reporting import _add_footer, resolve_time_axis

__all__ = [
    "AlignedNormalReference",
    "CosineDistance",
    "CosineWindowRetriever",
    "EuclideanDistance",
    "HistoricalAttributionMetrics",
    "HistoricalAttributionRequest",
    "LinearBlendReplacement",
    "MetricsConfig",
    "MultiDonorReplacement",
    "NearestNormalWindowRetriever",
    "NormalDonor",
    "NormalizedMAEDistance",
    "ReferenceMatch",
    "ReplacementCondition",
    "ReplacementContext",
    "ReplacementStrategy",
    "SimilarityStrategy",
    "analyze_single_window",
    "historical_summary",
]


# Five metrics


def point_metrics(
    truth, before, after, scale, *, control_truth=None, control_before=None, control_after=None, epsilon=1e-8
):
    """Positive reduction means better reconstruction of the FIXED observation.

    HI percent denominator is the original fault residual; near-zero -> None.
    Control effects are subtracted in physical units, NOT percentages with
    different denominators. No normal reference is used as the target truth.
    """
    if scale <= 0 or not np.isfinite([truth, before, after, scale]).all():
        raise ValueError("Invalid score inputs")
    r0, r1 = abs(truth - before), abs(truth - after)
    improvement = r0 - r1
    control_values = [control_truth, control_before, control_after]
    has_control = all(v is not None for v in control_values)
    if any(v is not None for v in control_values) and not has_control:
        raise ValueError("Provide all control values together, or none")
    if has_control and not np.isfinite(control_values).all():
        raise ValueError("Control observations and predictions must be finite")
    control_effect = (
        None if not has_control else abs(control_truth - control_before) - abs(control_truth - control_after)
    )
    adjusted = None if not has_control else improvement - control_effect
    return dict(
        residual_before=r0,
        residual_after=r1,
        score_before=r0 / scale,
        score_after=r1 / scale,
        historical_impact_pct=100 * improvement / r0 if r0 > epsilon else None,
        target_reconstruction_improvement=improvement,
        reconstruction_change=after - before,
        control_effect=control_effect,
        control_adjusted_impact=adjusted,
        control_adjusted_score=None if adjusted is None else adjusted / scale,
    )


def positive_shares(effects, epsilon=1e-8):
    x = np.asarray(effects, float)
    if not np.isfinite(x).all():
        return [None] * len(x), None
    positive = np.maximum(x, 0)
    total = float(positive.sum())
    return ([None] * len(x) if total <= epsilon else (100 * positive / total).tolist()), total


def average_ranks(values):
    """Descending average ranks with exact ties; deterministic/no SciPy dependency."""
    x = np.asarray(values, float)
    return np.array([1 + np.count_nonzero(x > v) + (np.count_nonzero(x == v) - 1) / 2 for v in x])


def rank_correlation(x, y):
    a, b = average_ranks(x), average_ranks(y)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def effect_stability(effects, reference, epsilon=1e-8):
    a = np.asarray(effects, float)

    def sign(v):
        return np.where(v > epsilon, 1, np.where(v < -epsilon, -1, 0))

    return dict(
        n=int(a.size),
        median=float(np.median(a)),
        minimum=float(a.min()),
        maximum=float(a.max()),
        q10=float(np.quantile(a, 0.1)),
        q90=float(np.quantile(a, 0.9)),
        positive_fraction=float((a > epsilon).mean()),
        sign_agreement=float((sign(a) == sign(reference)).mean()),
    )


@dataclass(frozen=True)
class MetricsConfig:
    version: str = "2.0.0"
    numerical_tolerance: float = 0.002
    small_residual_warning: float = 1.0
    practical_effect_threshold: float = 1.0
    minimum_rank_candidates: int = 3

    def __post_init__(self):
        for v in [self.numerical_tolerance, self.small_residual_warning, self.practical_effect_threshold]:
            if not np.isfinite(v) or v < 0:
                raise ValueError("Tolerances must be finite and nonnegative")
        if self.minimum_rank_candidates < 2:
            raise ValueError("At least two candidates required")
        if self.version != "2.0.0":
            raise ValueError("Unsupported metrics version")


def finite(v):
    try:
        return v is not None and np.isscalar(v) and bool(np.isfinite(v))
    except TypeError:
        return False


def state(status="ok", reason=None):
    return dict(status=status, reason=reason)


class HistoricalAttributionMetrics:
    """Compute HI/TRI/CAI, stability and positive CAI shares from saved outputs.

    Effects use original fault truth. H is signed displacement, not CAI.
    Quantiles and deterministic reruns are not statistical replicates.
    """

    FIELDS = ("rF0", "rFb", "rN0", "rNb", "HI", "TRI_F", "TRI_N", "CAI", "delta_F", "delta_N", "H", "normal_approach")

    def __init__(self, config=None):
        self.config = config if isinstance(config, MetricsConfig) else MetricsConfig(**(config or {}))

    def config_dict(self):
        return asdict(self.config)

    def _hi(self, before, improvement):
        if not finite(before) or not finite(improvement):
            return None, state("unavailable", "missing_fault_residual")
        if before == 0:
            return None, state("unavailable", "zero_baseline_residual")
        v = 100 * improvement / before
        if not np.isfinite(v):
            return None, state("unavailable", "percentage_overflow")
        return float(v), state(
            "warning", "small_baseline_residual"
        ) if before < self.config.small_residual_warning else state()

    @staticmethod
    def _series(v, n):
        if v is None:
            return [None] * n
        a = np.asarray(v, dtype=object)
        if a.ndim == 0:
            return [a.item()] * n
        if a.ndim != 1 or len(a) != n:
            raise ValueError("Every series must be one-dimensional and time-aligned")
        return a.tolist()

    def compute_effects(
        self,
        fault_truth,
        fault_before,
        fault_after,
        *,
        normal_truth=None,
        normal_before=None,
        normal_after=None,
        forecast_times=None,
        metadata=None,
    ):
        """Return one explicit value/status per future time, including missing data."""
        series = [fault_truth, fault_before, fault_after, normal_truth, normal_before, normal_after]
        lengths = [len(np.asarray(v)) for v in series if v is not None and np.asarray(v).ndim > 0]
        n = lengths[0] if lengths else 1
        try:
            values = [self._series(v, n) for v in series]
        except (ValueError, TypeError) as e:
            return dict(metrics_version=self.config.version, points=[], **state("invalid_input", str(e)))
        if forecast_times is None:
            forecast_times = list(range(n))
        if len(forecast_times) != n:
            return dict(
                metrics_version=self.config.version, points=[], **state("invalid_input", "time_labels_length_mismatch")
            )
        points = []
        for i, vals in enumerate(zip(*values, strict=True)):
            yf, f0, fb, yn, n0, nb = vals
            row = dict(metadata or {}, forecast_time=forecast_times[i], metrics_version=self.config.version)
            row.update(dict.fromkeys(self.FIELDS))
            statuses = {k: state("unavailable", "missing_or_nonfinite_input") for k in self.FIELDS}
            if all(finite(v) for v in [yf, f0, fb]):
                control = all(finite(v) for v in [yn, n0, nb])
                p = point_metrics(
                    yf,
                    f0,
                    fb,
                    1.0,
                    control_truth=yn if control else None,
                    control_before=n0 if control else None,
                    control_after=nb if control else None,
                    epsilon=0.0,
                )
                for field, key in [
                    ("rF0", "residual_before"),
                    ("rFb", "residual_after"),
                    ("TRI_F", "target_reconstruction_improvement"),
                    ("delta_F", "reconstruction_change"),
                ]:
                    row[field] = float(p[key])
                    statuses[field] = state()
                row["HI"], statuses["HI"] = self._hi(row["rF0"], row["TRI_F"])
                if control:
                    row["TRI_N"] = float(p["control_effect"])
                    row["CAI"] = float(p["control_adjusted_impact"])
                    statuses["TRI_N"] = state()
                    statuses["CAI"] = state()
                else:
                    statuses["CAI"] = state("unavailable", "missing_or_incomplete_normal_control")
            if all(finite(v) for v in [yn, n0, nb]):
                row["rN0"] = float(abs(n0 - yn))
                row["rNb"] = float(abs(nb - yn))
                row["TRI_N"] = row["rN0"] - row["rNb"]
                for k in ["rN0", "rNb", "TRI_N"]:
                    statuses[k] = state()
            if all(finite(v) for v in [f0, fb]):
                row["delta_F"] = float(fb - f0)
                statuses["delta_F"] = state()
            if all(finite(v) for v in [n0, nb]):
                row["delta_N"] = float(nb - n0)
                statuses["delta_N"] = state()
            if finite(row["delta_F"]) and finite(row["delta_N"]):
                row["H"] = row["delta_F"] - row["delta_N"]
                statuses["H"] = state()
            if all(finite(v) for v in [f0, fb, yn]):
                row["normal_approach"] = float(abs(f0 - yn) - abs(fb - yn))
                statuses["normal_approach"] = state()
            row["statuses"] = statuses
            row["status"] = (
                "ok"
                if all(v["status"] == "ok" for v in statuses.values())
                else "partial"
                if any(v["status"] == "unavailable" for v in statuses.values())
                else "warning"
            )
            row["reason"] = (
                None if row["status"] == "ok" else sorted({v["reason"] for v in statuses.values() if v["reason"]})
            )
            row["interpretation"] = self.interpret(row["TRI_F"], row["CAI"])
            row["CAI_exceeds_practical_threshold"] = (
                None if row["CAI"] is None else abs(row["CAI"]) >= self.config.practical_effect_threshold
            )
            points.append(row)
        return dict(metrics_version=self.config.version, points=points, **state())

    @staticmethod
    def interpret(tri, cai):
        if not finite(tri) or not finite(cai):
            return "not_evaluable"
        if cai > 0 and tri <= 0:
            return "less_deterioration_than_normal_control"
        if cai > 0:
            return "fault_error_improves_beyond_normal_control"
        return "no_positive_control_adjusted_improvement"

    def summarize_horizons(self, effects, horizons=(1, 5, 10, 20)):
        """Average per-time errors first; HI=100*mean(TRI_F)/mean(rF0)."""
        points = effects["points"] if isinstance(effects, dict) else effects
        results = []
        for h in horizons:
            if not isinstance(h, int) or h < 1:
                raise ValueError("Horizons must be positive integers")
            row = dict(points=h, metrics_version=self.config.version, statuses={})
            for field in self.FIELDS:
                if field == "HI":
                    continue
                vals = [p.get(field) for p in points[:h]]
                valid = len(vals) == h and all(finite(v) for v in vals)
                row[field] = float(np.mean(vals)) if valid else None
                row["statuses"][field] = state() if valid else state("unavailable", "incomplete_horizon")
            row["HI"], row["statuses"]["HI"] = self._hi(row["rF0"], row["TRI_F"])
            row["interpretation"] = self.interpret(row["TRI_F"], row["CAI"])
            row["forecast_start"] = points[0]["forecast_time"] if points else None
            row["forecast_end"] = points[h - 1]["forecast_time"] if len(points) >= h else None
            row["status"] = (
                "ok"
                if all(s["status"] == "ok" for s in row["statuses"].values())
                else "partial"
                if any(s["status"] == "unavailable" for s in row["statuses"].values())
                else "warning"
            )
            row["reason"] = (
                None
                if row["status"] == "ok"
                else sorted({s["reason"] for s in row["statuses"].values() if s["reason"]})
            )
            row["HI_aggregation"] = (
                "ratio of mean pointwise residual improvement to mean pointwise baseline residual; not mean of pointwise percentages"
            )
            results.append(row)
        return results

    @staticmethod
    def _group(records, group, expected_candidates):
        required = ["case_id", "run_id", "target", "unit", "replacement_source", "forecast_time"]
        kind = group.get("kind")
        if kind == "historical_blocks":
            required += ["scope"]
        elif kind == "signal_groups":
            required += ["block_start", "block_end", "g", "L"]
        else:
            raise ValueError("Explicit historical_blocks or signal_groups group required")
        if any(k not in group for k in required):
            raise ValueError("Incomplete group context")
        if len(set(expected_candidates)) != len(expected_candidates):
            raise ValueError("Duplicate expected candidate")
        required = list(dict.fromkeys(required + [k for k in group if k != "kind"]))
        by = {}
        for r in records:
            if any(r.get(k) != group[k] for k in required):
                raise ValueError("Mixed case/target/unit/source/time or candidate group")
            cid = r["candidate_id"]
            if cid not in expected_candidates or cid in by:
                raise ValueError("Unexpected or duplicate candidate")
            by[cid] = r
        return by

    def rank_candidates(self, records, *, group, expected_candidates, value_key="CAI", absolute=False):
        by = self._group(records, group, expected_candidates)
        ids = [i for i in expected_candidates if i in by and finite(by[i].get(value_key))]
        values = np.array([by[i][value_key] for i in ids], float)
        scores = np.abs(values) if absolute else values
        ranks = average_ranks(scores) if ids else []
        reason = (
            "missing_candidates"
            if len(ids) < len(expected_candidates)
            else "no_candidates"
            if not ids
            else "all_near_zero"
            if np.max(np.abs(values)) <= self.config.numerical_tolerance
            else "constant_scores"
            if np.ptp(scores) == 0
            else None
        )
        positions = sorted(range(len(ids)), key=lambda j: (-scores[j], expected_candidates.index(ids[j])))
        order = [ids[j] for j in positions]
        result = []
        for cid in expected_candidates:
            j = ids.index(cid) if cid in ids else None
            result.append(
                dict(
                    candidate_id=cid,
                    value=None if j is None else float(values[j]),
                    rank=None if j is None else float(ranks[j]),
                    status="unavailable" if j is None else "warning" if reason else "ok",
                    reason="missing_effect" if j is None else reason,
                )
            )
        top = {}
        for k in [3, 5]:
            if len(ids) < k:
                top[str(k)] = dict(members=None, **state("unavailable", "insufficient_candidates"))
            else:
                cutoff = scores[positions[k - 1]]
                tie = bool(len(ids) > k and scores[positions[k]] == cutoff)
                top[str(k)] = dict(
                    members=order[:k],
                    cutoff_tie=tie,
                    **state("warning" if tie or reason else "ok", "cutoff_tie" if tie else reason),
                )
        return dict(
            group=group,
            value_key=value_key,
            absolute=absolute,
            expected_candidates=list(expected_candidates),
            valid_count=len(ids),
            missing_count=len(expected_candidates) - len(ids),
            order=order,
            rows=result,
            top=top,
            **state("warning" if reason else "ok", reason),
        )

    def compute_contribution_share(self, records, *, group, expected_candidates):
        """Normalize complete-group positive CAI, never only top-k or partial data."""
        by = self._group(records, group, expected_candidates)
        missing = [i for i in expected_candidates if i not in by or not finite(by[i].get("CAI"))]
        total = None
        values = [None] * len(expected_candidates)
        if missing:
            status, reason = "unavailable", "incomplete_candidate_set"
        else:
            effects = [by[i]["CAI"] for i in expected_candidates]
            values, total = positive_shares(effects, epsilon=0.0)
            status, reason = (
                ("unavailable", "no_positive_control_adjusted_effect")
                if total == 0
                else ("warning", "small_positive_total")
                if total < self.config.practical_effect_threshold
                else ("ok", None)
            )
        return dict(
            group=group,
            expected_candidates=list(expected_candidates),
            positive_total=total,
            missing_candidates=missing,
            aggregation="positive CAI normalized across complete group; overlapping candidates are not additive causes",
            rows=[
                dict(
                    candidate_id=i,
                    share_pct=v,
                    CAI=by.get(i, {}).get("CAI"),
                    TRI_F=by.get(i, {}).get("TRI_F"),
                    TRI_N=by.get(i, {}).get("TRI_N"),
                    **state(status, reason),
                )
                for i, v in zip(expected_candidates, values, strict=True)
            ],
            **state(status, reason),
        )

    def compute_stability(
        self,
        replicates,
        *,
        axis,
        expected_candidates,
        reference_id=None,
        applicable=True,
        reason=None,
        rank_absolute=False,
    ):
        """Replicates are externally computed paired effects: {replicate:{id:value}}.

        No model calls. Ranking comparisons explicitly retain common candidates;
        deterministic top-k breaks exact ties in frozen candidate order.
        """
        if not applicable:
            return dict(axis=axis, replicates=[], candidates=[], rank_comparisons=[], **state("not_applicable", reason))
        expected = list(expected_candidates)
        if len(set(expected)) != len(expected):
            raise ValueError("Duplicate expected candidates")
        if not replicates:
            return dict(
                axis=axis, replicates=[], candidates=[], rank_comparisons=[], **state("unavailable", "no_replicates")
            )
        if any(set(values) - set(expected) for values in replicates.values()):
            raise ValueError("Unexpected comparison candidate")
        reference_id = reference_id or next(iter(replicates))
        if reference_id not in replicates:
            raise ValueError("Missing reference replicate")
        common = [c for c in expected if all(finite(v.get(c)) for v in replicates.values())]
        rank_maps = (
            {
                r: dict(zip(common, average_ranks([abs(v[c]) if rank_absolute else v[c] for c in common]), strict=True))
                for r, v in replicates.items()
            }
            if common
            else {}
        )
        summaries = []
        for cid in expected:
            values = {r: float(v[cid]) if finite(v.get(cid)) else None for r, v in replicates.items()}
            valid = [v for v in values.values() if v is not None]
            ref = values[reference_id]
            if valid:
                # Reuse legacy distribution calculation without changing its API.
                stat = effect_stability(
                    valid, ref if ref is not None else float(np.median(valid)), epsilon=self.config.numerical_tolerance
                )
                stat["sign_agreement"] = stat["sign_agreement"] if ref is not None else None
                pos = float(np.mean(np.array(valid) > self.config.numerical_tolerance))
                neg = float(np.mean(np.array(valid) < -self.config.numerical_tolerance))
                zero = 1 - pos - neg
                stat.update(
                    negative_fraction=neg,
                    near_zero_fraction=float(max(0, zero)),
                    modal_sign_agreement=max(pos, neg, zero),
                )
                reason0 = (
                    "insufficient_repeats"
                    if len(valid) < 2
                    else "all_near_zero"
                    if max(abs(v) for v in valid) <= self.config.numerical_tolerance
                    else None
                )
            else:
                stat = dict.fromkeys(
                    [
                        "median",
                        "minimum",
                        "maximum",
                        "q10",
                        "q90",
                        "positive_fraction",
                        "negative_fraction",
                        "near_zero_fraction",
                        "sign_agreement",
                        "modal_sign_agreement",
                    ]
                )
                reason0 = "no_valid_repeats"
            if reason0 is None and len(valid) < len(values):
                reason0 = "missing_replicates"
            ranks = {r: float(rm[cid]) if cid in rm else None for r, rm in rank_maps.items()}
            rv = [v for v in ranks.values() if v is not None]
            summaries.append(
                dict(
                    candidate_id=cid,
                    paired_effects=values,
                    paired_effect_statuses={
                        r: state() if v is not None else state("unavailable", "missing_replicate_effect")
                        for r, v in values.items()
                    },
                    valid_repeats=len(valid),
                    missing_repeats=len(values) - len(valid),
                    ranks_on_common_set=ranks,
                    rank_min=min(rv) if rv else None,
                    rank_max=max(rv) if rv else None,
                    **stat,
                    **state("warning" if reason0 else "ok", reason0),
                )
            )
        pairs = []
        for a, b in combinations(replicates, 2):
            ids = [c for c in expected if finite(replicates[a].get(c)) and finite(replicates[b].get(c))]
            av = np.array([replicates[a][c] for c in ids])
            bv = np.array([replicates[b][c] for c in ids])
            if rank_absolute:
                av, bv = np.abs(av), np.abs(bv)
            if len(ids) < self.config.minimum_rank_candidates:
                corr = None
                st = state("unavailable", "insufficient_candidates")
            elif (
                np.max(np.abs(av)) <= self.config.numerical_tolerance
                or np.max(np.abs(bv)) <= self.config.numerical_tolerance
            ):
                corr = None
                st = state("unavailable", "all_near_zero_ranking")
            else:
                corr = rank_correlation(av, bv)
                st = state() if corr is not None else state("unavailable", "constant_ranking")
            tops = {}
            for k in [3, 5]:
                if len(ids) < k:
                    tops[str(k)] = dict(overlap=None, fraction=None, **state("unavailable", "insufficient_candidates"))
                    continue
                ia = sorted(range(len(ids)), key=lambda j: (-av[j], expected.index(ids[j])))
                ib = sorted(range(len(ids)), key=lambda j: (-bv[j], expected.index(ids[j])))
                sa = [ids[j] for j in ia[:k]]
                sb = [ids[j] for j in ib[:k]]
                ties = bool(len(ids) > k and (av[ia[k - 1]] == av[ia[k]] or bv[ib[k - 1]] == bv[ib[k]]))
                overlap = len(set(sa) & set(sb))
                tops[str(k)] = dict(
                    a=sa,
                    b=sb,
                    overlap=overlap,
                    fraction=overlap / k,
                    cutoff_tie=ties,
                    **state(
                        "warning" if ties or st["status"] != "ok" else "ok", "cutoff_tie" if ties else st["reason"]
                    ),
                )
            pairs.append(
                dict(
                    a=a,
                    b=b,
                    common_candidates=ids,
                    excluded_candidates=[c for c in expected if c not in ids],
                    spearman=corr,
                    top=tops,
                    **st,
                )
            )
        return dict(
            axis=axis,
            rank_absolute=rank_absolute,
            replicates=list(replicates),
            reference_id=reference_id,
            expected_candidates=expected,
            rank_common_candidates=common,
            candidates=summaries,
            rank_comparisons=pairs,
            **state(
                "ok" if len(replicates) >= 2 else "warning", None if len(replicates) >= 2 else "insufficient_repeats"
            ),
        )


# Normal references and retrieval


def input_hash(values):
    a = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


class SimilarityStrategy(Protocol):
    """Compare already aligned, normal-calibrated arrays; never select donors.

    direction is "minimize" or "maximize". A nonfinite score (or None)
    means the comparison is undefined and is excluded, never admitted as zero.
    Optional tie_breakers name retrieval diagnostics used before source
    position; acceptance is always the retriever's independent rule.
    """

    name: str
    direction: str

    def compare(self, query: np.ndarray, candidate: np.ndarray) -> float | None: ...


@dataclass(frozen=True)
class NormalizedMAEDistance:
    name: str = "normalized_mae"
    direction: str = "minimize"
    definition: str = "mean absolute difference in normal Q95-Q05 units"

    def compare(self, query, candidate):
        return float(np.mean(np.mean(np.abs(query - candidate), axis=0)))


@dataclass(frozen=True)
class CosineDistance:
    name: str = "cosine_distance"
    direction: str = "minimize"
    definition: str = "one minus cosine of median-centered, normal Q95-Q05 scaled context"
    tie_breakers: tuple[str, ...] = ("normalized_difference",)
    invalid_query_reason: str = "undefined_cosine_zero_query"

    def validate_query(self, query):
        return np.linalg.norm(query) > 0

    def compare(self, query, candidate):
        q, c = query.ravel(), candidate.ravel()
        denominator = np.linalg.norm(q) * np.linalg.norm(c)
        if denominator == 0:
            return None
        return float(1 - np.clip(np.dot(q, c) / denominator, -1, 1))


@dataclass(frozen=True)
class EuclideanDistance:
    name: str = "euclidean_distance"
    direction: str = "minimize"
    definition: str = "L2 norm of differences in normal Q95-Q05 units"

    def compare(self, query, candidate):
        return float(np.linalg.norm(query - candidate))


# Only legacy string configuration uses this registry; analysis never branches
# on a method name. New comparators can be injected without editing this mapping.
_LEGACY_SIMILARITIES = {
    "normalized_mae": NormalizedMAEDistance,
    "cosine_distance": CosineDistance,
    "euclidean_distance": EuclideanDistance,
}


@dataclass
class NormalDonor:
    """One admitted, intact multivariate segment; control must be aligned."""

    values: np.ndarray
    source: dict
    control: np.ndarray | None = None
    accepted: bool = True


@dataclass
class ReferenceMatch:
    status: str
    values: np.ndarray | None = None
    source: dict = dataclass_field(default_factory=dict)
    candidates: list = dataclass_field(default_factory=list)
    control: np.ndarray | None = None
    reason: str | None = None
    donors: list[NormalDonor] = dataclass_field(default_factory=list)


class ReferenceResolver(Protocol):
    def resolve(self, values, columns, start, stop, visible, event): ...


@dataclass
class AlignedNormalReference:
    frame: pd.DataFrame
    source: dict
    alignment_verified: bool = False

    def resolve(self, values, columns, start, stop, visible, event):
        if not self.alignment_verified:
            return ReferenceMatch("unavailable", reason="baseline_alignment_not_verified")
        if list(self.frame.columns) != list(columns) or self.frame.shape != values.shape:
            raise ValueError("Normal reference feature order and time alignment must match the input")
        normal = self.frame.to_numpy(dtype=float)
        if not np.isfinite(normal).all():
            raise ValueError("Normal reference contains nonfinite values")
        return ReferenceMatch("accepted", normal[start:stop].copy(), dict(self.source), control=normal)


@dataclass
class CosineWindowRetriever:
    """Use declared normal intervals only, with an independently calibrated gate.

    Intervals are half-open source positions, never inferred from replacement effects.
    For same-source retrieval, all donors precede the alarm's model window; the
    model window and calibration rows are excluded from the retrieval bank.
    Difference <= .05 means each query feature's mean absolute difference is no
    more than 5% of its normal Q95-Q05 range. This is not cosine >= .95.
    """

    frame: pd.DataFrame
    normal_intervals: list
    calibration_intervals: list
    source: dict
    same_source: bool = True
    model_window_length: int = 100
    context_points: int = 8
    max_difference: float | None = 0.05
    top_n: int = 5
    ranking_metric: str = "cosine_distance"
    similarity: SimilarityStrategy | None = None

    def resolve(self, values, columns, start, stop, visible, event):
        if list(self.frame.columns) != list(columns):
            raise ValueError("Donor feature order differs from evaluation data")
        if not self.source.get("normal_evidence"):
            return ReferenceMatch("unavailable", reason="normal_bank_has_no_normality_evidence")
        if isinstance(self.top_n, bool | np.bool_) or not isinstance(self.top_n, int | np.integer) or self.top_n < 1:
            raise ValueError("top_n must be a positive integer")
        if (self.max_difference is not None and not 0 < self.max_difference < 1) or self.context_points < 2:
            raise ValueError("Invalid retrieval settings")
        strategy = self.similarity
        if strategy is None:
            try:
                strategy = _LEGACY_SIMILARITIES[self.ranking_metric]()
            except KeyError as exc:
                raise ValueError("Unknown retrieval distance") from exc
        if strategy.direction not in {"minimize", "maximize"} or not strategy.name:
            raise ValueError("Similarity requires a name and explicit minimize/maximize direction")
        values = np.asarray(values, float)
        visible = np.asarray(visible, bool)
        if (
            values.ndim != 2
            or values.shape[1] != len(columns)
            or visible.shape != (len(values),)
            or len(set(columns)) != len(columns)
            or not 0 <= start < stop <= event < len(values)
            or self.model_window_length < 1
        ):
            raise ValueError("Query features, visibility and historical window must align")
        bank = self.frame.to_numpy(dtype=float)
        normal = np.zeros(len(bank), dtype=bool)
        calibration = np.zeros(len(bank), dtype=bool)
        for mask, intervals in [(normal, self.normal_intervals), (calibration, self.calibration_intervals)]:
            for a, b in intervals:
                if not 0 <= a < b <= len(bank):
                    raise ValueError("Invalid declared normal interval")
                mask[a:b] = True
        if np.any(calibration & ~normal) or calibration.sum() < 10:
            return ReferenceMatch("unavailable", reason="insufficient_separate_normal_calibration")
        exclusion_start = (min(start, event) // self.model_window_length) * self.model_window_length
        if self.same_source:
            if np.any(calibration[exclusion_start:]):
                return ReferenceMatch("unavailable", reason="calibration_overlaps_evaluation_or_future")
            normal[exclusion_start:] = False
        allowed = normal & ~calibration
        cal = bank[calibration]
        if not np.isfinite(cal).all():
            return ReferenceMatch("unavailable", reason="nonfinite_normal_calibration")
        location = np.median(cal, axis=0)
        scale = np.quantile(cal, 0.95, axis=0) - np.quantile(cal, 0.05, axis=0)
        if np.any(scale <= 1e-12):
            return ReferenceMatch("unavailable", reason="constant_feature_needs_explicit_resolution_scale")
        indices = np.flatnonzero(visible[:start])[-self.context_points :]
        if len(indices) < self.context_points:
            return ReferenceMatch("unavailable", reason="insufficient_visible_pre_replacement_context")
        offsets = indices - start
        q = values[indices]
        if not np.isfinite(q).all():
            return ReferenceMatch("unavailable", reason="nonfinite_visible_query")
        qz = (q - location) / scale
        validate_query = getattr(strategy, "validate_query", lambda _query: True)
        if not validate_query(qz):
            return ReferenceMatch(
                "unavailable", reason=getattr(strategy, "invalid_query_reason", "undefined_similarity_query")
            )
        rows = []
        legal_count, undefined_count = 0, 0
        length = stop - start
        for ds in range(-int(offsets.min()), len(bank) - length + 1):
            used = np.r_[ds + offsets, np.arange(ds, ds + length)]
            if not allowed[used].all():
                continue
            context = bank[ds + offsets]
            segment = bank[ds : ds + length]
            if not np.isfinite(context).all() or not np.isfinite(segment).all():
                continue
            legal_count += 1
            dz = (context - location) / scale
            score = strategy.compare(qz, dz)
            if not finite(score):
                undefined_count += 1
                continue
            distance = CosineDistance().compare(qz, dz)
            per_feature = np.mean(np.abs(q - context) / scale, axis=0)
            difference = float(per_feature.max())
            rows.append(
                dict(
                    start=ds,
                    stop=ds + length,
                    cosine_distance=distance,
                    normalized_difference=difference,
                    normalized_mae=float(per_feature.mean()),
                    score=float(score),
                    method=strategy.name,
                    score_direction=strategy.direction,
                    accepted=self.max_difference is None or difference <= self.max_difference,
                    segment_hash=input_hash(segment),
                )
            )
        direction = 1 if strategy.direction == "minimize" else -1
        tie_breakers = getattr(strategy, "tie_breakers", ())
        rows.sort(key=lambda r: (direction * r["score"], *(r[k] for k in tie_breakers), r["start"]))
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
        eligible = [r for r in rows if r["accepted"]]
        details = dict(
            self.source,
            bank_count=len(rows),
            legal_window_count=legal_count,
            undefined_similarity_count=undefined_count,
            accepted_count=len(eligible),
            requested_donor_count=int(self.top_n),
            returned_donor_count=min(len(eligible), int(self.top_n)),
            acceptance_threshold=self.max_difference,
            acceptance_definition="max_feature_mean_absolute_difference / normal_Q95_Q05"
            if self.max_difference is not None
            else "eligible_normal_history_only_no_distance_cutoff",
            ranking_metric=strategy.name,
            similarity_method=strategy.name,
            score_direction=strategy.direction,
            distance_definition=getattr(strategy, "definition", "caller-supplied aligned-array comparison"),
            scale_definition="per-feature normal median location and normal Q95-Q05 span",
            search_step=1,
            context_points=self.context_points,
            reference_type="retrieved_normal_history_not_paired_baseline",
            calibration_intervals=self.calibration_intervals,
            calibration_location=location.tolist(),
            calibration_scale=scale.tolist(),
            query_positions=indices.tolist(),
            query_hash=input_hash(q),
        )
        if not eligible:
            return ReferenceMatch(
                "no_acceptable_replacement",
                source=details,
                candidates=rows[: self.top_n],
                reason=(
                    "no_eligible_normal_windows"
                    if not legal_count
                    else "no_defined_similarity_scores"
                    if not rows
                    else "no_donor_passed_normal_range_gate"
                ),
            )
        best = eligible[0]
        details.update(best)
        details["rank_among_all"] = rows.index(best) + 1
        details["donor_context_positions"] = (best["start"] + offsets).tolist()
        details["selected_score"] = best["score"]
        details["selected_distance"] = best["score"] if strategy.direction == "minimize" else None
        details["not_a_verified_matched_baseline"] = True
        donors = []
        for row in eligible[: self.top_n]:
            source = dict(details, **row)
            source["rank_among_all"] = row["rank"]
            source["donor_context_positions"] = (row["start"] + offsets).tolist()
            source["selected_score"] = row["score"]
            source["selected_distance"] = row["score"] if strategy.direction == "minimize" else None
            if isinstance(self.frame.index, pd.DatetimeIndex):
                source["start_time"] = str(self.frame.index[row["start"]])
                source["end_time"] = str(self.frame.index[row["stop"] - 1])
            donors.append(NormalDonor(bank[row["start"] : row["stop"]].copy(), source))
        return ReferenceMatch("accepted", donors[0].values.copy(), details, rows[: self.top_n], donors=donors)


@dataclass
class NearestNormalWindowRetriever(CosineWindowRetriever):
    """Pick the minimum normalized MAE from eligible normal history, without a gate.

    The normal Q95-Q05 span sets each feature's scale. Mean absolute differences
    are averaged over visible context positions and features; ties use the
    earliest donor start. All channels share one intact donor segment. Ranking
    uses only the pre-replacement visible context, never replacement effects.
    A nearest donor is not a verified paired normal baseline. The legacy cosine
    resolver remains available for reproducing older experiments.
    """

    max_difference: float | None = None
    ranking_metric: str = "normalized_mae"


# Replacement and SDK adapter


@dataclass
class HistoricalAttributionRequest:
    """Technical settings stay in code/config, not reader-facing PDF labels.

    gap=1 and length=20 select [alarm-20, alarm-1], inclusive. This is ONE
    intervention, not a sliding scan: twenty outputs are reconstruction responses
    at twenty positions, not twenty independent causes of the alarm.
    Blending is confined to the replacement block; edge donor weights are 1/3,
    2/3, then 1, with the symmetric taper at the other end. No outside endpoints
    are used. The native SDK conditioning and sampler are intentionally unchanged.

    signals optionally selects named features in original model order; otherwise
    scope selects target or all. replacement_strategy defaults to the existing
    single-donor blend. MultiDonorReplacement executes every admitted donor
    separately. The returned conditions each retain inputs, reconstructions and
    five metrics; legacy top-level fields describe the first condition.
    """

    resolver: ReferenceResolver | None = None
    event_position: int | None = None
    target: Hashable | None = None
    scope: str = "target"
    length: int = 20
    gap: int = 1
    blend_points: int = 2
    case_name: str = "SDK anomaly event"
    units: dict = dataclass_field(default_factory=dict)
    validation_repeats: bool = True
    event_source: str = "sdk"
    signals: list[Hashable] | None = None
    replacement_strategy: "ReplacementStrategy | None" = None


def _point_metrics(y, before, after, unit):
    cfg = MetricsConfig(numerical_tolerance=1e-8, small_residual_warning=1e-8, practical_effect_threshold=0.0)
    return HistoricalAttributionMetrics(cfg).compute_effects(y, before, after, metadata={"unit": unit})["points"][0]


def blended_replacement(values, segment, start, stop, channels, blend_points):
    if segment.shape != (stop - start, values.shape[1]) or not np.isfinite(segment).all():
        raise ValueError("Donor must be a finite coherent multivariate segment")
    if blend_points < 0 or 2 * blend_points > stop - start:
        raise ValueError("Invalid edge blending length")
    k = np.arange(stop - start)
    weights = np.minimum(1.0, np.minimum((k + 1) / (blend_points + 1), (stop - start - k) / (blend_points + 1)))
    result = values.copy()
    ix = np.ix_(np.arange(start, stop), channels)
    mixed = (1 - weights[:, None]) * values[ix] + weights[:, None] * segment[:, channels]
    result[ix] = np.where(values[ix] == segment[:, channels], values[ix], mixed)
    return result, weights


@dataclass(frozen=True)
class ReplacementContext:
    """Fixed event/window and read-only original input, in model feature order.

    visible is the native strategy mask. Native raw arrays retain hidden
    observations for backward compatibility; they are not retrieval context.
    Rules must not use hidden observations to choose donors, weights, endpoints
    or visible replacement values. Full raw and masked conditioning hashes are
    recorded separately: the native SDK's full-array scaling is unchanged.
    """

    values: np.ndarray
    columns: tuple[Hashable, ...]
    visible: np.ndarray
    event: int
    start: int
    stop: int
    channels: tuple[int, ...]
    time_labels: tuple[str, ...]
    reference: ReferenceMatch
    blend_points: int = 2


@dataclass
class ReplacementCondition:
    """An independent input, plus provenance and the same-rule normal control."""

    condition_id: str
    values: np.ndarray
    strategy: str
    parameters: dict
    donor: NormalDonor
    weights: np.ndarray | None = None
    control_values: np.ndarray | None = None


class ReplacementStrategy(Protocol):
    """Injectable RM object: produce conditions, leaving inference to the SDK.

    Pass an implementation directly for its normal execution, or compose
    MultiDonorReplacement(rule=implementation) to apply that same RM separately
    to every admitted donor. Donor count and the replacement rule are independent.
    """

    def generate(self, context: ReplacementContext) -> list[ReplacementCondition]: ...


def _accepted_donors(reference):
    if reference.status != "accepted":
        return []
    donors = reference.donors
    if not donors and reference.values is not None:
        donors = [NormalDonor(reference.values, reference.source, reference.control)]
    return [
        replace(d, control=d.control if d.control is not None else reference.control)
        for d in donors
        if d.accepted and d.source.get("accepted", True)
    ]


@dataclass(frozen=True)
class LinearBlendReplacement:
    """Legacy within-block taper; zero blend points directly inserts the donor.

    No neighboring truth or interpolation endpoint is read. Hidden block cells
    retain the legacy display/raw-array arithmetic, but cannot influence any
    visible replacement value. The SDK adapter still owns native masking/scaling.
    """

    blend_points: int | None = None

    def generate(self, context):
        donors = _accepted_donors(context.reference)
        if not donors:
            return []
        donor = donors[0]
        edge = context.blend_points if self.blend_points is None else self.blend_points
        values, weights = blended_replacement(
            context.values, donor.values, context.start, context.stop, context.channels, edge
        )
        control_values = None
        if donor.control is not None:
            control_values, _ = blended_replacement(
                donor.control, donor.values, context.start, context.stop, context.channels, edge
            )
        return [
            ReplacementCondition(
                "donor_1",
                values,
                "linear_blending",
                {"blend_points": edge},
                donor,
                weights,
                control_values,
            )
        ]


@dataclass(frozen=True)
class MultiDonorReplacement:
    """Existing multiple-donor meaning: apply one rule separately to each donor.

    This is not donor averaging. Top-N search metadata alone does not execute
    this strategy. Each rule receives the SAME original input and one admitted
    intact donor. Custom rules may themselves return several conditions.
    The rule argument is an injectable RM object, not a method-name selector:
    MultiDonorReplacement(rule=MyReplacementMethod()) reuses that method without
    changing analysis, retrieval or the inference callback.
    """

    rule: ReplacementStrategy = dataclass_field(default_factory=LinearBlendReplacement)

    def generate(self, context):
        conditions = []
        for index, donor in enumerate(_accepted_donors(context.reference), 1):
            reference = ReferenceMatch("accepted", donor.values, donor.source, control=donor.control, donors=[donor])
            donor_context = replace(context, reference=reference)
            for inner in self.rule.generate(donor_context):
                _validate_condition(inner, donor_context)
                conditions.append(
                    replace(
                        inner,
                        condition_id=f"donor_{index}:{inner.condition_id}",
                        strategy="multi_donor",
                        parameters=dict(inner.parameters, rule=inner.strategy, donor_mode="separate"),
                    )
                )
        return conditions


def _validate_condition(condition, context):
    variant = np.asarray(condition.values, float)
    if variant.shape != context.values.shape or not np.isfinite(variant).all():
        raise ValueError("Replacement condition must be finite and aligned with the original input")
    allowed = np.zeros(variant.shape, bool)
    allowed[np.ix_(np.arange(context.start, context.stop), context.channels)] = True
    if not np.array_equal(variant[~allowed], context.values[~allowed]):
        raise ValueError("Replacement may only modify requested signals inside the historical window")
    donors = _accepted_donors(context.reference)
    if not any(
        input_hash(d.values) == input_hash(condition.donor.values) and d.source == condition.donor.source
        for d in donors
    ):
        raise ValueError("Replacement condition must use an admitted donor from the resolver")
    if condition.weights is not None:
        weights = np.asarray(condition.weights, float)
        if weights.shape != (context.stop - context.start,) or not np.isfinite(weights).all():
            raise ValueError("Replacement weights must align with the historical window")
    donor_values = np.asarray(condition.donor.values, float)
    if donor_values.shape != (context.stop - context.start, variant.shape[1]) or not np.isfinite(donor_values).all():
        raise ValueError("Donor must be a finite coherent multivariate segment")
    matched = next(
        d
        for d in donors
        if input_hash(d.values) == input_hash(condition.donor.values) and d.source == condition.donor.source
    )
    if (matched.control is None) != (condition.donor.control is None) or (
        matched.control is not None and not np.array_equal(matched.control, condition.donor.control)
    ):
        raise ValueError("Replacement must retain the resolver's matched normal control")
    if condition.donor.control is not None and condition.control_values is not None:
        control = np.asarray(condition.donor.control, float)
        cv = np.asarray(condition.control_values, float)
        if control.shape != variant.shape or cv.shape != variant.shape or not np.isfinite(control).all():
            raise ValueError("Normal control must be finite and aligned with the original input")
        if not np.isfinite(cv).all() or not np.array_equal(control[~allowed], cv[~allowed]):
            raise ValueError("Normal replacement must preserve values outside the requested intervention")
    return variant


def _donor_identity(donor):
    """Treat reordered provenance mappings as the same donor, retaining source identity."""
    return (
        input_hash(np.asarray(donor["values"])),
        json.dumps(donor["source"], sort_keys=True, separators=(",", ":"), default=str),
    )


def _donor_stability(results):
    """Donors are replicates, not new historical candidates or seeds/masks."""
    metrics = HistoricalAttributionMetrics()
    ids = [str(p["position"]) for p in results[0]["points"]]
    controlled = all(r["metrics"]["CAI"] is not None for r in results)
    effect_key = "CAI" if controlled else "TRI_F"
    response_key = "control_adjusted_reconstruction_change" if controlled else "signed_change"
    seen, replicates, responses = set(), {}, {}
    for result in results:
        donor = result["condition"]["donor"]
        identity = _donor_identity(donor)
        if identity in seen:
            continue
        seen.add(identity)
        key = result["condition"]["id"]
        replicates[key] = {"fixed_block": result["metrics"][effect_key]}
        responses[key] = {str(p["position"]): p[response_key] for p in result["points"]}
    # Rank refers to output responses, never independent historical causes.
    alarm = metrics.compute_stability(replicates, axis="donor", expected_candidates=["fixed_block"])
    return dict(
        status=alarm["status"],
        reason=alarm["reason"],
        axis="donor",
        checked_axes=["donor"],
        unverified_axes=["seed", "mask", "boundary"],
        effect_metric=effect_key,
        distinct_donor_count=len(replicates),
        alarm_effect=alarm,
        output_response_ranking=metrics.compute_stability(
            responses, axis="donor", expected_candidates=ids, rank_absolute=True
        ),
        interpretation="Same historical block across donors; output-time ranks describe reconstruction responses",
    )


def analyze_single_window(
    values,
    columns,
    flags,
    scores,
    reconstruction,
    infer: Callable,
    request,
    *,
    visible,
    time_labels=None,
    threshold_trace=None,
    event_labels=None,
    time_axis=None,
):
    """Replace [p-gap-length+1, p-gap] once; compare reconstructions at its points.

    ``infer`` returns original-unit reconstruction of the complete unchanged-size
    input. Truth must never be overwritten for scoring. Native SDK normalization
    is retained for this first version; no hidden-truth invariance is claimed.
    """
    values = np.asarray(values, float)
    reconstruction = np.asarray(reconstruction, float)
    flags = np.asarray(flags, bool)
    if (
        values.ndim != 2
        or reconstruction.shape != values.shape
        or flags.shape != (len(values),)
        or np.asarray(scores).shape != (len(values),)
        or not np.isfinite(reconstruction).all()
    ):
        raise ValueError("Input, reconstruction and SDK alarms must be aligned")
    if len(columns) != values.shape[1] or len(set(columns)) != len(columns) or not np.isfinite(values).all():
        raise ValueError("Unique feature names and finite values are required")
    if request.scope not in {"target", "all", "signals"} or request.gap < 1 or request.length < 1:
        raise ValueError("Invalid historical replacement setting")
    if time_axis is None:
        time_axis = resolve_time_axis(
            pd.DataFrame(index=np.arange(len(values))) if time_labels is None else pd.DataFrame({"Time": time_labels}),
            None if time_labels is None else "Time",
        )
    time_axis = deepcopy(time_axis)
    if (
        time_axis.get("kind") not in {"datetime", "numeric", "row_position"}
        or len(time_axis.get("values", [])) != len(values)
        or time_axis.get("position_base") != 0
    ):
        raise ValueError("Time axis must identify aligned datetime, numeric time or zero-based row positions")
    times = [str(t) for t in time_axis["values"]]
    out = dict(
        schema_version="1.0",
        case_name=request.case_name,
        status="unavailable",
        technical_settings={
            k: getattr(request, k) for k in ("length", "gap", "blend_points", "scope", "validation_repeats", "signals")
        },
        feature_order=list(columns),
        input_hash=input_hash(values),
        time_basis={"datetime": "timestamp", "numeric": "numeric", "row_position": "sample_position_no_timestamp"}[
            time_axis["kind"]
        ],
        time_axis=time_axis,
        limitations=[
            "Single joint intervention; output positions are not independent contributions to the alarm.",
            "Native SDK normalization and DPM sampler are unchanged; hidden-truth leakage is not excluded.",
            "Visible future observations can condition this offline reconstruction.",
        ],
    )
    if request.event_source == "sdk":
        events = np.flatnonzero(flags)
        if not len(events):
            return dict(out, reason="no_sdk_anomaly")
        p = int(events[0] if request.event_position is None else request.event_position)
        if not 0 <= p < len(values) or not flags[p]:
            raise ValueError("Requested event must be an actual SDK anomaly")
    elif request.event_source == "dataset_label":
        # Dataset events are an explicit opt-in, never fabricated SDK alarms.
        labels = np.asarray(event_labels)
        if labels.shape != (len(values),) or not np.isin(labels, [0, 1]).all():
            raise ValueError("Aligned binary dataset event labels are required")
        if request.event_position is None or request.target is None:
            raise ValueError("A dataset-labelled event requires an explicit position and target")
        p = int(request.event_position)
        if not 0 <= p < len(values) or not labels[p]:
            raise ValueError("Requested event must have a positive dataset label")
    else:
        raise ValueError("Unknown event source")
    if len(times) != len(values):
        raise ValueError("Timestamps must align with all observations")
    target = (
        int(np.argmax(np.abs(values[p] - reconstruction[p])))
        if request.target is None
        else list(columns).index(request.target)
    )
    unit = request.units.get(
        columns[target], request.units.get(str(columns[target]), "Original scale (units unspecified)")
    )
    stop = p - request.gap + 1
    start = stop - request.length
    out.update(
        event=dict(
            position=p,
            time=times[p],
            source=request.event_source,
            sdk_detected=bool(flags[p]),
            sdk_score=float(scores[p]),
            target=str(columns[target]),
            target_index=target,
            target_selection="largest_original_raw_unit_residual" if request.target is None else "explicit_target",
            threshold_trace=threshold_trace,
        ),
        unit=unit,
        metrics={"status": "unavailable", "reason": "no_replacement_result"},
    )
    if start < 0:
        return dict(out, reason="insufficient_history_before_actual_alarm")
    visible = np.asarray(visible(p), bool)
    if visible.shape != (len(values),) or visible[p]:
        raise ValueError("Alarm target must be hidden in the selected SDK strategy")
    if request.signals is not None:
        signals = request.signals
        if (
            not isinstance(signals, list | tuple)
            or not signals
            or len(set(signals)) != len(signals)
            or any(s not in columns for s in signals)
            or request.scope == "all"
        ):
            raise ValueError("Requested signals must be a nonempty unique list of known feature names")
        channels = [i for i, name in enumerate(columns) if name in signals]
    elif request.scope == "signals":
        raise ValueError("Explicit signals are required for signals scope")
    else:
        channels = [target] if request.scope == "target" else list(range(values.shape[1]))
    strategy = request.replacement_strategy or LinearBlendReplacement()
    original = values.copy()
    original.setflags(write=False)
    context = ReplacementContext(
        original,
        tuple(columns),
        visible,
        p,
        start,
        stop,
        tuple(channels),
        tuple(times),
        ReferenceMatch("unavailable"),
        request.blend_points,
    )
    out.update(
        window=dict(
            start=start,
            stop=stop,
            start_label=times[start],
            end_label=times[stop - 1],
            positions=list(range(start, stop)),
            labels=times[start:stop],
            visible_positions=np.flatnonzero(visible[start:stop]).__add__(start).tolist(),
            visible_count=int(visible[start:stop].sum()),
        ),
        replaced_features=[str(columns[c]) for c in channels],
    )
    # Persist original curves even when no acceptable normal reference exists.
    out["curves"] = dict(labels=times, observed=values[:, target].tolist(), before=reconstruction[:, target].tolist())
    if request.validation_repeats:
        repeated = np.asarray(infer(values.copy()), float)
        if repeated.shape != values.shape or not np.isfinite(repeated).all():
            raise ValueError("Adapter must return finite original-unit reconstruction")
        self_reference = ReferenceMatch("accepted", values[start:stop].copy(), {"id": "self"})
        self_context = replace(context, reference=self_reference)
        self_conditions = strategy.generate(self_context)
        if not self_conditions:
            raise ValueError("Replacement strategy must support an admitted self-reference")
        self_checks = []
        for condition in self_conditions:
            identity = _validate_condition(condition, self_context)
            self_out = np.asarray(infer(identity.copy()), float)
            if self_out.shape != values.shape or not np.isfinite(self_out).all():
                raise ValueError("Adapter must return finite original-unit reconstruction")
            self_checks.append(
                dict(
                    condition_id=condition.condition_id,
                    input_equal=bool(np.array_equal(identity, values)),
                    max_abs_difference=float(np.abs(self_out - reconstruction).max()),
                )
            )
        out["validation"] = dict(
            repeat_max_abs_difference=float(np.abs(repeated - reconstruction).max()),
            self_input_equal=all(check["input_equal"] for check in self_checks),
            self_max_abs_difference=max(check["max_abs_difference"] for check in self_checks),
            self_conditions=self_checks,
            alarm_observation_unchanged=True,
            hidden_truth_invariance="not_verified_native_path_unchanged",
        )
    if request.resolver is None:
        return dict(out, reason="normal_reference_not_provided")
    match = request.resolver.resolve(values, columns, start, stop, visible, p)
    out["reference"] = dict(status=match.status, reason=match.reason, source=match.source, candidates=match.candidates)
    if match.status != "accepted":
        return dict(out, status=match.status, reason=match.reason)
    context = replace(context, reference=match)
    conditions = strategy.generate(context)
    if not conditions:
        return dict(out, status="no_acceptable_replacement", reason="strategy_produced_no_conditions")
    if len({c.condition_id for c in conditions}) != len(conditions):
        raise ValueError("Replacement condition IDs must be unique")
    for condition in conditions:
        _validate_condition(condition, context)
    results = []
    for condition in conditions:
        results.append(_analyze_condition(out, context, condition, reconstruction, infer))
    primary = deepcopy(results[0])
    if len(results) > 1:
        rules = [(r["condition"]["strategy"], r["condition"]["parameters"]) for r in results]
        if all(rule == rules[0] for rule in rules):
            stability = _donor_stability(results)
            for result in results:
                result["metrics"]["ExplanationStability"] = deepcopy(stability)
            primary["metrics"]["ExplanationStability"] = deepcopy(stability)
        else:
            reason = dict(status="unavailable", reason="nonmatched_replacement_rules", axis="donor")
            for result in results:
                result["metrics"]["ExplanationStability"] = reason.copy()
            primary["metrics"]["ExplanationStability"] = reason.copy()
    primary["conditions"] = results
    primary["condition_count"] = len(results)
    primary["distinct_donor_count"] = len({_donor_identity(r["condition"]["donor"]) for r in results})
    primary["primary_condition_id"] = results[0]["condition"]["id"]
    return primary


def _analyze_condition(base, context, condition, reconstruction, infer):
    """Infer and measure ANY admitted condition, without rule-name branches."""
    out = deepcopy(base)
    values, columns, visible = context.values, context.columns, context.visible
    p, start, stop, channels = context.event, context.start, context.stop, context.channels
    times = context.time_labels
    target = out["event"]["target_index"]
    unit = out["unit"]
    variant = np.asarray(condition.values, float)
    weights = condition.weights
    donor = condition.donor
    out["reference"]["source"] = dict(donor.source)
    out["condition"] = dict(
        id=condition.condition_id,
        strategy=condition.strategy,
        parameters=condition.parameters,
        donor=dict(source=donor.source, values=donor.values.tolist()),
        input_hash=input_hash(variant),
        conditioning_hash=input_hash(np.where(visible[:, None], variant, 0)),
        visibility=visible.tolist(),
        hidden_values="legacy native raw-array/display values; excluded from retrieval and visible blend decisions",
    )
    out["replacement_input"] = variant.tolist()
    after = np.asarray(infer(variant.copy()), float)
    if after.shape != values.shape or not np.isfinite(after).all():
        raise ValueError("Adapter must return finite original-unit reconstruction")
    out["reconstruction"] = after.tolist()
    delta = after[start:stop, target] - reconstruction[start:stop, target]
    changed = ~np.isclose(variant[start:stop], values[start:stop], rtol=0, atol=1e-12)
    out.update(
        status="ok",
        reason=None,
        replacement_hash=input_hash(variant),
        changes=dict(
            changed_cells=int(changed.sum()),
            changed_positions=(np.flatnonzero(changed.any(1)) + start).tolist(),
            changed_visible_cells=int(changed[visible[start:stop]].sum()),
            mean_absolute_input_change=float(np.abs(variant[start:stop] - values[start:stop]).mean()),
            weights=weights.tolist() if weights is not None else None,
        ),
    )
    out["curves"].update(replaced_input=variant[:, target].tolist(), after=after[:, target].tolist())
    changes = out["changes"]
    changes["changed_feature_indices"] = np.flatnonzero(changed.any(axis=0)).tolist()
    changes["changed_features"] = [str(columns[c]) for c in changes["changed_feature_indices"]]
    changes["changed_times"] = [times[t] for t in changes["changed_positions"]]
    changes["donor_segment_hash"] = input_hash(donor.values)
    changes["unblended_mean_absolute_input_change"] = float(
        np.abs(donor.values[:, channels] - values[start:stop, channels]).mean()
    )
    changes["per_feature"] = {}
    for c in channels:
        summaries = {"feature_index": int(c), "display_label": str(columns[c])}
        for key, array in [("before", values), ("after", variant)]:
            a = array[start:stop, c]
            summaries[key] = dict(
                mean=float(a.mean()),
                std=float(a.std()),
                left_seam=float(a[0] - array[start - 1, c]) if start and visible[start - 1] else None,
                right_seam=float(array[stop, c] - a[-1]) if stop < len(values) and visible[stop] else None,
            )
        changes["per_feature"][str(c)] = summaries
    out.setdefault("validation", {}).update(
        alarm_observation_unchanged=bool(np.array_equal(variant[p], values[p])),
        outside_window_unchanged=bool(
            np.array_equal(variant[:start], values[:start]) and np.array_equal(variant[stop:], values[stop:])
        ),
    )
    out["points"] = [
        dict(
            position=t,
            time=times[t],
            before=float(reconstruction[t, target]),
            after=float(after[t, target]),
            signed_change=float(d),
            historical_reconstruction_change_score=float(abs(d)),
        )
        for t, d in zip(range(start, stop), delta, strict=True)
    ]
    out["top_points"] = sorted(
        out["points"], key=lambda r: (-r["historical_reconstruction_change_score"], r["position"])
    )[:5]
    out["alarm_comparison"] = dict(
        observed=float(values[p, target]),
        before=float(reconstruction[p, target]),
        after=float(after[p, target]),
        signed_change=float(after[p, target] - reconstruction[p, target]),
    )
    out["metrics"] = _point_metrics(
        float(values[p, target]), float(reconstruction[p, target]), float(after[p, target]), unit
    )
    out["metrics"]["ExplanationStability"] = dict(
        status="unavailable",
        reason="single_donor",
        axis="donor",
        distinct_donor_count=1,
        checked_axes=[],
        unverified_axes=["seed", "mask", "boundary"],
    )
    out["metrics"]["HistoricalContributionShare"] = dict(status="not_applicable", reason="single_intervention")
    for key, x in [("before", values), ("after", variant)]:
        a = x[start:stop, target]
        out["changes"][key] = dict(
            mean=float(a.mean()),
            std=float(a.std()),
            left_seam=float(a[0] - x[start - 1, target]) if start else None,
            right_seam=float(x[stop, target] - a[-1]) if stop < len(values) and visible[stop] else None,
        )
    if donor.control is not None and condition.control_values is not None:
        control = np.asarray(donor.control, float)
        c0 = np.asarray(infer(control.copy()), float)
        cv = np.asarray(condition.control_values, float)
        c1 = np.asarray(infer(cv.copy()), float)
        if any(c.shape != values.shape or not np.isfinite(c).all() for c in (c0, c1)):
            raise ValueError("Adapter must return finite aligned normal-control reconstruction")
        out["normal_control"] = dict(
            input_hash=input_hash(control),
            replacement_hash=input_hash(cv),
            replacement_input=cv.tolist(),
            reconstruction=c1.tolist(),
            before=c0[:, target].tolist(),
            after=c1[:, target].tolist(),
            observed=control[:, target].tolist(),
            signed_change=(c1[start:stop, target] - c0[start:stop, target]).tolist(),
        )
        met = HistoricalAttributionMetrics(
            MetricsConfig(numerical_tolerance=1e-8, small_residual_warning=1e-8, practical_effect_threshold=0.0)
        ).compute_effects(
            values[p, target],
            reconstruction[p, target],
            after[p, target],
            normal_truth=control[p, target],
            normal_before=c0[p, target],
            normal_after=c1[p, target],
        )["points"][0]
        out["metrics"].update(met)
        normal_delta = c1[start:stop, target] - c0[start:stop, target]
        for point, normal_change in zip(out["points"], normal_delta, strict=True):
            point["normal_signed_change"] = float(normal_change)
            point["control_adjusted_reconstruction_change"] = point["signed_change"] - float(normal_change)
        out["validation"]["normal_self_input_equal"] = bool(np.array_equal(control, cv))
        out["validation"]["normal_self_max_abs_difference"] = float(np.abs(c1 - c0).max())
    else:
        out["normal_control"] = dict(
            status="unavailable",
            reason="no_matched_normal_control" if donor.control is None else "strategy_did_not_supply_matched_control",
        )
    out["metrics"]["ExplanationStability"]["effect_metric"] = "CAI" if out["metrics"]["CAI"] is not None else "TRI_F"
    return out


def run_sdk_attribution(
    frame,
    native,
    flags,
    scores,
    request,
    infer,
    model_path,
    config_path,
    time_labels=None,
    threshold_trace=None,
    event_labels=None,
    time_axis=None,
):
    import pandas as pd
    import torch
    from sdk.feature_adapter import FeatureAdapter
    from sdk.inference_ad import _get_checkpoint_preprocessing, _get_finetune_window_settings, load_config

    if isinstance(event_labels, pd.Series) and not event_labels.index.equals(frame.index):
        raise ValueError("Dataset event labels must align with the input row index and order")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config") or load_config(config_path)
    preprocessing = _get_checkpoint_preprocessing(checkpoint)
    length, split = _get_finetune_window_settings(preprocessing or {}, checkpoint, config)
    del checkpoint
    dataset = config.get("dataset", {})
    factor = float(dataset.get("scale_factor", 1.0))
    adapter = None
    if preprocessing is not None:
        adapter = FeatureAdapter.from_metadata(preprocessing)
        if adapter.uses_pca:
            raise ValueError("Historical attribution currently requires direct feature mapping; no PCA attribution")
        columns = preprocessing.get("columns")
        if columns:
            if len(columns) != adapter.input_dim or len(set(columns)) != len(columns):
                raise ValueError("Checkpoint preprocessing columns must uniquely identify the model inputs")
            if any(column not in frame.columns for column in columns):
                raise ValueError("Historical attribution input is missing checkpoint feature columns")
            frame = frame.loc[:, columns]
        if frame.shape[1] != adapter.input_dim:
            raise ValueError("Historical attribution features must match the checkpoint input dimension")
        factor = adapter.scale_factor
    if not np.isfinite(factor) or factor == 0:
        raise ValueError("Historical attribution requires a finite nonzero preprocessing scale factor")
    # Copy the built-in retriever: caller configuration must not weaken native-window exclusion.
    if isinstance(request.resolver, CosineWindowRetriever):
        request = replace(request, resolver=replace(request.resolver, model_window_length=length))
    values = frame.to_numpy(dtype=float)
    f = values.shape[1]
    audit = []

    def physical(x, result):
        y = np.asarray(result["target"], float)[: len(x), :f]
        rec = np.asarray(result["recon"], float)[: len(x), :f]
        low = x.min(axis=0) if adapter is None else np.asarray(adapter.min_[:f], float)
        high = x.max(axis=0) if adapter is None else np.asarray(adapter.max_[:f], float)
        span = high - low
        span = np.where(span == 0, 1.0, span)
        expected = (x - low) / span * factor if adapter is None else adapter.transform(x).numpy()[:, :f]
        if y.shape != x.shape or not np.allclose(y, expected, rtol=2e-5, atol=2e-5):
            raise ValueError("Native output is not the expected direct min/max feature mapping; no PCA attribution")
        audit.append(
            dict(
                external_min=low.tolist(),
                external_span=span.tolist(),
                scale_factor=factor,
                scaling_source="request_min_max" if adapter is None else "checkpoint_min_max",
                internal_location_scale="not_exposed_native_path_unchanged",
            )
        )
        return rec / factor * span + low

    before = physical(values, native)
    selected_request = request
    auto_select_target = request.event_source == "sdk" and request.target is None and np.any(flags)
    if auto_select_target:
        p = int(np.flatnonzero(flags)[0] if request.event_position is None else request.event_position)
        if not 0 <= p < len(frame) or not flags[p]:
            raise ValueError("Requested event must be an actual SDK anomaly")
        target = int(np.argmax(np.abs(np.asarray(native["target"])[p, :f] - np.asarray(native["recon"])[p, :f])))
        selected_request = replace(request, target=frame.columns[target])

    def reconstruct(x):
        return physical(x, infer(pd.DataFrame(x, columns=frame.columns, index=frame.index)))

    def visibility(p):
        pos = np.arange(len(values)) % length
        skip = length // split
        chunks = pos // skip
        pchunk = (p % length) // skip
        # cond_mask 1 = visible; use the complementary strategy hiding the alarm.
        return ((chunks % 2) != (pchunk % 2)) & (chunks < split)

    if time_axis is None and time_labels is None:
        time_axis = resolve_time_axis(frame)
    result = analyze_single_window(
        values,
        list(frame.columns),
        flags,
        scores,
        before,
        reconstruct,
        selected_request,
        visible=visibility,
        time_labels=time_labels,
        threshold_trace=threshold_trace,
        event_labels=event_labels,
        time_axis=time_axis,
    )
    if auto_select_target and "event" in result:
        result["event"]["target_selection"] = "largest_original_SDK_scaled_residual"
    result["native_settings"] = dict(
        window_length=length,
        split=split,
        scale_factor=factor,
        output_statistic="existing SDK sample median and hidden-strategy combination",
        future_context="Visible future positions remain available under the original interleaved masks",
        preprocessing_audit=audit,
    )
    return result


# PDF appendix

INK = "#17365D"


BODY = "#344054"


def timestamp_labels(values, *, expected_length, timezone=None):
    """Validate a complete ordered datetime axis; never interpret row IDs as epochs."""
    if len(values) != expected_length or any(isinstance(v, int | float | np.number) for v in values):
        raise ValueError("Expected one original datetime timestamp per case observation")
    times = pd.DatetimeIndex(pd.to_datetime(values, errors="raise", utc=timezone is not None))
    if timezone is not None:
        times = times.tz_convert(timezone)
    if times.hasnans or not times.is_monotonic_increasing or not times.is_unique:
        raise ValueError("Case timestamps must be complete, unique and chronological")
    return times


def format_timestamp(value):
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError("Missing timestamp")
    text = ts.strftime("%Y-%m-%d %H:%M:%S" if ts.second or ts.microsecond else "%Y-%m-%d %H:%M")
    return text + (ts.strftime(" %z") if ts.tz is not None else "")


def _case_time_axis(case):
    if case.get("time_axis") is not None:
        return case["time_axis"]
    return dict(
        kind="datetime" if case.get("timestamps") is not None else "row_position",
        values=case.get("timestamps", list(range(case["steps"]))),
        label="Time" if case.get("timestamps") is not None else "Row number (no timestamp column)",
    )


def alarm_label(result):
    case = result["case"]
    position = case.get("alarm_position", case["alarm_step"] - 1)
    time_axis = _case_time_axis(case)
    if time_axis["kind"] == "datetime":
        times = timestamp_labels(time_axis["values"], expected_length=case["steps"], timezone=time_axis.get("timezone"))
        return format_timestamp(times[position])
    if case.get("alarm_timestamp") is not None:
        return format_timestamp(case["alarm_timestamp"])
    return (
        f"row {position}"
        if time_axis["kind"] == "row_position"
        else f"{time_axis['label']} {time_axis['values'][position]}"
    )


def historical_axis(result):
    case = result["case"]
    steps = np.array([p["case_step"] for p in result["summary"]["primary"]["points"]], dtype=int)
    if (steps < 1).any() or (steps >= case["alarm_step"]).any():
        raise ValueError("Historical labels must precede the fixed alarm")
    time_axis = _case_time_axis(case)
    if time_axis["kind"] == "datetime":
        times = timestamp_labels(time_axis["values"], expected_length=case["steps"], timezone=time_axis.get("timezone"))
        x = np.asarray(mdates.date2num(times.to_pydatetime()))
        labels = {int(s): format_timestamp(times[s - 1]) for s in steps}
        return dict(
            is_datetime=True,
            timezone=times.tz,
            source_timestamps=times[np.r_[steps - 1, case["alarm_step"] - 1]].tolist(),
            x=x[steps - 1],
            alarm_x=float(x[case["alarm_step"] - 1]),
            alarm_label=alarm_label(result),
            labels=labels,
            column_label="Time",
            axis_label=time_axis["label"],
            range_label=f"{labels[int(steps[0])]} to {labels[int(steps[-1])]}",
        )
    x = np.asarray(time_axis["values"], dtype=float)
    if x.shape != (case["steps"],) or not np.isfinite(x).all():
        raise ValueError("Numeric/row time axis must contain one finite value per observation")
    labels = {int(s): str(time_axis["values"][s - 1]) for s in steps}
    return dict(
        is_datetime=False,
        time_kind=time_axis["kind"],
        x=x[steps - 1],
        alarm_x=float(x[case["alarm_step"] - 1]),
        alarm_label=alarm_label(result),
        labels=labels,
        column_label="Row number" if time_axis["kind"] == "row_position" else time_axis["label"],
        axis_label=time_axis["label"],
        range_label=f"{time_axis['label']}: {labels[int(steps[0])]} to {labels[int(steps[-1])]}",
    )


class _AdaptiveTimeLocator(Locator):
    """Keep as many source times as fit, re-evaluating font/width on every draw."""

    def __init__(self, values):
        self.values = np.unique(np.asarray(values, dtype=float))
        self.text_metrics = TextToPath()

    def __call__(self):
        low, high = sorted(self.axis.get_view_interval())
        values = self.values[(self.values >= low) & (self.values <= high)]
        if len(values) <= 2:
            return values
        axis = self.axis.axes
        font = self.axis.get_major_ticks(1)[0].label1.get_fontproperties()
        factor = axis.figure.dpi / 72
        labels = [self.axis.get_major_formatter()(value) for value in values]
        widths = np.array(
            [
                max(
                    self.text_metrics.get_text_width_height_descent(line, font, ismath=False)[0]
                    for line in label.split("\n")
                )
                * factor
                for label in labels
            ]
        )
        positions = axis.transData.transform(np.column_stack((values, np.zeros(len(values)))))[:, 0]
        left, right = positions - widths / 2, positions + widths / 2
        gap = 6 * factor
        # Reserve the interval endpoints, then use interval scheduling between them.
        selected, seen, cursor = [0], {labels[0], labels[-1]}, right[0]
        for index in sorted(range(1, len(values) - 1), key=lambda i: right[i]):
            if labels[index] not in seen and left[index] >= cursor + gap and right[index] + gap <= left[-1]:
                selected.append(index)
                seen.add(labels[index])
                cursor = right[index]
        return values[sorted([*selected, len(values) - 1])]


def style_time_axis(axis, info):
    x = info["x"]
    spacing = max(float(np.median(np.diff(x))) if len(x) > 1 else abs(info["alarm_x"] - x[0]), 1e-9)
    axis.set_xlim(x[0] - 0.9 * spacing, info["alarm_x"] + 0.8 * spacing)
    if info["is_datetime"]:
        if not set_short_timestamp_ticks(axis, info["source_timestamps"]):
            times = pd.DatetimeIndex(info["source_timestamps"])
            pattern = "%H:%M:%S" if (times.second != 0).any() else "%H:%M"
            if times[0].date() != times[-1].date():
                pattern = "%m-%d\n" + pattern
            axis.xaxis.set_major_formatter(mdates.DateFormatter(pattern, tz=info.get("timezone")))
            axis.xaxis.set_major_locator(_AdaptiveTimeLocator(np.r_[x, info["alarm_x"]]))
    else:
        # Thin display ticks only; retain every original time coordinate in the curve/JSON.
        row_axis = info.get("time_kind") == "row_position"
        low, high = min(x[0], info["alarm_x"]), max(x[0], info["alarm_x"])
        span = max(high - low, np.finfo(float).eps)
        offset = low if not row_axis and abs(low) > 1e4 * span else 0.0
        power = int(np.floor(np.log10(span)))
        scale = 10.0**power if not row_axis and (power < -1 or power > 3) else 1.0
        formatter = FuncFormatter(
            lambda value, _: f"{value:.0f}" if row_axis else f"{(value - offset) / scale:.2f}".rstrip("0").rstrip(".")
        )
        notation = ([f"x 1e{power:+d}"] if scale != 1 else []) + ([f"+ {offset:.15g}"] if offset else [])
        formatter.set_offset_string(" ".join(notation))
        axis.xaxis.set_major_formatter(formatter)
        axis.xaxis.set_major_locator(_AdaptiveTimeLocator(np.r_[x, info["alarm_x"]]))
    axis.set_xlabel(info["axis_label"], fontsize=9, labelpad=9)


def set_short_timestamp_ticks(axis, values):
    """Label sub-five-minute plots directly from source times, including seconds.

    Source ticks preserve irregular spacing; display density adapts to the plot
    and font size. No timezone or missing timestamps are invented.
    """
    times = pd.DatetimeIndex(pd.to_datetime(values))
    if len(times) < 2 or times[-1] - times[0] > pd.Timedelta(minutes=5):
        return False
    pattern = "%H:%M:%S" if times[0].date() == times[-1].date() else "%m-%d\n%H:%M:%S"
    axis.xaxis.set_major_formatter(mdates.DateFormatter(pattern, tz=times.tz))
    axis.xaxis.set_major_locator(_AdaptiveTimeLocator(mdates.date2num(times.to_pydatetime())))
    return True


def _text(fig, x, y, text, *, width=None, size=9, bold=False, color=BODY):
    if width:
        text = "\n".join(fill(line, width=width) for line in text.split("\n"))
    return fig.text(
        x, y, text, va="top", fontsize=size, weight="bold" if bold else "normal", color=color, linespacing=1.5
    )


def _page(title, subtitle, number):
    fig = Figure(figsize=(11, 8.5), facecolor="white")
    _text(fig, 0.07, 0.955, title, size=18, bold=True, color=INK)
    _text(fig, 0.07, 0.9, subtitle, size=9)
    _add_footer(fig, number)
    return fig


def _saved_primary(result):
    """Read the selected condition without altering saved analysis/metric data."""
    raw = result.get("historical_result", {})
    identifier = result["summary"]["primary"].get("condition_id")
    return next((c for c in raw.get("conditions", []) if c["condition"]["id"] == identifier), raw)


def _labelled_onset(result):
    """Use only recorded, alignable timestamps; labels are not inferred causes."""
    axis = _case_time_axis(result["case"])
    raw = result.get("historical_result", {})
    trace = raw.get("event", {}).get("threshold_trace") or {}
    value = trace.get("labelled_onset") if isinstance(trace, dict) else None
    if axis["kind"] != "datetime" or value is None or isinstance(value, int | float | np.number):
        return None
    try:
        onset, first = pd.Timestamp(value), pd.Timestamp(axis["values"][0])
        if pd.isna(onset) or (onset.tz is None) != (first.tz is None):
            return None
        return onset
    except (TypeError, ValueError):
        return None


def historical_summary(result):
    """Deterministic PDF prose from saved facts; no GPT or new inference is used.

    Return overview/result/takeaway paragraphs. Shares concern historical output
    responses after one joint intervention, not independent effects on the alarm.
    Missing facts are omitted; zero and tied responses have their own templates.
    """
    if result.get("schema_version") != "reconstruction_shares_v1":
        if result.get("status") != "ok":
            return dict(
                overview="Historical attribution is unavailable: " + str(result.get("reason", "missing result")),
                result="",
                takeaway="No response-based time priority is available.",
            )
        result = report_data(result)
    case = result["case"]
    primary = _saved_primary(result)
    time = historical_axis(result)
    points = result["summary"]["primary"]["points"]
    ranked = sorted((p for p in points if p["magnitude"] > 0), key=lambda p: (-p["magnitude"], p["case_step"]))[:5]
    total = sum(p["magnitude"] for p in points)
    row_axis = _case_time_axis(case)["kind"] == "row_position"
    event_time = (
        f"observation index {case.get('alarm_position', case['alarm_step'] - 1)}" if row_axis else time["alarm_label"]
    )
    source = {"sdk": "SDK alarm", "dataset_label": "dataset-labelled analysis point"}.get(
        case.get("event_source"), "analysis point"
    )
    count = len(primary.get("replaced_features", []))
    scope = (
        f"all {count} signals"
        if count and count == len(case.get("features", []))
        else f"{count} selected signal(s)"
        if count
        else "the selected signals"
    )
    interval = (
        f"observation indices {points[0]['case_step'] - 1} to {points[-1]['case_step'] - 1}"
        if row_axis
        else time["range_label"]
    )
    overview = (
        f"We examine the {source} at {event_time} for {case['target']} ({case.get('target_unit', 'original units')}). "
        f"One earlier block of {scope} is changed over {interval}, then target reconstructions are compared. "
        "The bars rank times; matching numbers link measurements to the relative-response timeline."
    )
    if not ranked:
        finding = "No historical model response was observed in this interval; relative percentages are undefined."
        takeaway = "No response-based time priority is available. This does not rule out a real fault."
    else:
        maximum = ranked[0]["magnitude"]
        share = 100 * maximum / total
        leaders = sum(p["magnitude"] == maximum for p in points)
        finding = (
            f"{leaders} times share the maximum response ({share:.1f}% each); there is no single leading time."
            if leaders > 1
            else f"Maximum relative response: {time['labels'][ranked[0]['case_step']]} ({share:.1f}% of the interval total)."
        )
        finding += f" The displayed Top-{len(ranked)} times account for {100 * sum(p['magnitude'] for p in ranked) / total:.1f}% of the response."
        takeaway = "Review measurements and available operating records around these times; they are investigation priorities, not confirmed fault sources."
    change = primary.get("alarm_comparison", {}).get("signed_change")
    if finite(change):
        finding += f" At the analysis point, reconstruction changes by {change:+.3g} {case.get('target_unit', 'original units')}."
    onset = _labelled_onset(result)
    if onset is not None:
        axis = _case_time_axis(case)
        first, last = (pd.Timestamp(axis["values"][p["case_step"] - 1]) for p in (points[0], points[-1]))
        relation = "before" if onset < first else "after" if onset > last else "within"
        takeaway += f" Labelled anomaly onset {format_timestamp(onset)} is {relation} the reviewed interval."
    return dict(overview=overview, result=finding, takeaway=takeaway)


def create_historical_pages(result, *, start_page=1):
    """Separate rankings/summary and full-width timelines; saved metrics are unchanged."""
    if result.get("schema_version") != "reconstruction_shares_v1":
        if result.get("status") != "ok":
            fig = _page("Historical attribution", result.get("case_name", "SDK event"), start_page)
            _text(fig, 0.07, 0.80, "Not available: " + str(result.get("reason", "missing replacement")))
            yield fig
            return
        result = report_data(result)
    case, summary = result["case"], result["summary"]
    primary = _saved_primary(result)
    points = summary["primary"]["points"]
    time = historical_axis(result)
    # The SDK records its final target selection after the independent conditions.
    event = result.get("historical_result", {}).get("event", primary.get("event", {}))
    target_info = f"Target: {case['target']}"
    contributors = result.get("alarm_contributors", {})
    if case.get("event_source") == "sdk" and (
        event.get("target_selection") == "largest_original_SDK_scaled_residual"
        or (contributors.get("target_rank") == 1 and event.get("target_selection") != "explicit_target")
    ):
        target_info += " | SDK Top-1 contributor at this alarm"
    row_axis = _case_time_axis(case)["kind"] == "row_position"
    event_time = (
        f"observation index {case.get('alarm_position', case['alarm_step'] - 1)}" if row_axis else time["alarm_label"]
    )
    interval = (
        f"Observation indices {points[0]['case_step'] - 1} to {points[-1]['case_step'] - 1}"
        if row_axis
        else time["range_label"]
    )
    event_label = "SDK alarm" if case.get("event_source") == "sdk" else "Dataset-labelled analysis point"
    subtitle = f"{case['name']} | {event_label}: {event_time}\n{target_info}\nHistorical interval: {interval}"
    fig = _page("Historical attribution", subtitle, start_page)
    _text(
        fig,
        0.07,
        0.795,
        "Highlights earlier signal times to inspect when investigating this alarm.",
        size=10,
        bold=True,
        color=INK,
    )
    magnitudes = np.array([p["magnitude"] for p in points], dtype=float)
    # Zero has no rankable response. Ties keep chronological order, with an explicit note.
    ranked = sorted(
        (i for i, v in enumerate(magnitudes) if v > 0), key=lambda i: (-magnitudes[i], points[i]["case_step"])
    )[:5]
    n = len(ranked)
    heading = f"Top-{n} historical " + (
        "timestamps" if time["is_datetime"] else "observation indices" if row_axis else "times"
    )
    _text(
        fig,
        0.07,
        0.738,
        heading if n else "Historical timestamps" if time["is_datetime"] else "Historical observations",
        size=11,
        bold=True,
        color="#175CD3",
    )
    bars = fig.add_axes([0.18, 0.43, 0.75, 0.25])
    bars.set_gid("historical_top_times")
    bars.set_xlim(0, 100)
    bars.set_xticks([0, 25, 50, 75, 100], labels=["0%", "25%", "50%", "75%", "100%"])
    bars.set_xlabel("Historical response share (%)", fontsize=9)
    bars.spines[["top", "right", "left"]].set_visible(False)
    bars.spines["bottom"].set_color("#CDD4DF")
    bars.grid(axis="x", alpha=0.2)
    bars.set_axisbelow(True)
    bars.tick_params(labelsize=8, colors=BODY)
    if n:
        shares = 100 * magnitudes[ranked] / magnitudes.sum()
        if time["is_datetime"]:
            times = timestamp_labels(
                _case_time_axis(case)["values"],
                expected_length=case["steps"],
                timezone=_case_time_axis(case).get("timezone"),
            )
            pattern = "%H:%M:%S" if (times.second != 0).any() else "%H:%M"
            if times[0].date() != times[-1].date():
                pattern = "%m-%d\n" + pattern
            labels = [times[points[i]["case_step"] - 1].strftime(pattern) for i in ranked]
        else:
            labels = [time["labels"][points[i]["case_step"]] for i in ranked]
        bars.barh(range(n), shares, color="#175CD3", height=0.55)
        bars.set_yticks(range(n), labels=[f"{i + 1}. {label}" for i, label in enumerate(labels)])
        bars.invert_yaxis()
        for i, share in enumerate(shares):
            bars.text(
                share + 2 if share < 85 else share - 2,
                i,
                f"{share:.1f}%",
                va="center",
                ha="left" if share < 85 else "right",
                fontsize=9,
                color=BODY,
            )
        note = f"Percentages use all {len(points)} historical points."
        if len(set(magnitudes)) == 1:
            note += "\nEqual responses; tied timestamps are listed in time order."
        elif len(set(magnitudes[ranked])) < n or (
            n < len(magnitudes) and np.count_nonzero(magnitudes == magnitudes[ranked[-1]]) > 1
        ):
            note += "\nTied responses share a rank; time order breaks display ties."
    else:
        bars.set_yticks([])
        bars.text(0.5, 0.5, "No observed model response.", transform=bars.transAxes, ha="center", color=BODY)
        note = "All historical changes are zero; percentages are undefined."
    _text(fig, 0.07, 0.355, note, size=8, width=143)
    prose = historical_summary(result)
    explanation = (
        "We replace the selected historical block with reference data and compare the target reconstructions "
        "before and after. Each timestamp\u2019s percentage is its absolute reconstruction change divided by the "
        "total absolute change across the entire historical interval."
    )
    _text(fig, 0.07, 0.31, "Summary", size=10, bold=True, color=INK)
    _text(fig, 0.07, 0.284, explanation + "\n\n" + "\n\n".join(prose.values()), size=8, width=143)
    yield fig

    fig = _page("Historical attribution", subtitle, start_page + 1)
    _text(fig, 0.07, 0.795, "Matching numbers refer to the Top-N timestamps on the previous page.", size=9)
    _text(fig, 0.07, 0.755, "Measured signal and historical response", size=11, bold=True, color="#175CD3")

    observed = primary.get("curves", {}).get("observed")
    signal = fig.add_axes([0.12, 0.495, 0.81, 0.205])
    response = fig.add_axes([0.12, 0.21, 0.81, 0.21], sharex=signal)
    signal.set_gid("historical_measured_target")
    response.set_gid("historical_time_response")
    axis = _case_time_axis(case)
    if time["is_datetime"]:
        times = timestamp_labels(axis["values"], expected_length=case["steps"], timezone=axis.get("timezone"))
        x = np.asarray(mdates.date2num(times.to_pydatetime()))
    else:
        x = np.asarray(axis["values"], dtype=float)
    positions = np.array([p["case_step"] - 1 for p in points])
    if observed is not None and len(observed) == case["steps"]:
        observed = np.asarray(observed, dtype=float)
        signal.plot(x, observed, color="#526580", linewidth=1.2)
        signal.margins(y=0.25)
        if n:
            signal.scatter(x[positions[ranked]], observed[positions[ranked]], color="#175CD3", s=18, zorder=3)
            # Stagger adjacent callouts to avoid colliding labels on dense timelines.
            chronological = sorted(range(n), key=lambda k: positions[ranked[k]])
            for order, rank in enumerate(chronological):
                i = ranked[rank]
                signal.annotate(
                    str(rank + 1),
                    (x[positions[i]], observed[positions[i]]),
                    xytext=(0, 10 if order % 2 == 0 else -12),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#175CD3",
                    bbox=dict(boxstyle="circle,pad=0.15", facecolor="white", edgecolor="#175CD3", linewidth=0.6),
                )
    else:
        signal.text(
            0.5, 0.5, "Original target signal was not saved.", transform=signal.transAxes, ha="center", fontsize=8
        )
    if n:
        percentages = 100 * magnitudes / magnitudes.sum()
        response.vlines(time["x"], 0, percentages, color="#175CD3", linewidth=1.1)
        response.scatter(time["x"], percentages, color="#175CD3", s=10, zorder=3)
        response.set_ylim(0, max(float(percentages.max()) * 1.45, 1.0))
        for rank, i in enumerate(ranked):
            response.annotate(
                str(rank + 1),
                (time["x"][i], percentages[i]),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#175CD3",
                bbox=dict(boxstyle="circle,pad=0.15", facecolor="white", edgecolor="#175CD3", linewidth=0.6),
            )
    else:
        response.set_ylim(0, 1)
        response.set_yticks([])
        response.text(
            0.5, 0.5, "No response; percentages are undefined.", transform=response.transAxes, ha="center", fontsize=7.5
        )
    _text(fig, 0.07, 0.47, "Relative historical response", size=10, bold=True, color="#175CD3")
    full_axis = dict(
        time, x=x, alarm_x=float(x[-1]), axis_label="Observation index" if row_axis else time["axis_label"]
    )
    if time["is_datetime"]:
        full_axis["source_timestamps"] = times.tolist()
    onset = _labelled_onset(result)
    onset_x = float(mdates.date2num(onset)) if onset is not None else None
    for ax in [signal, response]:
        style_time_axis(ax, full_axis)
        ax.axvspan(time["x"][0], time["x"][-1], color="#EEF2F6", zorder=0)
        ax.axvline(time["alarm_x"], color="#D92D20", ls=":", lw=1)
        if onset_x is not None and x[0] <= onset_x <= x[-1]:
            ax.axvline(onset_x, color="#F79009", ls="--", lw=0.9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#CDD4DF")
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(labelsize=8, colors=BODY)
    signal.set_xlabel("")
    signal.tick_params(labelbottom=False)
    signal.set_ylabel(f"{case['target']}\n({case.get('target_unit', 'original units')})", fontsize=9)
    signal.text(
        time["alarm_x"],
        1.07,
        "Analysis point" if case.get("event_source") == "dataset_label" else "Alarm",
        transform=signal.get_xaxis_transform(),
        ha="center",
        color="#D92D20",
        fontsize=8,
    )
    if onset_x is not None and x[0] <= onset_x <= x[-1]:
        signal.text(
            onset_x,
            1.07,
            "Labelled onset",
            transform=signal.get_xaxis_transform(),
            ha="center",
            color="#B96500",
            fontsize=8,
        )
    response.set_ylabel("Historical response share (%)", fontsize=9)
    _text(fig, 0.12, 0.12, "Numbers link the same times; blank areas were not ranked.", size=8)
    _text(
        fig,
        0.12,
        0.09,
        "The upper plot shows measured values in the sensor\u2019s original unit; the lower plot shows relative "
        "reconstruction-change shares (%), not input measurements.",
        width=135,
        size=8,
    )
    yield fig


def report_data(result):
    """Project every analyzed condition; donor conditions are not assumed seeds.

    The PDF displays the selected primary condition in its existing template.
    All condition metrics and output-time rankings remain available to callers.
    """
    conditions = result.get("conditions", [result])
    primary_id = result.get("primary_condition_id", conditions[0]["condition"]["id"])
    summaries, effects = {}, {}
    for condition in conditions:
        identifier = condition["condition"]["id"]
        total = sum(point["historical_reconstruction_change_score"] for point in condition["points"])
        points = [
            dict(
                position=point["position"],
                case_step=point["position"] + 1,
                signed_change=point["signed_change"],
                magnitude=point["historical_reconstruction_change_score"],
                share_pct=100 * point["historical_reconstruction_change_score"] / total if total else None,
            )
            for point in condition["points"]
        ]
        summaries[identifier] = dict(
            condition_id=identifier,
            donor_source=deepcopy(condition["condition"]["donor"]["source"]),
            points=points,
            time_rankings=sorted(points, key=lambda p: (-p["magnitude"], p["case_step"])),
        )
        effects[identifier] = deepcopy(condition["metrics"])
    if primary_id not in summaries:
        raise ValueError("Primary condition ID must identify an analyzed condition")
    primary = next(c for c in conditions if c["condition"]["id"] == primary_id)
    event, curves = primary["event"], primary["curves"]
    source = primary["reference"]["source"]
    control_reason = primary.get("normal_control", {}).get("reason")
    control_missing = control_reason == "strategy_did_not_supply_matched_control"
    data = dict(
        schema_version="reconstruction_shares_v1",
        case=dict(
            name=result["case_name"],
            steps=len(curves["labels"]),
            features=result["feature_order"],
            target=event["target"],
            target_unit=result["unit"],
            alarm_step=event["position"] + 1,
            alarm_position=event["position"],
            event_source=event["source"],
        ),
        summary=dict(
            primary_condition_id=primary_id,
            condition_count=len(conditions),
            primary=summaries[primary_id],
            conditions=summaries,
            alarm_metrics_by_condition=effects,
        ),
        report_context=dict(
            event_label="SDK alarm" if event["source"] == "sdk" else "Dataset-labelled analysis point",
            inference_note="Reconstruction from the configured inference callback; original inference settings retained.",
            distinct_donor_count=len({_donor_identity(c["condition"]["donor"]) for c in conditions}),
            control_unavailable_reason="matched-control replacement missing"
            if control_missing
            else "no matched normal control",
            control_note=(
                "An aligned normal control exists, but its replacement condition was not supplied by the strategy."
                if control_missing
                else "No matched normal control was supplied by the reference resolver."
                if effects[primary_id]["CAI"] is None
                else "The same replacement was evaluated on the verified aligned normal control."
            ),
            source_note=f"{result['case_name']}. Normal reference: {source.get('id', 'provided reference')}.",
            limits="Native normalization and interleaved masks are unchanged; visible future observations remain available.\n"
            "This is offline reconstruction sensitivity. Hidden-truth invariance and causal interpretation are not established.",
        ),
        historical_result=result,
    )
    if result.get("time_axis") is not None:
        data["case"]["time_axis"] = deepcopy(result["time_axis"])
    if result["time_basis"] == "timestamp":
        data["case"]["timestamps"] = curves["labels"]
    return data
