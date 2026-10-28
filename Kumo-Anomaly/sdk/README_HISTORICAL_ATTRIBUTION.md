# Historical attribution

Replace one historical block before a fixed SDK anomaly, rerun reconstruction,
and append attribution to the existing PDF. This describes model sensitivity,
not physical causation. Attribution is optional and dataset-independent.

## Design

**Event → normal reference/donor → replacement → reconstruction → metrics → PDF**.
The caller supplies dataset identity, normality evidence, feature names,
timestamps and units.

- `ReferenceResolver` supplies an aligned normal reference or eligible donors.
- `SimilarityStrategy` ranks aligned context; acceptance is separate.
- `ReplacementStrategy.generate(context)` returns independent conditions for
  the same inference and metrics workflow.

## Use the existing SDK

Verify normal-reference identity, feature order, units and time alignment.
Keep timestamps as the index; exclude time/label columns from feature matrices.
The SDK also supports configured datetime/numeric time columns. The PDF and
attribution JSON preserve the same time axis; without time metadata, positions
are zero-based DataFrame row numbers.

```python
from dataclasses import replace
from sdk.anomaly_analysis import perform_anomaly_analysis_with_diffusion
from sdk.historical_attribution import (
    AlignedNormalReference, HistoricalAttributionRequest,
)

resolver = AlignedNormalReference(
    normal_features,
    source={"id": "verified_normal_reference"},
    alignment_verified=True,
)
request = HistoricalAttributionRequest(resolver=resolver)
result = perform_anomaly_analysis_with_diffusion(
    observations, **sdk_settings, historical_attribution=request,
)
attribution = result.attrs["historical_attribution"]
```

Use your existing `observations` and `sdk_settings`. Defaults select the first
SDK anomaly and its largest original SDK-scaled residual feature. Override
`event_position` and `target` using an actual alarm and original feature key.

Defaults: `length=20`, `gap=1`, `blend_points=2`, `scope="target"`.
Use `scope="all"`, or `scope="signals", signals=feature_keys`, to change the
replacement scope independently of the scoring target.

Attribution requires a direct mapping to the original features; PCA and external
preprocessing models are not supported. For fine-tuned checkpoints with saved
direct-feature preprocessing, the SDK preserves training feature order, min/max
scaling, and window settings. Reference and donor frames must contain those
training features in the same order. Extra serving columns are excluded from
attribution according to the checkpoint's feature schema.

## Retrieval and interchangeable methods

Without a paired reference, configure a normal-history bank:

```python
from sdk.historical_attribution import (
    NearestNormalWindowRetriever, CosineDistance, EuclideanDistance,
    LinearBlendReplacement, MultiDonorReplacement,
)

bank_resolver = NearestNormalWindowRetriever(
    normal_bank_features,
    normal_intervals=normal_intervals,
    calibration_intervals=calibration_intervals,
    source=normal_source,  # Include verified normal_evidence.
    same_source=False,    # Only for a separate normal trajectory.
)
retrieved = replace(request, resolver=bank_resolver)

# Default ranking is normalized MAE; other methods use the same interface.
cosine = replace(retrieved, resolver=replace(bank_resolver, similarity=CosineDistance()))
euclidean = replace(retrieved, resolver=replace(bank_resolver, similarity=EuclideanDistance()))

# Apply the same RM independently to up to five eligible donors.
multiple = replace(
    retrieved,
    resolver=replace(bank_resolver, top_n=5),
    replacement_strategy=MultiDonorReplacement(rule=LinearBlendReplacement(blend_points=2)),
)
```

Intervals use `[start, stop)`. Calibration rows are excluded from donors.
For the same source, use `same_source=True`; the actual SDK evaluation window
and its future are excluded. Nearest retrieval has no default distance cutoff.
Resolvers do not automatically switch; select one for your data. No alarm,
insufficient history or no eligible donor returns an unavailable result.

`top_n` has no three-donor limit. Multi-donor inference requires
`MultiDonorReplacement`; retrieval alone only supplies candidates. Each donor
starts from the original input; donors are neither averaged nor stitched.
With `top_n=N`, up to N eligible donors are analyzed automatically. If fewer
are available, only those are used; no duplicates are added to reach N.

`LinearBlendReplacement(blend_points=0)` directly inserts the donor. Custom RMs
implement `generate(context) -> list[ReplacementCondition]`; custom similarity
objects expose `name`, `direction` (`"minimize"`/`"maximize"`), and
`compare(query, candidate)`. RM2 means single-donor retrieval; RM3 evaluates
donors separately. Donor-free interpolation (RM1) is not implemented.

## Results and reporting

`attribution["conditions"]` saves every input, reconstruction, donor, audit and
metric result. `condition_count` and `distinct_donor_count` record actual counts;
a custom RM may produce multiple conditions per donor. Top-level fields describe
the first condition. Report JSON indexes metrics by condition ID, without
assuming a seed. Previously saved seed-based reports remain readable.

| Metric | Meaning |
| --- | --- |
| Historical Impact | Percentage reduction in absolute target error at the fixed event. |
| Target Reconstruction Improvement | Reduction in absolute target error, in the original unit. |
| Control-Adjusted Impact | Target improvement minus matched normal-control improvement. |
| Explanation Stability | Effect and output-time rank consistency across evaluated distinct donors. |
| Historical Contribution Share | Positive control-adjusted effects normalized across a defined candidate set. |

Missing controls/comparisons remain unavailable with reasons. Donor stability
does not establish seed/mask/boundary stability. PDF historical response shares
normalize absolute changes between reconstructions over the whole interval;
they are not independent causes or the Historical Contribution Share metric.
Measured curves retain the sensor units supplied in metadata.

Set the SDK config's `report_path` to include two attribution pages (rankings/summary
and full-width aligned timelines) before the interpretation page and save
`<report-stem>.historical.json`. Native masks/scaling remain unchanged; future
observations may remain visible. Omitting attribution preserves SDK behavior.

## Tests

```bash
cd Kumo-Anomaly
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" python -m pytest \
  -q -o addopts= -p no:cacheprovider \
  sdk/tests/test_historical_attribution.py \
  sdk/tests/test_reporting.py \
  sdk/tests/test_anomaly_analysis.py
```

PDF tests use deterministic fixtures and write `historical_attribution_test_report.pdf`
with JSON/CSV companions in pytest temporary directories.
