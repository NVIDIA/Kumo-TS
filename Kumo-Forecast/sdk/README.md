# Kumo-Forecast SDK

Programmatic entry point for running `perform_forecasting()` on pandas DataFrames. Supports multivariate forecasting, context-enhanced (DARR) predictions, and model-agnostic interpretability with horizon-specific lag attributions.

## Features

- **DataFrame-first API**: Works with pandas DataFrames; automatically detects numeric columns as features while keeping the provided timestamp + target.
- **Autoregressive horizon extension**: Automatically repeats the model prediction window when `forecast_horizon` exceeds the model's native horizon.
- **DARR mode**: Blends direct model output with kNN-based context memory, with configurable `alpha`, `k`, and `temperature`.
- **Robust preprocessing**: Converts timestamps, fills numeric NULLs with zeros, enforces minimum sequence length (`seq_len`), and standardizes input using saved standardizer metadata.
- **Column alignment**: Automatically handles datasets with different feature sets by aligning to common columns, preventing broadcasting errors.
- **Diverse output**: Produces hybrid, direct, and kNN forecasts when context is provided; otherwise returns only the direct forecast column.
- **Built-in interpretability**: Opt-in `interpretability=True` produces horizon-resolved lag attribution, semantic-flow diagnostics, latent-trajectory stability, JSON/CSV/PNG artifacts, and a self-contained PDF report with feature-axis embedding stability for multivariate inputs. Output format is selectable via `interpretability_output` (`"json"`, `"pdf"`, or `None` for both).
- **Feature-axis interpretability**: For multivariate inputs, the SDK automatically adds batched Jacobian channel-by-horizon attribution to the PDF and artifact bundle.
- **Feature-axis embedding stability**: Multivariate interpretability runs perturb one input feature at a time and report Lipschitz-style embedding sensitivity in the feature-axis JSON block, `feature_axis_embedding_stability.csv`, and a dedicated PDF page.

## Installation

The SDK is shipped with `kumo-forecast`. Install (or update) the package and dependencies using:

```bash
uv sync
```

## Quick start

### Standard inference

```python
from sdk.forecasting import ForecastingConfig, perform_forecasting
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "timestamp": pd.date_range("2023-01-01", periods=600, freq="H"),
    "target": np.sin(np.linspace(0, 4 * np.pi, 600)),
    "feature_a": np.random.randn(600),
    "feature_b": np.random.randn(600),
})

config = ForecastingConfig(
    seq_len=512,
    forecast_horizon=72,
)
forecasts = perform_forecasting(df=df, config=config)

# Result contains a `target_forecast` column with `forecast_horizon` rows
```

### YAML-backed inference config

When repeated calls need the same inference knobs, put them in YAML and pass the
file path instead of expanding a long argument list. A fully commented template
is available at `sdk/forecasting_inference_config.yaml`.

```yaml
inference:
  # Number of historical rows consumed for the input context window.
  seq_len: 512
  # Number of future rows to predict.
  forecast_horizon: 72
  # Native horizon used when training the checkpoint.
  model_horizon: 72
  # Enable cross-channel attention for multivariate forecasting.
  use_cross_channel: true
```

```python
forecasts = perform_forecasting(df=df, config="forecasting_inference_config.yaml")
```

The YAML may be either a flat mapping of `ForecastingConfig` fields or an
`inference:` section inside a larger spec. Unknown config keys raise
`ValueError` so misspellings do not silently fall back to defaults.

### DARR (context-enhanced) inference

```python
context = pd.DataFrame({
    "timestamp": pd.date_range("2022-01-01", periods=2500, freq="H"),
    "target": historical_target_series,
    "feature_a": historical_feature_a,
})

darr_config = ForecastingConfig(
    seq_len=512,
    forecast_horizon=72,
    alpha=0.2,  # 20% direct, 80% kNN
    k=64,
    temperature=0.05,
)
darr_result = perform_forecasting(df=df, config=darr_config, context_df=context)

# DARR output includes hybrid, direct, and kNN forecast columns
```

#### Context data expectations

The `context_df` you supply should contain the same `timestamp` and `target_column` as your input data, but **does not need to have identical feature columns**. The SDK automatically handles column mismatches by:

1. **Detecting differences**: Identifies when input and context datasets have different feature sets
2. **Finding common features**: Determines the intersection of feature columns between datasets  
3. **Automatic alignment**: Uses only common features for consistent predictions
4. **Clear reporting**: Provides warnings about column mismatches and alignment decisions

```python
# Input dataset with many features
df = pd.DataFrame({
    'timestamp': timestamps,
    'target': values,
    'feature_A': data_A,
    'feature_B': data_B, 
    'feature_C': data_C,
})

# Context dataset with fewer features - this is now supported!
context_df = pd.DataFrame({
    'timestamp': historical_timestamps,
    'target': historical_values,
    'feature_A': historical_A,  # Only this feature in common
})

# SDK automatically aligns to use only feature_A
result = perform_forecasting(df=df, context_df=context_df, ...)
```

**Requirements**:
- Both datasets must have the same `timestamp_column` and `target_column`
- At least one common feature column must exist (besides the target)
- Context dataset must have at least `seq_len + max(model_horizon, forecast_horizon)` rows so DARR can retrieve the full requested continuation

For this release, the blending weight `alpha` defaults to `0.01` and can be adjusted via the `alpha` parameter—the hybrid forecast uses that value to combine direct predictions with context-derived neighbors (`alpha * direct + (1 - alpha) * kNN`).

### Interpretability inference

Setting `interpretability=True` runs the loaded model on the trailing `seq_len` window and produces a horizon-resolved explanation alongside the forecast. The artifacts are written to a UTC-stamped subdirectory under `interpretability_out_dir`.

```python
from pathlib import Path

interp_config = ForecastingConfig(
    seq_len=512,
    forecast_horizon=100,
    interpretability=True,
    interpretability_output=None,                 # "json", "pdf", or None for both
    interpretability_out_dir=Path("interpretability_output"),
    interpretability_dataset_name="my_dataset.csv",
    n_lags=128,
    softmax_tau=1.0,
    integrated_gradients=True,
)
interp_result = perform_forecasting(df=df, config=interp_config)
```

`interpretability_output` selects which artifacts are written:

| Value | Files written under `<interpretability_out_dir>/run_<UTC>/` |
|-------|-------------------------------------------------------------|
| `"json"` | `forecast.csv`, `explanation.json` |
| `"pdf"` | `forecast.csv`, lag/semantic-flow artifacts, feature-axis attribution and embedding-stability artifacts for multivariate input, optional integrated-gradients artifacts, `explanation_report.pdf` |
| `None`  | All of the above |

The returned DataFrame is the explanation-aligned forecast (single forward pass, so it lines up 1:1 with the attribution matrix). PDF / heatmap require `matplotlib`; if it's missing, those steps are skipped with a warning while JSON output continues to work.

If the input DataFrame has more than one numeric column, the feature-axis decomposition runs automatically: the report gains channel-by-horizon attribution and feature-axis embedding-stability pages. By default attribution is measured in latent space (output-agnostic); set `channel_output_aware=True` to attribute the target channel's forecast directly.

### Missingness-aware forecasting

By default NULLs are zero-filled before the base forecaster runs. Set
`handle_missingness=True` to route the call to the missingness-aware
missingness-aware model (`impute/`, Backbone-LF+ CRS) instead:

```python
config = ForecastingConfig(
    target_column="OT",
    forecast_horizon=96,
    handle_missingness=True,
    # impute_ckpt defaults to the released checkpoint hf://nvidia/Kumo-Forecast/kumo-forecast-1.2.0
)
forecasts = perform_forecasting(df=df_with_nans, config=config)
```

or in YAML:

```yaml
inference:
  handle_missingness: true
  impute_ckpt: hf://nvidia/Kumo-Forecast@main/kumo-forecast-1.2.0   # optional; pin a revision in production
```

#### Where the impute checkpoint comes from

`impute_ckpt` accepts:

- `None` (default): the released checkpoint, `hf://nvidia/Kumo-Forecast/kumo-forecast-1.2.0`.
- `hf://<org>/<repo>[@revision][/subfolder]`: a Hugging Face Hub location, e.g. `hf://nvidia/Kumo-Forecast@<tag-or-commit>/kumo-forecast-1.2.0`. Only `best_model.pt` and the config (`config_base.json`, or `config.json`) in that folder are downloaded, into the standard Hugging Face cache (`~/.cache/huggingface/hub`, or `HF_HOME` / `HF_HUB_CACHE`). They are resolved once per process and reused.
- A local `.pt` file, or a folder containing one plus `config.json` or `config_base.json` (e.g. a training run's output).

Private or gated repos use your Hugging Face login (`huggingface-cli login`) or `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`. Set `local_files_only=True` to run offline from the cache after one online download. Pin `@revision` (a tag or commit) in production so a repo update can't change forecasts underneath you.

To publish a new impute checkpoint, upload `best_model.pt` and an inference-only `config_base.json` (the fields the loader reads: model type and sizes, `seq_len` / `pred_len` / `patch_len`, `cross_channel_n_heads`, module switches, `target_col`, and `meta.n_channels` / `meta.channels` / `meta.source_union_channels`) into one folder of the repo. The released `kumo-forecast-1.2.0/` folder contains exactly these two files. Add `meta.dataset_metadata` to `config_base.json` only if you want `impute_normalization="checkpoint"` to be available.

What happens on this path (it mirrors how the missingness model's training pipeline prepares data):

1. **Channel alignment by name.** The target goes to the checkpoint's target slot and every feature to the slot its column name had in training (read from `config.json`: `meta.source_union_channels` for pooled multi-CSV training, `meta.channels` for single-CSV training). Training channels missing from your input are fed as padding (`valid_channel_mask=0`), exactly like pooled training pads sources that lack a column. Outputs are mapped back to your column names.
2. **Fixed, training-style normalization.** Each channel is standardized with one fixed scaler fit on *observed* values only (the same maths as the training-split `ChannelObservedScaler`) — never re-fit per window, so the adaptive normalizer's learned priors keep their training meaning.
3. Per-channel / per-timestep masks are built from the real NULLs **before** anything is filled; gaps are then zero-filled in scaled space.
4. The window's missing rate (over the valid channels) is passed to the model so the adaptive normalizer engages at high missingness (> 50%).
5. Forecasts are mapped back to original units. The output format is identical to the default path (`{target_column}_forecast`, or every input channel with `return_all_channels=True`).

Impute-specific settings:

| Field | Default | Meaning |
|-------|---------|---------|
| `impute_ckpt` | `None` | Checkpoint location: `None` = released `hf://nvidia/Kumo-Forecast/kumo-forecast-1.2.0`; `hf://org/repo[@rev][/subfolder]`; or a local `.pt` / folder with `config.json` or `config_base.json`. See "Where the impute checkpoint comes from". |
| `impute_normalization` | `"history"` | `"history"`: fit the scaler on the observed rows of `df` — pass real history, not just the last `seq_len` rows (a warning is logged below `2 * seq_len` rows). `"checkpoint"`: reuse the training standardizer saved in the checkpoint (for data from a training source dataset). `"provided"`: use `impute_scaler_stats`. |
| `impute_history_rows` | `None` | `"history"` only: fit the scaler on the last N rows of `df` (must be `>= seq_len`), so very old regimes don't skew it and results don't depend on how much history a caller sends. `None` uses every row. |
| `impute_scaler_stats` | `None` | `"provided"` only: stored reference statistics `{"mean": {column: value}, "std": {column: value}}` for every input column (std > 0). Entries for columns not in the input are ignored with a warning. |
| `impute_source_dataset` | `None` | With `"checkpoint"` normalization, which training source's standardizer to use, by CSV stem (e.g. `"ETTh1"`). Optional when the checkpoint has a single source. |
| `impute_channel_alignment` | `"name"` | `"name"`: align to the training schema; unknown columns raise `ValueError`. `"positional"`: feed channels in input order (channel identities and per-channel priors then follow position — use only for data whose columns don't match the training names). |

> **Custom datasets:** the default `impute_channel_alignment="name"` only accepts feature columns whose names
> match the checkpoint's training channels (for the released ETT checkpoint: `HUFL`, `HULL`, `LUFL`, `LULL`,
> `MUFL`, `MULL`). For any other dataset, pass `impute_channel_alignment="positional"`: the `target_column` is Y
> (slot 0) and the remaining numeric columns are the X channels, fed in the order given. Keep that column order
> fixed across requests, and send a temporarily missing sensor as an all-NULL column instead of dropping it, so
> every other channel keeps its position.
>
> ```python
> config = ForecastingConfig(
>     target_column="sales",
>     handle_missingness=True,
>     impute_channel_alignment="positional",
> )
> ```

For deployments, compute reference statistics once on a clean period, store them, and reuse them on every
request (the same idea as the training-split scaler, for series the checkpoint wasn't trained on):

```python
from sdk import ForecastingConfig, fit_impute_scaler_stats, perform_forecasting

stats = fit_impute_scaler_stats(reference_df, target_column="OT")   # observed values only; plain JSON/YAML
config = ForecastingConfig(
    target_column="OT",
    handle_missingness=True,
    impute_normalization="provided",
    impute_scaler_stats=stats,
)
forecasts = perform_forecasting(df=latest_rows, config=config)   # only the last seq_len rows are needed
```

Other rules:

- `seq_len` must match the checkpoint. If `seq_len` is left at the SDK default (`512`), the checkpoint's value is used automatically; any other mismatch raises `ValueError`.
- `forecast_horizon` longer than the checkpoint's `pred_len` is produced autoregressively (predictions are appended as observed history; the scaler stays fixed).
- A feature named like the checkpoint's target channel while `target_column` is something else raises `ValueError`.
- With `"positional"` alignment on a CRS checkpoint, channels beyond the trained channel-embedding size share one identity embedding (a warning is logged).
- The target must have at least one observed value in the input window.
- Not supported yet together with DARR (`context_df`) or `interpretability=True`; both raise `ValueError`.

## Function signature

```python
perform_forecasting(
    df: pd.DataFrame,
    *,
    config: ForecastingConfig | str | Path | None = None,
    context_df: pd.DataFrame | None = None,
) -> pd.DataFrame
```

### Key parameters

| Parameter | Description |
|-----------|-------------|
| `df` | Input DataFrame containing time series data |
| `config` | `ForecastingConfig`, YAML path, or `None` for defaults. The YAML can be flat or use a top-level `inference` section |
| `context_df` | Enables DARR mode when supplied |

All model, data-column, DARR, output, and interpretability settings are fields
on `ForecastingConfig` and are documented in the commented YAML template.

| Parameter | Description |
|-----------|-------------|
| `interpretability` | Master switch. When `True`, the SDK skips the standard / DARR inference path and runs the interpretability explanation pipeline against the loaded model |
| `interpretability_output` | `"json"` for explanation JSON only, `"pdf"` for the heatmap + PDF bundle, `None` for both. Invalid values raise `ValueError` |
| `interpretability_out_dir` | Parent directory for the run subfolder; created if missing |
| `interpretability_run_name` | Override the auto-stamped `run_<UTC>` folder name |
| `interpretability_top_k` | How many top lag steps per horizon to render in the PDF table (default `5`) |
| `interpretability_dataset_name` | Free-form label embedded in the JSON metadata and the PDF cover page |
| `n_lags` | Number of past steps the lag-attribution matrix resolves (default `128`) |
| `softmax_tau` | Temperature applied when softmaxing scores into per-horizon attribution |
| `channel_output_aware` | Use target-specific directional input-Jacobian effects for feature-axis attribution (default `False`) |
| `integrated_gradients` | Add embedding integrated-gradients attribution to the JSON/PDF report and export its CSV/PNG artifacts |
| `integrated_gradients_baseline` | IG reference input; `"noise"` is the default and avoids the constant-baseline degeneracy of instance normalization |
| `integrated_gradients_steps` | Number of midpoint integration steps (default `64`) |
| `integrated_gradients_n_baselines` | Number of noise baselines averaged as Expected Gradients (default `1`) |
| `integrated_gradients_internal_batch_size` | Maximum interpolation points embedded per batch; `None` processes all steps together |
| `integrated_gradients_reduce` | Embedding scalar objective: `"l2"`, `"sum"`, or `"mean"` |
| `integrated_gradients_grad_through_norm` | Include RevIN statistics in the attribution gradient while leaving forward outputs unchanged |
| `return_all_channels` | When `True`, the result contains one `{column}_forecast` column per processed channel (target column first, then the remaining numeric features) instead of only `{target_column}_forecast`, including in interpretability mode |

## Preprocessing expectations

- Timestamp column must be parseable by pandas and free of NULLs.
- Target column must be numeric; NULLs are filled with zeros (or, with `handle_missingness=True`, passed to the missingness-aware model as masks — see above).
- All numeric features are automatically included; NULLs become zeros (same `handle_missingness` option).
- Numeric feature columns should match the set and order the checkpoint/standardizer was trained with; a different order silently misattributes forecasts, and a different column count fails at standardization with a broadcasting error.
- Input length must be at least `seq_len`; otherwise `ValueError` is raised.

## Outputs

- **Standard mode**: `{target_column}_forecast` containing the requested `forecast_horizon` predictions with timestamps inferred from the input frequency.
- **DARR mode**: Returns hybrid predictions in `{target_column}_forecast` (direct and kNN components are computed internally).
- **Interpretability mode**: Returns the explanation-aligned forecast in `{target_column}_forecast` and writes the artifact bundle to `<interpretability_out_dir>/run_<UTC>/`.

With `return_all_channels=True`, the single `{target_column}_forecast` column is replaced by one `{column}_forecast` column per processed channel (target column first, then the remaining numeric features in input-column order). Each forward pass already predicts every channel, so this avoids re-running the SDK once per column when downstream code needs multivariate forecasts. Autoregressive extension (`forecast_horizon > model_horizon`) and DARR blending apply to every channel the same way they apply to the target column; in DARR mode with mismatched input/context columns, only the aligned common columns are forecast (see Column Mismatch Handling).

> **Channel order matters.** The checkpoint and standardizer are trained against a specific channel layout. At inference time the numeric columns of the input DataFrame should match the training-time set and order. With the same columns in a different order the model still runs, but the forecasts can be semantically wrong — especially visible with `return_all_channels=True`, where every channel becomes an output column. With extra or missing numeric columns the channel count no longer matches the standardizer, and the call fails at standardization with a broadcasting error before inference runs.

### Output DataFrame structure

| Column | Description |
|--------|-------------|
| `{timestamp_column}` | Forecasted timestamps starting after the last input timestamp |
| `{target_column}_forecast` | Predicted values for the forecast horizon |
| `{column}_forecast` | (Only with `return_all_channels=True`) Predicted values for each additional numeric feature channel |

### Interpretability artifact bundle

When `interpretability=True`, the run directory contains the following files (selected by `interpretability_output`):

| File | Written when | Contents |
|------|--------------|----------|
| `forecast.csv` | json, pdf, both | The returned forecast DataFrame |
| `explanation.json` | json, both | Forecast + full explanation payload, including lag×horizon values, semantic-flow diagnostics, trajectory stability, a multivariate `feature_axis` block containing attribution and `embedding_stability` results, optional Integrated Gradients, and dataset metadata |
| `lag_horizon_attributions.csv` | pdf, both | Wide K×H attribution matrix |
| `lag_horizon_long.csv` | pdf, both | Tidy `(lag, horizon, attribution[, score])` table |
| `lag_horizon_heatmap.png` | pdf, both *(needs matplotlib)* | Visual heatmap, viridis cmap |
| `semantic_flow.csv` | pdf, both | Tidy `(transition_index, segment, flow_magnitude)` table, where `segment` is `history` for transitions fully inside the input window, `forecast` for transitions whose window extends into the model-generated future, and `tail` for any trailing transitions outside both segments |
| `channel_horizon_attributions.csv` | multivariate pdf, both | Feature-axis `C×H` attribution matrix |
| `channel_horizon_heatmap.png` | multivariate pdf, both *(needs matplotlib)* | Feature-axis channel-by-horizon heatmap |
| `feature_axis_embedding_stability.csv` | multivariate pdf, both | Complete feature-level embedding-stability report: mean, p50, p95, and max Lipschitz-style ratios, retained trial counts, sampled-window count, and the unperturbed latent-step reference scale |
| `integrated_gradients_attributions.csv` | `integrated_gradients=True` with pdf or both | Signed embedding IG value for every channel and context position |
| `integrated_gradients_channel_summary.csv` | `integrated_gradients=True` with pdf or both | Signed effect, absolute effect, and absolute share by channel |
| `integrated_gradients_heatmap.png` | `integrated_gradients=True` with pdf or both *(needs matplotlib)* | Signed channel-by-context IG heatmap |
| `explanation_report.pdf` | pdf, both *(needs matplotlib)* | Multi-page report covering forecast, lag×horizon attribution, semantic flow, latent stability, feature-axis attribution, feature-axis embedding stability, and optional embedding Integrated Gradients |

The run directory path is printed on stdout when the call completes.

## Error handling

### Column Mismatch Handling

The SDK automatically detects and handles column mismatches between input and context datasets:

```python
# Example: Input has 7 features, context has 3 features
# Before fix: "operands could not be broadcast together with shapes (1,7,24) (1,3,24)"
# After fix: Automatic alignment using common features

Warning: Column mismatch detected between input and context datasets
  Input dataset columns: ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'OT']
  Context dataset columns: ['HULL', 'MULL', 'LUFL']
  Common columns (input order): ['HULL', 'MULL', 'LUFL']
  Using only common columns for consistent predictions: ['HULL', 'MULL', 'LUFL']
```

**Behavior**:
- **Automatic detection**: Identifies when datasets have different feature sets
- **Order-sensitive alignment**: Realignment also triggers when both datasets contain the same columns in a different order — otherwise the direct and kNN predictions would blend mismatched channels
- **One canonical order**: Both temporary datasets are rewritten with the common feature columns in input-DataFrame order (target column first), so the caller-facing channel layout is preserved for the channels that remain
- **Clear warnings**: Reports what columns are being used and why
- **Graceful fallback**: Prevents cryptic NumPy broadcasting errors

### Common Error Scenarios

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: No common numeric columns found between input and context datasets` | Datasets share no feature columns (only target) | Ensure context dataset has at least one feature column in common with input |
| `ValueError: Shape mismatch between direct and kNN predictions` | Input and context columns could not be aligned | Check that both datasets have valid numeric columns |
| `ValueError: DataFrame has X rows but seq_len requires at least Y rows` | Insufficient data points | Provide more data or reduce `seq_len` |
| `ValueError: Context DataFrame has X rows but requires at least Y rows` | Context dataset too small | Context needs `seq_len + max(model_horizon, forecast_horizon)` rows minimum |
| `ValueError: forecast_horizon must be <= 512` | Forecast horizon too large | Reduce `forecast_horizon` or make multiple calls |
| `ValueError: Target column 'X' not found in DataFrame` | Missing target column | Verify column name or set `target_column` on `ForecastingConfig` |
| `ValueError: interpretability_output must be one of None, 'json', 'pdf'` | Bad value for the artifact selector | Use `None`, `"json"`, or `"pdf"` |
| `Interpretability PDF report skipped: matplotlib is not installed.` | PDF/heatmap path attempted without matplotlib | `uv add matplotlib`, or set `interpretability_output="json"` |

### Data Validation

Common safeguards raised as `ValueError` include:

- Missing or NULL timestamp column
- Unparseable timestamps  
- Missing target column
- Non-numeric target data
- Too few rows for the configured `seq_len`
- Invalid `model_horizon` or `forecast_horizon` (must be positive)
- `forecast_horizon` exceeds maximum limit of 512
- **NEW**: No common feature columns between input and context datasets

## Examples and tests

See `sdk/tests/test_forecasting.py` for unit test coverage with mockers and `sdk/quick_example.py` for an end-to-end script.
The missingness-aware path is covered by `sdk/tests/test_imputation.py` (real-checkpoint tests are opt-in via `KUMO_IMPUTE_CKPT`, `KUMO_IMPUTE_HF`, `KUMO_IMPUTE_CSV`) and demonstrated in `examples/missingness_quickstart.py`.
