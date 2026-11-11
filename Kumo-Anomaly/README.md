# Kumo-Anomaly

Inference uses DPM-Solver++ exclusively. Training, noise schedules, and checkpoint
weights are unchanged; existing checkpoints do not require retraining.

The low-level inference APIs retain `dpm_steps=20` but no longer accept
`use_dpm_solver`. Remove that argument from existing calls. The DDPM, DDIM, and
DDPM intermediate-step sampling methods are no longer exposed.

A package for anomaly detection using Kumo-Anomaly diffusion models.

## Features

- **Diffusion-based Anomaly Detection**: Uses advanced diffusion models for robust anomaly detection
- **Adaptive Thresholding**: Implements SCS (Segmented Confidence Sequences) and MACS (Multi-Scale Adaptive Confidence Segments) methods
- **Multi-GPU Support**: Automatic multi-GPU inference with shared memory optimization
- **Fast Inference**: Supports DPM-Solver for 50-100x speedup over standard diffusion
- **Preprocessing Pipeline**: Complete TSB-AD compatible preprocessing with domain adaptation
- **Model-agnostic Explainability**: Attributes each detected anomaly's MAE to its largest reconstruction errors
- **Auto-download from Hugging Face**: Pretrained weights are fetched automatically from [`nvidia/Kumo-Anomaly`](https://huggingface.co/nvidia/Kumo-Anomaly) on first use
- **Simple Structure**: Organized as modules without Python package complexity - easy to use and modify

## Quick Start

**Requirements**: Python 3.12+ and uv package manager

First, set up the environment:
```bash
# Verify Python version
python --version  # Should be 3.12.0+

# Install and sync dependencies
uv sync

# Or run directly without activation
uv run python your_script.py
```

Copy [sdk/sdk_config.yaml](sdk/sdk_config.yaml) and pass it to the SDK.
The [explainability and reporting](#explainability-and-reporting) settings below
control the optional explanations and PDF output.

```python
import os
import sys

import pandas as pd

sys.path.append("/path/to/Kumo-Anomaly")
from sdk.anomaly_analysis import perform_anomaly_analysis_with_diffusion

df = pd.read_csv("your_data.csv")

# Perform anomaly detection — omit model_path/model_config_path to auto-download
# the pretrained weights from Hugging Face (nvidia/Kumo-Anomaly).
results = perform_anomaly_analysis_with_diffusion(
    df=df,
    threshold_strategy="scs",  # or "macs"
    # model_path="path/to/your/model.pth",         # optional; defaults to final_model.pth
    # model_config_path="path/to/config.yaml",     # optional; defaults to curriculum_medium.yaml
    nsample=15,
    sdk_config="path/to/sdk_config.yaml",
    feature_embedding_mode="shared_mean_norm_matched_zero_pad",  # or "positional"
)
print(f"Detected {results['Anomaly'].sum()} anomalies")
```

`feature_embedding_mode` defaults to `shared_mean_norm_matched_zero_pad`, which
shares a norm-matched checkpoint embedding across active features and zeros only
SDK-identified padding positions. Use `positional` to retain the checkpoint's
original position-specific feature embeddings.

The same embedding mode is used for the original inference and every repair
re-scoring pass when reconstruction diagnosis is enabled.

## Explainability and reporting

Part 1 explains reconstruction errors and tests repair hypotheses. Part 2 measures
how reconstruction responds to replacing earlier observations with normal history.
Both describe model behavior and sensitivity; neither establishes physical causation.
They are optional and can be used together in the same SDK call and report.

Use YAML or an `ADDiffusionConfig` object for part 1 and reporting settings:

```yaml
sdk:
  explain: true
  explanation_top_k: 3
  diagnose_reconstruction: true
  report_path: output/pdf/kumo-anomaly-report.pdf
  timestamp_column: timestamp
  ground_truth_column: GT
  report_title: Kumo-Anomaly Report
  report_explanation_csv_path: output/pdf/kumo-anomaly-report_explanations.csv
  report_max_pages: 10
  report_consolidated_top_k: 5
```

### Part 1: Reconstruction attribution and diagnosis

`explain=True` adds `TopContributors`, `ContributionShares`, `ExplanationCoverage`,
and `ExplanationMethod` from the existing target and reconstruction, without another
inference pass. Directly mapped features retain their original names. Padding is
excluded from MAE, L2, thresholding, and shares; low-level results expose
`valid_feature_mask` and `score_feature_count` while retaining full-width `target`
and `recon`. PCA or engineered features use `component_N` labels.

`diagnose_reconstruction=True` requires `explain=True` and directly mapped inputs
without an external preprocessing model. Four additional inference passes test
level-shift, trend-change, transient-spike, and variance-burst repairs, returning
`LikelyReconstructionIssue`, `RepairImpact`, `DiagnosticConfidence`, and
`ExplanationConsistency`.

### Part 2: Historical attribution

Pass `historical_attribution=HistoricalAttributionRequest(...)` directly to the SDK;
it is independent of the YAML explanation switches and runs additional inference.
Supply a verified normal reference aligned to the input's feature order, rows,
and units, excluding timestamp and label columns from the reference feature matrix:

```python
from sdk.anomaly_analysis import ADDiffusionConfig, perform_anomaly_analysis_with_diffusion
from sdk.historical_attribution import AlignedNormalReference, HistoricalAttributionRequest

request = HistoricalAttributionRequest(
    resolver=AlignedNormalReference(
        normal_features,  # Verified normal data aligned to the input feature matrix.
        source={"id": "verified_normal_reference"},
        alignment_verified=True,
    ),
)
results = perform_anomaly_analysis_with_diffusion(
    df,
    threshold_strategy="scs",
    historical_attribution=request,
    sdk_config=ADDiffusionConfig(explain=True, report_path="output/pdf/kumo-anomaly-report.pdf"),
)
attribution = results.attrs["historical_attribution"]
```

Defaults select the first SDK anomaly and the feature with its largest SDK-scaled
residual, replacing a block with `length=20`, `gap=1`, `blend_points=2`, and
`scope="target"`. Override `event_position` and `target` with an actual alarm and
original feature key; `scope="all"` or `scope="signals", signals=[...]` controls
which features are replaced. Supply sensor units through `units={feature: unit}`.
For an explicitly labelled event, set `event_source="dataset_label"` with a positive
binary ground-truth label, explicit position, and target.

Attribution requires direct feature mapping; PCA and external preprocessing models
are unsupported. Fine-tuned checkpoints retain their training feature order,
min/max scaling, and window settings. Reference and donor frames must use that
training schema; extra serving columns are excluded from attribution.

#### Normal-history retrieval and multiple donors

Without an aligned reference, use a declared normal-history bank. Retrieval ranks
eligible donors separately from acceptance; normalized MAE is the default ranking
and has no distance cutoff. Cosine and Euclidean distances are also available:

```python
from dataclasses import replace
from sdk.historical_attribution import (
    CosineDistance, EuclideanDistance, LinearBlendReplacement,
    MultiDonorReplacement, NearestNormalWindowRetriever,
)

retriever = NearestNormalWindowRetriever(
    normal_bank_features,
    normal_intervals=normal_intervals,
    calibration_intervals=calibration_intervals,
    source=normal_source,  # Include an identifier and verified normal_evidence.
    same_source=False,    # Only for a separate normal trajectory.
    top_n=5,
)
# Optional: retriever = replace(retriever, similarity=CosineDistance())  # or EuclideanDistance()
request = replace(
    request,
    resolver=retriever,
    replacement_strategy=MultiDonorReplacement(rule=LinearBlendReplacement(blend_points=2)),
)
```

Intervals are `[start, stop)`; calibration rows cannot be donors. Set
`same_source=True` for the same trajectory to exclude the event's model window
and future. Resolvers do not switch automatically. No alarm, insufficient history,
or no eligible donor returns an unavailable result with its reason.

`top_n` supplies up to that many eligible donors; `MultiDonorReplacement` is required
to evaluate them independently. Each condition starts from the original input,
without averaging donors, stitching features, or duplicating donors to fill a quota.
`LinearBlendReplacement(blend_points=0)` inserts a donor directly. Custom replacements
implement `generate(context) -> list[ReplacementCondition]`; custom similarities
expose `name`, `direction` (`"minimize"` or `"maximize"`), and `compare(query, candidate)`.

#### Historical results

| Metric | Meaning |
| --- | --- |
| Historical Impact | Percentage reduction in absolute target error at the fixed event. |
| Target Reconstruction Improvement | Reduction in absolute target error in the original unit. |
| Control-Adjusted Impact | Target improvement minus matched normal-control improvement. |
| Explanation Stability | Effect and output-time rank consistency across distinct donors. |
| Historical Contribution Share | Positive control-adjusted effects normalized across a defined candidate set. |

Missing controls or comparisons remain unavailable with reasons. Donor stability
does not establish seed, mask, or boundary stability. The PDF's historical response
shares normalize reconstruction changes over the replaced interval; they are not
the Historical Contribution Share metric or independent causal contributions.

`attribution["conditions"]` records each input, reconstruction, donor, audit, and
metric result. `condition_count` and `distinct_donor_count` record actual counts;
top-level fields describe the first condition. Report metrics use condition IDs,
and previously saved seed-based reports remain readable. Native masks and scaling
are retained, so visible future observations may still condition reconstruction.

### Shared PDF and data outputs

Set `report_path` to generate a PDF with original signals, detected anomalies,
MAE, and available ground truth. Reporting or historical attribution excludes
configured/inferred timestamp and ground-truth columns from model features while
preserving them in the returned DataFrame. Datetime indices and numeric time columns
are supported; without time metadata, positions are zero-based row numbers. PDF and
historical JSON use the same time axis, and historical curves retain supplied units.

With `explain=True`, the PDF adds contributor-share graphs and exports every detected
anomaly to a companion CSV. If detailed charts exceed `report_max_pages` (default 10),
they become one MAE-weighted top-contributors page; the CSV remains complete.
Historical attribution adds ranking and timeline pages, or one unavailable-result
page, within the same page budget, plus `<report-stem>.historical.json`. A page budget
too small for the required sections raises an error. Omitting the attribution request
preserves the existing inference path; omitting `report_path` disables PDF generation.

Render an existing result or CSV after inference with the same reporter:

```python
from sdk.reporting import generate_anomaly_detection_report

generate_anomaly_detection_report(results, "output/pdf/kumo-anomaly-report.pdf")
```

Pass `historical_attribution=attribution` to include saved historical results.
The reporter infers common timestamp and label names such as `timestamp`, `GT`,
`ground_truth`, `is_anomaly`, and `label`.

## Pretrained Weights

The pretrained Kumo-Anomaly model is hosted on Hugging Face:

- **Repository**: [`nvidia/Kumo-Anomaly`](https://huggingface.co/nvidia/Kumo-Anomaly)
- **Checkpoint**: `final_model.pth`
- **Config**: `curriculum_medium.yaml`

### Auto-download (recommended)

The SDK downloads these two files into the current working directory on first use,
so the simplest path is to just call `perform_anomaly_analysis_with_diffusion`
(or `inference_ad_tesseract2` / `inference_ad_tesseract2_mp`) **without** passing
`model_path`/`config_path`. Subsequent runs will reuse the local copies.

You can also pre-download them explicitly:

```python
from sdk.inference_ad import download_model_weights

model_path, config_path = download_model_weights(
    model_path="weights/final_model.pth",          # optional custom location
    config_path="weights/curriculum_medium.yaml",  # optional custom location
)
```

Or via the example CLI:

```bash
uv run python examples/quick_example.py --download-weights
```

### Hugging Face authentication

Weights are published on the public [`nvidia/Kumo-Anomaly`](https://huggingface.co/nvidia/Kumo-Anomaly) repository and download without authentication. If a download fails with `401`/`403`, accept the model license on the repo page or authenticate before retrying:

```bash
# Option A: interactive login (writes ~/.cache/huggingface/token)
uv add "huggingface_hub[cli]"
huggingface-cli login

# Option B: non-interactive via env var (great for CI and containers)
export HUGGINGFACE_HUB_TOKEN="hf_xxx_your_token_here"
```

You can create a token at <https://huggingface.co/settings/tokens>. A read-only
token is sufficient for downloading weights.

If you see a `401`/`403` error from `download_model_weights`, accept the model
license on the repo page and, if needed, log in with a Hugging Face token:

## Installation

### Prerequisites
- Python 3.12 or higher
- `uv` package manager (install from https://docs.astral.sh/uv/)

### From Source
```bash
# Clone or download the Kumo-Anomaly directory
cd Kumo-Anomaly

# Install dependencies
uv sync

# Run example
uv run python examples/quick_example.py
```

### Development Setup
```bash
# Install with dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Run linting
uv run ruff check .
```

Run the focused explainability and reporting regressions with:

```bash
uv run pytest sdk/tests/test_historical_attribution.py \
  sdk/tests/test_historical_sdk_integration.py sdk/tests/test_reporting.py \
  sdk/tests/test_anomaly_analysis.py -q
```

PDF tests use deterministic fixtures and write reports with JSON/CSV companions
in pytest temporary directories.

## Package Structure

```
Kumo-Anomaly/
├── sdk/                        # Main inference functions
│   ├── anomaly_analysis.py     # Main API function
│   ├── explainability.py       # Reconstruction-error contribution explanations
│   ├── historical_attribution.py # Historical replacement, metrics, and PDF pages
│   ├── inference_ad.py         # Core inference engine
│   ├── inference_worker.py     # Multi-GPU worker
│   ├── reporting.py            # PDF and companion CSV report generation
│   └── thresholds.py          # Threshold strategies (SCS, MACS)
├── models/                     # Diffusion model implementations
│   ├── main_model.py          # TSDiffuser_Generic model
│   ├── diff_models.py         # Diffusion model components
│   └── utils.py               # Model evaluation utilities
├── utils/                      # Utility functions and tools
│   ├── tsb_ad_preprocessor.py # Data preprocessing
│   ├── adaptive_normalizer.py # Distribution-aware normalization
│   ├── json_utils.py          # Model loading/saving
│   ├── adaptive_threshold.py  # SCS and MACS implementations
│   └── dpm_solver_pytorch.py  # DPM-Solver for fast inference
├── examples/                   # Examples and sample data
│   ├── quick_example.py       # Complete usage example
│   ├── finetune_example.py    # CSV fine-tuning example
│   └── datasets/              # Sample datasets for testing
├── README.md                  # This file
├── pyproject.toml            # Project configuration
└── uv.lock                   # Dependency lock file
```

## Dataset Usage

Supports both synthetic datasets (auto-generated with ground truth) and custom CSV files. Use `--dataset-path` to specify your own data.

📂 **See [examples/datasets/README.md](examples/datasets/README.md) for detailed dataset usage, format requirements, and examples.**

## Quick Start Example

Run the included example to get started:

```bash
cd Kumo-Anomaly
uv run python examples/quick_example.py --help

# Auto-downloads final_model.pth + curriculum_medium.yaml from Hugging Face
# on first run, then performs anomaly detection on synthetic data.
uv run python examples/quick_example.py

# Use your own dataset for anomaly detection:
uv run python examples/quick_example.py --dataset-path /path/to/your/data.csv

# Pre-download the weights only (e.g. to warm the cache):
uv run python examples/quick_example.py --download-weights

# Use a checkpoint you already have locally:
uv run python examples/quick_example.py --model-path /path/to/final_model.pth

# Combine custom dataset with custom model:
uv run python examples/quick_example.py --dataset-path data.csv --model-path model.pth
```

This example demonstrates:
- Auto-downloading pretrained weights from Hugging Face (`nvidia/Kumo-Anomaly`)
- Loading time series data (synthetic generation or custom CSV files)
- Running anomaly detection with diffusion models
- Applying adaptive thresholding (SCS/MACS)
- Using both synthetic and real-world datasets
- Evaluating results against ground truth

## Fine-tuning

Use `examples/finetune_example.py` to fine-tune Kumo-Anomaly on normal windows from your own data. The CSV should contain mostly normal behavior. Numeric feature columns are used for training; use `--timestamp-col`, `--label-col`, and `--drop-cols` to remove metadata columns from the feature matrix.

```bash
uv run python examples/finetune_example.py \
  --csv /path/to/normal_training_data.csv \
  --timestamp-col timestamp \
  --label-col is_anomaly \
  --epochs 10 \
  --batch-size 16 \
  --lr 1e-5 \
  --output-dir artifacts/finetune_my_data
```

When using the default temporal validation split, the validation partition must contain at least `--window-length` rows. The default `--val-ratio 0.3` works with the bundled 500-row sample dataset and default `--window-length 100`. If you lower `--val-ratio`, either increase the dataset size, pass a separate `--val-csv`, or reduce `--window-length`.

To use a separate validation file instead of a temporal split:

```bash
uv run python examples/finetune_example.py \
  --csv /path/to/train_normal.csv \
  --val-csv /path/to/val_normal.csv \
  --timestamp-col timestamp \
  --epochs 10 \
  --output-dir artifacts/finetune_my_data
```

The output directory contains:

- `best_finetuned_model.pth` - best validation checkpoint, compatible with the SDK inference loader
- `final_finetuned_model.pth` - final epoch checkpoint
- `metrics.json` - training and validation losses
- `finetune_config.yaml` - model configuration used for fine-tuning

Fine-tuned checkpoints contain the fitted feature-column order, min-max statistics, PCA projection (when used),
scale factor, and window settings. The SDK replays that preprocessing automatically instead of fitting a new
transform on each inference batch. Extra columns are ignored, while missing training features produce a clear error.

Single-device and multi-GPU inference use the same saved window length and mask split. Checkpoint preprocessing
metadata takes precedence, followed by saved fine-tuning arguments and then the model dataset config
(defaults: window length 100, split 4). An explicit `preprocess_model_dir` overrides the feature transform,
while retaining these saved window settings. Fine-tuning writes effective CLI overrides to both checkpoints
and `finetune_config.yaml`.

Run anomaly detection with the fine-tuned checkpoint. The SDK selects the feature columns recorded during fine-tuning;
dropping timestamp, label, and other metadata columns explicitly is still recommended for clarity.

```python
analysis_df = df.select_dtypes(include="number").drop(columns=["is_anomaly"], errors="ignore")

results = perform_anomaly_analysis_with_diffusion(
    df=analysis_df,
    threshold_strategy="scs",
    model_path="artifacts/finetune_my_data/best_finetuned_model.pth",
    model_config_path="artifacts/finetune_my_data/finetune_config.yaml",
    nsample=15,
)
```

## System Requirements

### Hardware
- **CPU**: Any modern multi-core processor
- **Memory**: 8GB+ RAM recommended (16GB+ for large datasets)
- **GPU**: Optional but recommended (NVIDIA with CUDA support)

### Software
- **Python**: 3.12 or higher (required for modern typing features)
- **Operating System**: Linux, macOS, or Windows

## Python Dependencies

Core dependencies (see `pyproject.toml` for exact versions):
- `torch>=2.7.0` - Deep learning framework (with CUDA support if GPU available)
- NumPy, Pandas, SciPy
- scikit-learn (for PCA and normalization)
- PyYAML (for configuration files)
- tqdm (for progress bars)
- psutil (for system resource information)
- huggingface_hub (for model downloads)
- matplotlib (for PDF reports)

Development dependencies:
- pytest (testing)
- ruff (linting and formatting)

All dependencies are managed automatically by `uv`.

## Usage

### Basic Usage

The main entry point is the `perform_anomaly_analysis_with_diffusion` function:

```python
import sys, os
sys.path.append('/path/to/Kumo-Anomaly')
from sdk.anomaly_analysis import perform_anomaly_analysis_with_diffusion
```

### Parameters

- **df** (pd.DataFrame): Input time series data (all columns must be numeric)
- **threshold_strategy** (str): "scs" or "macs" for adaptive thresholding
- **model_path** (str|Path, optional): Path to the Kumo-Anomaly model checkpoint.
  If omitted or the file doesn't exist, `final_model.pth` is auto-downloaded from
  `nvidia/Kumo-Anomaly` on Hugging Face.
- **config_path** (str|Path, optional): Path to the model config YAML. If omitted
  or missing, `curriculum_medium.yaml` is auto-downloaded from the same repo.
- **nsample** (int): Number of diffusion samples (default: 15)
- **preprocess_model_dir** (str|Path): Optional preprocessing model directory
- **sdk_config** (ADDiffusionConfig|str|Path, optional): Explainability and
  reporting configuration object or path to an SDK YAML file. The default
  disables explanations, reconstruction diagnosis, and PDF generation.
- **historical_attribution** (HistoricalAttributionRequest, optional): Normal
  reference, event, and replacement settings for historical attribution. Results
  are returned in `DataFrame.attrs["historical_attribution"]`.

### Advanced Usage

For direct access to inference functions:

```python
import sys, os
sys.path.append('/path/to/Kumo-Anomaly')
from sdk.inference_ad import inference_ad_tesseract2, inference_ad_tesseract2_mp

# Single GPU inference (auto-downloads weights if needed)
results = inference_ad_tesseract2(
    data=df,
    # model_path omitted -> final_model.pth auto-downloaded
    nsample=30,
    dpm_steps=20,
)

# Multi-GPU inference (if available)
results = inference_ad_tesseract2_mp(
    data=df,
    model_path="final_model.pth",  # use a specific local checkpoint if desired
    nsample=30,
)
```

## Running

### With uv (Recommended)
```bash
# Run any script with dependencies
uv run python examples/quick_example.py

# Run with custom arguments
uv run python examples/quick_example.py --model-path /path/to/model.pth

# Use your own dataset
uv run python examples/quick_example.py --dataset-path /path/to/your/data.csv
```

### Manual Python
```bash
# Activate virtual environment first
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate     # On Windows

# Then run normally
python examples/quick_example.py
```


## Performance Notes

- **DPM-Solver++**: Inference always uses DPM-Solver++; adjust `dpm_steps` to set the sampling budget (default: 20).
- **Multi-GPU**: Automatically detected and used when available
- **Memory**: Large datasets are processed in chunks to manage memory usage
- **Preprocessing**: Models are cached to avoid repeated loading

## Troubleshooting

### Import Issues
If you get import errors, ensure the path is added correctly:
```python
import sys, os
# Adjust this path to where you installed Kumo-Anomaly
sys.path.append('/full/path/to/Kumo-Anomaly')
```

### CUDA Issues
For GPU-related problems:
```bash
# Check PyTorch CUDA installation
uv run python -c "import torch; print(torch.cuda.is_available())"

# Force CPU mode if needed
export CUDA_VISIBLE_DEVICES=""
```

### Memory Issues
For large datasets:
- Reduce `nsample` parameter (try 5-10 instead of 15)
- Process data in smaller chunks
- Use DPM-Solver for reduced memory usage

### Hugging Face Download Issues
If `download_model_weights` raises a `401` / `403` error:

```bash
# Accept the model license on the repo page, then log in if needed
uv add "huggingface_hub[cli]"
huggingface-cli login

# Or set the token in the current shell
export HUGGINGFACE_HUB_TOKEN="hf_xxx"
```

Then accept the model license on the repo page if prompted:
<https://huggingface.co/nvidia/Kumo-Anomaly>

If downloads consistently fail with network errors, you can pre-fetch the files
manually with `huggingface-cli download nvidia/Kumo-Anomaly
final_model.pth curriculum_medium.yaml --local-dir .` and then pass the local
paths explicitly via `model_path=`/`config_path=`.

### Dependencies
If you encounter dependency conflicts:
```bash
# Clear cache and reinstall
uv cache clean
uv sync --reinstall
```
