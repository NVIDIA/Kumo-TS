# Changelog

All notable changes to Kumo-TS (formerly NV-Tesseract) are documented in this file.

## v1.0.1 - Unreleased

Changes since v1.0.0. Release date will be recorded when published.

### Added

- Anomaly explanation metrics, per-feature contributions, companion CSV exports, and consolidated PDF reports. ([#68](https://github.com/NVIDIA/Kumo-TS/pull/68), [#72](https://github.com/NVIDIA/Kumo-TS/pull/72))
- YAML-configured anomaly explanation settings and optional reconstruction diagnosis. ([#75](https://github.com/NVIDIA/Kumo-TS/pull/75), [#77](https://github.com/NVIDIA/Kumo-TS/pull/77))
- Historical attribution for anomaly explanations, including event validation and report timestamp preservation. ([#93](https://github.com/NVIDIA/Kumo-TS/pull/93))
- Configurable shared feature embeddings for anomaly detection, with a positional compatibility mode. ([#80](https://github.com/NVIDIA/Kumo-TS/pull/80))
- Missingness-aware forecasting in the forecasting SDK: `ForecastingConfig(handle_missingness=True)` routes inputs with
  NaN gaps to a missingness-aware model (`impute/`, Backbone-LF+ CRS) that reads per-channel/per-timestep observation
  masks instead of zero-filling. Options: `impute_ckpt`, `impute_normalization` (`history` / `provided` / `checkpoint`),
  `impute_history_rows`, `impute_scaler_stats`, `impute_source_dataset`, `impute_channel_alignment` (`name` / `positional`),
  plus `fit_impute_scaler_stats`.
- The released missingness checkpoint is loaded from `hf://nvidia/Kumo-Forecast/kumo-forecast-1.2.0` (any
  `hf://org/repo[@revision][/subfolder]` reference or local folder is accepted).
- `examples/missingness_quickstart.py` and `sdk/tests/test_imputation.py` (real-checkpoint tests opt-in via
  `KUMO_IMPUTE_CKPT` / `KUMO_IMPUTE_HF` / `KUMO_IMPUTE_CSV`).

### Changed

- Renamed the repository to Kumo-TS, forecasting to Kumo-Forecast, and AD Diffusion to Kumo-Anomaly; updated package names, directories, artifact prefixes, model references, and documentation. ([#83](https://github.com/NVIDIA/Kumo-TS/pull/83), [#84](https://github.com/NVIDIA/Kumo-TS/pull/84), [#85](https://github.com/NVIDIA/Kumo-TS/pull/85), [#86](https://github.com/NVIDIA/Kumo-TS/pull/86))
- Added published Kumo blogs and research papers to the repository README. ([#91](https://github.com/NVIDIA/Kumo-TS/pull/91))
- The NULL zero-fill warning now points to `handle_missingness=True`; `clear_model_cache()` also clears cached
  missingness models.

### Fixed

- Replayed saved fine-tuning feature order and preprocessing during anomaly inference rather than fitting transforms to the scoring batch. ([#32](https://github.com/NVIDIA/Kumo-TS/pull/32))
- Corrected `preprocess_simple` target-dimension forwarding. ([#44](https://github.com/NVIDIA/Kumo-TS/pull/44))
- Made forecasting RevIN statistics request-local to prevent interference between concurrent calls. ([#70](https://github.com/NVIDIA/Kumo-TS/pull/70))
- Removed GPU-count-dependent diffusion random-number behavior. ([#78](https://github.com/NVIDIA/Kumo-TS/pull/78))
- Corrected DPM-Solver++ timesteps, clipped schedules, and cumulative unconditional noise; anomaly inference now uses DPM-Solver++ exclusively. ([#89](https://github.com/NVIDIA/Kumo-TS/pull/89))
- Restored fine-tuning run-config support for validation data, labels, dropped columns, pretrained models, and model configuration. ([#90](https://github.com/NVIDIA/Kumo-TS/pull/90))
- Aligned anomaly inference window, split, and scale settings with saved fine-tuned checkpoints, including preprocessing overrides and multi-device paths. ([#92](https://github.com/NVIDIA/Kumo-TS/pull/92))

### Notes

- Repository release version: `v1.0.1`; package versions: `kumo-forecast` 1.0.1 and `kumo-anomaly` 1.0.1.
- Missingness-aware forecasting is introduced by [#87](https://github.com/NVIDIA/Kumo-TS/pull/87). Its `kumo-forecast-1.2.0` model asset version is independent of SDK release 1.0.1.
- Anomaly output remains int64 (`0` = normal, `1` = anomaly). Explanation and historical attribution are opt-in.
- Missingness-aware forecasting is inference-only in this release and cannot be combined with forecasting interpretability.

## v1.0.0 - 2026-09-21

Changes since v0.1.0.

### Added

- Forecasting output for all input channels through `ForecastingConfig(return_all_channels=True)`. ([#26](https://github.com/NVIDIA/NV-Tesseract/pull/26))
- Hugging Face Hub model wrappers for both forecasting and AD Diffusion, with `from_pretrained()` loading. ([#48](https://github.com/NVIDIA/NV-Tesseract/pull/48))
- TAO AutoML run-config support and scalar `metrics.json` outputs in both fine-tuning examples; per-epoch results are available in `epoch_metrics.json`. ([#54](https://github.com/NVIDIA/NV-Tesseract/pull/54), [#59](https://github.com/NVIDIA/NV-Tesseract/pull/59))
- Automatic multi-GPU distributed fine-tuning in both examples, with `--num-gpus` to limit GPU use. ([#61](https://github.com/NVIDIA/NV-Tesseract/pull/61))
- Forecasting feature-axis attribution, embedding Integrated Gradients, and embedding stability reporting, with JSON, CSV, and PDF artifacts. ([#56](https://github.com/NVIDIA/NV-Tesseract/pull/56), [#62](https://github.com/NVIDIA/NV-Tesseract/pull/62))
- Multi-GPU execution for DARR context embedding and channel-flow Jacobian analysis. ([#67](https://github.com/NVIDIA/NV-Tesseract/pull/67))
- Optional AD Diffusion PDF reports with signal plots, anomaly scores, and ground truth, configured through `ADDiffusionConfig` or `sdk_config.yaml`. ([#76](https://github.com/NVIDIA/NV-Tesseract/pull/76))
- Automatic MPS device detection for AD inference on Apple Silicon. ([#55](https://github.com/NVIDIA/NV-Tesseract/pull/55))

### Changed

- **Breaking:** `perform_forecasting()` now accepts `config=ForecastingConfig(...)` or a YAML path in place of flat inference arguments. Keep `df` and optional `context_df` as direct inputs; move settings such as `target_column`, `forecast_horizon`, and `ckpt` into the config. See the [forecasting SDK guide](forecasting/sdk/README.md) and [YAML template](forecasting/sdk/forecasting_inference_config.yaml). ([#74](https://github.com/NVIDIA/NV-Tesseract/pull/74))
- **Breaking:** AD analysis renames `config_path` to `model_config_path`. Reporting options belong in `sdk_config=ADDiffusionConfig(...)` or a YAML file. See the [AD Diffusion guide](ad_diffusion/README.md). ([#76](https://github.com/NVIDIA/NV-Tesseract/pull/76))
- Simplified forecasting interpretability around batched Jacobian analysis and embedding Integrated Gradients; removed Shapley, coupling, and legacy sharded paths, and corrected semantic-flow report layout. ([#60](https://github.com/NVIDIA/NV-Tesseract/pull/60))
- Contributions target `dev`; `main` is reserved for release-ready code. ([#65](https://github.com/NVIDIA/NV-Tesseract/pull/65))

### Fixed

- Isolated temporary forecasting CSVs per call to prevent collisions between concurrent requests. ([#34](https://github.com/NVIDIA/NV-Tesseract/pull/34))
- Aligned AD inference normalization with training and returned reconstructions in the same scale as targets. ([#41](https://github.com/NVIDIA/NV-Tesseract/pull/41))
- Restored AD preprocessing artifacts with the reusable `AdaptiveNormalizer`, preserving fitted transforms through JSON save/load; added round-trip coverage for normalizers, scalers, transformers, and PCA. ([#71](https://github.com/NVIDIA/NV-Tesseract/pull/71))
- Kept shared memory open until AD worker inference completes. ([#47](https://github.com/NVIDIA/NV-Tesseract/pull/47))
- Aligned DARR input and context channel ordering when the same columns arrive in different orders. ([#52](https://github.com/NVIDIA/NV-Tesseract/pull/52))
- Corrected base/cross-channel checkpoint selection and inferred cross-channel mode from checkpoint weights. ([#73](https://github.com/NVIDIA/NV-Tesseract/pull/73))
- Excluded SDK-added padding from AD MAE, L2, and thresholding, while retaining full-width target and reconstruction outputs. Low-level results include `valid_feature_mask` and `score_feature_count`. Scores may differ for padded inputs. ([#79](https://github.com/NVIDIA/NV-Tesseract/pull/79))
- Verify downloaded model weight files before reporting a successful download in both SDKs. ([#82](https://github.com/NVIDIA/NV-Tesseract/pull/82))
- Clarified that AD `Anomaly` output uses int64 values (`0` = normal, `1` = anomaly); detection behavior is unchanged.
- Synchronized forecasting lockfile metadata with its dependency declarations without changing locked package versions.

### Notes

- Repository release version: `v1.0.0`.
- Package metadata versions: `tesseract_forecasting` 1.0.0 and `ad-diffusion-oss` 1.0.0.

## v0.1.0 - 2026-07-07

First public release of NV-Tesseract.

### Added

- Forecasting package with DataFrame-first inference, DARR context-enhanced forecasting, cross-channel forecasting support, and Hugging Face weight loading.
- Forecasting interpretability framework with input attributions, semantic-flow diagnostics, forecast-vs-history ratios, trajectory stability metrics, and optional PDF/JSON artifacts.
- AD Diffusion package for multivariate anomaly detection with SCS and MACS adaptive thresholding, DPM-Solver inference, and Hugging Face weight loading.
- Fine-tuning examples for both forecasting and AD Diffusion.
- Lightweight CI covering linting, SPDX checks, forecasting tests, AD Diffusion tests, and example tests.
- Public documentation for installation, examples, dataset expectations, model assets, contribution flow, security reporting, and third-party notices.

### Changed

- Raised the PyTorch dependency floor to `torch>=2.7.0` for Blackwell GPU support.
- Removed unused forecasting audio/vision PyTorch dependencies and the legacy `mac-mps` extra.
- Switched user-facing progress output from direct `print` calls to logging where appropriate.
- Updated documentation to reference public Hugging Face model repositories.

### Fixed

- Fixed DARR retrieval behavior when forecast horizons exceed the model horizon.
- Fixed AD Diffusion complementary mask aggregation so each target mask selects reconstructions from its own strategy.
- Fixed AD thresholding, packaging, and README examples.
- Fixed pandas frequency deprecation warnings in README examples.

### Notes

- Repository release version: `v0.1.0`.
- Package metadata versions at this release: `forecasting` is `0.1.0`; `ad-diffusion-oss` is `1.0.0`.
