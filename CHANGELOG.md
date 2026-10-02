# Changelog

All notable changes to Kumo-TS (formerly NV-Tesseract) are documented in this file.

## Unreleased

### Added

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

- The NULL zero-fill warning now points to `handle_missingness=True`; `clear_model_cache()` also clears cached
  missingness models.

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
