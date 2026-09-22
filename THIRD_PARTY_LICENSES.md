# Third-party notices and licenses

This document satisfies distribution requirements for third-party open source components used **by** NV-Tesseract (dependencies and vendored source). It complements the **project license** for NVIDIA-authored material.

## NVIDIA-authored project license (Apache License 2.0)

NV-Tesseract source code authored by NVIDIA is offered under the Apache License, Version 2.0. The full license text is in the repository root:

- **[`LICENSE`](LICENSE)**

Redistribution notices under Apache 2.0 §4(d) are in **[`NOTICE`](NOTICE)**. The **MIT license** for included DPM-Solver material appears both in [`LICENSE`](LICENSE) (**ADDITIONAL THIRD-PARTY LICENSES**) and in [`third_party/dpm-solver/LICENSE`](third_party/dpm-solver/LICENSE).

## Per-file notices

Python modules use SPDX license tags (`SPDX-License-Identifier`) and NVIDIA copyright lines where applicable, consistent with SPDX Specification v2.3, Annex E (short identifiers in source files). Files that incorporate or modify third-party source include the upstream copyright and license notice in addition to NVIDIA attribution — for example [`ad_diffusion/utils/dpm_solver_pytorch.py`](ad_diffusion/utils/dpm_solver_pytorch.py).

## Third-party source included in this repository

Upstream license texts for **vendored or fork-derived** source are stored under **[`third_party/`](third_party/)** (one subdirectory per upstream project). See [`third_party/README.md`](third_party/README.md).

### DPM-Solver (PyTorch)

- **Upstream:** [LuChengTHU/dpm-solver](https://github.com/LuChengTHU/dpm-solver) — `dpm_solver_pytorch.py`
- **License:** MIT License — **full text:** [`third_party/dpm-solver/LICENSE`](third_party/dpm-solver/LICENSE)
- **Modified source in tree:** [`ad_diffusion/utils/dpm_solver_pytorch.py`](ad_diffusion/utils/dpm_solver_pytorch.py) (includes the MIT notice in-file per license terms)

## Runtime dependencies (PyPI)

The packages below are **direct** runtime dependencies declared in [`Kumo-Forecast/pyproject.toml`](Kumo-Forecast/pyproject.toml) and [`ad_diffusion/pyproject.toml`](ad_diffusion/pyproject.toml). Each package is subject to its own license (typically included in the package metadata on PyPI and in installed distributions). SPDX or common names are shown for convenience; refer to the package for authoritative terms.

### `Kumo-Forecast/` (`tesseract_forecasting`)

| Component | Declared requirement | Typical license (verify on PyPI) |
|-----------|----------------------|----------------------------------|
| datasetsforecast | `>=1.0.0` | MIT |
| joblib | `>=1.5.2` | BSD-3-Clause |
| pandas | `>=2.1.0` | BSD-3-Clause |
| pytest-xdist | `>=3.8.0` | MIT |
| numpy | `>=1.24.0` | BSD-3-Clause |
| torch | `>=2.0.0` | BSD-style (PyTorch) |
| torchvision | `>=0.15.0` | BSD-style |
| torchaudio | `>=2.0.0` | BSD-style |
| tqdm | `>=4.65.0` | MPL-2.0 / MIT |
| huggingface_hub | `>=0.17.0` | Apache-2.0 |
| transformers | `>=4.36.0` | Apache-2.0 |

### `ad_diffusion/` (`ad-diffusion-oss`)

| Component | Declared requirement | Typical license (verify on PyPI) |
|-----------|----------------------|----------------------------------|
| torch | `>=1.13.0` | BSD-style (PyTorch) |
| numpy | `>=1.21.0,<2` | BSD-3-Clause |
| pandas | `>=1.5.0,<3` | BSD-3-Clause |
| scikit-learn | `>=1.1.0,<2` | BSD-3-Clause |
| scipy | `>=1.9.0,<2` | BSD-3-Clause |
| pyyaml | `>=6.0,<7` | MIT |
| tqdm | `>=4.64.0,<5` | MPL-2.0 / MIT |
| psutil | `>=5.8.0,<6` | BSD-3-Clause |
| huggingface_hub | `>=0.23.0,<1` | Apache-2.0 |

**Transitive** dependencies also apply when you install either package; they are licensed under their respective terms (BSD, MIT, Apache-2.0, PSF, MPL, ISC, etc.). To generate a machine-readable inventory of **all** installed packages and detected licenses (for a given virtual environment), you can run:

```bash
uvx pip-licenses --python /path/to/venv/bin/python --from=mixed --format=markdown --with-authors --with-urls -o name
```

## Trademarks

NVIDIA and other parties’ trademarks and logos are subject to their respective policies.
