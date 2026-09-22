# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

.DEFAULT_GOAL := help

# Ruff config and roots (keep Ruff spec in sync with [dependency-groups] dev in Kumo-Forecast/pyproject.toml)
RUFF_CONFIG := Kumo-Forecast/pyproject.toml
RUFF_SPEC := ruff>=0.8.0
LINT_PATHS := Kumo-Forecast ad_diffusion
UVX_RUFF := uvx --from "$(RUFF_SPEC)" ruff

.PHONY: help lint lint-fix spdx spdx-check

help:
	@echo "Targets:"
	@echo "  make lint       - Ruff lint + format check ($(LINT_PATHS))"
	@echo "  make lint-fix   - Auto-fix Ruff issues and apply formatting ($(LINT_PATHS))"
	@echo "  make spdx       - Insert SPDX headers into Python files under Kumo-Forecast/, ad_diffusion/, scripts/"
	@echo "  make spdx-check - Fail if any Python file is missing SPDX-License-Identifier"
	@echo "Requires: uv (https://docs.astral.sh/uv/)"

lint:
	$(UVX_RUFF) check $(LINT_PATHS) --config $(RUFF_CONFIG)
	$(UVX_RUFF) format --check $(LINT_PATHS) --config $(RUFF_CONFIG)

lint-fix:
	$(UVX_RUFF) check --fix $(LINT_PATHS) --config $(RUFF_CONFIG)
	$(UVX_RUFF) format $(LINT_PATHS) --config $(RUFF_CONFIG)

spdx:
	python3 scripts/add_spdx_headers.py

spdx-check:
	python3 scripts/add_spdx_headers.py --check
