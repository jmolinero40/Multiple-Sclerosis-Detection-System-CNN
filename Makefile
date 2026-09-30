# Convenience targets. Run `make help` for the list.
.PHONY: help install test lint format preprocess splits train evaluate figures clean

PYTHON  ?= python
CONFIG  ?= configs/multimodal.yaml
RUN_DIR ?= runs/multimodal

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with dev dependencies
	$(PYTHON) -m pip install -e ".[dev]"

test:  ## Run the test suite
	pytest

lint:  ## Check formatting and lint rules
	ruff check src tests
	black --check src tests

format:  ## Apply formatting
	black src tests
	ruff check --fix src tests

preprocess:  ## Stage 1: NIfTI volumes -> .npy slices
	$(PYTHON) -m msseg.data.preprocess --raw-root data/raw --out-root data/processed

splits:  ## Stage 2: build the split manifest
	$(PYTHON) -m msseg.data.splits --processed-root data/processed \
		--out data/manifests/splits.csv

train:  ## Train with CONFIG (default: configs/multimodal.yaml)
	$(PYTHON) -m msseg.train --config $(CONFIG)

evaluate:  ## Evaluate the best checkpoint on the test split
	$(PYTHON) -m msseg.evaluate --checkpoint $(RUN_DIR)/best.pt --split test

figures:  ## Generate report figures
	$(PYTHON) -m msseg.figures --checkpoint $(RUN_DIR)/best.pt \
		--results results/multimodal/metrics.json --output-dir docs/images

clean:  ## Remove caches and build artefacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist *.egg-info
