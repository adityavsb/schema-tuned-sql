.PHONY: help install lint test warehouse data eval clean

PYTHON = python3
PYTEST = pytest
RUFF = ruff

help:
	@echo "schema-tuned-sql: Fine-tuning small open LLMs for analytics text-to-SQL"
	@echo "Available commands:"
	@echo "  make install     Install package in editable mode"
	@echo "  make lint        Run ruff check and format inspection"
	@echo "  make test        Run CPU-only offline test suite"
	@echo "  make warehouse   Build and seed DuckDB analytics warehouse"
	@echo "  make data        Run execution-verified distillation under budget cap"
	@echo "  make eval        Run offline evaluation harness"
	@echo "  make clean       Remove caches, build artifacts, and local database files"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(RUFF) check src warehouse tests
	$(RUFF) format --check src warehouse tests

test:
	$(PYTEST) -v tests/

warehouse:
	$(PYTHON) -m warehouse.generate_data --seed 42 --db-path warehouse/analytics.duckdb

data:
	$(PYTHON) -m src.datagen.generator --config configs/data_gen.yaml

eval:
	$(PYTHON) -m src.eval.runner --config configs/eval.yaml --mode offline

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache *.egg-info warehouse/analytics.duckdb* .cache
