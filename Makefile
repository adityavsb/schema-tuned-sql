.PHONY: help install lint test warehouse data eval check-model-1 check-model-2 eval-openrouter distill-openrouter clean

PYTHON = python3
PYTEST = pytest
RUFF = ruff

help:
	@echo "schema-tuned-sql: Fine-tuning small open LLMs for analytics text-to-SQL"
	@echo "Available commands:"
	@echo "  make install            Install package in editable mode"
	@echo "  make lint               Run ruff check and format inspection"
	@echo "  make test               Run CPU-only offline test suite"
	@echo "  make warehouse          Build and seed DuckDB analytics warehouse"
	@echo "  make check-model-1      Stage 1 smoke check: Model 1 (DeepSeek) on 1 eval + 1 batch distill"
	@echo "  make check-model-2      Stage 2 smoke check: Model 2 (Qwen-32B) on 1 eval case"
	@echo "  make eval-openrouter    Stage 3a: Full 220-item test evaluation on OpenRouter APIs (T1, T2)"
	@echo "  make distill-openrouter Stage 3b: Full 2,000-pair batched distillation (~295 calls)"
	@echo "  make eval               Run offline smoke evaluation harness"
	@echo "  make clean              Remove caches, build artifacts, and local database files"

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
	$(PYTHON) -m src.datagen.generator --config configs/data_gen.yaml --mode dry_run

check-model-1:
	$(PYTHON) scripts/run_openrouter_eval.py --stage check-one

check-model-2:
	$(PYTHON) scripts/run_openrouter_eval.py --stage check-remaining

eval-openrouter:
	$(PYTHON) scripts/run_openrouter_eval.py --stage full-eval

distill-openrouter:
	$(PYTHON) scripts/run_openrouter_eval.py --stage full-distill

eval:
	$(PYTHON) -m src.eval.runner --config configs/eval.yaml --mode offline_smoke

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache *.egg-info warehouse/analytics.duckdb* .cache
