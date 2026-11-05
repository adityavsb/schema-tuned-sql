"""Builds and verifies local distillation manifests, general-SQL holdout, and dry-run telemetry."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.datagen.generator import run_distillation_pipeline  # noqa: E402
from src.eval.general_holdout import generate_general_holdout_items  # noqa: E402
from src.eval.runner import run_evaluation_suite  # noqa: E402
from src.serve.bench import run_serving_benchmark  # noqa: E402
from src.train.sft import run_sft  # noqa: E402
from src.utils.config import DataGenConfig, EvalConfig, ServeConfig, SFTConfig  # noqa: E402


def main() -> None:
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build 200-item General SQL Holdout
    holdout_path = REPO_ROOT / "data" / "general_sql_holdout.jsonl"
    holdout_items = generate_general_holdout_items(out_path=holdout_path)
    holdout_sha = hashlib.sha256(holdout_path.read_bytes()).hexdigest()
    print(
        f"[1/5] Built general_sql_holdout.jsonl ({len(holdout_items)} items, sha256={holdout_sha[:12]}...)"
    )

    # 2. Initialize Distillation Manifest (pending live OpenRouter run)
    dg_cfg = DataGenConfig.from_yaml(REPO_ROOT / "configs" / "data_gen.yaml")
    train_dir = REPO_ROOT / "data" / "train"
    dg_cfg.output_dir = str(train_dir)
    # Remove any leftover offline/template JSONL splits so only real OpenRouter splits live in data/train/
    for old_split in train_dir.glob("train_*.jsonl"):
        old_split.unlink()

    manifest = run_distillation_pipeline(dg_cfg, mode="dry_run")
    (train_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (results_dir / "distillation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("[2/5] Initialized distillation manifest (status=pending_openrouter_run).")

    # 3. Verify T4 SFT configurations (pending live T4 GPU run)
    sft_checks = {}
    for cfg_name in ("sft_qwen15b_lora.yaml", "sft_qwen3b_qlora.yaml", "sft_qwen7b_qlora.yaml"):
        sft_cfg = SFTConfig.from_yaml(REPO_ROOT / "configs" / cfg_name)
        sft_cfg.train_data_path = str(train_dir / "train_2000.jsonl")
        sft_checks[sft_cfg.run_name] = run_sft(sft_cfg, dry_run=True)
    (results_dir / "t4_sft_dryrun_manifest.json").write_text(
        json.dumps(sft_checks, indent=2), encoding="utf-8"
    )
    print(f"[3/5] Verified T4 SFT configurations: {list(sft_checks.keys())}")

    # 4. Run Serving Cost & Break-Even Analysis
    serve_cfg = ServeConfig.from_yaml(REPO_ROOT / "configs" / "serve.yaml")
    serve_report = run_serving_benchmark(serve_cfg, mode="dry_run")
    (results_dir / "serving_cost_analysis.json").write_text(
        json.dumps(serve_report, indent=2), encoding="utf-8"
    )
    print("[4/5] Saved serving cost model to results/serving_cost_analysis.json")

    # 5. Run Offline Smoke Evaluation Suite across all 220 frozen test items
    eval_cfg = EvalConfig.from_yaml(REPO_ROOT / "configs" / "eval.yaml")
    eval_cfg.test_path = str(REPO_ROOT / "data" / "test_gold.jsonl")
    eval_cfg.dev_path = str(REPO_ROOT / "data" / "dev.jsonl")
    eval_report = run_evaluation_suite(eval_cfg, mode="offline_smoke")
    eval_report["general_holdout_sha256"] = holdout_sha
    (results_dir / "offline_smoke_summary.json").write_text(
        json.dumps(eval_report, indent=2), encoding="utf-8"
    )
    print("[5/5] Saved offline smoke evaluation report to results/offline_smoke_summary.json")


if __name__ == "__main__":
    main()
