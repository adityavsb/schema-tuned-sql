#!/usr/bin/env python3
"""Staged OpenRouter preflight and evaluation harness for schema-tuned-sql.

Mirrors the staged gate in narrative-audit-jev:
  1. --stage check-one       : Smoke test Model 1 (T1: deepseek/deepseek-chat) on 1 eval case + 1 batch distill call
  2. --stage check-remaining : Smoke test Model 2 (T2: qwen/qwen-2.5-coder-32b-instruct) on 1 eval case
  3. --stage full-eval       : Full 220-item frozen test evaluation across T1 and T2
  4. --stage full-distill    : Full 2,000-pair execution-verified distillation (batched 8 Qs/call, ~295 calls)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datagen.generator import (  # noqa: E402
    format_batch_teacher_prompt,
    parse_batched_sql_completions,
    run_distillation_pipeline,
)
from src.datagen.templates import generate_candidate_candidates  # noqa: E402
from src.datagen.verifier import SQLVerifier  # noqa: E402
from src.eval.rag import SYSTEM_INSTRUCTION, SchemaDefinitionRetriever  # noqa: E402
from src.eval.runner import evaluate_single_item, run_evaluation_suite  # noqa: E402
from src.llm.pricing import BudgetTracker  # noqa: E402
from src.llm.provider import call_llm  # noqa: E402
from src.utils.config import DataGenConfig, EvalConfig, load_dotenv  # noqa: E402
from src.utils.db import ReadOnlySQLGuardrail  # noqa: E402
from warehouse.generate_data import generate_synthetic_warehouse  # noqa: E402


def verify_api_key_present() -> str:
    """Loads .env and verifies OPENROUTER_API_KEY is present before dispatching calls."""
    load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add OPENROUTER_API_KEY=sk-or-... to .env before running."
        )
    return key


def run_single_eval_smoke(
    model_id: str,
    system_label: str,
    eval_cfg: EvalConfig,
) -> dict[str, Any]:
    """Dispatches 1 evaluation query to verify model connectivity, SQL extraction, and DuckDB execution."""
    test_path = REPO_ROOT / eval_cfg.test_path
    first_item = json.loads(test_path.read_text(encoding="utf-8").splitlines()[0])

    conn = generate_synthetic_warehouse(
        db_path=":memory:",
        seed=eval_cfg.warehouse_seed,
        num_accounts=100,
        num_events=15000,
    )
    guardrail = ReadOnlySQLGuardrail()
    retriever = SchemaDefinitionRetriever(dev_path=REPO_ROOT / eval_cfg.dev_path)

    user_prompt = retriever.format_prompt(question=first_item["question"], mode="zero_shot")
    print(f"\n[*] DISPATCHING EVAL CHECK-RUN ({system_label}):")
    print(f"    Model:      {model_id}")
    print(f"    Item ID:    {first_item['id']} ({first_item['difficulty']})")
    print(f"    Question:   {first_item['question']}")
    print(f"    Gold SQL:   {first_item['gold_sql']}")

    resp = call_llm(
        model=model_id,
        prompt=user_prompt,
        system_prompt=SYSTEM_INSTRUCTION,
        temperature=eval_cfg.temperature,
        max_tokens=eval_cfg.max_tokens,
        cache_dir=str(REPO_ROOT / eval_cfg.cache_dir),
        max_usd=eval_cfg.max_usd,
        budget_ledger_path=str(REPO_ROOT / eval_cfg.budget_ledger_path),
    )

    rec = evaluate_single_item(
        item=first_item,
        raw_completion=resp.content,
        conn=conn,
        guardrail=guardrail,
        float_tol=eval_cfg.float_tolerance,
        latency_ms=resp.latency_ms,
        completion_tokens=resp.completion_tokens,
        cost_usd=resp.cost_usd,
    )
    conn.close()

    print("\n[+] API EVAL CHECK-RUN SUCCEEDED:")
    print(f"    Pred SQL:   {rec['pred_sql']}")
    print(f"    Valid SQL:  {rec['is_valid_sql']}")
    print(f"    Exec Match: {'PASS' if rec['is_match'] else 'MISMATCH'}")
    print(f"    Cached:     {resp.cached}")
    print(f"    Latency:    {resp.latency_ms:.1f} ms")
    print(f"    Tokens:     In={resp.prompt_tokens}, Out={resp.completion_tokens}")
    print(f"    Call Cost:  ${resp.cost_usd:.6f}")
    return rec


def run_single_batch_distill_smoke(dg_cfg: DataGenConfig) -> dict[str, Any]:
    """Dispatches 1 batched distillation call (8 questions) to verify JSON batch parsing and SQLVerifier."""
    conn = generate_synthetic_warehouse(
        db_path=":memory:",
        seed=dg_cfg.seed,
        num_accounts=100,
        num_events=15000,
    )
    verifier = SQLVerifier(conn)
    retriever = SchemaDefinitionRetriever(dev_path=REPO_ROOT / "data" / "dev.jsonl")

    batch_cands = generate_candidate_candidates(seed=dg_cfg.seed, count=dg_cfg.questions_per_call)
    questions = [c["question"] for c in batch_cands]
    prompt = format_batch_teacher_prompt(questions, retriever)

    print("\n[*] DISPATCHING BATCHED DISTILLATION CHECK-RUN (1 Call = 8 Questions):")
    print(f"    Teacher:    {dg_cfg.primary_teacher}")
    print(f"    Batch Size: {len(questions)} questions")

    resp = call_llm(
        model=dg_cfg.primary_teacher,
        prompt=prompt,
        system_prompt=SYSTEM_INSTRUCTION,
        temperature=dg_cfg.temperature,
        max_tokens=dg_cfg.max_tokens,
        cache_dir=str(REPO_ROOT / dg_cfg.cache_dir),
        max_usd=dg_cfg.max_usd,
        budget_ledger_path=str(REPO_ROOT / dg_cfg.budget_ledger_path),
    )

    sql_list = parse_batched_sql_completions(resp.content, expected_count=len(batch_cands))
    accepted_count = 0
    for cand, raw_sql in zip(batch_cands, sql_list, strict=True):
        ok, _, _, _ = verifier.verify_pair(
            question=cand["question"],
            raw_sql_or_completion=raw_sql,
            is_negative_query=bool(cand.get("is_negative_query", False)),
        )
        if ok:
            accepted_count += 1

    conn.close()
    print("\n[+] BATCHED DISTILLATION CHECK-RUN SUCCEEDED:")
    print(f"    Parsed SQLs:    {sum(1 for s in sql_list if s)} / {len(batch_cands)}")
    print(f"    Exec Verified:  {accepted_count} / {len(batch_cands)} passed DuckDB execution")
    print(f"    Sample SQL #1:  {sql_list[0][:100] if sql_list else ''}")
    print(f"    Cached:         {resp.cached}")
    print(f"    Latency:        {resp.latency_ms:.1f} ms")
    print(f"    Tokens:         In={resp.prompt_tokens}, Out={resp.completion_tokens}")
    print(f"    Call Cost:      ${resp.cost_usd:.6f}")

    return {
        "teacher": dg_cfg.primary_teacher,
        "batch_size": len(batch_cands),
        "accepted_count": accepted_count,
        "prompt_tokens": resp.prompt_tokens,
        "completion_tokens": resp.completion_tokens,
        "cost_usd": resp.cost_usd,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Staged OpenRouter runner for schema-tuned-sql")
    parser.add_argument(
        "--stage",
        choices=["check-one", "check-remaining", "full-eval", "full-distill"],
        required=True,
        help="Execution stage: check-one (Model 1), check-remaining (Model 2), full-eval, or full-distill",
    )
    parser.add_argument(
        "--systems",
        default="T1,T2",
        help="Comma-separated API system IDs for full-eval (default: T1,T2)",
    )
    parser.add_argument(
        "--target-pairs",
        type=int,
        default=None,
        help="Optional override for target verified pairs in full-distill",
    )
    args = parser.parse_args()

    verify_api_key_present()
    eval_cfg = EvalConfig.from_yaml(REPO_ROOT / "configs" / "eval.yaml")
    eval_cfg.test_path = str(REPO_ROOT / "data" / "test_gold.jsonl")
    eval_cfg.dev_path = str(REPO_ROOT / "data" / "dev.jsonl")
    eval_cfg.cache_dir = str(REPO_ROOT / ".cache" / "llm_responses")
    eval_cfg.budget_ledger_path = str(REPO_ROOT / ".cache" / "budget_ledger.json")

    dg_cfg = DataGenConfig.from_yaml(REPO_ROOT / "configs" / "data_gen.yaml")
    dg_cfg.output_dir = str(REPO_ROOT / "data" / "train")
    dg_cfg.cache_dir = str(REPO_ROOT / ".cache" / "llm_responses")
    dg_cfg.budget_ledger_path = str(REPO_ROOT / ".cache" / "budget_ledger.json")

    if args.stage == "check-one":
        run_single_eval_smoke("deepseek/deepseek-chat", "T1: DeepSeek Teacher", eval_cfg)
        run_single_batch_distill_smoke(dg_cfg)
    elif args.stage == "check-remaining":
        run_single_eval_smoke(
            "qwen/qwen-2.5-coder-32b-instruct", "T2: Qwen-2.5-Coder-32B", eval_cfg
        )
    elif args.stage == "full-eval":
        sys_ids = [s.strip() for s in args.systems.split(",") if s.strip()]
        print(
            f"\n[*] Running Full OpenRouter Evaluation for systems {sys_ids} on 220 test items..."
        )
        report = run_evaluation_suite(eval_cfg, mode="live", system_ids=sys_ids)
        out_path = REPO_ROOT / "results" / "openrouter_eval_summary.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[+] Saved live OpenRouter evaluation summary to {out_path}")
        for sid, sdata in report["systems"].items():
            ci = sdata.get("execution_accuracy_ci95", [0.0, 0.0])
            ex = sdata.get("execution_accuracy", 0.0)
            cost = sdata.get("total_cost_usd", 0.0)
            print(
                f"    {sid} ({sdata.get('system_name', sid)}): EX={ex * 100:.2f}% "
                f"[{ci[0] * 100:.2f}%, {ci[1] * 100:.2f}%] "
                f"| Cost=${cost:.4f}"
            )
    elif args.stage == "full-distill":
        target = args.target_pairs or dg_cfg.target_verified_pairs
        print(
            f"\n[*] Running Full Batched Distillation ({dg_cfg.questions_per_call} Qs/call) up to {target} verified pairs..."
        )
        manifest = run_distillation_pipeline(dg_cfg, mode="live", target_override=target)
        for split_info in manifest["scaling_splits"].values():
            p_str = str(split_info["path"])
            if str(REPO_ROOT) in p_str:
                split_info["path"] = p_str.replace(str(REPO_ROOT) + "/", "")
        (REPO_ROOT / "data" / "train" / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        (REPO_ROOT / "results" / "distillation_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(
            f"[+] Completed distillation: {manifest['total_verified_pairs']} verified pairs in {manifest['total_api_calls']} API calls."
        )

    ledger = BudgetTracker(max_usd=eval_cfg.max_usd, ledger_path=eval_cfg.budget_ledger_path)
    print(
        f"\n[*] Cumulative Budget Ledger: ${ledger.total_spent_usd:.4f} / ${eval_cfg.max_usd:.2f}"
    )


if __name__ == "__main__":
    main()
