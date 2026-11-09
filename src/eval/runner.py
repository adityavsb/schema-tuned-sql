"""Unified evaluation runner across prompting baselines, teacher APIs, and fine-tuned students."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import sqlglot
from sqlglot import exp

from src.eval.execution import evaluate_execution
from src.eval.metrics import aggregate_system_metrics, compute_mcnemar_test
from src.eval.rag import SYSTEM_INSTRUCTION, SchemaDefinitionRetriever
from src.llm.provider import call_llm
from src.utils.config import EvalConfig, SystemSpec
from src.utils.db import ReadOnlySQLGuardrail
from warehouse.generate_data import generate_synthetic_warehouse

REFUSAL_PATTERNS = [
    r"REFUSAL",
    r"UNANSWERABLE",
    r"cannot\s+be\s+answered",
    r"not\s+(?:available|present|stored)\s+in\s+the\s+(?:schema|database|tables)",
    r"no\s+(?:column|table|data)\s+for",
]


def is_refusal_response(text: str) -> bool:
    """Returns True if model output explicitly refuses or flags an unanswerable query."""
    if not text or not text.strip():
        return False
    for pat in REFUSAL_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False


def extract_sql(text: str) -> str:
    """Extracts clean SQL from a markdown code block or raw completion."""
    if not text:
        return ""
    fenced = re.search(r"```(?:sql|duckdb)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    else:
        candidate = text.strip()

    # Strip trailing semicolon
    if candidate.endswith(";"):
        candidate = candidate[:-1].strip()
    return candidate


def canonicalize_sql(sql: str) -> str:
    """Normalizes SQL via sqlglot AST for exact-structure comparison."""
    if not sql.strip():
        return ""
    try:
        parsed = sqlglot.parse_one(sql, read="duckdb")
        return parsed.sql(dialect="duckdb", normalize=True)
    except Exception:
        return ""


def classify_failure(
    gold_sql: str,
    pred_sql: str,
    is_valid_sql: bool,
    exec_msg: str,
    is_negative_query: bool,
    refusal_detected: bool,
) -> str:
    """Categorizes SQL generation failure into an interpretable error taxonomy."""
    if is_negative_query:
        return "correct_refusal" if refusal_detected else "hallucinated_on_unanswerable"
    if refusal_detected:
        return "false_refusal"
    if not is_valid_sql:
        if "Binder Error" in exec_msg or "Referenced column" in exec_msg or "Table" in exec_msg:
            return "hallucinated_column"
        return "syntax_or_guardrail_error"

    try:
        g_ast = sqlglot.parse_one(gold_sql, read="duckdb")
        p_ast = sqlglot.parse_one(pred_sql, read="duckdb")
        g_tables = {t.name.lower() for t in g_ast.find_all(exp.Table) if t.name}
        p_tables = {t.name.lower() for t in p_ast.find_all(exp.Table) if t.name}
        if g_tables != p_tables:
            return "wrong_join"

        g_sql_up = gold_sql.upper()
        p_sql_up = pred_sql.upper()
        if any(k in g_sql_up for k in ("INTERVAL", "DATE_TRUNC", "EXTRACT")) and not any(
            k in p_sql_up for k in ("INTERVAL", "DATE_TRUNC", "EXTRACT")
        ):
            return "wrong_date_window"

        if (
            ("STATUS = 'ACTIVE'" in g_sql_up and "STATUS = 'ACTIVE'" not in p_sql_up)
            or ("DASHBOARD_CREATE" in g_sql_up and "DASHBOARD_CREATE" not in p_sql_up)
            or ("14 DAYS" in g_sql_up and "14 DAYS" not in p_sql_up)
        ):
            return "definition_violated"
    except Exception:
        pass

    return "aggregation_error"


def evaluate_single_item(
    item: dict[str, Any],
    raw_completion: str,
    conn: Any,
    guardrail: ReadOnlySQLGuardrail,
    float_tol: float = 1e-4,
    latency_ms: float = 0.0,
    completion_tokens: int = 0,
    cost_usd: float = 0.0,
) -> dict[str, Any]:
    """Evaluates a single model completion against the gold item and DuckDB warehouse."""
    q_id = item["id"]
    is_neg = bool(item.get("is_negative_query", False))
    gold_sql = item.get("gold_sql", "")

    refusal = is_refusal_response(raw_completion)
    pred_sql = "" if refusal else extract_sql(raw_completion)

    if is_neg:
        is_match = refusal
        is_valid = False
        exact_ast = False
        failure_cat = classify_failure(gold_sql, pred_sql, is_valid, "", is_neg, refusal)
        return {
            "id": q_id,
            "difficulty": item.get("difficulty", "hard"),
            "category": item.get("category", "refusal"),
            "is_negative_query": True,
            "refusal_detected": refusal,
            "is_match": is_match,
            "is_valid_sql": is_valid,
            "exact_ast_match": exact_ast,
            "failure_category": None if is_match else failure_cat,
            "latency_ms": latency_ms,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "pred_sql": pred_sql,
        }

    if refusal or not pred_sql:
        return {
            "id": q_id,
            "difficulty": item.get("difficulty", "medium"),
            "category": item.get("category", "join"),
            "is_negative_query": False,
            "refusal_detected": refusal,
            "is_match": False,
            "is_valid_sql": False,
            "exact_ast_match": False,
            "failure_category": "false_refusal" if refusal else "syntax_or_guardrail_error",
            "latency_ms": latency_ms,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "pred_sql": pred_sql,
        }

    is_safe = guardrail.is_safe(pred_sql)
    if not is_safe:
        return {
            "id": q_id,
            "difficulty": item.get("difficulty", "medium"),
            "category": item.get("category", "join"),
            "is_negative_query": False,
            "refusal_detected": False,
            "is_match": False,
            "is_valid_sql": False,
            "exact_ast_match": False,
            "failure_category": "syntax_or_guardrail_error",
            "latency_ms": latency_ms,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "pred_sql": pred_sql,
        }

    is_match, msg, _ = evaluate_execution(gold_sql, pred_sql, conn, float_tol=float_tol)
    is_valid = not msg.startswith("Predicted SQL Execution Error")
    exact_ast = canonicalize_sql(gold_sql) == canonicalize_sql(pred_sql) and bool(pred_sql)
    failure_cat = (
        None if is_match else classify_failure(gold_sql, pred_sql, is_valid, msg, False, False)
    )

    return {
        "id": q_id,
        "difficulty": item.get("difficulty", "medium"),
        "category": item.get("category", "join"),
        "is_negative_query": False,
        "refusal_detected": False,
        "is_match": is_match,
        "is_valid_sql": is_valid,
        "exact_ast_match": exact_ast,
        "failure_category": failure_cat,
        "latency_ms": latency_ms,
        "completion_tokens": completion_tokens,
        "cost_usd": cost_usd,
        "pred_sql": pred_sql,
    }


class LocalInferenceResponse:
    """Standardized completion container for local PyTorch/PEFT CUDA inference."""

    def __init__(
        self,
        content: str,
        latency_ms: float = 0.0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        self.content = content
        self.latency_ms = latency_ms
        self.completion_tokens = completion_tokens
        self.cost_usd = cost_usd


def run_cuda_local_inference(
    sys_spec: SystemSpec,
    prompts: list[str],
    max_tokens: int = 512,
    batch_size: int = 4,
) -> list[LocalInferenceResponse]:
    """Runs batch inference on a local CUDA GPU using Hugging Face transformers & PEFT."""
    import torch
    # Neutralize incompatible pre-installed torchao in remote cloud GPU environments
    try:
        import peft.import_utils

        peft.import_utils.is_torchao_available = lambda: False
    except Exception:
        pass
    try:
        import peft.tuners.lora.torchao

        peft.tuners.lora.torchao.is_torchao_available = lambda: False
    except Exception:
        pass

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA required for cuda_live evaluation of system {sys_spec.id}.")

    print(f"\n[*] [CUDA] Initializing {sys_spec.id} ({sys_spec.name})...")
    tokenizer = AutoTokenizer.from_pretrained(sys_spec.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_4bit = (
        "3B" in sys_spec.model
        or "7B" in sys_spec.model
        or (sys_spec.adapter_path is not None and "qlora" in sys_spec.adapter_path.lower())
    )
    bnb_config = None
    if is_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        sys_spec.model,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True,
    )

    if sys_spec.adapter_path:
        ad_path = Path(sys_spec.adapter_path)
        if not (ad_path / "adapter_config.json").exists():
            seed_dirs = sorted(ad_path.glob("seed_*"))
            if seed_dirs:
                ad_path = seed_dirs[0]
        if (ad_path / "adapter_config.json").exists():
            print(f"[*] Attaching PEFT adapter from {ad_path}...")
            model = PeftModel.from_pretrained(model, str(ad_path))
        else:
            print(f"[!] Warning: No adapter found at {ad_path}, evaluating base model.")

    model.eval()

    responses: list[LocalInferenceResponse] = []
    total = len(prompts)
    print(f"[*] Generating SQL predictions on CUDA ({total} prompts, batch_size={batch_size})...")

    for i in range(0, total, batch_size):
        batch = prompts[i : i + batch_size]
        chat_texts = []
        for p in batch:
            msgs = [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": p},
            ]
            chat_texts.append(
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            )

        inputs = tokenizer(
            chat_texts, return_tensors="pt", padding=True, truncation=True, max_length=3584
        ).to("cuda")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        avg_lat_ms = dt_ms / max(1, len(batch))

        for j, out_ids in enumerate(outputs):
            in_len = inputs["input_ids"][j].shape[0]
            gen_ids = out_ids[in_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            responses.append(
                LocalInferenceResponse(
                    content=gen_text,
                    latency_ms=round(avg_lat_ms, 2),
                    completion_tokens=int(len(gen_ids)),
                    cost_usd=0.0,
                )
            )

        if (i + batch_size) % 40 == 0 or (i + len(batch)) == total:
            print(
                f"    [{sys_spec.id}] Progress: {min(total, i + len(batch))}/{total} ({min(total, i + len(batch)) / total * 100:.1f}%)"
            )

    del model
    del tokenizer
    torch.cuda.empty_cache()

    return responses


def summarize_system_run(
    system: SystemSpec,
    records: list[dict[str, Any]],
    n_bootstrap: int = 2000,
    seed: int = 42,
    mode: str = "live",
) -> dict[str, Any]:
    """Builds aggregate metrics, latency percentiles, cost per 1k queries, and error taxonomy."""
    core = aggregate_system_metrics(records, n_bootstrap=n_bootstrap, seed=seed)
    latencies = [float(r.get("latency_ms", 0.0)) for r in records]
    tokens = [int(r.get("completion_tokens", 0)) for r in records]
    costs = [float(r.get("cost_usd", 0.0)) for r in records]
    ans_records = [r for r in records if not r.get("is_negative_query")]
    exact_matches = [1.0 if r.get("exact_ast_match") else 0.0 for r in ans_records]

    failure_counts: Counter[str] = Counter(
        r["failure_category"] for r in records if r.get("failure_category")
    )

    total_queries = max(1, len(records))
    usd_per_1k = round((sum(costs) / total_queries) * 1000.0, 4)

    return {
        "system_id": system.id,
        "system_name": system.name,
        "kind": system.kind,
        "model": system.model,
        "mode": mode,
        "is_model_evidence": mode in ("live", "cuda_live"),
        **core,
        "exact_ast_match_rate": round(float(np.mean(exact_matches)) if exact_matches else 0.0, 4),
        "p50_latency_ms": round(float(np.percentile(latencies, 50)) if latencies else 0.0, 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)) if latencies else 0.0, 2),
        "mean_output_tokens": round(float(np.mean(tokens)) if tokens else 0.0, 1),
        "total_cost_usd": round(sum(costs), 6),
        "usd_per_1k_queries": usd_per_1k,
        "error_taxonomy": dict(failure_counts),
    }


def run_evaluation_suite(
    config: EvalConfig,
    mode: str = "offline_smoke",
    limit: int | None = None,
    system_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Executes configured systems across the test set and computes paired McNemar statistics."""
    test_path = Path(config.test_path)
    items = [
        json.loads(line)
        for line in test_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        items = items[:limit]

    conn = generate_synthetic_warehouse(
        db_path=":memory:",
        seed=config.warehouse_seed,
        num_accounts=100,
        num_events=15000,
    )
    guardrail = ReadOnlySQLGuardrail()
    retriever = SchemaDefinitionRetriever(dev_path=Path(config.dev_path))

    all_system_records: dict[str, list[dict[str, Any]]] = {}
    system_summaries: dict[str, dict[str, Any]] = {}

    active_systems = (
        [s for s in config.systems if s.id in set(system_ids)]
        if system_ids is not None
        else config.systems
    )

    for sys_spec in active_systems:
        records: list[dict[str, Any]] = []
        prompt_mode = (
            "schema_rag"
            if sys_spec.kind in ("schema_rag", "sft_rag")
            else ("few_shot" if sys_spec.kind == "few_shot" else "zero_shot")
        )

        formatted_prompts = [
            retriever.format_prompt(
                question=item["question"],
                mode=prompt_mode,
                few_shot_k=sys_spec.few_shot_k,
                top_tables=sys_spec.rag_top_tables,
                top_defs=sys_spec.rag_top_defs,
                top_examples=sys_spec.rag_top_examples,
            )
            for item in items
        ]

        if mode == "offline_smoke":
            for item in items:
                t0 = time.perf_counter()
                if item.get("is_negative_query"):
                    raw_out = "REFUSAL: UNANSWERABLE_FROM_SCHEMA"
                else:
                    raw_out = f"```sql\n{item['gold_sql']}\n```"
                lat_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                rec = evaluate_single_item(
                    item=item,
                    raw_completion=raw_out,
                    conn=conn,
                    guardrail=guardrail,
                    float_tol=config.float_tolerance,
                    latency_ms=lat_ms,
                    completion_tokens=len(raw_out.split()),
                    cost_usd=0.0,
                )
                records.append(rec)
        elif mode == "cuda_live" and sys_spec.kind in (
            "sft_student",
            "sft_rag",
            "zero_shot",
            "few_shot",
            "schema_rag",
        ):
            cuda_resps = run_cuda_local_inference(
                sys_spec=sys_spec,
                prompts=formatted_prompts,
                max_tokens=config.max_tokens,
                batch_size=4,
            )
            for item, c_resp in zip(items, cuda_resps, strict=True):
                rec = evaluate_single_item(
                    item=item,
                    raw_completion=c_resp.content,
                    conn=conn,
                    guardrail=guardrail,
                    float_tol=config.float_tolerance,
                    latency_ms=c_resp.latency_ms,
                    completion_tokens=c_resp.completion_tokens,
                    cost_usd=c_resp.cost_usd,
                )
                records.append(rec)
        else:
            print(f"\n[*] Evaluating system {sys_spec.id} ({sys_spec.name}) across {len(items)} items...")
            responses: list[Any] = [None] * len(items)

            def _fetch_completion(idx: int, p: str) -> tuple[int, Any]:
                res = call_llm(
                    model=sys_spec.model,
                    prompt=p,
                    system_prompt=SYSTEM_INSTRUCTION,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    cache_dir=config.cache_dir,
                    max_usd=config.max_usd,
                    budget_ledger_path=config.budget_ledger_path,
                )
                return idx, res

            workers = 5
            completed_count = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(_fetch_completion, idx, p)
                    for idx, p in enumerate(formatted_prompts)
                ]
                for fut in concurrent.futures.as_completed(futures):
                    idx, resp = fut.result()
                    responses[idx] = resp
                    completed_count += 1
                    if completed_count % 25 == 0 or completed_count == len(items):
                        print(f"    [{sys_spec.id}] Progress: {completed_count}/{len(items)} items fetched ({completed_count / len(items) * 100:.1f}%)")

            # Evaluate execution in DuckDB deterministically in order
            for item, resp in zip(items, responses, strict=True):
                rec = evaluate_single_item(
                    item=item,
                    raw_completion=resp.content,
                    conn=conn,
                    guardrail=guardrail,
                    float_tol=config.float_tolerance,
                    latency_ms=resp.latency_ms,
                    completion_tokens=resp.completion_tokens,
                    cost_usd=resp.cost_usd,
                )
                records.append(rec)

        all_system_records[sys_spec.id] = records
        system_summaries[sys_spec.id] = summarize_system_run(
            system=sys_spec,
            records=records,
            n_bootstrap=config.bootstrap_resamples,
            seed=config.warehouse_seed,
            mode=mode,
        )

        # Catastrophic forgetting check on cross-domain general SQL holdout
        if (
            mode == "cuda_live"
            and sys_spec.kind in ("sft_student", "sft_rag")
            and Path(config.general_holdout_path).is_file()
        ):
            from src.eval.general_holdout import GENERAL_SCHEMAS_DDL, evaluate_general_holdout

            h_items = [
                json.loads(line)
                for line in Path(config.general_holdout_path)
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            h_prompts = [
                f"### Database Schema:\n{GENERAL_SCHEMAS_DDL}\n\n### Task:\nWrite a clean DuckDB SQL query inside a ```sql ... ``` block to answer:\n{hit['question']}"
                for hit in h_items
            ]
            print(
                f"[*] Evaluating catastrophic forgetting on general holdout ({len(h_items)} items)..."
            )
            h_resps = run_cuda_local_inference(
                sys_spec=sys_spec,
                prompts=h_prompts,
                max_tokens=config.max_tokens,
                batch_size=4,
            )
            h_preds = {
                h_items[k]["id"]: extract_sql(h_resps[k].content) for k in range(len(h_items))
            }
            h_report = evaluate_general_holdout(h_preds, holdout_path=config.general_holdout_path)
            system_summaries[sys_spec.id]["general_sql_holdout_ex"] = h_report["execution_accuracy"]
            system_summaries[sys_spec.id]["general_sql_holdout_ci95"] = h_report["ci95"]

    # Compute paired McNemar tests on answerable items
    mcnemar_results: dict[str, Any] = {}
    for id_a, id_b in config.mcnemar_pairs:
        if id_a in all_system_records and id_b in all_system_records:
            recs_a = [r for r in all_system_records[id_a] if not r.get("is_negative_query")]
            recs_b = [r for r in all_system_records[id_b] if not r.get("is_negative_query")]
            preds_a = [bool(r["is_match"]) for r in recs_a]
            preds_b = [bool(r["is_match"]) for r in recs_b]
            mcnemar_results[f"{id_a}_vs_{id_b}"] = compute_mcnemar_test(preds_a, preds_b)

    holdout_p = Path(config.general_holdout_path)
    holdout_sha = (
        hashlib.sha256(holdout_p.read_bytes()).hexdigest() if holdout_p.is_file() else None
    )

    conn.close()
    return {
        "run_name": config.run_name,
        "mode": mode,
        "is_model_evidence": mode in ("live", "cuda_live"),
        "test_path": config.test_path,
        "general_holdout_sha256": holdout_sha,
        "systems": system_summaries,
        "mcnemar_tests": mcnemar_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run schema-tuned-sql evaluation suite")
    parser.add_argument("--config", default="configs/eval.yaml", help="Path to eval YAML config")
    parser.add_argument(
        "--mode",
        choices=["offline_smoke", "live", "cuda_live"],
        default="offline_smoke",
        help="Evaluation mode ('offline_smoke' for wiring verification, 'live' for OpenRouter, 'cuda_live' for local T4 GPU)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on test items")
    parser.add_argument(
        "--systems",
        default=None,
        help="Optional comma-separated system IDs to evaluate (e.g. 'T1' or 'T1,T2')",
    )
    parser.add_argument("--out", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    cfg = EvalConfig.from_yaml(args.config)
    sys_ids = [s.strip() for s in args.systems.split(",") if s.strip()] if args.systems else None
    report = run_evaluation_suite(cfg, mode=args.mode, limit=args.limit, system_ids=sys_ids)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved evaluation report to {out_path}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
