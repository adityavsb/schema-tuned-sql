"""Distillation orchestrator producing verified scaling splits (500, 1k, 2k, 5k) and contamination telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.datagen.dedupe import ContaminationFilter
from src.datagen.templates import generate_candidate_candidates
from src.datagen.verifier import SQLVerifier
from src.eval.rag import SYSTEM_INSTRUCTION, SchemaDefinitionRetriever
from src.llm.provider import call_llm
from src.utils.config import DataGenConfig
from warehouse.generate_data import generate_synthetic_warehouse


def format_batch_teacher_prompt(
    questions: list[str],
    retriever: SchemaDefinitionRetriever,
) -> str:
    """Formats a single multi-question teacher prompt sharing the schema and business definitions header."""
    if len(questions) == 1:
        return retriever.format_prompt(question=questions[0], mode="zero_shot")

    base_header = retriever.format_prompt(question="__PLACEHOLDER__", mode="zero_shot").split(
        "### Question:\n__PLACEHOLDER__"
    )[0]
    numbered = "\n".join(f"{idx}. {q}" for idx, q in enumerate(questions, start=1))
    return (
        f"{base_header}"
        f"### Batched Questions ({len(questions)} items):\n"
        f"{numbered}\n\n"
        f"### Output Format:\n"
        f"Return a JSON array of {len(questions)} objects in exact order:\n"
        f'[{{"index": 1, "sql": "SELECT ..."}}, {{"index": 2, "sql": "REFUSAL: UNANSWERABLE_FROM_SCHEMA"}}, ...]'
    )


def parse_batched_sql_completions(raw_text: str, expected_count: int) -> list[str]:
    """Extracts individual SQL or refusal strings from a batched JSON or numbered response."""
    if expected_count == 1:
        return [raw_text.strip()]

    cleaned = raw_text.strip()
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    json_candidate = fenced.group(1) if fenced else cleaned

    # Try JSON array extraction first
    bracket_match = re.search(r"\[.*\]", json_candidate, flags=re.DOTALL)
    if bracket_match:
        try:
            arr = json.loads(bracket_match.group(0))
            if isinstance(arr, list):
                results: list[str] = []
                for item in arr:
                    if isinstance(item, dict):
                        results.append(str(item.get("sql", "")).strip())
                    elif isinstance(item, str):
                        results.append(item.strip())
                if len(results) < expected_count:
                    results.extend([""] * (expected_count - len(results)))
                return results[:expected_count]
        except Exception:
            pass

    # Fallback: extract fenced code blocks in order
    blocks = re.findall(r"```(?:sql|duckdb)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        if len(blocks) < expected_count:
            blocks.extend([""] * (expected_count - len(blocks)))
        return [b.strip() for b in blocks[:expected_count]]

    return [cleaned] + [""] * (expected_count - 1)


def run_distillation_pipeline(
    config: DataGenConfig,
    mode: str = "dry_run",
    target_override: int | None = None,
) -> dict[str, Any]:
    """Generates, verifies, deduplicates, and writes training data splits."""
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "dry_run":
        empty_bucket = {
            "attempted": None,
            "accepted": None,
            "acceptance_rate": None,
            "rejected_execution": None,
            "rejected_contamination": None,
            "rejected_duplicate": None,
        }
        manifest = {
            "run_name": config.run_name,
            "mode": "dry_run",
            "status": "pending_openrouter_run",
            "is_live_teacher_distillation": False,
            "is_model_evidence": False,
            "primary_teacher": config.primary_teacher,
            "secondary_teacher": config.secondary_teacher
            if config.require_dual_agreement
            else None,
            "questions_per_call": config.questions_per_call,
            "total_api_calls": None,
            "total_verified_pairs": None,
            "contamination_threshold": config.contamination_sim_threshold,
            "contaminated_removed": None,
            "duplicates_removed": None,
            "rejection_breakdown": None,
            "acceptance_by_difficulty": {
                "easy": dict(empty_bucket),
                "medium": dict(empty_bucket),
                "hard": dict(empty_bucket),
            },
            "scaling_splits": {},
        }
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    target_n = target_override or config.target_verified_pairs

    conn = generate_synthetic_warehouse(
        db_path=":memory:",
        seed=config.seed,
        num_accounts=100,
        num_events=15000,
    )
    verifier = SQLVerifier(conn)
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    contamination = ContaminationFilter(
        test_path=REPO_ROOT / "data" / "test_gold.jsonl",
        dev_path=REPO_ROOT / "data" / "dev.jsonl",
        similarity_threshold=config.contamination_sim_threshold,
    )
    retriever = SchemaDefinitionRetriever()

    # Generate candidate pool with headroom for filtering
    pool_size = int(target_n * 1.45) + 500
    candidates = generate_candidate_candidates(seed=config.seed, count=pool_size)

    verified_items: list[dict[str, Any]] = []
    unfiltered_items: list[dict[str, Any]] = []

    stats_by_diff: dict[str, Counter[str]] = {
        "easy": Counter(),
        "medium": Counter(),
        "hard": Counter(),
    }
    rejection_reasons: Counter[str] = Counter()
    contaminated_removed = 0
    duplicates_removed = 0
    total_api_calls = 0

    batch_size = config.questions_per_call if mode == "live" else 1
    for start_idx in range(0, len(candidates), batch_size):
        if len(verified_items) >= target_n:
            break
        batch_cands = candidates[start_idx : start_idx + batch_size]

        if mode == "live":
            batch_questions = [c["question"] for c in batch_cands]
            prompt = format_batch_teacher_prompt(batch_questions, retriever)
            resp1 = call_llm(
                model=config.primary_teacher,
                prompt=prompt,
                system_prompt=SYSTEM_INSTRUCTION,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                cache_dir=config.cache_dir,
                max_usd=config.max_usd,
                budget_ledger_path=config.budget_ledger_path,
            )
            total_api_calls += 1
            primary_completions = parse_batched_sql_completions(resp1.content, len(batch_cands))

            if config.require_dual_agreement:
                resp2 = call_llm(
                    model=config.secondary_teacher,
                    prompt=prompt,
                    system_prompt=SYSTEM_INSTRUCTION,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    cache_dir=config.cache_dir,
                    max_usd=config.max_usd,
                    budget_ledger_path=config.budget_ledger_path,
                )
                total_api_calls += 1
                secondary_completions: list[str | None] = list(
                    parse_batched_sql_completions(resp2.content, len(batch_cands))
                )
            else:
                secondary_completions = [None] * len(batch_cands)
            teacher_source = config.primary_teacher
        else:
            primary_completions = [c["candidate_sql"] for c in batch_cands]
            secondary_completions = [None] * len(batch_cands)
            teacher_source = "template_bootstrap"

        for cand, raw_completion, sec_completion in zip(
            batch_cands, primary_completions, secondary_completions, strict=True
        ):
            diff = cand["difficulty"]
            is_neg = bool(cand.get("is_negative_query", False))
            question = cand["question"]
            stats_by_diff[diff]["attempted"] += 1

            # Save to unfiltered pool (for ablation comparing execution-filtered vs unfiltered)
            if len(unfiltered_items) < target_n:
                unfiltered_items.append(
                    {
                        "id": f"unfiltered_{len(unfiltered_items) + 1:05d}",
                        "difficulty": diff,
                        "category": cand["category"],
                        "question": question,
                        "sql": raw_completion,
                        "is_negative_query": is_neg,
                        "source": teacher_source,
                    }
                )

            accepted, reason, clean_sql, _ = verifier.verify_pair(
                question=question,
                raw_sql_or_completion=raw_completion,
                is_negative_query=is_neg,
                secondary_completion=sec_completion if not is_neg else None,
            )
            if not accepted:
                rejection_reasons[reason] += 1
                stats_by_diff[diff]["rejected_execution"] += 1
                continue

            is_clean, dedupe_reason, max_sim = contamination.check_candidate(question, clean_sql)
            if not is_clean:
                rejection_reasons[dedupe_reason] += 1
                if dedupe_reason.startswith("contaminated"):
                    contaminated_removed += 1
                    stats_by_diff[diff]["rejected_contamination"] += 1
                else:
                    duplicates_removed += 1
                    stats_by_diff[diff]["rejected_duplicate"] += 1
                continue

            stats_by_diff[diff]["accepted"] += 1
            verified_items.append(
                {
                    "id": f"train_{len(verified_items) + 1:05d}",
                    "difficulty": diff,
                    "category": cand["category"],
                    "persona": cand["persona"],
                    "question": question,
                    "sql": clean_sql,
                    "is_negative_query": is_neg,
                    "max_holdout_sim": max_sim,
                    "source": teacher_source,
                }
            )
            if len(verified_items) >= target_n:
                break

        if mode == "live" and (total_api_calls % 10 == 0 or len(verified_items) >= target_n):
            tot_rej = sum(rejection_reasons.values())
            pct = (len(verified_items) / target_n) * 100.0
            print(
                f"    [Distill] API Calls: {total_api_calls} | Verified: {len(verified_items)}/{target_n} "
                f"({pct:.1f}%) | Rejected: {tot_rej}"
            )

    # Write scaling splits and record SHA-256 hashes
    split_hashes: dict[str, dict[str, Any]] = {}
    for split_size in config.scaling_splits:
        if split_size <= len(verified_items):
            subset = verified_items[:split_size]
            split_file = out_dir / f"train_{split_size}.jsonl"
            content = "\n".join(json.dumps(x) for x in subset) + "\n"
            split_file.write_text(content, encoding="utf-8")
            sha = hashlib.sha256(split_file.read_bytes()).hexdigest()
            split_hashes[f"train_{split_size}"] = {
                "path": str(split_file),
                "count": len(subset),
                "sha256": sha,
            }

    acceptance_rates: dict[str, Any] = {}
    for diff, counts in stats_by_diff.items():
        att = max(1, counts["attempted"])
        acc = counts["accepted"]
        acceptance_rates[diff] = {
            "attempted": counts["attempted"],
            "accepted": acc,
            "acceptance_rate": round(acc / att, 4),
            "rejected_execution": counts["rejected_execution"],
            "rejected_contamination": counts["rejected_contamination"],
            "rejected_duplicate": counts["rejected_duplicate"],
        }

    manifest = {
        "run_name": config.run_name,
        "mode": mode,
        "is_live_teacher_distillation": mode == "live",
        "primary_teacher": config.primary_teacher if mode == "live" else "template_bootstrap",
        "secondary_teacher": config.secondary_teacher if config.require_dual_agreement else None,
        "questions_per_call": config.questions_per_call,
        "total_api_calls": total_api_calls if mode == "live" else 0,
        "total_verified_pairs": len(verified_items),
        "contamination_threshold": config.contamination_sim_threshold,
        "contaminated_removed": contaminated_removed,
        "duplicates_removed": duplicates_removed,
        "rejection_breakdown": dict(rejection_reasons),
        "acceptance_by_difficulty": acceptance_rates,
        "scaling_splits": split_hashes,
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    conn.close()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run execution-verified SQL distillation")
    parser.add_argument("--config", default="configs/data_gen.yaml", help="Path to data_gen YAML")
    parser.add_argument(
        "--mode",
        choices=["dry_run", "template_bootstrap", "live"],
        default="dry_run",
        help="Generation mode ('dry_run' for pending status, 'template_bootstrap' for offline smoke, 'live' for OpenRouter teacher)",
    )
    parser.add_argument("--target", type=int, default=None, help="Override target pair count")
    args = parser.parse_args()

    cfg = DataGenConfig.from_yaml(args.config)
    manifest = run_distillation_pipeline(cfg, mode=args.mode, target_override=args.target)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
