"""Distillation orchestrator producing verified scaling splits (500, 1k, 2k, 5k) and contamination telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
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

    for cand in candidates:
        diff = cand["difficulty"]
        is_neg = bool(cand.get("is_negative_query", False))
        question = cand["question"]
        stats_by_diff[diff]["attempted"] += 1

        if mode == "live":
            prompt = retriever.format_prompt(question=question, mode="zero_shot")
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
            raw_completion = resp1.content
            sec_completion = None
            if config.require_dual_agreement and not is_neg:
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
                sec_completion = resp2.content
            teacher_source = config.primary_teacher
        else:
            raw_completion = cand["candidate_sql"]
            sec_completion = None
            teacher_source = "template_bootstrap"

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
            secondary_completion=sec_completion,
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
