"""Stretch goal (S5): Execution-labeled preference pair builder for DPO / ORPO tuning."""

from __future__ import annotations

from typing import Any

from src.eval.rag import SYSTEM_INSTRUCTION, SchemaDefinitionRetriever


def build_execution_preference_pairs(
    verified_items: list[dict[str, Any]],
    rejected_candidates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Constructs (prompt, chosen, rejected) triples from execution-verified vs failed SQL."""
    retriever = SchemaDefinitionRetriever()
    rejected_by_q = {
        r["question"].strip().lower(): r.get("sql", "") for r in rejected_candidates if r.get("sql")
    }

    pairs: list[dict[str, str]] = []
    for item in verified_items:
        if item.get("is_negative_query"):
            continue
        q = item["question"]
        q_key = q.strip().lower()
        chosen_sql = item.get("sql") or item.get("gold_sql") or ""
        if not chosen_sql:
            continue

        # Use recorded failed candidate SQL if available, else synthesize a plausible unverified corruption
        rejected_sql = rejected_by_q.get(q_key)
        if not rejected_sql:
            if "WHERE" in chosen_sql:
                # Corrupt by dropping WHERE filter (simulates definition-violating teacher output)
                rejected_sql = chosen_sql.split("WHERE")[0].strip()
            else:
                rejected_sql = chosen_sql.replace("COUNT(*)", "COUNT(hallucinated_col)")

        if rejected_sql.strip() == chosen_sql.strip():
            continue

        user_prompt = retriever.format_prompt(question=q, mode="zero_shot")
        full_prompt = (
            f"<|im_start|>system\n{SYSTEM_INSTRUCTION}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        pairs.append(
            {
                "prompt": full_prompt,
                "chosen": f"```sql\n{chosen_sql.strip()}\n```<|im_end|>\n",
                "rejected": f"```sql\n{rejected_sql.strip()}\n```<|im_end|>\n",
            }
        )
    return pairs
