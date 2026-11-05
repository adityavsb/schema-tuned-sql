"""Deduplication and test/dev contamination filter for synthetic training data."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import sqlglot


def normalize_sql_ast(sql: str) -> str:
    """Produces a canonical DuckDB SQL string via sqlglot for exact-AST dedupe."""
    if not sql or not sql.strip() or sql.startswith("REFUSAL"):
        return ""
    try:
        parsed = sqlglot.parse_one(sql, read="duckdb")
        return parsed.sql(dialect="duckdb", normalize=True).strip().lower()
    except Exception:
        return re.sub(r"\s+", " ", sql.strip().lower())


def char_trigram_vector(text: str) -> Counter[str]:
    """Computes character 3-gram frequency vector for fast cosine similarity."""
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    if len(cleaned) < 3:
        return Counter([cleaned])
    return Counter(cleaned[i : i + 3] for i in range(len(cleaned) - 2))


def cosine_similarity(vec_a: Counter[str], vec_b: Counter[str]) -> float:
    """Computes cosine similarity between two sparse Counter vectors."""
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a.keys()) & set(vec_b.keys())
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class ContaminationFilter:
    """Filters out near-duplicates and any candidate too similar to frozen test or dev items."""

    def __init__(
        self,
        test_path: str | Path = "data/test_gold.jsonl",
        dev_path: str | Path = "data/dev.jsonl",
        similarity_threshold: float = 0.85,
    ):
        self.similarity_threshold = similarity_threshold
        self.forbidden_questions: list[str] = []
        self.forbidden_q_vecs: list[Counter[str]] = []
        self.forbidden_asts: set[str] = set()

        for path in (Path(test_path), Path(dev_path)):
            if path.is_file():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    q = item["question"].strip().lower()
                    self.forbidden_questions.append(q)
                    self.forbidden_q_vecs.append(char_trigram_vector(q))
                    ast_norm = normalize_sql_ast(item.get("gold_sql", ""))
                    if ast_norm:
                        self.forbidden_asts.add(ast_norm)

        self.seen_questions: set[str] = set()
        self.seen_question_Sql_pairs: set[str] = set()

    def check_candidate(self, question: str, sql: str) -> tuple[bool, str, float]:
        """Checks whether a candidate (question, sql) is clean and unique.

        Returns (is_clean, reason, max_similarity_to_holdout).
        """
        q_norm = re.sub(r"\s+", " ", question.strip().lower())
        ast_norm = normalize_sql_ast(sql)

        # 1. Exact AST match against frozen test/dev splits
        if ast_norm and ast_norm in self.forbidden_asts:
            return False, "contaminated_exact_ast", 1.0

        # 2. Question similarity against frozen test/dev splits
        q_vec = char_trigram_vector(q_norm)
        max_sim = 0.0
        for holdout_vec in self.forbidden_q_vecs:
            sim = cosine_similarity(q_vec, holdout_vec)
            if sim > max_sim:
                max_sim = sim
            if sim >= self.similarity_threshold:
                return False, "contaminated_question_similarity", round(sim, 4)

        # 3. Internal deduplication within training set
        pair_key = f"{q_norm}||{ast_norm}"
        if q_norm in self.seen_questions or pair_key in self.seen_question_Sql_pairs:
            return False, "internal_duplicate", round(max_sim, 4)

        self.seen_questions.add(q_norm)
        self.seen_question_Sql_pairs.add(pair_key)
        return True, "clean", round(max_sim, 4)
