"""Statistical metrics computation for text-to-SQL evaluations.

Implements Execution Accuracy (EX), Valid SQL rate, Refusal Accuracy,
2,000-resample bootstrap 95% confidence intervals, and paired McNemar significance tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import binomtest


def compute_bootstrap_ci(
    values: list[float] | np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Calculates percentile bootstrap confidence interval for mean of values."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return float(arr[0]), float(arr[0])

    rng = np.random.default_rng(seed)
    # Resample with replacement
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    bootstrap_means = np.mean(arr[indices], axis=1)

    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    low = float(np.percentile(bootstrap_means, lower_pct))
    high = float(np.percentile(bootstrap_means, upper_pct))
    return round(low, 4), round(high, 4)


def compute_mcnemar_test(
    predictions_a: list[bool] | np.ndarray,
    predictions_b: list[bool] | np.ndarray,
) -> dict[str, Any]:
    """Computes exact two-sided McNemar test for paired binary classification.

    b: Model A correct, Model B incorrect
    c: Model A incorrect, Model B correct
    """
    arr_a = np.asarray(predictions_a, dtype=bool)
    arr_b = np.asarray(predictions_b, dtype=bool)
    if len(arr_a) != len(arr_b):
        raise ValueError(f"Array lengths must match: {len(arr_a)} vs {len(arr_b)}")

    b = int(np.sum(arr_a & ~arr_b))
    c = int(np.sum(~arr_a & arr_b))
    discordant_total = b + c

    if discordant_total == 0:
        p_val = 1.0
    else:
        res = binomtest(b, n=discordant_total, p=0.5, alternative="two-sided")
        p_val = float(res.pvalue)

    return {
        "b_a_only": b,
        "c_b_only": c,
        "discordant_pairs": discordant_total,
        "p_value": round(p_val, 5),
        "is_significant_05": p_val < 0.05,
    }


def aggregate_system_metrics(
    records: list[dict[str, Any]],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Computes aggregate execution accuracy, valid SQL rate, refusal rate, and bootstrap CIs."""
    if not records:
        return {
            "total_samples": 0,
            "execution_accuracy": 0.0,
            "execution_accuracy_ci95": (0.0, 0.0),
            "valid_sql_rate": 0.0,
            "refusal_accuracy": 0.0,
            "by_difficulty": {},
        }

    total_samples = len(records)
    ex_scores = [1.0 if r.get("is_match") else 0.0 for r in records]
    valid_sql_scores = [1.0 if r.get("is_valid_sql") else 0.0 for r in records]

    # Negative / refusal queries
    refusal_records = [r for r in records if r.get("is_negative_query")]
    refusal_acc = (
        sum(1.0 for r in refusal_records if r.get("refusal_detected")) / len(refusal_records)
        if refusal_records
        else 1.0
    )

    # Standard answerable queries (exclude negative queries for primary EX denominator)
    answerable_records = [r for r in records if not r.get("is_negative_query")]
    ans_ex_scores = (
        [1.0 if r.get("is_match") else 0.0 for r in answerable_records]
        if answerable_records
        else ex_scores
    )

    ex_mean = float(np.mean(ans_ex_scores)) if ans_ex_scores else 0.0
    ex_ci = compute_bootstrap_ci(ans_ex_scores, n_bootstrap=n_bootstrap, seed=seed)

    valid_sql_mean = float(np.mean(valid_sql_scores))

    # Difficulty breakdown
    by_diff: dict[str, dict[str, Any]] = {}
    for diff in ["easy", "medium", "hard"]:
        diff_recs = [r for r in answerable_records if r.get("difficulty") == diff]
        if diff_recs:
            diff_scores = [1.0 if r.get("is_match") else 0.0 for r in diff_recs]
            diff_mean = float(np.mean(diff_scores))
            diff_ci = compute_bootstrap_ci(diff_scores, n_bootstrap=n_bootstrap, seed=seed)
            by_diff[diff] = {
                "count": len(diff_recs),
                "execution_accuracy": round(diff_mean, 4),
                "ci95": diff_ci,
            }

    return {
        "total_samples": total_samples,
        "answerable_samples": len(answerable_records),
        "negative_samples": len(refusal_records),
        "execution_accuracy": round(ex_mean, 4),
        "execution_accuracy_ci95": ex_ci,
        "valid_sql_rate": round(valid_sql_mean, 4),
        "refusal_accuracy": round(refusal_acc, 4),
        "by_difficulty": by_diff,
    }
