"""OpenRouter token pricing table and cumulative hard-dollar budget tracker."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

PRICE_TABLE_DATE = "2026-11-05"

# USD per 1,000,000 tokens: (input_usd_per_1m, output_usd_per_1m)
# Restricted to cheap OpenRouter models (DeepSeek and Qwen families) + local student weights.
MODEL_PRICE_TABLE_USD: dict[str, tuple[float, float]] = {
    "deepseek/deepseek-chat": (0.14, 0.28),
    "deepseek/deepseek-v3": (0.14, 0.28),
    "deepseek/deepseek-v4.1-flash": (0.14, 0.28),
    "qwen/qwen-2.5-coder-32b-instruct": (0.07, 0.16),
    "qwen/qwen-2.5-72b-instruct": (0.12, 0.30),
    "qwen/qwen-2.5-coder-7b-instruct": (0.04, 0.10),
    "qwen/qwen-2.5-7b-instruct": (0.04, 0.10),
    "qwen/qwen3.7-flash": (0.05, 0.15),
    "qwen/qwen2.5-coder-1.5b-instruct": (0.0, 0.0),
    "qwen/qwen2.5-coder-3b-instruct": (0.0, 0.0),
    "qwen/qwen2.5-coder-7b-instruct": (0.0, 0.0),
    "local": (0.0, 0.0),
    "offline": (0.0, 0.0),
}


class BudgetExceededError(RuntimeError):
    """Raised when an API call would exceed the configured hard dollar cap."""

    pass


def get_model_rates_usd(model_name: str) -> tuple[float, float]:
    """Returns (input_usd_per_1m, output_usd_per_1m) for a model slug."""
    norm = model_name.strip().lower().replace("openrouter/", "")
    if norm in MODEL_PRICE_TABLE_USD:
        return MODEL_PRICE_TABLE_USD[norm]

    if ":free" in norm or norm.startswith("local/") or norm.startswith("hf/"):
        return 0.0, 0.0

    for key, rates in MODEL_PRICE_TABLE_USD.items():
        if key in norm:
            return rates

    if "deepseek" in norm:
        return 0.14, 0.28
    if "qwen" in norm:
        return 0.07, 0.16

    raise KeyError(
        f"Unregistered model '{model_name}' in MODEL_PRICE_TABLE_USD (dated {PRICE_TABLE_DATE}). "
        "Only cheap DeepSeek/Qwen OpenRouter models or local student checkpoints are permitted."
    )


def estimate_cost_usd(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Computes exact USD cost from prompt and completion token counts."""
    in_rate, out_rate = get_model_rates_usd(model_name)
    cost = (prompt_tokens / 1_000_000.0) * in_rate + (completion_tokens / 1_000_000.0) * out_rate
    return round(cost, 6)


class BudgetTracker:
    """Persistent cumulative cost ledger enforcing a hard USD budget cap."""

    _global_lock = threading.Lock()

    def __init__(
        self,
        max_usd: float = 15.0,
        ledger_path: str | Path = ".cache/budget_ledger.json",
    ):
        self.max_usd = max_usd
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self._global_lock:
            self._ensure_ledger()

    def _ensure_ledger(self) -> None:
        if not self.ledger_path.is_file():
            self._write_state(
                {
                    "total_spent_usd": 0.0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_calls": 0,
                }
            )

    def _read_state(self) -> dict[str, Any]:
        try:
            return json.loads(self.ledger_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {
                "total_spent_usd": 0.0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_calls": 0,
            }

    def _write_state(self, state: dict[str, Any]) -> None:
        self.ledger_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    @property
    def total_spent_usd(self) -> float:
        with self._global_lock:
            return float(self._read_state().get("total_spent_usd", 0.0))

    def check_can_spend(self, projected_cost_usd: float = 0.0) -> None:
        """Raises BudgetExceededError if current + projected spend exceeds max_usd."""
        current = self.total_spent_usd
        if current >= self.max_usd or (current + projected_cost_usd) > self.max_usd:
            raise BudgetExceededError(
                f"Budget cap exceeded: current spend ${current:.4f} + projected ${projected_cost_usd:.4f} "
                f"exceeds cap ${self.max_usd:.2f}."
            )

    def record_spend(
        self, prompt_tokens: int, completion_tokens: int, cost_usd: float
    ) -> dict[str, Any]:
        """Records token usage and cost, enforcing the budget cap."""
        with self._global_lock:
            state = self._read_state()
            new_total = round(float(state.get("total_spent_usd", 0.0)) + cost_usd, 6)
            state["total_spent_usd"] = new_total
            state["total_prompt_tokens"] = int(state.get("total_prompt_tokens", 0)) + prompt_tokens
            state["total_completion_tokens"] = (
                int(state.get("total_completion_tokens", 0)) + completion_tokens
            )
            state["total_calls"] = int(state.get("total_calls", 0)) + 1
            self._write_state(state)
            return state
        return state
