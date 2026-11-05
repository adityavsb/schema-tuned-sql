"""Unit tests for LLM disk caching, OpenRouter pricing, secret redaction, and hard USD budget caps."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.llm.cache import DiskResponseCache
from src.llm.pricing import (
    BudgetExceededError,
    BudgetTracker,
    estimate_cost_usd,
    get_model_rates_usd,
)
from src.llm.provider import call_llm, sanitize_error_detail


def test_disk_cache_roundtrip(tmp_path: Path):
    cache = DiskResponseCache(cache_dir=tmp_path / "cache")
    assert cache.get("deepseek/deepseek-chat", "SELECT 1", temperature=0.0) is None

    payload = {
        "content": "```sql\nSELECT 1\n```",
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
        "cost_usd": 0.0001,
    }
    cache.set("deepseek/deepseek-chat", "SELECT 1", payload, temperature=0.0)
    hit = cache.get("deepseek/deepseek-chat", "SELECT 1", temperature=0.0)
    assert hit is not None
    assert hit["content"] == "```sql\nSELECT 1\n```"

    removed = cache.clear()
    assert removed == 1
    assert cache.get("deepseek/deepseek-chat", "SELECT 1", temperature=0.0) is None


def test_pricing_table_and_unregistered_model_rejection():
    in_rate, out_rate = get_model_rates_usd("deepseek/deepseek-chat")
    assert in_rate == 0.14
    assert out_rate == 0.28

    cost = estimate_cost_usd("deepseek/deepseek-chat", prompt_tokens=950, completion_tokens=110)
    assert cost == pytest.approx(0.000164, abs=1e-6)

    with pytest.raises(KeyError, match="Unregistered model"):
        get_model_rates_usd("anthropic/claude-3-opus-expensive")


def test_hard_budget_cap_enforcement(tmp_path: Path):
    ledger = tmp_path / "budget.json"
    tracker = BudgetTracker(max_usd=0.05, ledger_path=ledger)
    assert tracker.total_spent_usd == 0.0

    tracker.record_spend(prompt_tokens=10000, completion_tokens=2000, cost_usd=0.04)
    assert tracker.total_spent_usd == pytest.approx(0.04)

    # Next spend of $0.02 would exceed $0.05 cap
    with pytest.raises(BudgetExceededError, match="Budget cap exceeded"):
        tracker.check_can_spend(projected_cost_usd=0.02)


def test_secret_redaction_in_error_messages():
    raw_err = "HTTP 401: Authorization: Bearer sk-or-v1-998877665544332211 failed with key sk-1234567890abcdef"
    clean = sanitize_error_detail(raw_err)
    assert "sk-or-v1" not in clean
    assert "sk-1234567890abcdef" not in clean
    assert "[REDACTED]" in clean


def test_call_llm_returns_cached_without_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cache_dir = tmp_path / "llm_cache"
    cache = DiskResponseCache(cache_dir=cache_dir)
    cache.set(
        model="deepseek/deepseek-chat",
        prompt="SYS:sys\nUSER:hello",
        response={
            "content": "SELECT 42",
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "total_tokens": 15,
        },
        temperature=0.0,
    )

    resp = call_llm(
        model="deepseek/deepseek-chat",
        prompt="hello",
        system_prompt="sys",
        temperature=0.0,
        cache_dir=str(cache_dir),
        budget_ledger_path=str(tmp_path / "ledger.json"),
    )
    assert resp.cached is True
    assert resp.cost_usd == 0.0
    assert resp.content == "SELECT 42"


def test_call_llm_missing_key_raises_on_cache_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY environment variable is missing"):
        call_llm(
            model="deepseek/deepseek-chat",
            prompt="uncached prompt",
            cache_dir=str(tmp_path / "empty_cache"),
            budget_ledger_path=str(tmp_path / "ledger.json"),
        )
