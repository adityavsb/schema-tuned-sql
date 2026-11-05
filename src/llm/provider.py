"""LiteLLM -> OpenRouter provider layer with disk cache, cost logging, and hard USD cap."""

from __future__ import annotations

import os
import re
import time
from typing import Any

from pydantic import BaseModel, Field

from src.llm.cache import DiskResponseCache
from src.llm.pricing import BudgetTracker, estimate_cost_usd
from src.utils.config import load_dotenv

load_dotenv()


def sanitize_error_detail(text: str, max_len: int = 200) -> str:
    """Redacts any Bearer tokens or sk-* API keys from provider error strings."""
    redacted = re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", text, flags=re.IGNORECASE)
    redacted = re.sub(r"sk-[a-zA-Z0-9_-]{6,}", "sk-[REDACTED]", redacted)
    if len(redacted) > max_len:
        return redacted[:max_len] + "..."
    return redacted


class LLMResponse(BaseModel):
    """Structured LLM completion with token counts, USD cost, and cache provenance."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cached: bool = False
    model: str
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


def call_llm(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: float = 90.0,
    max_retries: int = 3,
    use_cache: bool = True,
    cache_dir: str = ".cache/llm_responses",
    max_usd: float = 15.0,
    budget_ledger_path: str = ".cache/budget_ledger.json",
) -> LLMResponse:
    """Calls an OpenRouter model (DeepSeek / Qwen) with disk caching and hard budget enforcement."""
    full_prompt_key = f"SYS:{system_prompt or ''}\nUSER:{prompt}"
    cache = DiskResponseCache(cache_dir=cache_dir) if use_cache else None

    if cache is not None:
        hit = cache.get(model=model, prompt=full_prompt_key, temperature=temperature)
        if hit is not None:
            return LLMResponse(
                content=hit["content"],
                prompt_tokens=hit.get("prompt_tokens", 0),
                completion_tokens=hit.get("completion_tokens", 0),
                total_tokens=hit.get("total_tokens", 0),
                cost_usd=0.0,
                latency_ms=0.0,
                cached=True,
                model=model,
                raw_metadata=hit.get("raw_metadata", {}),
            )

    budget = BudgetTracker(max_usd=max_usd, ledger_path=budget_ledger_path)
    # Estimate conservative pre-call cost (~1500 prompt + max_tokens completion)
    projected_cost = estimate_cost_usd(model, 1500, max_tokens)
    budget.check_can_spend(projected_cost)

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is missing. "
            "Set it in your local .env (gitignored) or environment before running live API calls."
        )

    import litellm

    litellm.suppress_debug_info = True

    model_slug = model if model.startswith("openrouter/") else f"openrouter/{model}"
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    extra_headers = {
        "HTTP-Referer": "https://github.com/adityavsb/schema-tuned-sql",
        "X-Title": "schema-tuned-sql",
    }

    last_error: Exception | None = None
    response: Any = None
    t0 = time.perf_counter()

    for attempt in range(max_retries):
        try:
            response = litellm.completion(
                model=model_slug,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                api_key=api_key,
                extra_headers=extra_headers,
            )
            break
        except Exception as e:
            last_error = e
            err_lower = str(e).lower()
            if attempt < max_retries - 1 and (
                "429" in err_lower
                or "503" in err_lower
                or "rate" in err_lower
                or "overloaded" in err_lower
            ):
                time.sleep(2.0 * (attempt + 1))
                continue
            break

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    if response is None:
        err_cls = type(last_error).__name__ if last_error else "UnknownError"
        detail = sanitize_error_detail(str(last_error)) if last_error else ""
        raise RuntimeError(
            f"OpenRouter call failed for '{model_slug}' after {max_retries} attempts ({err_cls}): {detail}"
        ) from last_error

    choice = response.choices[0]
    content = getattr(choice.message, "content", "") or ""
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or len(full_prompt_key.split()))
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or len(content.split()))
    total_tokens = prompt_tokens + completion_tokens

    cost_usd = estimate_cost_usd(model, prompt_tokens, completion_tokens)
    budget.record_spend(prompt_tokens, completion_tokens, cost_usd)

    payload = {
        "content": content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "latency_ms": latency_ms,
        "model": model,
        "raw_metadata": {"provider": "openrouter"},
    }
    if cache is not None:
        cache.set(model=model, prompt=full_prompt_key, response=payload, temperature=temperature)

    return LLMResponse(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        cached=False,
        model=model,
        raw_metadata={"provider": "openrouter"},
    )
