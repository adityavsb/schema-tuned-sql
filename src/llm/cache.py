"""Deterministic on-disk response cache keyed by SHA-256 of (model, prompt, temperature)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class DiskResponseCache:
    """Persists LLM responses on disk to prevent duplicate OpenRouter spend."""

    def __init__(self, cache_dir: str | Path = ".cache/llm_responses"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_key(model: str, prompt: str, temperature: float = 0.0) -> str:
        """Computes a deterministic SHA-256 cache key."""
        normalized = f"{model.strip().lower()}|temp={temperature:.3f}|{prompt.strip()}"
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, model: str, prompt: str, temperature: float = 0.0) -> dict[str, Any] | None:
        key = self.compute_key(model, prompt, temperature)
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.is_file():
            return None
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def set(
        self,
        model: str,
        prompt: str,
        response: dict[str, Any],
        temperature: float = 0.0,
    ) -> None:
        key = self.compute_key(model, prompt, temperature)
        cache_file = self.cache_dir / f"{key}.json"
        try:
            cache_file.write_text(json.dumps(response, indent=2), encoding="utf-8")
        except OSError:
            pass

    def clear(self) -> int:
        removed = 0
        for item in self.cache_dir.glob("*.json"):
            try:
                item.unlink()
                removed += 1
            except OSError:
                pass
        return removed
