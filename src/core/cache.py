"""Caching layer.

TTLCache      - small LRU + TTL store that also remembers each entry's age, so an
                expired entry can still be served as a labelled fallback.
ResponseCache - LLM answer cache keyed by a hash of the *whole* prompt. Because
                tool output (quotes, portfolio metrics, retrieved passages) is part
                of the prompt, the key changes whenever the underlying data changes,
                so a cached answer can never describe stale numbers.

Both are in-process. To scale past one worker, swap the dict for Redis: the
interface (get / set / stats) is all the rest of the app depends on.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

_WS = re.compile(r"\s+")


class TTLCache:
    def __init__(self, max_items: int = 2000, clock: Callable[[], float] = time.monotonic):
        self._data: OrderedDict[Any, tuple[float, Any]] = OrderedDict()
        self._max, self._clock = max_items, clock

    def get(self, key) -> tuple[Any, float] | None:
        """Return (value, age_seconds) or None. Expiry is the caller's decision."""
        hit = self._data.get(key)
        if hit is None:
            return None
        self._data.move_to_end(key)
        stored_at, value = hit
        return value, self._clock() - stored_at

    def set(self, key, value) -> None:
        self._data[key] = (self._clock(), value)
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)

    def delete(self, key) -> None:
        self._data.pop(key, None)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


def normalize(text: str) -> str:
    """'What is an ETF?' and '  what is an etf ' should share one cache entry."""
    return _WS.sub(" ", text).strip().lower().rstrip("?!. ")


def prompt_key(messages: list[tuple[str, str]], namespace: str = "") -> str:
    """Stable hash of a prompt. The final user turn is normalised; everything else is exact."""
    parts = [[role, content] for role, content in messages]
    if parts and parts[-1][0] == "user":
        parts[-1][1] = normalize(parts[-1][1])
    blob = json.dumps([namespace, parts], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    """Exact-match LLM response cache with TTL, LRU eviction and hit/miss counters."""

    def __init__(self, ttl: float = 3600.0, max_items: int = 1000, clock: Callable[[], float] = time.monotonic):
        self.ttl = ttl
        self._store = TTLCache(max_items, clock)
        self.hits = self.misses = self.writes = 0

    @property
    def enabled(self) -> bool:
        return self.ttl > 0

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        hit = self._store.get(key)
        if hit is None or hit[1] >= self.ttl:
            if hit is not None:
                self._store.delete(key)
            self.misses += 1
            return None
        self.hits += 1
        return hit[0]

    def set(self, key: str, value: Any) -> None:
        if self.enabled and value:
            self._store.set(key, value)
            self.writes += 1

    def clear(self) -> None:
        self._store.clear()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "enabled": self.enabled, "ttl_seconds": self.ttl, "entries": len(self._store),
            "hits": self.hits, "misses": self.misses, "writes": self.writes,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }
