"""In-process counters for the System tab and /api/metrics: routing mix, cache, fallbacks, latency, cost."""
from __future__ import annotations

from collections import Counter, deque


def estimate_tokens(text: str) -> int:
    """~4 characters per token is close enough for a cost estimate."""
    return max(1, len(text) // 4)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct * (len(ordered) - 1)))]


class Metrics:
    def __init__(self, pricing: dict | None = None, baseline_model: str = "", window: int = 500):
        self.pricing = pricing or {}
        self.baseline_model = baseline_model  # "what it would cost if every call used the smart model"
        self.calls = self.cache_hits = self.retries = self.fallbacks = self.degraded = self.continuations = 0
        self.by_tier: Counter[str] = Counter()
        self.by_model: Counter[str] = Counter()
        self.by_agent: Counter[str] = Counter()
        self.cost = self.cost_all_smart = 0.0
        self._latency: deque[float] = deque(maxlen=window)
        self._first_token: deque[float] = deque(maxlen=window)
        self.recent: deque[dict] = deque(maxlen=25)

    def price(self, model: str, tokens_in: int, tokens_out: int) -> float:
        p = self.pricing.get(model.split(":")[-1])
        if not p:
            return 0.0
        return (tokens_in * float(p.get("input", 0)) + tokens_out * float(p.get("output", 0))) / 1_000_000

    def record(self, trace) -> None:
        self.calls += 1
        self.by_agent[trace.agent or "unknown"] += 1
        self.retries += trace.retries
        self.fallbacks += len(trace.fallbacks)
        self.continuations += int(trace.continued)
        self._latency.append(trace.latency_ms)
        if trace.first_token_ms is not None:
            self._first_token.append(trace.first_token_ms)
        if trace.cache_hit:
            self.cache_hits += 1
        elif trace.degraded:
            self.degraded += 1
        else:
            self.by_tier[trace.tier] += 1
            self.by_model[trace.model] += 1
            self.cost += self.price(trace.model, trace.tokens_in, trace.tokens_out)
        # A cache hit costs nothing, but would have cost a smart-model call without router + cache.
        self.cost_all_smart += self.price(self.baseline_model, trace.tokens_in, trace.tokens_out)
        self.recent.appendleft(trace.as_dict())

    def snapshot(self) -> dict:
        lat, first = list(self._latency), list(self._first_token)
        return {
            "llm_calls": self.calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(self.cache_hits / self.calls, 4) if self.calls else 0.0,
            "by_tier": dict(self.by_tier),
            "by_model": dict(self.by_model),
            "by_agent": dict(self.by_agent),
            "retries": self.retries,
            "fallbacks": self.fallbacks,
            "continuations": self.continuations,
            "degraded_answers": self.degraded,
            "latency_ms": {"p50": round(_percentile(lat, 0.5)), "p95": round(_percentile(lat, 0.95))},
            "first_token_ms": {"p50": round(_percentile(first, 0.5)), "p95": round(_percentile(first, 0.95))},
            "estimated_cost_usd": round(self.cost, 6),
            "estimated_cost_all_smart_usd": round(self.cost_all_smart, 6),
            "estimated_savings_usd": round(max(self.cost_all_smart - self.cost, 0.0), 6),
            "recent": list(self.recent),
        }
