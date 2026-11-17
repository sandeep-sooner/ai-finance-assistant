"""LLM gateway: every model call in the app goes through here.

One `stream()` call does, in order:

  1. route    - ModelRouter picks the `fast` or `smart` tier for this question
  2. cache    - an identical prompt answered recently is replayed (no API call)
  3. call     - stream from the tier's model, with a per-token timeout
  4. retry    - on failure, retry the same model with exponential backoff + jitter
  5. switch   - then try the next model in the chain (other tier, other provider);
                a circuit breaker skips models that keep failing
  6. continue - if a model dies mid-answer, the next one is asked to continue from
                the text already shown, so the user never sees a restart
  7. degrade  - if every model is down, reply with a deterministic answer built
                from the retrieved passages / computed numbers

`stream()` therefore never raises: users always get an answer, never an error.
Details of what happened are written to a CallTrace for logs, metrics and the UI.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass, field

from src.core.cache import ResponseCache, prompt_key
from src.core.metrics import Metrics, estimate_tokens
from src.core.model_router import FAST, SMART, ModelRouter, RouteDecision

log = logging.getLogger(__name__)

Message = tuple[str, str]

CONTINUE_PROMPT = (
    "Your previous answer was cut off. Continue it from exactly where it stopped. "
    "Do not repeat anything already written and do not apologise or mention the interruption."
)
GENERIC_FALLBACK = (
    "I can't reach the language model right now, so I can't write a full answer. "
    "Your question was received and nothing was lost. Please try again in a moment. "
    "The Portfolio, Market and Goals tabs keep working, because their numbers are computed without the model."
)
CUT_SHORT_NOTE = "\n\n*(The answer stops here because the language model became unavailable. Please ask again to continue.)*"


@dataclass
class ModelSpec:
    name: str      # e.g. "openai:gpt-4o-mini"
    tier: str      # "fast" | "smart"
    llm: object    # anything with .stream(messages) -> AsyncIterator[str]


class CircuitBreaker:
    """After `max_failures` consecutive failures, skip the model for `cooldown` seconds."""

    def __init__(self, max_failures: int = 3, cooldown: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.max_failures, self.cooldown, self._clock = max_failures, cooldown, clock
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def open(self) -> bool:
        if self._opened_at is None:
            return False
        if self._clock() - self._opened_at >= self.cooldown:  # half-open: allow one trial call
            self._opened_at = None
            self._failures = self.max_failures - 1
            return False
        return True

    def success(self) -> None:
        self._failures, self._opened_at = 0, None

    def failure(self) -> None:
        self._failures += 1
        if self._failures >= self.max_failures:
            self._opened_at = self._clock()


@dataclass
class CallTrace:
    """What happened during one gateway call. Filled in as the call progresses."""

    agent: str = ""
    tier: str = ""
    model: str = ""
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    cache_hit: bool = False
    retries: int = 0
    fallbacks: list[str] = field(default_factory=list)  # models that failed before the one that answered
    continued: bool = False     # a second model finished an answer the first one started
    degraded: bool = False      # no model answered; the deterministic fallback was used
    latency_ms: float = 0.0
    first_token_ms: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["latency_ms"] = round(self.latency_ms)
        d["first_token_ms"] = None if self.first_token_ms is None else round(self.first_token_ms)
        return d


def _chunks(text: str, size: int = 24):
    """Replay cached / fallback text in small pieces so the UI still streams smoothly."""
    for i in range(0, len(text), size):
        yield text[i:i + size]


class LLMGateway:
    def __init__(
        self,
        models: list[ModelSpec],
        router: ModelRouter | None = None,
        cache: ResponseCache | None = None,
        metrics: Metrics | None = None,
        *,
        retries: int = 2,
        backoff_base: float = 0.5,
        token_timeout: float = 30.0,
        breaker_failures: int = 3,
        breaker_cooldown: float = 60.0,
        sleep: Callable = asyncio.sleep,
    ):
        if not models:
            raise ValueError("LLMGateway needs at least one model")
        self.models = models
        self.router = router or ModelRouter()
        self.cache = cache or ResponseCache(ttl=0)
        self.metrics = metrics or Metrics()
        self.retries, self.backoff_base, self.token_timeout = retries, backoff_base, token_timeout
        self._sleep = sleep
        self._breakers = {m.name: CircuitBreaker(breaker_failures, breaker_cooldown) for m in models}

    @classmethod
    def single(cls, llm, name: str = "single", **kwargs) -> "LLMGateway":
        """Wrap one plain LLM (tests, demo mode): no cache, no retries, same safety net."""
        kwargs.setdefault("retries", 0)
        return cls([ModelSpec(name, FAST, llm)], **kwargs)

    # ---------- helpers ----------

    def chain_for(self, tier: str) -> list[ModelSpec]:
        """Models to try, in order: the chosen tier first, then everything else as fallback."""
        preferred = [m for m in self.models if m.tier == tier]
        rest = [m for m in self.models if m.tier != tier]
        return preferred + rest

    def describe(self) -> dict:
        return {
            "models": [{"name": m.name, "tier": m.tier, "circuit_open": self._breakers[m.name].open} for m in self.models],
            "threshold": self.router.threshold,
            "retries": self.retries,
        }

    async def _tokens(self, llm, messages: list[Message]) -> AsyncIterator[str]:
        """Stream from one model; a stalled stream raises TimeoutError instead of hanging the request."""
        iterator = llm.stream(messages).__aiter__()
        try:
            while True:
                try:
                    token = await asyncio.wait_for(iterator.__anext__(), self.token_timeout)
                except StopAsyncIteration:
                    return
                yield token
        finally:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception:  # pragma: no cover - best effort cleanup
                    pass

    # ---------- public API ----------

    async def stream(
        self,
        messages: list[Message],
        *,
        agent: str = "",
        question: str = "",
        trace: CallTrace | None = None,
        fallback_text: str | None = None,
        tier: str | None = None,
        advice_request: bool = False,
        multi_agent: bool = False,
        cacheable: bool = True,
    ) -> AsyncIterator[str]:
        trace = trace if trace is not None else CallTrace()
        started = time.perf_counter()
        system_chars = sum(len(c) for r, c in messages if r == "system")
        history_len = max(len(messages) - 2, 0)

        decision = (
            RouteDecision(tier, 1.0 if tier == SMART else 0.0, ("forced",)) if tier
            else self.router.decide(question, agent, system_chars, history_len, advice_request, multi_agent)
        )
        trace.agent, trace.tier, trace.score, trace.reasons = agent, decision.tier, decision.score, list(decision.reasons)
        trace.tokens_in = estimate_tokens("".join(c for _, c in messages))

        key = prompt_key(messages, namespace=decision.tier)
        emitted: list[str] = []

        def finish() -> None:
            trace.latency_ms = (time.perf_counter() - started) * 1000
            trace.tokens_out = estimate_tokens("".join(emitted))
            self.metrics.record(trace)
            log.info(
                "llm agent=%s tier=%s model=%s score=%.2f cache=%s retries=%d fallbacks=%s degraded=%s ms=%.0f",
                trace.agent, trace.tier, trace.model, trace.score, trace.cache_hit, trace.retries,
                trace.fallbacks, trace.degraded, trace.latency_ms,
            )

        def mark_first() -> None:
            if trace.first_token_ms is None:
                trace.first_token_ms = (time.perf_counter() - started) * 1000

        try:
            cached = self.cache.get(key) if cacheable else None
            if cached is not None:
                trace.cache_hit, trace.model = True, cached["model"]
                for piece in _chunks(cached["text"]):
                    mark_first()
                    emitted.append(piece)
                    yield piece
                return

            for spec in self.chain_for(decision.tier):
                breaker = self._breakers[spec.name]
                if breaker.open:
                    trace.fallbacks.append(f"{spec.name} (circuit open)")
                    continue
                for attempt in range(self.retries + 1):
                    partial = "".join(emitted)
                    prompt = messages if not partial else [*messages, ("assistant", partial), ("user", CONTINUE_PROMPT)]
                    try:
                        async for token in self._tokens(spec.llm, prompt):
                            if token:
                                mark_first()
                                emitted.append(token)
                                yield token
                        if not "".join(emitted).strip():
                            raise RuntimeError("model returned an empty answer")
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        breaker.failure()
                        log.warning("llm %s attempt %d failed: %s", spec.name, attempt + 1, exc)
                        if attempt < self.retries and not breaker.open:
                            trace.retries += 1
                            await self._sleep(self.backoff_base * (2 ** attempt) + random.uniform(0, self.backoff_base / 2))
                            continue
                        trace.fallbacks.append(spec.name)
                        break
                    else:
                        breaker.success()
                        trace.model = spec.name
                        trace.continued = bool(partial)
                        if spec.tier != decision.tier:
                            trace.tier = spec.tier
                        if cacheable and not trace.continued:
                            self.cache.set(key, {"text": "".join(emitted), "model": spec.name})
                        return

            # Every model failed. Never show an error: degrade to a deterministic answer.
            trace.degraded, trace.model = True, "offline-fallback"
            text = CUT_SHORT_NOTE if emitted else (fallback_text or GENERIC_FALLBACK)
            for piece in _chunks(text):
                mark_first()
                emitted.append(piece)
                yield piece
        finally:
            finish()

    async def complete(self, messages: list[Message], **kwargs) -> tuple[str, CallTrace]:
        """Non-streaming convenience: returns (text, trace)."""
        trace = kwargs.pop("trace", None) or CallTrace()
        parts = [t async for t in self.stream(messages, trace=trace, **kwargs)]
        return "".join(parts).strip(), trace
