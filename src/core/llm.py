"""LLM access: one tiny interface, OpenAI and Gemini implementations, and a fake for demos and tests."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import AsyncIterator, Protocol

from src.core.config import Settings

log = logging.getLogger(__name__)

Message = tuple[str, str]  # (role, content), role in {"system", "user", "assistant"}


class LLM(Protocol):
    def stream(self, messages: list[Message]) -> AsyncIterator[str]: ...


class RateLimiter:
    """Async sliding-window limiter: at most max_calls per period seconds."""

    def __init__(self, max_calls: int, period: float = 60.0):
        self.max_calls, self.period = max_calls, period
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                wait = self.period - (now - self._calls[0])
            await asyncio.sleep(max(wait, 0.05))


class FakeLLM:
    """Canned streaming replies so the whole stack runs without an API key."""

    def __init__(self, delay: float = 0.02):
        self.delay = delay

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        question = next((c for r, c in reversed(messages) if r == "user"), "")
        text = (
            f"Demo mode: no LLM API key is set, so this is a canned reply. "
            f'You asked: "{question}". Streaming, routing and sessions are all working. '
            f"Add OPENAI_API_KEY (or GOOGLE_API_KEY) to .env to get real answers."
        )
        for word in text.split(" "):
            await asyncio.sleep(self.delay)
            yield word + " "


class _LangChainLLM:
    """Shared streaming + rate limiting around any LangChain chat model.

    Retries, model switching and fallbacks live one level up, in LLMGateway, so `retries`
    defaults to 1 (a single attempt) here.
    """

    def __init__(self, model, limiter: RateLimiter, retries: int = 1):
        self._model, self._limiter, self._retries = model, limiter, retries

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        for attempt in range(self._retries):
            await self._limiter.acquire()
            emitted = False
            try:
                async for chunk in self._model.astream(messages):
                    text = _chunk_text(chunk.content)
                    if text:
                        emitted = True
                        yield text
                return
            except Exception as exc:
                # Retry only if nothing has been sent to the user yet.
                if emitted or attempt == self._retries - 1:
                    raise
                delay = 2**attempt
                log.warning("LLM call failed (%s); retrying in %ss", exc, delay)
                await asyncio.sleep(delay)


class OpenAILLM(_LangChainLLM):
    def __init__(self, settings: Settings, limiter: RateLimiter, retries: int = 1, model_name: str | None = None):
        from langchain_openai import ChatOpenAI  # imported lazily

        model = ChatOpenAI(
            model=model_name or settings.llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=settings.temperature,
        )
        super().__init__(model, limiter, retries)


class GeminiLLM(_LangChainLLM):
    def __init__(self, settings: Settings, limiter: RateLimiter, retries: int = 1, model_name: str | None = None):
        from langchain_google_genai import ChatGoogleGenerativeAI  # imported lazily

        model = ChatGoogleGenerativeAI(
            model=model_name or settings.llm_model,
            google_api_key=settings.google_api_key,
            temperature=settings.temperature,
        )
        super().__init__(model, limiter, retries)


def _chunk_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # multi-part content
        return "".join(p if isinstance(p, str) else p.get("text", "") for p in content)
    return ""


def create_llm(settings: Settings) -> LLM:
    limiter = RateLimiter(settings.llm_calls_per_min)
    if settings.llm_provider == "fake":
        log.warning("Running in demo mode with FakeLLM")
        return FakeLLM()
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai needs OPENAI_API_KEY (or set LLM_PROVIDER=fake)")
        return OpenAILLM(settings, limiter)
    if settings.llm_provider == "gemini":
        if not settings.google_api_key:
            raise RuntimeError("LLM_PROVIDER=gemini needs GOOGLE_API_KEY (or set LLM_PROVIDER=fake)")
        return GeminiLLM(settings, limiter)
    raise RuntimeError(f"Unknown LLM_PROVIDER={settings.llm_provider!r}; use openai, gemini or fake")


def create_gateway(settings: Settings, metrics=None):
    """Build the LLMGateway: model tiers, router, response cache and the fallback chain.

    Chain order = primary provider (fast, smart), then the other provider's models if its
    key is also configured. The gateway starts from the routed tier and walks the rest.
    """
    from src.core.cache import ResponseCache
    from src.core.gateway import LLMGateway, ModelSpec
    from src.core.metrics import Metrics
    from src.core.model_router import FAST, SMART, ModelRouter

    limiter = RateLimiter(settings.llm_calls_per_min)
    provider = settings.llm_provider
    if provider not in ("openai", "gemini", "fake"):
        raise RuntimeError(f"Unknown LLM_PROVIDER={provider!r}; use openai, gemini or fake")
    if provider == "openai" and not settings.openai_api_key:
        raise RuntimeError("LLM_PROVIDER=openai needs OPENAI_API_KEY (or set LLM_PROVIDER=fake)")
    if provider == "gemini" and not settings.google_api_key:
        raise RuntimeError("LLM_PROVIDER=gemini needs GOOGLE_API_KEY (or set LLM_PROVIDER=fake)")

    def openai_models(fast: str, smart: str) -> list:
        return [ModelSpec(f"openai:{fast}", FAST, OpenAILLM(settings, limiter, model_name=fast)),
                ModelSpec(f"openai:{smart}", SMART, OpenAILLM(settings, limiter, model_name=smart))]

    def gemini_models(fast: str, smart: str) -> list:
        return [ModelSpec(f"gemini:{fast}", FAST, GeminiLLM(settings, limiter, model_name=fast)),
                ModelSpec(f"gemini:{smart}", SMART, GeminiLLM(settings, limiter, model_name=smart))]

    tiers = settings.tier_models
    models: list = []
    if provider == "fake":
        log.warning("Running in demo mode with FakeLLM")
        models = [ModelSpec("fake:demo", FAST, FakeLLM())]
    elif provider == "openai":
        models = openai_models(settings.llm_model, settings.llm_smart_model)
        if settings.google_api_key:  # cross-provider fallback
            models += gemini_models(tiers["fast"]["gemini"], tiers["smart"]["gemini"])
    else:
        models = gemini_models(settings.llm_model, settings.llm_smart_model)
        if settings.openai_api_key:
            models += openai_models(tiers["fast"]["openai"], tiers["smart"]["openai"])

    # Two specs for the same model (fast == smart) would just retry the same thing twice.
    seen: set[str] = set()
    models = [m for m in models if not (m.name in seen or seen.add(m.name))]

    metrics = metrics or Metrics(settings.pricing, baseline_model=settings.llm_smart_model)
    return LLMGateway(
        models,
        router=ModelRouter(settings.model_router_threshold),
        cache=ResponseCache(settings.llm_cache_ttl if provider != "fake" else 0, settings.llm_cache_max_items),
        metrics=metrics,
        retries=settings.llm_retries if provider != "fake" else 0,
        backoff_base=settings.llm_backoff_base,
        token_timeout=settings.llm_token_timeout,
        breaker_failures=settings.breaker_failures,
        breaker_cooldown=settings.breaker_cooldown,
    )
