import asyncio
import time

from src.core.llm import FakeLLM, RateLimiter


def collect(llm, messages):
    async def run():
        return [t async for t in llm.stream(messages)]

    return asyncio.run(run())


def test_fake_llm_streams_and_echoes_question():
    tokens = collect(FakeLLM(delay=0), [("system", "s"), ("user", "What is an ETF?")])
    assert len(tokens) > 5
    assert "What is an ETF?" in "".join(tokens)


def test_rate_limiter_allows_up_to_max_then_waits():
    async def run():
        rl = RateLimiter(max_calls=2, period=0.3)
        start = time.monotonic()
        for _ in range(3):
            await rl.acquire()
        return time.monotonic() - start

    assert asyncio.run(run()) >= 0.25  # third call had to wait for the window
