"""Model router, response cache and the fallback chain (LLMGateway)."""
import asyncio

import pytest

from evals.suites import ScriptedLLM
from src.core.cache import ResponseCache, TTLCache, normalize, prompt_key
from src.core.gateway import GENERIC_FALLBACK, CallTrace, CircuitBreaker, LLMGateway, ModelSpec
from src.core.metrics import Metrics, estimate_tokens
from src.core.model_router import FAST, SMART, ModelRouter


async def _no_sleep(_):
    return None


def gateway(*specs, **kw):
    kw.setdefault("retries", 1)
    kw.setdefault("backoff_base", 0)
    return LLMGateway(list(specs), sleep=_no_sleep, **kw)


def ask(gw, user_text="q", **kw):
    trace = CallTrace()

    async def go():
        return "".join([t async for t in gw.stream([("system", "s"), ("user", user_text)], trace=trace, **kw)])

    return asyncio.run(go()), trace


# ---------- model router ----------

@pytest.mark.parametrize("question,agent,tier", [
    ("What is an ETF?", "finance_qa", FAST),
    ("What is AAPL trading at?", "market", FAST),
    ("Compare index funds versus active funds and explain the trade-offs and why costs matter", "finance_qa", SMART),
    ("Analyze my portfolio and explain how I should think about rebalancing: 10 AAPL, 5 VTI, 20 BND", "portfolio", SMART),
])
def test_router_sends_easy_to_fast_and_hard_to_smart(question, agent, tier):
    assert ModelRouter().decide(question, agent).tier == tier


def test_router_explains_itself_and_respects_threshold():
    d = ModelRouter().decide("What is an ETF?", "finance_qa")
    assert d.score < 0.5 and any("short definition" in r for r in d.reasons)
    assert d.as_dict()["tier"] == FAST
    assert ModelRouter(threshold=0.0).decide("What is an ETF?", "finance_qa").tier == SMART
    assert ModelRouter(threshold=1.1).decide("Compare A versus B and why", "tax", advice_request=True).tier == FAST


def test_router_context_signals_raise_the_score():
    r = ModelRouter()
    base = r.decide("Tell me about bonds", "finance_qa").score
    assert r.decide("Tell me about bonds", "finance_qa", context_chars=5000, history_len=8).score > base
    assert r.decide("Tell me about bonds", "finance_qa", multi_agent=True).score > base
    assert 0 <= r.decide("why " * 200, "portfolio", advice_request=True, multi_agent=True).score <= 1


# ---------- cache ----------

def test_ttl_cache_lru_and_age():
    now = [0.0]
    c = TTLCache(max_items=2, clock=lambda: now[0])
    c.set("a", 1); c.set("b", 2)
    now[0] = 5
    assert c.get("a") == (1, 5)
    c.set("c", 3)  # evicts b, the least recently used
    assert c.get("b") is None and len(c) == 2
    c.delete("a"); c.clear()
    assert len(c) == 0


def test_prompt_key_normalises_only_the_last_user_turn():
    a = prompt_key([("system", "S"), ("user", "What is an ETF?")])
    assert a == prompt_key([("system", "S"), ("user", "  what is an   etf ")])
    assert a != prompt_key([("system", "s"), ("user", "What is an ETF?")])       # system prompt is exact
    assert a != prompt_key([("system", "S"), ("user", "What is an ETF?")], "smart")  # namespaced by tier
    assert normalize("  Hello   World?! ") == "hello world"


def test_response_cache_ttl_stats_and_disable():
    now = [0.0]
    c = ResponseCache(ttl=10, clock=lambda: now[0])
    assert c.get("k") is None
    c.set("k", "v")
    assert c.get("k") == "v"
    now[0] = 11
    assert c.get("k") is None  # expired
    assert c.stats() == {"enabled": True, "ttl_seconds": 10, "entries": 0, "hits": 1, "misses": 2, "writes": 1,
                         "hit_rate": 0.3333}
    off = ResponseCache(ttl=0)
    off.set("k", "v")
    assert off.get("k") is None and not off.enabled


# ---------- gateway ----------

def test_gateway_routes_by_tier_and_records_trace():
    fast, smart = ScriptedLLM("ok", text="fast answer"), ScriptedLLM("ok", text="smart answer")
    gw = gateway(ModelSpec("f", FAST, fast), ModelSpec("s", SMART, smart))
    text, tr = ask(gw, "What is an ETF?", agent="finance_qa", question="What is an ETF?")
    assert text == "fast answer" and tr.tier == FAST and tr.model == "f" and tr.first_token_ms is not None
    text, tr = ask(gw, "hard", agent="finance_qa", question="Compare A versus B, why, and the trade-offs?")
    assert text == "smart answer" and tr.tier == SMART and tr.as_dict()["latency_ms"] >= 0
    assert ask(gw, "forced", tier=SMART)[1].reasons == ["forced"]


def test_gateway_retries_then_switches_then_degrades():
    flaky = ScriptedLLM("fail", "ok")
    assert ask(gateway(ModelSpec("f", FAST, flaky)))[1].retries == 1

    text, tr = ask(gateway(ModelSpec("f", FAST, ScriptedLLM("fail")), ModelSpec("s", SMART, ScriptedLLM("ok", text="backup"))))
    assert text == "backup" and tr.fallbacks == ["f"] and tr.tier == SMART

    text, tr = ask(gateway(ModelSpec("f", FAST, ScriptedLLM("fail"))), fallback_text="from the knowledge base")
    assert text == "from the knowledge base" and tr.degraded and tr.model == "offline-fallback"
    assert ask(gateway(ModelSpec("f", FAST, ScriptedLLM("fail"))))[0] == GENERIC_FALLBACK


def test_gateway_continues_an_answer_that_died_mid_stream():
    first, second = ScriptedLLM("partial", "fail"), ScriptedLLM("ok", text="forty two.")
    text, tr = ask(gateway(ModelSpec("f", FAST, first), ModelSpec("s", SMART, second)))
    assert text == "The answer forty two." and tr.continued
    assert second.prompts[0][-2] == ("assistant", "The answer ")
    # Nobody left to continue: the partial text is kept and a note is appended instead of an error.
    text, tr = ask(gateway(ModelSpec("f", FAST, ScriptedLLM("partial"))))
    assert text.startswith("The answer ") and "stops here" in text and tr.degraded


def test_gateway_cache_hit_and_no_caching_of_fallbacks():
    llm = ScriptedLLM("ok")
    gw = gateway(ModelSpec("f", FAST, llm), cache=ResponseCache(ttl=60))
    first, _ = ask(gw, "What is an ETF?")
    second, tr = ask(gw, "what is an etf")
    assert first == second and tr.cache_hit and llm.calls == 1
    ask(gw, "What is an ETF?", cacheable=False)
    assert llm.calls == 2

    down = gateway(ModelSpec("f", FAST, ScriptedLLM("fail")), cache=ResponseCache(ttl=60))
    ask(down, fallback_text="FB")
    assert down.cache.stats()["writes"] == 0


def test_gateway_times_out_a_stalled_model():
    class Stalled:
        async def stream(self, messages):
            await asyncio.sleep(5)
            yield "never"

    gw = gateway(ModelSpec("f", FAST, Stalled()), ModelSpec("s", SMART, ScriptedLLM("ok")), retries=0, token_timeout=0.05)
    text, tr = ask(gw)
    assert text == "The answer is forty two." and tr.fallbacks == ["f"]


def test_circuit_breaker_opens_and_half_opens():
    now = [0.0]
    b = CircuitBreaker(max_failures=2, cooldown=10, clock=lambda: now[0])
    b.failure()
    assert not b.open
    b.failure()
    assert b.open
    now[0] = 11
    assert not b.open      # half-open: one trial allowed
    b.failure()
    assert b.open          # the trial failed: straight back to open
    now[0] = 30
    assert not b.open
    b.success()
    b.failure()
    assert not b.open      # counter was reset by the success


def test_gateway_describe_complete_and_validation():
    gw = gateway(ModelSpec("f", FAST, ScriptedLLM("ok")))
    assert gw.describe()["models"] == [{"name": "f", "tier": FAST, "circuit_open": False}]
    text, trace = asyncio.run(gw.complete([("user", "q")], agent="router"))
    assert text == "The answer is forty two." and trace.agent == "router"
    with pytest.raises(ValueError):
        LLMGateway([])
    assert LLMGateway.single(ScriptedLLM("ok")).retries == 0


# ---------- metrics ----------

def test_metrics_cost_and_savings():
    m = Metrics({"cheap": {"input": 1, "output": 2}, "big": {"input": 10, "output": 20}}, baseline_model="big")
    live = CallTrace(agent="finance_qa", tier=FAST, model="openai:cheap", tokens_in=1_000_000, tokens_out=1_000_000, latency_ms=100, first_token_ms=20)
    cached = CallTrace(agent="finance_qa", tier=FAST, model="openai:cheap", cache_hit=True, tokens_in=1_000_000, tokens_out=1_000_000, latency_ms=5)
    down = CallTrace(agent="tax", degraded=True, model="offline-fallback", latency_ms=50, fallbacks=["a", "b"], retries=2)
    for t in (live, cached, down):
        m.record(t)
    s = m.snapshot()
    assert s["llm_calls"] == 3 and s["cache_hits"] == 1 and s["degraded_answers"] == 1
    assert s["by_tier"] == {FAST: 1} and s["fallbacks"] == 2 and s["retries"] == 2
    assert s["estimated_cost_usd"] == 3.0 and s["estimated_cost_all_smart_usd"] == 60.0
    assert s["estimated_savings_usd"] == 57.0 and s["latency_ms"]["p95"] == 100
    assert estimate_tokens("abcdefgh") == 2 and Metrics().price("unknown", 10, 10) == 0.0
    assert Metrics().snapshot()["latency_ms"] == {"p50": 0, "p95": 0}
