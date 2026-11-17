"""Config, market provider chain, new API endpoints, MCP server and the eval gate."""
import asyncio
import json

import pytest

from src.core.config import Settings, load_yaml
from src.core.schemas import Holding
from src.market.providers import (
    AlphaVantageProvider, FallbackProvider, ProviderError, SampleProvider, SymbolNotFound,
)
from src.market.service import MarketDataService, MarketDataUnavailable, create_market_service
from src.portfolio.analysis import analyze, diversification_score, risk_score


# ---------- config ----------

def test_settings_come_from_yaml_then_env(monkeypatch, tmp_path):
    for var in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "LLM_FAST_MODEL", "LLM_SMART_MODEL",
                "MODEL_ROUTER_THRESHOLD", "ROUTER_LLM_FALLBACK", "MARKET_DATA_PROVIDER", "RAG_TOP_K"):
        monkeypatch.delenv(var, raising=False)
    s = Settings.load({"model_router": {"smart_threshold": 0.7}, "rag": {"top_k": 6}})
    assert s.llm_provider == "fake" and s.model_router_threshold == 0.7 and s.rag_top_k == 6
    assert s.market_data_provider == "yfinance" and s.router_llm_fallback is True

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MODEL_ROUTER_THRESHOLD", "0.3")
    monkeypatch.setenv("ROUTER_LLM_FALLBACK", "false")
    s = Settings.load({"model_router": {"smart_threshold": 0.7}})
    assert s.llm_provider == "openai" and s.llm_model == "gpt-4o-mini" and s.llm_smart_model == "gpt-4o"
    assert s.model_router_threshold == 0.3 and s.router_llm_fallback is False   # env wins

    monkeypatch.setenv("LLM_SMART_MODEL", "big-model")
    assert Settings.load({}).llm_smart_model == "big-model"
    assert load_yaml(tmp_path / "missing.yaml") == {}
    assert "tiers" in load_yaml()["llm"]                                        # the shipped config.yaml parses


def test_create_gateway_builds_the_chain(monkeypatch):
    from src.core.llm import create_gateway

    monkeypatch.setenv("LLM_PROVIDER", "fake")
    gw = create_gateway(Settings.load({}))
    assert [m.name for m in gw.models] == ["fake:demo"] and not gw.cache.enabled

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    gw = create_gateway(Settings.load({}))
    assert [m.name for m in gw.models] == ["openai:gpt-4o-mini", "openai:gpt-4o", "gemini:gemini-2.0-flash", "gemini:gemini-2.5-pro"]
    assert [m.name for m in gw.chain_for("smart")][:2] == ["openai:gpt-4o", "gemini:gemini-2.5-pro"]
    assert gw.cache.enabled and gw.retries == 2

    monkeypatch.setenv("LLM_SMART_MODEL", "gpt-4o-mini")                        # same model twice is deduplicated
    monkeypatch.delenv("GOOGLE_API_KEY")
    assert len(create_gateway(Settings.load({})).models) == 1

    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        create_gateway(Settings.load({}))
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(RuntimeError, match="Unknown"):
        create_gateway(Settings.load({}))


# ---------- Alpha Vantage ----------

def av(payloads):
    calls = []

    def http_get(url, params, timeout):
        calls.append(params)
        result = payloads[params["function"]]
        if isinstance(result, Exception):
            raise result
        return result

    return AlphaVantageProvider("key", http_get=http_get), calls


def test_alpha_vantage_quote_history_news_profile():
    from datetime import date, timedelta

    recent, old = (date.today() - timedelta(days=3)).isoformat(), (date.today() - timedelta(days=400)).isoformat()
    provider, calls = av({
        "GLOBAL_QUOTE": {"Global Quote": {"05. price": "101.50", "08. previous close": "100.00"}},
        "TIME_SERIES_DAILY": {"Time Series (Daily)": {recent: {"4. close": "101.5"}, old: {"4. close": "80"}}},
        "NEWS_SENTIMENT": {"feed": [{"title": "Apple up", "url": "https://n.test/1", "source": "Wire",
                                     "time_published": "20260102T093000"}, {"title": ""}, {"title": "No date", "time_published": "x"}]},
        "OVERVIEW": {"Symbol": "AAPL", "Name": "Apple Inc", "AssetType": "Common Stock", "Sector": "TECHNOLOGY"},
    })
    q = provider.quote("AAPL")
    assert (q.price, q.previous_close) == (101.5, 100.0) and calls[0]["apikey"] == "key"
    assert [p.date for p in provider.history("AAPL", "1mo")] == [recent]          # old bar is outside the period
    news = provider.news("AAPL", 5)
    assert [n.title for n in news] == ["Apple up", "No date"] and news[0].published_at.startswith("2026-01-02T09:30")
    p = provider.profile("AAPL")
    assert (p.asset_class, p.sector, p.name) == ("Stock", "Technology", "Apple Inc")


def test_alpha_vantage_error_mapping():
    provider, _ = av({"GLOBAL_QUOTE": {"Note": "Thank you for using Alpha Vantage! Rate limit reached."},
                      "TIME_SERIES_WEEKLY": {"Error Message": "Invalid API call"},
                      "NEWS_SENTIMENT": OSError("connection reset"), "OVERVIEW": {}})
    with pytest.raises(ProviderError, match="Rate limit"):
        provider.quote("AAPL")                                 # rate limit is retryable, not "not found"
    with pytest.raises(SymbolNotFound):
        provider.history("ZZZZ", "1y")
    with pytest.raises(ProviderError):
        provider.news("AAPL", 3)
    with pytest.raises(SymbolNotFound):
        provider.profile("ZZZZ")
    empty, _ = av({"GLOBAL_QUOTE": {"Global Quote": {}}, "TIME_SERIES_DAILY": {}})
    with pytest.raises(SymbolNotFound):
        empty.quote("ZZZZ")
    with pytest.raises(SymbolNotFound):
        empty.history("ZZZZ", "1mo")
    with pytest.raises(ValueError):
        AlphaVantageProvider("")


# ---------- provider chain and service resilience ----------

class Down(SampleProvider):
    name = "down"

    def __init__(self):
        self.calls = 0

    def quote(self, symbol):
        self.calls += 1
        raise ProviderError("outage")


def test_fallback_provider_switches_on_outage_but_trusts_not_found():
    chain = FallbackProvider([Down(), SampleProvider()])
    assert chain.quote("AAPL").price == 230.0 and chain.switches == 1 and chain.chain == ["down", "sample"]
    assert chain.history("AAPL", "1mo") and chain.profile("AAPL").name and chain.news("AAPL", 3) == []
    with pytest.raises(SymbolNotFound):
        chain.quote("ZZZZ")
    with pytest.raises(ProviderError):
        FallbackProvider([Down(), Down()]).quote("AAPL")
    with pytest.raises(ValueError):
        FallbackProvider([])


def test_service_retries_with_backoff_then_counts_everything():
    class FlakyOnce(SampleProvider):
        calls = 0

        def quote(self, symbol):
            FlakyOnce.calls += 1
            if FlakyOnce.calls == 1:
                raise ProviderError("blip")
            return super().quote(symbol)

    delays = []

    async def sleep(d):
        delays.append(d)

    svc = MarketDataService(FlakyOnce(), retries=2, backoff_base=0.5, sleep=sleep)
    assert asyncio.run(svc.quote("AAPL")).price == 230.0 and delays == [0.5]
    asyncio.run(svc.quote("AAPL"))
    s = svc.stats()
    assert (s["retries"], s["cache_hits"], s["provider_calls"], s["provider"]) == (1, 1, 2, "sample")

    down = MarketDataService(Down(), retries=1, backoff_base=0, sleep=sleep)
    q = asyncio.run(down.quote("AAPL"))                       # known ticker: labelled sample price, not an error
    assert q.is_sample and down.stats()["sample_served"] == 1 and down.provider.calls == 2
    with pytest.raises(MarketDataUnavailable):
        asyncio.run(down.quote("ZZZZ"))


def test_create_market_service_from_settings(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "sample")
    assert create_market_service(Settings.load({})).source == "sample"
    assert create_market_service("sample").source == "sample"
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "alphavantage")
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ALPHAVANTAGE_API_KEY"):
        create_market_service(Settings.load({}))
    with pytest.raises(RuntimeError, match="Unknown"):
        create_market_service("bloomberg")


# ---------- portfolio metrics added for the brief ----------

def test_portfolio_scores_and_fund_costs():
    single = analyze([Holding(symbol="TSLA", shares=10, avg_cost=100)])
    assert (single.diversification_score, single.risk_score, single.risk_level) == (0, 100, "Very high")
    assert single.expense_ratio is None and single.equity_weight == 1.0

    bonds = analyze([Holding(symbol="BND", shares=10, avg_cost=70)])
    assert bonds.risk_level == "Low" and bonds.diversification_score >= 90 and bonds.equity_weight == 0.0
    assert bonds.expense_ratio == 0.0003 and bonds.annual_fund_cost == round(700 * 0.0003, 2)

    pricey = analyze([Holding(symbol="GLD", shares=10, avg_cost=200), Holding(symbol="QQQ", shares=1, avg_cost=10)])
    assert "high_fund_costs" not in {f.code for f in pricey.flags}        # 0.40% is under the 0.50% line
    assert analyze([Holding(symbol="ZZZZ", shares=1)]).risk_level == "Unknown"
    assert diversification_score([], [], set()) == 0 and risk_score([], set()) == 0


# ---------- API ----------

@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "sample")
    monkeypatch.setenv("RAG_EMBEDDER", "local")
    monkeypatch.setenv("API_REQUESTS_PER_MIN", "1000")
    monkeypatch.setenv("CHAT_REQUESTS_PER_MIN", "1000")
    from src.web_app.api.main import app

    with TestClient(app) as c:
        yield c


def frames(text):
    out = []
    for block in text.strip().split("\n\n"):
        event, data = block.split("\n")
        out.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def test_chat_returns_sources_meta_and_never_errors(client):
    r = client.post("/api/chat", json={"session_id": "abcdef123456", "message": "What is an ETF?", "level": "intermediate"}).json()
    assert r["agent_used"] == "finance_qa" and r["sources"] and r["sources"][0]["title"]
    assert r["meta"]["calls"][0]["tier"] == "fast" and r["meta"]["disclaimer"]
    events = frames(client.post("/api/chat/stream", json={"session_id": "abcdef123456", "message": "hello"}).text)
    assert [e for e, _ in events][-2:] == ["meta", "done"]
    assert client.post("/api/chat", json={"session_id": "abcdef123456", "message": "x", "level": "expert"}).status_code == 422


def test_workflow_crash_becomes_a_polite_answer(client, monkeypatch):
    class Boom:
        def astream(self, *a, **k):
            raise RuntimeError("graph exploded")

    monkeypatch.setattr(client.app.state, "graph", Boom())
    r = client.post("/api/chat", json={"session_id": "abcdef123456", "message": "What is an ETF?"})
    assert r.status_code == 200 and "something went wrong on my side" in r.json()["answer"]
    events = frames(client.post("/api/chat/stream", json={"session_id": "abcdef123456", "message": "hi"}).text)
    assert "error" not in [e for e, _ in events] and events[-1] == ("done", {})


def test_knowledge_endpoints(client):
    r = client.get("/api/knowledge/search", params={"q": "what is an etf", "k": 2}).json()
    assert len(r["results"]) <= 2 and r["results"][0]["score"] > 0.2
    tax = client.get("/api/knowledge/search", params={"q": "roth ira", "category": "tax"}).json()
    assert all(x["category"] == "tax" for x in tax["results"])
    a = client.get("/api/knowledge/articles").json()
    assert 50 <= a["count"] <= 100 and len(a["categories"]) == 8
    assert client.get("/api/knowledge/articles", params={"category": "bonds"}).json()["count"] == 9
    assert client.get("/api/knowledge/glossary", params={"q": "expense ratio"}).json()["terms"][0]["term"] == "Expense ratio"
    assert client.get("/api/knowledge/glossary").json()["count"] >= 50
    assert client.get("/api/knowledge/search", params={"q": "x"}).status_code == 422


def test_system_metrics_route_preview_health_and_samples(client):
    client.post("/api/chat", json={"session_id": "abcdef123456", "message": "What is an ETF?"})
    m = client.get("/api/system/metrics").json()
    assert m["llm"]["llm_calls"] == 1 and m["llm"]["by_agent"] == {"finance_qa": 1}
    assert m["retrieval"]["articles"] > 100 and m["market"]["provider"] == "sample" and m["gateway"]["models"]
    p = client.get("/api/system/route-preview", params={"q": "Compare A versus B and explain why", "agent": "tax"}).json()
    assert p["tier"] == "smart" and p["model_chain"] == ["fake:demo"] and p["reasons"]
    h = client.get("/api/health").json()
    assert h["rag"] == "KnowledgeBaseRetriever" and h["models"] == ["fake:demo"]
    samples = client.get("/api/portfolio/samples").json()["portfolios"]
    assert len(samples) == 5
    analysis = client.post("/api/portfolio/analyze", json={"holdings": samples[1]["holdings"]}).json()
    assert analysis["diversification_score"] >= 90 and analysis["expense_ratio"] > 0


def test_insight_streams_emit_meta(client):
    body = {"holdings": [{"symbol": "AAPL", "shares": 10}]}
    events = frames(client.post("/api/portfolio/insights/stream", json=body).text)
    assert [e for e, _ in events][-3:] == ["sources", "meta", "done"]
    assert dict(events)["meta"]["calls"][0]["agent"] == "portfolio"


# ---------- MCP server ----------

def test_mcp_server_tools(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "sample")
    monkeypatch.setenv("RAG_EMBEDDER", "local")
    from src.mcp_server import server

    for cached in (server._settings, server._market, server._retriever, server._graph):
        cached.cache_clear()
    names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert names == {"ask_finance_assistant", "search_knowledge_base", "lookup_glossary", "get_stock_quote",
                     "get_price_history", "get_market_news", "analyze_portfolio", "plan_savings_goal"}
    assert server.plan_savings_goal(50_000, 10, 5_000, 300)["on_track"] is False
    assert server.lookup_glossary("expense")[0]["term"] == "Expense ratio"
    assert server.search_knowledge_base("what is an etf", top_k=2)[0]["score"] > 0.2
    assert asyncio.run(server.get_stock_quote("AAPL"))["price"] == 230.0
    assert "error" in asyncio.run(server.get_stock_quote("ZZZZ"))
    assert asyncio.run(server.get_price_history("SPY", "1mo"))["points"]
    assert "error" in asyncio.run(server.get_price_history("ZZZZ"))
    assert asyncio.run(server.get_market_news("SPY")) == []
    assert asyncio.run(server.analyze_portfolio([{"symbol": "AAPL", "shares": 10}]))["total_value"] == 2300.0
    answer = asyncio.run(server.ask_finance_assistant("What is an ETF?"))
    assert answer["agents"] == ["finance_qa"] and answer["sources"] and answer["session_id"].startswith("mcp-")
    for cached in (server._settings, server._market, server._retriever, server._graph):
        cached.cache_clear()


# ---------- eval gate ----------

def test_eval_gate_flags_floors_and_regressions():
    from evals.run import evaluate, render_markdown
    from evals.suites import SuiteResult

    results = [SuiteResult("retrieval", {"hit_at_3": 0.91, "mrr": 0.70, "hit_at_1": 0.9}, 10, ["a case"])]
    thresholds = {"regression_tolerance": 0.02, "default_floor": 0.5, "floors": {"retrieval": {"mrr": 0.8, "hit_at_3": 0.85}}}
    rows = {r["metric"]: r["status"] for r in evaluate(results, thresholds, {"retrieval.hit_at_3": 0.98, "retrieval.hit_at_1": 0.91})}
    assert rows == {"retrieval.hit_at_3": "regression",     # clears the 0.85 floor but dropped 7 points
                    "retrieval.mrr": "below_floor",
                    "retrieval.hit_at_1": "pass"}           # within tolerance of the baseline
    md = render_markdown(evaluate(results, thresholds, {}), results, 1.0, False)
    assert "FAIL" in md and "BELOW FLOOR" in md and "a case" in md


def test_offline_eval_suites_pass_their_floors():
    """The deploy gate itself, run as a test: a routing/retrieval/safety regression fails CI here too."""
    from evals.run import evaluate, load_thresholds
    from evals.suites import OFFLINE

    rows = evaluate([fn() for fn in OFFLINE.values()], load_thresholds(), {})
    assert [r for r in rows if r["status"] != "pass"] == []
