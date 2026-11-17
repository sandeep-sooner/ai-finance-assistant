import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("langgraph")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "sample")
    monkeypatch.setenv("API_REQUESTS_PER_MIN", "1000")
    monkeypatch.setenv("CHAT_REQUESTS_PER_MIN", "3")
    from src.web_app.api.main import app

    with TestClient(app) as c:
        yield c


def frames(text):
    out = []
    for block in text.strip().split("\n\n"):
        event, data = block.split("\n")
        out.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


BODY = {"session_id": "abcdef123456", "message": "What is an ETF?"}


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_stream_contract_matches_react_hook(client):
    r = client.post("/api/chat/stream", json=BODY)
    assert r.headers["content-type"].startswith("text/event-stream")
    events = frames(r.text)
    assert events[0] == ("agent", "finance_qa")
    assert events[-1] == ("done", {})
    assert any(e == "token" for e, _ in events) and any(e == "sources" for e, _ in events)


def test_non_stream_chat(client):
    r = client.post("/api/chat", json=BODY).json()
    assert r["agent_used"] == "finance_qa" and r["answer"]


def test_validation_error(client):
    assert client.post("/api/chat/stream", json={"session_id": "x", "message": ""}).status_code == 422


def test_rate_limit(client):
    codes = [client.post("/api/chat", json=BODY).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]


PORTFOLIO = {"holdings": [{"symbol": "AAPL", "shares": 10, "avg_cost": 200}, {"symbol": "BND", "shares": 20}]}


def test_portfolio_analyze(client):
    r = client.post("/api/portfolio/analyze", json=PORTFOLIO).json()
    assert r["total_value"] == 10 * 230 + 20 * 73  # offline sample prices
    assert {h["price_source"] for h in r["holdings"]} == {"sample"}
    assert {h["symbol"] for h in r["holdings"]} == {"AAPL", "BND"}


def test_portfolio_insights_stream(client):
    events = frames(client.post("/api/portfolio/insights/stream", json=PORTFOLIO).text)
    assert events[0] == ("agent", "portfolio") and events[-1] == ("done", {})
    assert any(e == "token" for e, _ in events)


def test_portfolio_validation(client):
    assert client.post("/api/portfolio/analyze", json={"holdings": []}).status_code == 422
    assert client.post("/api/portfolio/analyze", json={"holdings": [{"symbol": "AAPL", "shares": -1}]}).status_code == 422


def test_errors_use_standard_shape(client):
    r = client.post("/api/portfolio/analyze", json={"holdings": []})
    assert r.status_code == 422
    body = r.json()
    assert body["error_code"] == "validation_error" and body["message"].startswith("holdings")


def test_market_overview_and_quotes(client):
    r = client.get("/api/market/overview").json()
    assert [q["symbol"] for q in r["quotes"]][:3] == ["SPY", "QQQ", "DIA"] and r["errors"] == []
    r = client.get("/api/market/quotes", params={"symbols": "aapl,ZZZZ"}).json()
    assert [q["symbol"] for q in r["quotes"]] == ["AAPL"]
    assert r["errors"][0]["error_code"] == "symbol_not_found"


def test_market_symbol_detail_history_and_errors(client):
    d = client.get("/api/market/symbols/brk.b").json()
    assert d["quote"]["symbol"] == "BRK-B" and d["profile"]["sector"] == "Financials"
    h = client.get("/api/market/history/SPY", params={"period": "6mo"}).json()
    assert h["period"] == "6mo" and len(h["points"]) > 100 and h["is_sample"]

    r = client.get("/api/market/symbols/ZZZZ")
    assert r.status_code == 404 and r.json()["error_code"] == "symbol_not_found"
    assert client.get("/api/market/history/SPY", params={"period": "2d"}).status_code == 422
    assert client.get("/api/market/symbols/bad!sym").status_code == 400


def test_market_news(client):
    r = client.get("/api/market/news").json()
    assert r["symbol"] is None and r["items"] == []  # sample provider has no news


GOAL = {"target_amount": 50000, "years": 10, "current_savings": 5000, "monthly_contribution": 300}


def test_goal_plan_and_insights(client):
    r = client.post("/api/goals/plan", json=GOAL).json()
    assert r["on_track"] is False and len(r["projection"]) == 11
    events = frames(client.post("/api/goals/insights/stream", json=GOAL).text)
    assert events[0] == ("agent", "goal_planning") and events[-1] == ("done", {})
    assert client.post("/api/goals/plan", json={**GOAL, "years": 0}).status_code == 422


def test_delete_session_forgets_history(client):
    client.post("/api/chat", json=BODY)
    graph = client.app.state.graph
    cfg = {"configurable": {"thread_id": BODY["session_id"]}}
    assert graph.get_state(cfg).values.get("messages")
    assert client.delete(f"/api/sessions/{BODY['session_id']}").status_code == 204
    assert not graph.get_state(cfg).values.get("messages")
    assert client.delete("/api/sessions/x").status_code == 422
