"""The LangGraph workflow.

    START -> guardrail_in -+-> blocked ----------------------------+
                           +-> router -> <agent> -> [<agent>] -----+-> guardrail_out -> END

<agent> is one of: finance_qa, portfolio, market, goal_planning, news, tax. The router
writes a plan of one or two agents into the state; after each agent a conditional edge
either runs the next agent in the plan or moves on to guardrail_out.

Agents do not talk to each other directly. They communicate through the shared
AssistantState: each appends its section to `parts`, its citations to `sources` and its
model-call trace to `traces`. guardrail_out merges those into one assistant message.

Nodes publish live events (agent, token, sources, meta) with the LangGraph stream
writer; the API turns them into server-sent events (stream_mode="custom").
"""
from __future__ import annotations

import logging

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from src.agents.base_agent import DISCLAIMER, AgentReply, BaseAgent
from src.agents.finance_qa_agent import FinanceQAAgent
from src.agents.goal_planning_agent import GoalPlanningAgent
from src.agents.market_agent import MarketAgent
from src.agents.news_agent import NewsAgent
from src.agents.portfolio_agent import PortfolioAgent
from src.agents.tax_agent import TaxAgent
from src.core.gateway import CallTrace, LLMGateway
from src.core.guardrails import check_input, check_output
from src.core.llm import Message
from src.core.tracing import annotate, trace_tags
from src.market.service import MarketDataService
from src.rag.retriever import Retriever
from src.workflow.router import route
from src.workflow.state import AssistantState

log = logging.getLogger(__name__)

FAILED_TEXT = "Sorry, I could not prepare an answer just now. Please try again."


def _text(msg) -> str:
    return getattr(msg, "content", None) or (msg[1] if isinstance(msg, tuple) else "")


def _role(msg) -> str:
    t = getattr(msg, "type", None)
    return "assistant" if t == "ai" else "user"


def as_gateway(llm) -> LLMGateway:
    """Accept either a ready LLMGateway or any plain LLM (wrapped as a one-model gateway)."""
    return llm if isinstance(llm, LLMGateway) else LLMGateway.single(llm)


def build_agents(llm, retriever: Retriever | None, market: MarketDataService | None) -> dict[str, BaseAgent]:
    gateway = as_gateway(llm)
    agents: list[BaseAgent] = [
        FinanceQAAgent(gateway, retriever),
        PortfolioAgent(gateway, retriever, market),
        MarketAgent(gateway, retriever, market),
        GoalPlanningAgent(gateway, retriever),
        NewsAgent(gateway, retriever, market),
        TaxAgent(gateway, retriever),
    ]
    return {a.name: a for a in agents}


def build_graph(llm, retriever: Retriever | None = None, checkpointer=None,
                market: MarketDataService | None = None, max_history: int = 10, max_chars: int = 2000,
                llm_routing: bool = False):
    gateway = as_gateway(llm)
    agents = build_agents(gateway, retriever, market)

    def history_of(state: AssistantState) -> list[Message]:
        return [(_role(m), _text(m)) for m in state["messages"][:-1][-max_history:]]

    def question_of(state: AssistantState) -> str:
        return _text(state["messages"][-1])

    # ---------- nodes ----------

    def guardrail_in(state: AssistantState) -> dict:
        result = check_input(question_of(state), max_chars)
        if not result.ok:
            annotate({"block_reason": result.reason}, ["blocked"])
        elif result.advice_request:
            annotate(tags=["advice_request"])
        # Per-turn fields are reset here so nothing leaks from the previous turn.
        return {"blocked": not result.ok, "block_reason": result.reason, "advice_request": result.advice_request,
                "parts": [], "sources": [], "traces": [], "plan": [], "step": 0}

    async def router(state: AssistantState) -> dict:
        decision = await route(question_of(state), state.get("intent"), gateway, use_llm=llm_routing)
        plan = [i for i in decision.intents if i in agents] or ["finance_qa"]
        log.info("route plan=%s method=%s", plan, decision.method)
        annotate({"route_plan": plan, "route_method": decision.method}, [f"route:{decision.method}"])
        return {"intent": plan[0], "plan": plan, "step": 0, "route_method": decision.method}

    def blocked(state: AssistantState) -> dict:
        write = get_stream_writer()
        write({"event": "agent", "data": "guardrail"})
        text = state.get("block_reason") or "I can't help with that."
        write({"event": "token", "data": text})
        return {"parts": [{"agent": "guardrail", "text": text, "canned": True}], "plan": ["guardrail"]}

    async def run_agent(agent: BaseAgent, state: AssistantState) -> dict:
        write = get_stream_writer()
        plan, step = state.get("plan") or [agent.name], state.get("step", 0)
        parts, multi = list(state.get("parts") or []), len(state.get("plan") or []) > 1
        advice = state.get("advice_request", False)
        done = {"step": step + 1}

        try:
            reply: AgentReply = await agent.respond(
                history_of(state), question_of(state), advice, state.get("level", "beginner")
            )
        except Exception:
            # A tool failure must not kill the stream: degrade to a polite message.
            log.exception("agent %s failed to prepare a reply", agent.name)
            reply = AgentReply.canned(FAILED_TEXT)

        is_last = step + 1 >= len(plan)
        shown = [p for p in parts if not p["canned"]]
        if reply.text is not None:
            # In a multi-agent plan, an agent that only has "please give me X" to say steps aside
            # for the others; it speaks only if nobody else produced an answer.
            parts.append({"agent": agent.name, "text": reply.text, "canned": True})
            if multi and not (is_last and not shown):
                return {**done, "parts": parts}
            speaker = parts[0] if multi else parts[-1]  # the primary agent's request for input wins
            write({"event": "agent", "data": speaker["agent"]})
            write({"event": "token", "data": speaker["text"]})
            speaker["shown"] = True
            return {**done, "parts": parts}

        active = [p["agent"] for p in shown] + [agent.name]
        write({"event": "agent", "data": active if len(active) > 1 else agent.name})
        heading = f"### {agent.label}\n\n" if shown else ""  # only a second section needs a heading
        if heading:
            write({"event": "token", "data": ("\n\n" if shown else "") + heading})

        trace = CallTrace()
        tokens: list[str] = []
        async for token in gateway.stream(
            reply.messages or [], agent=agent.name, question=question_of(state), trace=trace,
            fallback_text=reply.fallback_text, advice_request=advice, multi_agent=multi,
        ):
            tokens.append(token)
            write({"event": "token", "data": token})
        parts.append({"agent": agent.name, "text": heading + "".join(tokens).strip(), "canned": False, "shown": True})
        call = trace.as_dict()
        annotate(  # visible and filterable on this node's run in LangSmith
            {"model_tier": call["tier"], "model": call["model"], "complexity_score": call["score"],
             "routing_reasons": call["reasons"], "cache_hit": call["cache_hit"], "retries": call["retries"],
             "fallbacks": call["fallbacks"], "continued": call["continued"], "degraded": call["degraded"],
             "first_token_ms": call["first_token_ms"], "sources": len(reply.sources)},
            trace_tags(call),
        )
        return {
            **done, "parts": parts,
            "sources": [*(state.get("sources") or []), *reply.sources],
            "traces": [*(state.get("traces") or []), call],
        }

    def make_node(agent: BaseAgent):
        async def node(state: AssistantState) -> dict:
            return await run_agent(agent, state)
        node.__name__ = agent.name
        return node

    def guardrail_out(state: AssistantState) -> dict:
        write = get_stream_writer()
        shown = [p for p in state.get("parts") or [] if p.get("shown") or p["agent"] == "guardrail"]
        answer = "\n\n".join(p["text"] for p in shown).strip()
        check = check_output(answer)
        sources = _dedupe(state.get("sources") or [])
        if check.flags:
            annotate({"output_flags": check.flags}, ["output_flagged"])
        write({"event": "sources", "data": sources})
        write({"event": "meta", "data": {
            "agents": [p["agent"] for p in shown],
            "route": {"plan": state.get("plan") or [], "method": state.get("route_method", "rules")},
            "calls": state.get("traces") or [],
            "disclaimer": check.disclaimer if not state.get("blocked") else "",
            "flags": check.flags,
        }})
        return {"messages": [("assistant", answer)], "sources": sources, "final_answer": answer}

    def next_step(state: AssistantState) -> str:
        plan, step = state.get("plan") or [], state.get("step", 0)
        return plan[step] if step < len(plan) and plan[step] in agents else "guardrail_out"

    # ---------- wiring ----------

    g = StateGraph(AssistantState)
    g.add_node("guardrail_in", guardrail_in)
    g.add_node("router", router)
    g.add_node("blocked", blocked)
    g.add_node("guardrail_out", guardrail_out)
    targets = {name: name for name in agents} | {"guardrail_out": "guardrail_out"}
    for name, agent in agents.items():
        g.add_node(name, make_node(agent))
        g.add_conditional_edges(name, next_step, targets)

    g.add_edge(START, "guardrail_in")
    g.add_conditional_edges("guardrail_in", lambda s: "blocked" if s.get("blocked") else "router",
                            {"blocked": "blocked", "router": "router"})
    g.add_conditional_edges("router", next_step, targets)
    g.add_edge("blocked", "guardrail_out")
    g.add_edge("guardrail_out", END)
    return g.compile(checkpointer=checkpointer)


def _dedupe(sources: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out = []
    for s in sources:
        key = (s.get("title"), s.get("url"))
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


__all__ = ["build_graph", "build_agents", "as_gateway", "DISCLAIMER", "FAILED_TEXT"]
