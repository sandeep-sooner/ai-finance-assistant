"""Model router: easy questions go to a cheap, fast model; hard ones go to a smarter one.

The decision is a transparent, deterministic complexity score (no extra LLM call, so
routing adds ~0 ms and ~$0). Each signal adds or removes weight, and the reasons are
returned with the decision so they can be logged, shown in the UI and asserted on by
the eval suite (`evals/datasets/model_routing.jsonl`).

    score >= threshold  ->  "smart"      score < threshold  ->  "fast"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

FAST, SMART = "fast", "smart"

# How demanding each agent's job is before looking at the question itself.
AGENT_BASE = {
    "finance_qa": 0.10,     # mostly definitions grounded in retrieved passages
    "market": 0.10,         # restating fetched quotes
    "news": 0.25,           # summarising headlines
    "goal_planning": 0.30,  # explaining a computed projection and its trade-offs
    "portfolio": 0.35,      # interpreting several metrics together
    "tax": 0.35,            # accuracy-sensitive rules
    "router": 0.0,          # intent classification is always a cheap-model job
}

_DEFINITION = re.compile(r"^\s*(what(?:'s| is| are| does)|define|meaning of|who is|explain what)\b", re.I)
_REASONING = re.compile(
    r"\b(compare|comparison|versus|vs\.?|trade-?offs?|pros and cons|difference between|why|how should|"
    r"which is better|better (?:for|than)|strategy|strategies|scenario|what if|what happens if|implications?|"
    r"optimi[sz]e|rebalanc\w*|evaluate|step[- ]by[- ]step|walk me through|in detail|analy[sz]e|"
    r"relationship between|impact of|depends?)\b", re.I)
_MULTI_PART = re.compile(r"\?.*\?|\b(and also|as well as|additionally|then explain|, and (?:how|why|what))\b", re.I | re.S)
_NUMBER = re.compile(r"\d[\d,.]*")


@dataclass(frozen=True)
class RouteDecision:
    tier: str
    score: float
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {"tier": self.tier, "score": self.score, "reasons": list(self.reasons)}


class ModelRouter:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def decide(self, question: str, agent: str = "", context_chars: int = 0, history_len: int = 0,
               advice_request: bool = False, multi_agent: bool = False) -> RouteDecision:
        q = (question or "").strip()
        words = len(q.split())
        score = AGENT_BASE.get(agent, 0.15)
        reasons = [f"agent:{agent or 'unknown'}+{score:.2f}"]

        def add(points: float, why: str) -> None:
            nonlocal score
            score += points
            reasons.append(f"{why}{points:+.2f}")

        if _DEFINITION.search(q) and words <= 12:
            add(-0.15, "short definition")
        reasoning_hits = len({m.group(0).lower() for m in _REASONING.finditer(q)})
        if reasoning_hits:
            add(min(0.20 * reasoning_hits, 0.40), f"reasoning words x{reasoning_hits}")
        if _MULTI_PART.search(q):
            add(0.15, "multi-part question")
        if words > 60:
            add(0.25, "very long question")
        elif words > 25:
            add(0.15, "long question")
        if len(_NUMBER.findall(q)) >= 3:
            add(0.10, "several numbers")
        if context_chars > 4000:
            add(0.10, "large context")
        if history_len >= 6:
            add(0.05, "long conversation")
        if advice_request:
            add(0.15, "advice-sensitive")
        if multi_agent:
            add(0.20, "multi-agent synthesis")

        score = round(max(0.0, min(score, 1.0)), 2)
        return RouteDecision(SMART if score >= self.threshold else FAST, score, tuple(reasons))
