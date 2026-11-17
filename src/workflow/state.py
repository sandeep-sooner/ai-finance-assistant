from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AssistantState(TypedDict, total=False):
    """Shared state passed between graph nodes. The checkpointer stores it per session (thread_id)."""

    messages: Annotated[list, add_messages]  # chat history, restored by the checkpointer
    level: str                # "beginner" | "intermediate": how much jargon answers may assume
    intent: str               # primary agent of the latest turn (kept for follow-up routing)
    plan: list[str]           # agents to run this turn, in order
    step: int                 # index into plan of the next agent to run
    route_method: str         # how the plan was chosen: rules | follow_up | default | llm
    blocked: bool
    block_reason: str
    advice_request: bool
    parts: list[dict]         # this turn's output so far: {"agent", "text", "canned"}
    sources: list[dict]
    traces: list[dict]        # one LLM gateway trace per model call this turn
    final_answer: str
