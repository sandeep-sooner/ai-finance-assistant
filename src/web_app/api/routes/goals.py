from fastapi import APIRouter, Depends, Request

from src.core.schemas import GoalPlan, GoalRequest
from src.goals.planner import plan_goal
from src.web_app.api.deps import api_rate_limit, llm_rate_limit, sse_response, stream_reply

router = APIRouter(prefix="/api/goals", tags=["goals"])

QUESTION = "Explain this savings plan for me."


@router.post("/plan", response_model=GoalPlan, dependencies=[Depends(api_rate_limit)])
async def plan(req: GoalRequest):
    """Deterministic projection (no LLM): target grown by inflation, three risk scenarios, required saving."""
    return plan_goal(req)


@router.post("/insights/stream", dependencies=[Depends(llm_rate_limit)])
async def insights(req: GoalRequest, request: Request):
    """Streams the Goal planning agent's plain-language explanation of the same projection."""
    agent = request.app.state.agents["goal_planning"]
    reply = agent.explain([], QUESTION, plan_goal(req), req)
    return sse_response(stream_reply(request.app.state.gateway, agent.name, reply, QUESTION))
