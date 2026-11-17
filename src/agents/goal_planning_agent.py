from src.agents.base_agent import AgentReply, BaseAgent, recent_user_texts
from src.core.llm import Message
from src.core.schemas import GoalPlan, GoalRequest
from src.goals.planner import parse_goal, plan_goal, to_prompt_context

HELP_TEXT = (
    "I can project a savings goal for you. Tell me **how much** you need, **in how many years**, and "
    "optionally what you have saved and can add each month. For example: *\"Save $50,000 in 10 years. "
    "I have $5,000 and can add $300 a month.\"* You can also use the **Goals** tab to try different numbers."
)


class GoalPlanningAgent(BaseAgent):
    name = "goal_planning"
    rag_k = 2
    label = "Goal planning"
    rag_category = "planning"
    system_prompt = (
        "You are the Goal planning agent of an educational assistant for beginners. You are given a "
        "savings projection that was already computed. Explain it in plain language: whether the user is on "
        "track, why the target grows with inflation, how much comes from saving versus growth, and what the "
        "three scenarios show about risk and return. Then give two or three general levers the user controls "
        "(saving more, more time, a different target) without recommending specific products. Use only the "
        "numbers given and stress that returns are assumptions, not promises. Keep it under about 250 words, "
        "with short headings or bullets."
    )

    def explain(
        self, history: list[Message], question: str, plan: GoalPlan, req: GoalRequest, advice_request: bool = False,
        level: str = "beginner",
    ) -> AgentReply:
        return self.prepare(history, question, advice_request, extra_context=to_prompt_context(plan, req), level=level)

    async def respond(self, history: list[Message], question: str, advice_request: bool = False,
                      level: str = "beginner") -> AgentReply:
        # A follow-up ("what if I add $100 more?") may not restate the goal: fall back to earlier messages.
        req = parse_goal(question) or next((r for r in map(parse_goal, recent_user_texts(history)) if r), None)
        if req is None:
            return AgentReply.canned(HELP_TEXT)
        return self.explain(history, question, plan_goal(req), req, advice_request, level)
