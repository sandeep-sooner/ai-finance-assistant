from src.agents.base_agent import BaseAgent


class FinanceQAAgent(BaseAgent):
    name = "finance_qa"
    label = "Finance Q&A"
    system_prompt = (
        "You are the Finance Q&A agent of an educational assistant for beginners. "
        "Use plain language, define any jargon, and give one short example. "
        "You provide education, not personalised financial advice. "
        "Never recommend a specific security or promise returns."
    )
