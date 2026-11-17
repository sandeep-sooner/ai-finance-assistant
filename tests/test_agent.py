from src.agents.base_agent import build_fallback, recent_user_texts
from src.agents.finance_qa_agent import FinanceQAAgent
from src.core.llm import FakeLLM


class OneDocRetriever:
    def search(self, query, category=None):
        return [{"title": "What is an ETF", "url": None, "category": "etfs", "source": "SEC",
                 "text": "An ETF is a basket. It trades all day. Third sentence.", "score": 0.9}]


def test_no_docs_means_no_sources_and_no_citation_instruction():
    reply = FinanceQAAgent(FakeLLM()).prepare([], "hi")
    assert reply.sources == [] and reply.fallback_text is None
    assert "do not cite" in reply.messages[0][1]
    assert reply.messages[-1] == ("user", "hi")


def test_docs_become_numbered_context_and_sources():
    reply = FinanceQAAgent(FakeLLM(), OneDocRetriever()).prepare([("user", "a"), ("assistant", "b")], "q")
    assert "[1] What is an ETF" in reply.messages[0][1]
    assert reply.sources[0]["title"] == "What is an ETF" and reply.sources[0]["category"] == "SEC"
    assert [m[0] for m in reply.messages] == ["system", "user", "assistant", "user"]


def test_advice_request_adds_instruction():
    reply = FinanceQAAgent(FakeLLM()).prepare([], "should I buy X", advice_request=True)
    assert "Do not give one" in reply.messages[0][1]


def test_every_prompt_carries_the_no_advice_rules_and_the_level():
    agent = FinanceQAAgent(FakeLLM())
    beginner = agent.prepare([], "q").messages[0][1]
    assert "not personalised financial advice" in beginner and "beginner" in beginner
    assert "knows the basics" in agent.prepare([], "q", level="intermediate").messages[0][1]
    assert "beginner" in agent.prepare([], "q", level="nonsense").messages[0][1]  # unknown level -> safe default


def test_fallback_answer_is_built_from_passages_and_numbers_without_an_llm():
    reply = FinanceQAAgent(FakeLLM(), OneDocRetriever()).prepare([], "q", extra_context="Total value: $100\nRisk: Low")
    text = reply.fallback_text
    assert "An ETF is a basket. It trades all day." in text and "Third sentence" not in text  # two-sentence lead
    assert "- Total value: $100" in text and "not financial" in text
    assert build_fallback([], "") is None


def test_recent_user_texts_newest_first():
    history = [("user", "one"), ("assistant", "x"), ("user", "two"), ("assistant", "y")]
    assert recent_user_texts(history) == ["two", "one"]
    assert recent_user_texts(history, limit=1) == ["two"]
