from src.agents.base_agent import AgentReply, BaseAgent, recent_user_texts
from src.core.llm import LLM, Message
from src.core.schemas import PortfolioAnalysis
from src.market.service import MarketDataService
from src.portfolio.analysis import analyze_live, parse_holdings, to_prompt_context
from src.rag.retriever import Retriever

HELP_TEXT = (
    "I can analyze your portfolio, but I could not find any holdings in your message. "
    "List them as quantity and ticker, for example: **10 AAPL, 5 VTI @ 250, 20 BND**. "
    "The `@ price` part is your average cost and is optional. "
    "You can also enter holdings in the **Portfolio** tab."
)


class PortfolioAgent(BaseAgent):
    name = "portfolio"
    rag_k = 2
    label = "Portfolio analysis"
    system_prompt = (
        "You are the Portfolio analysis agent of an educational assistant for beginners. "
        "You are given metrics that were already computed for the user's holdings. "
        "Explain them in plain language: what the asset mix and sector mix look like, how concentrated "
        "the portfolio is (define 'effective number of holdings' simply), what the diversification score "
        "and risk level say, what the funds cost per year if fund costs are given, and what each flag means. "
        "Then list two or three general questions the user could think about, such as timeline or "
        "risk comfort. Use only the numbers given; never invent prices or returns. Mention where the "
        "prices came from, as stated in the price note. Use short headings or bullets and keep it under "
        "about 250 words. You provide education, not personalised financial advice: never tell the user "
        "to buy, sell or switch a specific security, and never promise returns."
    )

    def __init__(self, llm: LLM, retriever: Retriever | None = None, market: MarketDataService | None = None):
        super().__init__(llm, retriever)
        self.market = market

    def explain(
        self, history: list[Message], question: str, analysis: PortfolioAnalysis, advice_request: bool = False,
        level: str = "beginner",
    ) -> AgentReply:
        return self.prepare(history, question, advice_request, extra_context=to_prompt_context(analysis), level=level)

    async def respond(self, history: list[Message], question: str, advice_request: bool = False,
                      level: str = "beginner") -> AgentReply:
        # A follow-up ("is that too risky?") may not repeat the holdings: reuse the ones given earlier.
        holdings = parse_holdings(question) or next(
            (h for h in map(parse_holdings, recent_user_texts(history)) if h), []
        )
        if not holdings or self.market is None:
            return AgentReply.canned(HELP_TEXT)
        analysis = await analyze_live(holdings, self.market)
        return self.explain(history, question, analysis, advice_request, level)
