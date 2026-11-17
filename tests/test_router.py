import pytest

from src.workflow.router import classify


@pytest.mark.parametrize(
    "text,intent",
    [
        ("What is an ETF?", "finance_qa"),
        ("Explain diversification with an example", "finance_qa"),
        ("What is SPY trading at today?", "market"),
        ("price of $AAPL", "market"),
        ("Plan my savings for a 10-year goal", "goal_planning"),
        ("How much to retire at 60?", "goal_planning"),
        ("Analyze my portfolio please", "portfolio"),
        ("I own 10 AAPL and 5 VTI, is that diversified?", "portfolio"),
        ("I want to save $50,000 in 10 years", "goal_planning"),
        ("How is the stock market doing?", "market"),
        ("How is NVDA doing", "market"),
        ("How is the economy doing?", "finance_qa"),
        ("Latest headlines about Apple", "news"),
        ("What is a Roth IRA?", "tax"),
        ("Any news on interest rates?", "news"),
    ],
)
def test_classify(text, intent):
    assert classify(text) == intent


def test_etf_is_not_mistaken_for_a_ticker():
    assert classify("What is an ETF and how is it different from a mutual fund?") == "finance_qa"
