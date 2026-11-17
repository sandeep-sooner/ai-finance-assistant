"""Find ticker symbols in free text: "$aapl", "AAPL", or common names like "apple"."""
from __future__ import annotations

import re

from src.market.instruments import ALIASES, normalize_symbol

_DOLLAR = re.compile(r"\$([A-Za-z]{1,5}(?:[.-][A-Za-z])?)\b")
_UPPER = re.compile(r"(?<![\w$])([A-Z]{1,5}(?:[.-][A-Z])?)\b")
_ALIAS = re.compile(r"\b(" + "|".join(re.escape(a) for a in sorted(ALIASES, key=len, reverse=True)) + r")\b", re.I)

# Upper-case words that are not tickers in a finance chat.
STOPWORDS = frozenset({
    "A", "I", "AI", "ETF", "ETFS", "IRA", "ROTH", "USD", "INR", "EUR", "GBP", "US", "USA", "UK", "EU",
    "CEO", "CFO", "IPO", "GDP", "CPI", "FED", "SEC", "PE", "EPS", "ROI", "NAV", "OK", "FAQ", "YTD",
    "SIP", "NPS", "PPF", "ELSS", "TAX", "NEWS", "WHAT", "HOW", "IS", "THE", "AND", "OR", "VS", "DO",
    "K", "M", "B", "X", "S", "P", "Q", "MY", "ME", "IT", "TO", "IN", "ON", "AT", "OF", "FOR", "BY",
})


def extract_symbols(text: str, limit: int = 5) -> list[str]:
    found: list[str] = []
    for m in _DOLLAR.finditer(text):
        found.append(m.group(1))
    for m in _ALIAS.finditer(text):
        found.append(ALIASES[m.group(1).lower()])
    for m in _UPPER.finditer(text):
        if m.group(1) not in STOPWORDS:
            found.append(m.group(1))
    return list(dict.fromkeys(normalize_symbol(s) for s in found))[:limit]
