"""Static reference data for common tickers.

Used for (1) classifying holdings without a network call and (2) offline sample
prices when the market data provider is unavailable. Sample prices are
illustrative and always labelled as such; they are never presented as quotes.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    name: str
    asset_class: str     # "Stock", "Broad equity fund", "Bond fund", ...
    sector: str          # "Diversified" for broad funds, "Fixed income" for bonds
    diversified: bool    # True for funds that already hold many companies
    sample_price: float  # offline fallback only
    expense_ratio: float | None = None  # yearly fund cost as a fraction (approximate, from fund fact sheets)


def _stock(name: str, sector: str, price: float) -> Instrument:
    return Instrument(name, "Stock", sector, False, price)


INSTRUMENTS: dict[str, Instrument] = {
    "AAPL": _stock("Apple Inc.", "Technology", 230.0),
    "MSFT": _stock("Microsoft Corporation", "Technology", 420.0),
    "NVDA": _stock("NVIDIA Corporation", "Technology", 130.0),
    "GOOGL": _stock("Alphabet Inc. Class A", "Communication", 175.0),
    "GOOG": _stock("Alphabet Inc. Class C", "Communication", 176.0),
    "META": _stock("Meta Platforms, Inc.", "Communication", 560.0),
    "NFLX": _stock("Netflix, Inc.", "Communication", 700.0),
    "AMZN": _stock("Amazon.com, Inc.", "Consumer discretionary", 190.0),
    "TSLA": _stock("Tesla, Inc.", "Consumer discretionary", 250.0),
    "JPM": _stock("JPMorgan Chase & Co.", "Financials", 220.0),
    "V": _stock("Visa Inc.", "Financials", 280.0),
    "MA": _stock("Mastercard Incorporated", "Financials", 500.0),
    "BRK-B": _stock("Berkshire Hathaway Inc. Class B", "Financials", 450.0),
    "JNJ": _stock("Johnson & Johnson", "Health care", 160.0),
    "UNH": _stock("UnitedHealth Group Incorporated", "Health care", 560.0),
    "PFE": _stock("Pfizer Inc.", "Health care", 28.0),
    "XOM": _stock("Exxon Mobil Corporation", "Energy", 115.0),
    "CVX": _stock("Chevron Corporation", "Energy", 150.0),
    "PG": _stock("Procter & Gamble Company", "Consumer staples", 165.0),
    "KO": _stock("Coca-Cola Company", "Consumer staples", 65.0),
    "WMT": _stock("Walmart Inc.", "Consumer staples", 80.0),
    "VTI": Instrument("Vanguard Total Stock Market ETF", "Broad equity fund", "Diversified", True, 280.0, 0.0003),
    "VOO": Instrument("Vanguard S&P 500 ETF", "Broad equity fund", "Diversified", True, 510.0, 0.0003),
    "SPY": Instrument("SPDR S&P 500 ETF Trust", "Broad equity fund", "Diversified", True, 560.0, 0.0009),
    "IVV": Instrument("iShares Core S&P 500 ETF", "Broad equity fund", "Diversified", True, 565.0, 0.0003),
    "DIA": Instrument("SPDR Dow Jones Industrial Average ETF", "Broad equity fund", "Diversified", True, 420.0, 0.0016),
    "IWM": Instrument("iShares Russell 2000 ETF", "Broad equity fund", "Diversified", True, 220.0, 0.0019),
    "SCHD": Instrument("Schwab U.S. Dividend Equity ETF", "Broad equity fund", "Diversified", True, 28.0, 0.0006),
    "QQQ": Instrument("Invesco QQQ Trust", "Equity fund", "Technology", False, 480.0, 0.002),  # Nasdaq-100: tech-heavy
    "VXUS": Instrument("Vanguard Total International Stock ETF", "International equity fund", "Diversified", True, 62.0, 0.0005),
    "VEA": Instrument("Vanguard FTSE Developed Markets ETF", "International equity fund", "Diversified", True, 50.0, 0.0003),
    "VWO": Instrument("Vanguard FTSE Emerging Markets ETF", "International equity fund", "Diversified", True, 44.0, 0.0007),
    "BND": Instrument("Vanguard Total Bond Market ETF", "Bond fund", "Fixed income", True, 73.0, 0.0003),
    "AGG": Instrument("iShares Core U.S. Aggregate Bond ETF", "Bond fund", "Fixed income", True, 99.0, 0.0003),
    "TLT": Instrument("iShares 20+ Year Treasury Bond ETF", "Bond fund", "Fixed income", True, 92.0, 0.0015),
    "VYM": Instrument("Vanguard High Dividend Yield ETF", "Broad equity fund", "Diversified", True, 125.0, 0.0006),
    "VNQ": Instrument("Vanguard Real Estate ETF", "Equity fund", "Real estate", False, 90.0, 0.0013),
    "GLD": Instrument("SPDR Gold Shares", "Commodity fund", "Commodities", False, 240.0, 0.004),
}

# Common names people type instead of tickers.
ALIASES: dict[str, str] = {
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "google": "GOOGL", "alphabet": "GOOGL",
    "meta": "META", "facebook": "META", "netflix": "NFLX", "amazon": "AMZN", "tesla": "TSLA",
    "berkshire": "BRK-B", "s&p 500": "SPY", "s&p": "SPY", "sp500": "SPY", "nasdaq": "QQQ",
    "dow jones": "DIA", "dow": "DIA", "russell 2000": "IWM", "gold": "GLD",
}

# What the Market tab and "how is the market today?" show.
OVERVIEW_SYMBOLS = ("SPY", "QQQ", "DIA", "IWM", "VXUS", "BND", "GLD")


def normalize_symbol(symbol: str) -> str:
    """Upper-case and use Yahoo's dash for share classes (BRK.B -> BRK-B)."""
    return symbol.strip().upper().replace(".", "-")


def lookup(symbol: str) -> Instrument | None:
    return INSTRUMENTS.get(normalize_symbol(symbol))
