# controller/central_controller.py
import re

from utils.llama3_ollama import ask_llama3

VALID_ROUTES = frozenset({"budget", "anomaly", "stock", "portfolio"})

# High-confidence phrases — checked before LLM so routing works even if Ollama is flaky
_STRONG_ROUTE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(red flags?|unusual transactions?|suspicious|outliers?|"
            r"fraud|anomalies|weird charges?)\b",
            re.I,
        ),
        "anomaly",
    ),
    (
        re.compile(
            r"\b(forecast|next month|budget|overspend|overspending|"
            r"how much (will|should) i spend)\b",
            re.I,
        ),
        "budget",
    ),
    (
        re.compile(
            r"\b(portfolio|holdings|rebalance|overexposed|diversif)\b",
            re.I,
        ),
        "portfolio",
    ),
    (
        re.compile(
            r"\b(stock sentiment|outlook on|should i buy|how is .+ stock)\b",
            re.I,
        ),
        "stock",
    ),
    (re.compile(r"\b(AAPL|TSLA|NVDA|MSFT|GOOG|AMZN|META|NFLX)\b"), "stock"),
]


def normalize_route(raw: str) -> str:
    """Map noisy LLM router output to a single valid agent id."""
    text = raw.strip().lower()
    if text in VALID_ROUTES:
        return text
    if text == "unknown":
        return "unknown"

    for route in VALID_ROUTES:
        if re.search(rf"\b{route}\b", text):
            return route

    return "unknown"


def keyword_route(user_query: str) -> str | None:
    """Deterministic routing for clear phrases (fast fallback)."""
    for pattern, route in _STRONG_ROUTE_PATTERNS:
        if pattern.search(user_query):
            return route
    return None


def route_user_query(user_query: str) -> str:
    """
    Route based on THIS message only (ignores prior agent context).
    Returns: budget | anomaly | stock | portfolio | unknown
    """
    keyword = keyword_route(user_query)
    if keyword:
        return keyword

    prompt = f"""
You are a routing agent in a financial assistant. Route based on THIS message only,
even if it would change topic from a prior turn. Pick exactly ONE domain:

- budget — forecasting, monthly spend, savings, category budgets
- anomaly — unusual charges, red flags, suspicious or outlier transactions
- stock — a specific ticker, stock price, news sentiment, buy/hold/sell for one company
- portfolio — multiple holdings, diversification, portfolio review or rebalance

Respond with ONLY one word: budget, anomaly, stock, portfolio, or unknown

User query: "{user_query}"
Route:"""
    llm_route = normalize_route(ask_llama3(prompt))
    if llm_route != "unknown":
        return llm_route
    return keyword or "unknown"