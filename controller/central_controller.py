# controller/central_controller.py
import re

from utils.llama3_ollama import ask_llama3

VALID_ROUTES = frozenset({"budget", "anomaly", "stock", "portfolio"})


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


def route_user_query(user_query: str) -> str:
    """
    Use LLaMA 3 to decide which agent should handle the query.
    Returns: budget | anomaly | stock | portfolio | unknown
    """
    prompt = f"""
You are a routing agent in a financial assistant. Pick exactly ONE domain:

- budget — forecasting, monthly spend, savings, category budgets
- anomaly — unusual charges, red flags, suspicious or outlier transactions
- stock — a specific ticker, stock price, news sentiment, buy/hold/sell for one company
- portfolio — multiple holdings, diversification, portfolio review or rebalance

Respond with ONLY one word: budget, anomaly, stock, portfolio, or unknown

User query: "{user_query}"
Route:"""
    category = normalize_route(ask_llama3(prompt))
    return category
