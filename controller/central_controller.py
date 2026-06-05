# controller/central_controller.py
from utils.llama3_ollama import ask_llama3


def route_user_query(user_query: str) -> str:
    """
    Use LLaMA 3 to decide which agent to run.
    Returns: "budget", "anomaly", "stock", "portfolio" or "unknown"
    """
    prompt = f"""
You are a routing agent in a CLI-based financial assistant. Based on the user's question, decide which domain it belongs to:
1. budget - for forecasting and expense-related questions
2. anomaly - for spotting unusual or outlier transactions
3. stock - for analyzing the current stock price and news sentiment of a company
4. portfolio - for analyzing user's stock portfolio or user's stock holdings

Respond ONLY with one of these: budget, anomaly, stock, portfolio unknown

User query: "{user_query}"
Your response:
"""
    category = ask_llama3(prompt).strip().lower()
    if category not in ["budget", "anomaly", "stock", "portfolio"]:
        category = "unknown"
    return category
