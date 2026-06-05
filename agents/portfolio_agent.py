import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "stock_sentiment_analysis",
            "master_service",
        )
    )
)

import pandas as pd
import requests

try:
    import streamlit as st
except ImportError:
    st = None

from stock_sentiment_analysis.master_service.master_agent.agents.price import PriceAgent
from stock_sentiment_analysis.master_service.master_agent.agents.alpha_vantage_agent import (
    AlphaVantageNewsAgent,
)


def run_portfolio_agent_loop(csv_path: str, streamlit_mode=False):
    # Session memory setup
    if streamlit_mode and st:
        if "agent_conversations" not in st.session_state:
            st.session_state.agent_conversations = {}
        if "portfolio" not in st.session_state.agent_conversations:
            st.session_state.agent_conversations["portfolio"] = []
        memory = st.session_state.agent_conversations["portfolio"]
        user_input = st.session_state.current_input
    else:
        if not hasattr(run_portfolio_agent_loop, "memory"):
            run_portfolio_agent_loop.memory = []
        memory = run_portfolio_agent_loop.memory

        print("\n📊 Entering Stock Portfolio Analyzer")
        print("Reading your portfolio and fetching market insights...")

    # First message = build system prompt
    if not memory:
        df = pd.read_csv(csv_path)
        if df.empty or not all(
            col in df.columns for col in ["Ticker", "Holding", "Profit percentage"]
        ):
            error_msg = "❌ CSV format is invalid or empty."
            return error_msg if streamlit_mode else print(error_msg)

        price_agent = PriceAgent()
        news_agent = AlphaVantageNewsAgent()
        portfolio_summary = []

        for _, row in df.iterrows():
            data = {
                "ticker": row["Ticker"],
                "holding": row["Holding"],
                "profit_pct": row["Profit percentage"],
            }
            data = price_agent.run(data)
            data = news_agent.run(data)
            portfolio_summary.append(data)

        system_prompt = """
You are a financial portfolio advisor. Your job is to analyze the user's holdings based on current prices, profit %, and recent news.

The dataset contains:
- Ticker: Stock ticker
- Holding: Number of shares held
- Profit percentage: Gain or loss since purchase

Give:
- Suggestions on whether to hold/sell
- Highlight risky or overperforming stocks
- Tips for balancing the portfolio
- Mention if any stocks are overexposed or trending negatively

In the end, include:
"Investments in the stock market are subject to market risks. Please perform your own research before investing."
"""

        analysis_input = "### User Portfolio:\n"
        for stock in portfolio_summary:
            analysis_input += f"""
{stock['ticker']}:
- Holding: {stock['holding']}
- Profit %: {stock['profit_pct']}%
- Price: ${stock.get('price', 'N/A')}
- Recent News:
"""
            for article in stock.get("sources", [])[:2]:
                analysis_input += f"    • {article['title']} ({article['sentiment_label']})\n"

        memory.extend([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": analysis_input}
        ])

        response = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": "llama3", "messages": memory, "stream": False},
            headers={"Content-Type": "application/json"},
        ).json()["message"]["content"]

        memory.append({"role": "assistant", "content": response})
        if streamlit_mode:
            return response
        else:
            print(f"\n💬 {response}\n")

    # CLI Follow-up loop (always run after first response)
    if not streamlit_mode:
        while True:
            user_input = input("PortfolioAgent> ").strip()
            if user_input.lower() in ["exit", "quit", "back"]:
                print("↩️ Returning to FinFluent main menu.\n")
                break

            memory.append({"role": "user", "content": user_input})
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={"model": "llama3", "messages": memory, "stream": False},
                headers={"Content-Type": "application/json"},
            ).json()["message"]["content"]

            memory.append({"role": "assistant", "content": response})
            print(f"\n💬 {response}\n")

    # Streamlit follow-up (one turn)
    elif streamlit_mode:
        if user_input.lower() in ["exit", "quit", "back"]:
            return None

        memory.append({"role": "user", "content": user_input})
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": "llama3", "messages": memory, "stream": False},
            headers={"Content-Type": "application/json"},
        ).json()["message"]["content"]

        memory.append({"role": "assistant", "content": response})
        return response
