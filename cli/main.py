# cli/main.py
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from controller.central_controller import route_user_query
from agents.budget_agent import run_budget_agent_loop
from agents.anomaly_agent import run_anomaly_agent_loop
from agents.stock_agent import run_stock_agent_loop
from agents.portfolio_agent import run_portfolio_agent_loop


def main():
    print("""👋 Welcome to FinFluent - your personal, AI-powered financial advisor!
          
Here's what I can do:
🔮 Budget Forecasting — See where your money is headed next month  
🚨 Anomaly Detection — Spot unusual or suspicious transactions  
📈 Stock Sentiment — Get the latest trends and news on your favorite stocks
📊 Portfolio Review - Get insights on your stock portfolio 
          
How can I help you today?
""")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("FinFluent> ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye 👋")
            break

        route = route_user_query(user_input)
        print(f"Connecting you to a specialist: {route}")

        if route == "budget":
            run_budget_agent_loop(str(ROOT / "data" / "user_3.csv"))
            response = None
        elif route == "anomaly":
            run_anomaly_agent_loop(str(ROOT / "data" / "user_1.csv"))
            response = None
        elif route == "stock":
            run_stock_agent_loop()
            response = None  
        elif route == "portfolio":
            run_portfolio_agent_loop(str(ROOT / "data" / "sample_portfolio.csv"))
            response = None

        else:
            response = "Sorry, I didn't understand. Please ask about your budget, anomalies, stocks, or portfolio analysis."

        if response:
            print(f"\n{response}\n")


if __name__ == "__main__":
    main()
