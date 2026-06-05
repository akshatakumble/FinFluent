import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.tseries.offsets import MonthEnd
import requests

try:
    import streamlit as st
except ImportError:
    st = None


def forecast_sarima(data, steps=1):
    model = SARIMAX(
        data,
        order=(3, 0, 0),
        seasonal_order=(1, 0, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    model_fit = model.fit(disp=False)
    return model_fit.forecast(steps=steps)


def run_budget_agent_loop(transactions_path: str, streamlit_mode=False):
    # === Memory setup ===
    if streamlit_mode and st:
        if "agent_conversations" not in st.session_state:
            st.session_state.agent_conversations = {}
        if "budget" not in st.session_state.agent_conversations:
            st.session_state.agent_conversations["budget"] = []
        memory = st.session_state.agent_conversations["budget"]
        user_input = st.session_state.current_input
    else:
        if not hasattr(run_budget_agent_loop, "memory"):
            run_budget_agent_loop.memory = []
        memory = run_budget_agent_loop.memory

        print("\n💰 Entering Budget Forecasting Mode")
        print("Ask questions about your spending forecast. Type 'exit' to return.\n")

        # Prompt for first input (after greeting)
        user_input = "Please analyze my forecast and offer suggestions."

    # === Initial setup: Forecast and context ===
    if not memory:
        df = pd.read_csv(transactions_path, parse_dates=["Date"])
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        df_salary = df[
            (df["Category"] == "Salary") &
            (df["Transaction Type"].str.lower() == "credit")
        ]
        latest_year = df_salary["Date"].dt.year.max()
        latest_salary_entry = df_salary[df_salary["Date"].dt.year == latest_year].sort_values("Date", ascending=False).head(1)
        user_salary = latest_salary_entry["Amount"].values[0] if not latest_salary_entry.empty else "Unknown"

        debit_categories = {
            "Shopping", "Entertainment", "Restaurants", "Travel expenses",
            "Mortgage & Rent", "Grocery shopping", "Utilities", "Heating fuel",
        }
        df = df[df["Category"].isin(debit_categories)]
        df["Amount"] = df["Amount"].abs()
        df["Month"] = (df["Date"] + MonthEnd(0)).dt.to_period("M").dt.to_timestamp()
        monthly_spending = df.groupby(["Month", "Category"])["Amount"].sum().unstack().fillna(0)
        monthly_spending.index = pd.date_range(start=monthly_spending.index.min(), periods=len(monthly_spending), freq="MS")

        future_spending = {
            category: forecast_sarima(monthly_spending[category]).iloc[0]
            for category in monthly_spending.columns
        }

        forecast_text = "\n".join(f"- {cat}: ${amt:.2f}" for cat, amt in future_spending.items())

        system_prompt = f"""
You are an AI-powered Financial Advisor. Your job is to provide accurate, data-driven financial guidance.

## User Information:
- Monthly Salary: ${user_salary}
- Predicted Spending for Next Month:
{forecast_text}

## Instructions:
1. Next Be specific and data-driven: - When answering questions, refer to the user’s salary and predicted spending.  
   - Use numbers, percentages, and trends instead of generic advice.
2. Recommend savings strategies: - If spending in a category is higher than 30% of salary, suggest ways to reduce it.  
   - Recommend savings and investment strategies based on spending habits.
3. Warn about high-risk categories - If spending in a category has increased significantly, explain why it might be a problem.  
   - Identify risky spending patterns.
4. Use plain text. No bold or italics or any special characters
"""
        memory.extend([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
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

    # === CLI Conversation loop ===
    if not streamlit_mode:
        while True:
            user_input = input("BudgetAgent> ").strip()
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

    # === Streamlit follow-up ===
    elif streamlit_mode:
        memory.append({"role": "user", "content": user_input})
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": "llama3", "messages": memory, "stream": False},
            headers={"Content-Type": "application/json"},
        ).json()["message"]["content"]
        memory.append({"role": "assistant", "content": response})
        return response
