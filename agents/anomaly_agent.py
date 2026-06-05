from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import requests

try:
    import streamlit as st
except ImportError:
    st = None  # For CLI


def run_anomaly_agent_loop(transactions_path: str, streamlit_mode=False):
    # Memory setup
    if streamlit_mode and st:
        if "agent_conversations" not in st.session_state:
            st.session_state.agent_conversations = {}
        if "anomaly" not in st.session_state.agent_conversations:
            st.session_state.agent_conversations["anomaly"] = []
        memory = st.session_state.agent_conversations["anomaly"]
        user_input = st.session_state.current_input
    else:
        if not hasattr(run_anomaly_agent_loop, "memory"):
            run_anomaly_agent_loop.memory = []
        memory = run_anomaly_agent_loop.memory

        print("\n🚨 Entering Anomaly Detection Mode")
        print("I've scanned your debit transactions for unusual spending.")
        print("Ask about any transaction, category, or pattern. Type 'exit' to return.\n")

        user_input = "Please analyze these transactions and tell me what's unusual."

    # First run: scan transactions and send summary
    if not memory:
        df = pd.read_csv(transactions_path)
        df_debit = df[df["Transaction Type"].str.lower() == "debit"].copy()

        if df_debit.empty or "Amount" not in df_debit.columns:
            msg = "❌ No debit transactions found or missing 'Amount' column."
            if streamlit_mode:
                return msg
            print(msg)
            return

        scaler = StandardScaler()
        df_debit["Amount_scaled"] = scaler.fit_transform(df_debit[["Amount"]])

        model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        df_debit["outlier_flag"] = model.fit_predict(df_debit[["Amount_scaled"]])
        df_debit["is_outlier"] = df_debit["outlier_flag"] == -1

        outliers = df_debit[df_debit["is_outlier"]]
        outliers = outliers[outliers["Category"] != "Mortgage & Rent"]
        outliers = outliers.sort_values(by="Amount", ascending=False)

        if outliers.empty:
            msg = "✅ No major spending anomalies detected this month. You're all good!"
            if streamlit_mode:
                return msg
            print(msg)
            return

        outlier_summary = ""
        for _, row in outliers.head(5).iterrows():
            outlier_summary += f"- {row['Date']}: ${row['Amount']:.2f} for {row['Category']} ({row['Description']})\n"

        system_prompt = f"""
You are a smart financial assistant. Below are unusual debit transactions detected by an Isolation Forest algorithm.

## Detected Anomalies:
{outlier_summary}

## Instructions:
1. Summarize the potential concerns in a friendly tone.
2. Mention if these seem risky or need user attention.
3. Suggest follow-up steps or questions to ask the user.
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

    # CLI mode conversation loop
    if not streamlit_mode:
        while True:
            user_input = input("AnomalyAgent> ").strip()
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

    # Streamlit follow-up
    elif streamlit_mode:
        memory.append({"role": "user", "content": user_input})
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": "llama3", "messages": memory, "stream": False},
            headers={"Content-Type": "application/json"},
        ).json()["message"]["content"]
        memory.append({"role": "assistant", "content": response})
        return response
