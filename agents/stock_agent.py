import subprocess
import time
import requests

try:
    import streamlit as st
except ImportError:
    st = None  # CLI-safe


def run_stock_agent_loop(streamlit_mode=False):
    # ✅ Set up memory
    if streamlit_mode and st:
        if "agent_conversations" not in st.session_state:
            st.session_state.agent_conversations = {}
        if "stock" not in st.session_state.agent_conversations:
            st.session_state.agent_conversations["stock"] = []
        memory = st.session_state.agent_conversations["stock"]
        user_input = st.session_state.get("current_input", "").strip()
    else:
        if not hasattr(run_stock_agent_loop, "memory"):
            run_stock_agent_loop.memory = []
        memory = run_stock_agent_loop.memory
        user_input = None

        print("\n📈 Entering Stock Sentiment Mode")
        print("Ask about any stock ticker (e.g., TSLA, AAPL, NVDA).")
        print("Type 'exit' to return to the main FinFluent menu.\n")

    # ✅ Start subprocesses once per session
    if not memory:
        print("🚀 Starting stock sentiment services...")
        subprocess.Popen(["bash", "stock_sentiment_analysis/llm_service/script.sh"])
        subprocess.Popen(["bash", "stock_sentiment_analysis/master_service/script.sh"])
        time.sleep(5)

    # ✅ Streamlit Mode
    if streamlit_mode and st:
        if user_input.lower() in ["exit", "quit", "back"]:
            st.session_state.agent_conversations["stock"] = []
            return "↩️ Exited Stock Agent. Ask something else to continue."

        ticker = next(
            (
                word
                for word in user_input.split()
                if word.isupper() and 2 <= len(word) <= 5
            ),
            None,
        )

        if not ticker:
            return "❗ Please enter a valid stock ticker (e.g., AAPL, TSLA)."

        try:
            output = subprocess.check_output(
                [
                    "python3",
                    "stock_sentiment_analysis/run_analysis.py",
                    "--ticker",
                    ticker,
                ],
                stderr=subprocess.STDOUT,
            )
            response = output.decode("utf-8")
            memory.append({"role": "user", "content": user_input})
            memory.append({"role": "assistant", "content": response})
            return response

        except subprocess.CalledProcessError as e:
            return f"❌ Stock analysis failed.\n\n{e.output.decode('utf-8') if e.output else 'No output'}"

    # ✅ CLI Mode
    else:
        while True:
            user_input = input("StockAgent> ").strip()
            if user_input.lower() in ["exit", "quit", "back"]:
                print("↩️ Returning to FinFluent main menu.\n")
                break

            ticker = next(
                (
                    word
                    for word in user_input.split()
                    if word.isupper() and 2 <= len(word) <= 5
                ),
                None,
            )

            if not ticker:
                print("❗ Please include a valid stock ticker (e.g., AAPL, TSLA).")
                continue

            try:
                output = subprocess.check_output(
                    [
                        "python3",
                        "stock_sentiment_analysis/run_analysis.py",
                        "--ticker",
                        ticker,
                    ],
                    stderr=subprocess.STDOUT,
                )
                response = output.decode("utf-8")
                memory.append({"role": "user", "content": user_input})
                memory.append({"role": "assistant", "content": response})
                print(f"\n💬 {response}\n")

            except subprocess.CalledProcessError as e:
                print("❌ Stock analysis failed.")
                print(f"Command: {e.cmd}")
                print(f"Output:\n{e.output.decode('utf-8') if e.output else 'No output'}")
