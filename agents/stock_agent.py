import subprocess
import sys
import time
from pathlib import Path

import requests

from utils.agent_response import AgentResponse

try:
    import streamlit as st
except ImportError:
    st = None  # CLI-safe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_SERVICE_DIR = PROJECT_ROOT / "stock_sentiment_analysis" / "master_service"
LLM_SERVICE_DIR = PROJECT_ROOT / "stock_sentiment_analysis" / "llm_service"


def _service_ready(url: str) -> bool:
    try:
        requests.get(url, timeout=2)
        return True
    except Exception:
        return False


def _ensure_stock_services(streamlit_mode: bool) -> None:
    if _service_ready("http://127.0.0.1:8000/docs"):
        return

    if streamlit_mode and st:
        if st.session_state.get("stock_services_started"):
            return
        st.session_state.stock_services_started = True
    elif not streamlit_mode:
        print("Starting stock sentiment services...")

    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app", "--port", "8000"],
        cwd=str(MASTER_SERVICE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "llm_app:app", "--port", "8001"],
        cwd=str(LLM_SERVICE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(5)


def _run_stock_analysis(ticker: str) -> str:
    output = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "stock_sentiment_analysis.run_analysis",
            "--ticker",
            ticker,
        ],
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return output.strip()


def run_stock_agent_loop(streamlit_mode=False):    # ✅ Set up memory
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

        print("\nEntering Stock Sentiment Mode")
        print("Ask about any stock ticker (e.g., TSLA, AAPL, NVDA).")
        print("Type 'exit' to return to the main FinFluent menu.\n")

    if not memory:
        _ensure_stock_services(streamlit_mode)
    # ✅ Streamlit Mode
    if streamlit_mode and st:
        if user_input.lower() in ["exit", "quit", "back"]:
            st.session_state.agent_conversations["stock"] = []
            return AgentResponse.from_text(
                "Exited Stock Agent. Ask something else to continue."
            )
        ticker = next(
            (
                word
                for word in user_input.split()
                if word.isupper() and 2 <= len(word) <= 5
            ),
            None,
        )

        if not ticker:
            return AgentResponse.from_text(
                "Please enter a valid stock ticker (e.g., AAPL, TSLA)."
            )

        if not _service_ready("http://127.0.0.1:8000/docs"):
            return AgentResponse.from_text(
                "Stock services are not running. Start them manually:\n"
                "1. In stock_sentiment_analysis/llm_service: "
                "`uvicorn llm_app:app --port 8001`\n"
                "2. In stock_sentiment_analysis/master_service: "
                "`uvicorn server:app --port 8000`"
            )

        try:
            response = _run_stock_analysis(ticker)
            memory.append({"role": "user", "content": user_input})
            memory.append({"role": "assistant", "content": response})
            return AgentResponse.from_text(response)

        except subprocess.CalledProcessError as e:
            detail = e.output if isinstance(e.output, str) else (
                e.output.decode("utf-8", errors="replace") if e.output else "No output"
            )
            return AgentResponse.from_text(f"Stock analysis failed.\n\n{detail}")
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
                print("Please include a valid stock ticker (e.g., AAPL, TSLA).")
                continue

            try:
                response = _run_stock_analysis(ticker)
                memory.append({"role": "user", "content": user_input})
                memory.append({"role": "assistant", "content": response})
                print(f"\n{response}\n")

            except subprocess.CalledProcessError as e:
                print("Stock analysis failed.")
                print(f"Command: {e.cmd}")
                detail = e.output if isinstance(e.output, str) else (
                    e.output.decode("utf-8", errors="replace") if e.output else "No output"
                )
                print(f"Output:\n{detail}")