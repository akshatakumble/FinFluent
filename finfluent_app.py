import html
import hashlib
import streamlit as st
import os
from pathlib import Path

from utils.agent_response import AgentResponse
from utils.encryption import SecureFileStore, secure_delete
from controller.central_controller import route_user_query
from agents.budget_agent import run_budget_agent_loop
from agents.anomaly_agent import run_anomaly_agent_loop
from agents.stock_agent import run_stock_agent_loop
from agents.portfolio_agent import run_portfolio_agent_loop

# ==============================
# 🔒 Default fallback CSV paths
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BUDGET_PATH = str(PROJECT_ROOT / "data" / "user_1.csv")
DEFAULT_ANOMALY_PATH = str(PROJECT_ROOT / "data" / "user_1.csv")
DEFAULT_PORTFOLIO_PATH = str(PROJECT_ROOT / "data" / "sample_portfolio.csv")

# ==============================
# ⚙️ Streamlit config & app title
# ==============================

st.set_page_config(page_title="FinFluent", layout="wide")
st.title("💰 FinFluent - Your AI Financial Advisor")

# ==============================
# 🧠 Session state initialization
# ==============================

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": """👋 Welcome to FinFluent!

Here’s what I can do:

🔮 Budget Forecasting — See where your money is headed next month  
🚨 Anomaly Detection — Spot unusual or suspicious transactions  
📈 Stock Sentiment — Get recent stock news and trends  
📊 Portfolio Review — Analyze your current holdings

You can also upload your own files.
"""
    }]

if "active_agent" not in st.session_state:
    st.session_state.active_agent = None

if "agent_conversations" not in st.session_state:
    st.session_state.agent_conversations = {
        "budget": [],
        "anomaly": [],
        "stock": [],
        "portfolio": [],
    }

if "secure_store" not in st.session_state:
    st.session_state.secure_store = SecureFileStore()

if "encrypted_paths" not in st.session_state:
    st.session_state.encrypted_paths = []

# ==============================
# 📁 Upload section (always at top)
# ==============================

st.markdown("### 📁 Upload Your Files")

st.markdown("Upload a bank statement for budget & anomaly detection:")
budget_file = st.file_uploader("📄 Bank Statement CSV", type=["csv"], key="budget_file", label_visibility="collapsed")

st.markdown("Upload a stock portfolio for portfolio analysis:")
portfolio_file = st.file_uploader("📊 Portfolio CSV", type=["csv"], key="portfolio_file", label_visibility="collapsed")

st.markdown(
    "<div style='margin-top: -5px; margin-bottom: 15px; font-size: 0.85rem; color: gray;'>"
    "🔐 Uploads are encrypted at rest with Fernet (AES-128). "
    "Files are decrypted only in memory while an agent analyzes them."
    "</div>",
    unsafe_allow_html=True,
)


def _replace_encrypted_upload(file_bytes: bytes, label: str) -> str:
    """Encrypt upload; replace prior encrypted file for this label only."""
    store: SecureFileStore = st.session_state.secure_store
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    hash_key = f"upload_hash_{label}"
    if st.session_state.get(hash_key) != file_hash:
        st.session_state[hash_key] = file_hash
        if label == "budget":
            st.session_state.pop("budget_evidence", None)
            st.session_state.pop("anomaly_evidence", None)
            st.session_state.agent_conversations["budget"] = []
            st.session_state.agent_conversations["anomaly"] = []
    old_path = st.session_state.get(f"encrypted_{label}")
    if old_path:
        secure_delete(old_path)
        if old_path in st.session_state.encrypted_paths:
            st.session_state.encrypted_paths.remove(old_path)
    enc_path = store.encrypt_bytes(file_bytes)
    st.session_state.encrypted_paths.append(enc_path)
    st.session_state[f"encrypted_{label}"] = enc_path
    return enc_path


# Handle uploaded files → encrypt; sample data stays plaintext under data/
if budget_file:
    BUDGET_DATA_PATH = _replace_encrypted_upload(budget_file.getvalue(), "budget")
    ANOMALY_DATA_PATH = BUDGET_DATA_PATH
elif st.session_state.get("encrypted_budget"):
    BUDGET_DATA_PATH = st.session_state["encrypted_budget"]
    ANOMALY_DATA_PATH = BUDGET_DATA_PATH
else:
    BUDGET_DATA_PATH = DEFAULT_BUDGET_PATH
    ANOMALY_DATA_PATH = DEFAULT_ANOMALY_PATH

if portfolio_file:
    PORTFOLIO_DATA_PATH = _replace_encrypted_upload(portfolio_file.getvalue(), "portfolio")
elif st.session_state.get("encrypted_portfolio"):
    PORTFOLIO_DATA_PATH = st.session_state["encrypted_portfolio"]
else:
    PORTFOLIO_DATA_PATH = DEFAULT_PORTFOLIO_PATH

# ==============================
# 🧼 Markdown sanitization helper
# ==============================

def safe_markdown(text: str) -> str:
    # Escape $ so Streamlit does not treat "$2000 ... $333" as LaTeX (spaces vanish).
    return (
        text.replace("\\", "\\\\")
            .replace("$", "\\$")
            .replace("_", "\\_")
            .replace("*", "\\*")
            .replace("`", "\\`")
    )


def normalize_response(raw) -> AgentResponse:
    if isinstance(raw, AgentResponse):
        return raw
    return AgentResponse.from_text(str(raw))


def render_thinking(thinking: str) -> None:
    escaped = html.escape(thinking).replace("\n", "<br>")
    st.markdown(
        "<div style=\"color:#9ca3af;font-size:0.85rem;line-height:1.6;"
        "margin-bottom:0.85rem;padding:0.65rem 0 0.65rem 0.85rem;"
        "border-left:2px solid #374151;\">"
        "<span style=\"font-size:0.72rem;text-transform:uppercase;"
        "letter-spacing:0.05em;color:#6b7280;\">Thinking</span><br>"
        f"{escaped}</div>",
        unsafe_allow_html=True,
    )


def render_assistant_message(msg: dict) -> None:
    if msg.get("thinking"):
        render_thinking(msg["thinking"])
    st.markdown(safe_markdown(msg["content"]))


# ==============================
# 💬 Render chat history
# ==============================

st.divider()
st.markdown("### 💬 FinFluent Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_assistant_message(msg)
        else:
            st.markdown(safe_markdown(msg["content"]))

# ==============================
# 📤 Input + Agent routing
# ==============================

def _reset_agent_session(agent: str | None) -> None:
    """Clear cached evidence and conversation for one agent."""
    if not agent:
        return
    st.session_state.agent_conversations[agent] = []
    if agent == "anomaly":
        st.session_state.pop("anomaly_evidence", None)
    if agent == "budget":
        st.session_state.pop("budget_evidence", None)


def _resolve_agent(user_input: str) -> tuple[str | None, bool]:
    """
    Route every message. Returns (agent_id, switched).
    If router is unknown, stay on current agent (likely a vague follow-up).
    """
    predicted = route_user_query(user_input)
    current = st.session_state.active_agent

    if predicted == "unknown":
        return current, False

    switched = predicted != current
    if switched and current:
        _reset_agent_session(current)
    if switched and predicted in ("budget", "anomaly"):
        _reset_agent_session(predicted)

    st.session_state.active_agent = predicted
    return predicted, switched


def get_active_agent_response(agent, message):
    st.session_state.current_input = message

    store = st.session_state.secure_store
    if agent == "budget":
        return run_budget_agent_loop(
            BUDGET_DATA_PATH, streamlit_mode=True, secure_store=store
        )
    elif agent == "anomaly":
        return run_anomaly_agent_loop(
            ANOMALY_DATA_PATH, streamlit_mode=True, secure_store=store
        )
    elif agent == "stock":
        return run_stock_agent_loop(streamlit_mode=True)
    elif agent == "portfolio":
        return run_portfolio_agent_loop(PORTFOLIO_DATA_PATH, streamlit_mode=True)
    return AgentResponse.from_text("❌ Unknown agent")

# Agent status indicator
agent_labels = {
    "budget": "🔮 Budget Forecasting Agent",
    "anomaly": "🚨 Anomaly Detection Agent",
    "stock": "📈 Stock Sentiment Agent",
    "portfolio": "📊 Portfolio Review Agent"
}
active = st.session_state.active_agent
if active:
    st.markdown(
        f"<div style='margin-top: 0.25rem; font-size: 0.9rem;'>🧠 <b>Currently talking to:</b> <code>{agent_labels.get(active, active)}</code></div>",
        unsafe_allow_html=True
    )

# Chat input field
user_input = st.chat_input("Ask about your finances — routing is automatic")

if user_input:
    user_lower = user_input.lower().strip()
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(safe_markdown(user_input))

    with st.chat_message("assistant"):
        # Optional shortcuts to clear session without sending to an agent
        if user_lower in ["exit", "quit", "back"]:
            _reset_agent_session(st.session_state.active_agent)
            st.session_state.active_agent = None
            result = AgentResponse.from_text(
                "Session cleared. Ask anything and we'll route you to the right specialist."
            )
        else:
            prev_agent = st.session_state.active_agent
            with st.spinner("Routing..."):
                agent, switched = _resolve_agent(user_input)

            if agent is None:
                result = AgentResponse.from_text(
                    "I couldn't tell whether that's about budgeting, anomalies, a stock, "
                    "or your portfolio. Try rephrasing with a bit more context."
                )
            else:
                if switched and prev_agent:
                    st.markdown(
                        f"🔁 Switched from **`{prev_agent}`** → **`{agent}`** agent..."
                    )
                elif switched or len(st.session_state.agent_conversations.get(agent, [])) == 0:
                    st.markdown(f"🔁 Routing to **`{agent}`** agent...")
                try:
                    with st.spinner("Thinking..."):
                        result = normalize_response(
                            get_active_agent_response(agent, user_input)
                        )
                except Exception as e:
                    result = AgentResponse.from_text(
                        f"❌ An error occurred while processing your request.\n\n{e}"
                    )

        if result.thinking:
            render_thinking(result.thinking)
        st.markdown(safe_markdown(result.answer))
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.answer,
            "thinking": result.thinking,
        })
