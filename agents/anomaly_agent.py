from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd

from utils.agent_response import AgentResponse
from utils.encryption import SecureFileStore, open_csv_path
from utils.llm_cot import ANOMALY_SYSTEM_PROMPT, cot_financial_response

try:
    import streamlit as st
except ImportError:
    st = None


def _get_secure_store(streamlit_mode: bool) -> SecureFileStore | None:
    if streamlit_mode and st:
        return st.session_state.get("secure_store")
    return None


def _is_material_outlier(amount: float, debit_amounts: pd.Series) -> bool:
    """Keep Isolation Forest hits that are clearly above normal spending."""
    if debit_amounts.empty:
        return amount >= 2000.0
    p95 = float(debit_amounts.quantile(0.95))
    p99 = float(debit_amounts.quantile(0.99))
    median = float(debit_amounts.median())
    threshold = max(1.75 * p99, 4.0 * p95, 2000.0)
    if amount >= threshold:
        return True
    # Catch likely data-entry errors (e.g. 192222 vs 192.22)
    if median > 0 and amount / median >= 100:
        return True
    return False


def _material_outlier_threshold(debit_amounts: pd.Series) -> float:
    if debit_amounts.empty:
        return 2000.0
    p95 = float(debit_amounts.quantile(0.95))
    p99 = float(debit_amounts.quantile(0.99))
    return max(1.75 * p99, 4.0 * p95, 2000.0)


def _build_anomaly_evidence(transactions_path: str, secure_store: SecureFileStore | None) -> str | None:
    with open_csv_path(transactions_path, secure_store) as path:
        df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df_debit = df[df["Transaction Type"].str.lower() == "debit"].copy()

    if df_debit.empty or "Amount" not in df_debit.columns:
        return None

    df_debit["Amount"] = df_debit["Amount"].abs()
    threshold = _material_outlier_threshold(df_debit["Amount"])

    scaler = StandardScaler()
    df_debit["Amount_scaled"] = scaler.fit_transform(df_debit[["Amount"]])

    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    df_debit["outlier_flag"] = model.fit_predict(df_debit[["Amount_scaled"]])
    df_debit["is_outlier"] = df_debit["outlier_flag"] == -1

    outliers = df_debit[df_debit["is_outlier"]]
    outliers = outliers[outliers["Category"] != "Mortgage & Rent"]
    outliers = outliers[
        outliers["Amount"].apply(lambda amt: _is_material_outlier(amt, df_debit["Amount"]))
    ]
    outliers = outliers.sort_values(by="Amount", ascending=False)

    if outliers.empty:
        return ""

    lines = []
    for _, row in outliers.head(5).iterrows():
        merchant = str(row.get("Description", "Unknown")).strip()
        category = str(row.get("Category", "Unknown")).strip()
        lines.append(
            f"- {row['Date']}: ${row['Amount']:.2f} at merchant \"{merchant}\" "
            f"(statement category label: {category})"
        )
    return (
        "## Detected Anomalies:\n"
        + "\n".join(lines)
        + "\n\nNote: Detected by Isolation Forest on transaction amount. "
        f"Material outlier threshold ~${threshold:,.0f}. "
        "Category labels come from the bank statement and may not match the merchant."
    )


def _store_evidence(streamlit_mode: bool, evidence: str) -> None:
    if streamlit_mode and st:
        st.session_state.anomaly_evidence = evidence
    else:
        run_anomaly_agent_loop.cached_evidence = evidence


def _get_stored_evidence(streamlit_mode: bool) -> str | None:
    if streamlit_mode and st:
        return st.session_state.get("anomaly_evidence")
    return getattr(run_anomaly_agent_loop, "cached_evidence", None)


def run_anomaly_agent_loop(
    transactions_path: str,
    streamlit_mode=False,
    secure_store: SecureFileStore | None = None,
):
    if secure_store is None:
        secure_store = _get_secure_store(streamlit_mode)

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

    evidence = _get_stored_evidence(streamlit_mode)
    if not evidence:
        evidence = _build_anomaly_evidence(transactions_path, secure_store)

        if evidence is None:
            msg = "❌ No debit transactions found or missing 'Amount' column."
            if streamlit_mode:
                return AgentResponse.from_text(msg)
            print(msg)
            return

        if evidence == "":
            msg = "✅ No major spending anomalies detected in your statement. You're all good!"
            if streamlit_mode:
                return AgentResponse.from_text(msg)
            print(msg)
            return

        _store_evidence(streamlit_mode, evidence)

    result = cot_financial_response(
        evidence=evidence,
        user_query=user_input,
        domain="spending anomaly detection",
        response_style="anomaly",
        system_prompt=ANOMALY_SYSTEM_PROMPT,
        answer_sentences=(4, 6),
        thinking_lines=(5, 8),
    )

    if not memory:
        memory.extend([
            {"role": "system", "content": ANOMALY_SYSTEM_PROMPT},
            {"role": "assistant", "content": result.answer},
        ])
    else:
        memory.append({"role": "user", "content": user_input})
        memory.append({"role": "assistant", "content": result.answer})

    if streamlit_mode:
        return result
    print(f"\n💬 {result.answer}\n")

    if not streamlit_mode:
        while True:
            user_input = input("AnomalyAgent> ").strip()
            if user_input.lower() in ["exit", "quit", "back"]:
                print("↩️ Returning to FinFluent main menu.\n")
                break

            follow_up = cot_financial_response(
                evidence=evidence,
                user_query=user_input,
                domain="spending anomaly detection",
                response_style="anomaly",
                system_prompt=ANOMALY_SYSTEM_PROMPT,
                answer_sentences=(4, 6),
                thinking_lines=(5, 8),
            )
            memory.append({"role": "user", "content": user_input})
            memory.append({"role": "assistant", "content": follow_up.answer})
            print(f"\n💬 {follow_up.answer}\n")
