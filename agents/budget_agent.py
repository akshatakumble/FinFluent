import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.tseries.offsets import MonthEnd

from utils.encryption import SecureFileStore, open_csv_path
from utils.llm_cot import BUDGET_SYSTEM_PROMPT, cot_financial_response

try:
    import streamlit as st
except ImportError:
    st = None


def _get_secure_store(streamlit_mode: bool) -> SecureFileStore | None:
    if streamlit_mode and st:
        return st.session_state.get("secure_store")
    return None


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


def _merge_scale_from_salary(df_salary: pd.DataFrame) -> int:
    """Detect multi-account merged CSVs (e.g. 5 salary deposits per month)."""
    if df_salary.empty:
        return 1
    monthly_counts = (
        df_salary.assign(Month=df_salary["Date"].dt.to_period("M"))
        .groupby("Month")
        .size()
    )
    if monthly_counts.empty:
        return 1
    typical = int(monthly_counts.mode().iloc[0])
    if typical > 1 and (monthly_counts == typical).mean() >= 0.75:
        return typical
    return 1


def _build_budget_evidence(
    transactions_path: str,
    secure_store: SecureFileStore | None,
) -> tuple[str, float | str]:
    """Run SARIMA pipeline and return evidence text for CoT (SARIMA logic unchanged)."""
    with open_csv_path(transactions_path, secure_store) as path:
        df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "Date" not in df.columns:
        raise ValueError(
            "CSV must include a 'Date' column. "
            f"Found columns: {', '.join(df.columns.astype(str))}"
        )
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df_salary = df[
        (df["Category"] == "Salary")
        & (df["Transaction Type"].str.lower() == "credit")
    ]
    latest_year = df_salary["Date"].dt.year.max()
    latest_salary_entry = df_salary[df_salary["Date"].dt.year == latest_year].sort_values(
        "Date", ascending=False
    ).head(1)
    user_salary = (
        latest_salary_entry["Amount"].values[0]
        if not latest_salary_entry.empty
        else "Unknown"
    )
    merge_scale = _merge_scale_from_salary(df_salary)

    debit_categories = {
        "Shopping",
        "Entertainment",
        "Restaurants",
        "Travel expenses",
        "Mortgage & Rent",
        "Grocery shopping",
        "Utilities",
        "Heating fuel",
    }
    df = df[df["Category"].isin(debit_categories)]
    df["Amount"] = df["Amount"].abs()
    if merge_scale > 1:
        df["Amount"] = df["Amount"] / merge_scale
    df["Month"] = (df["Date"] + MonthEnd(0)).dt.to_period("M").dt.to_timestamp()
    monthly_spending = df.groupby(["Month", "Category"])["Amount"].sum().unstack().fillna(0)
    monthly_spending.index = pd.date_range(
        start=monthly_spending.index.min(),
        periods=len(monthly_spending),
        freq="MS",
    )

    future_spending = {
        category: forecast_sarima(monthly_spending[category]).iloc[0]
        for category in monthly_spending.columns
    }

    forecast_text = "\n".join(
        f"- {cat}: ${amt:.2f}" for cat, amt in future_spending.items()
    )
    total_forecast = sum(future_spending.values())
    if isinstance(user_salary, (int, float)):
        surplus = user_salary - total_forecast
        user_info = (
            "## User Information:\n"
            f"- Monthly Salary: ${user_salary:.2f}\n"
            "- Predicted Spending for Next Month:\n"
            f"{forecast_text}\n"
            f"- Total predicted spending: ${total_forecast:.2f}\n"
            f"- Estimated monthly surplus: ${surplus:.2f}\n"
        )
    else:
        user_info = (
            "## User Information:\n"
            f"- Monthly Salary: {user_salary}\n"
            "- Predicted Spending for Next Month:\n"
            f"{forecast_text}\n"
        )

    if merge_scale > 1:
        user_info += (
            f"- Note: Statement combines ~{merge_scale} accounts; "
            "spending is scaled to a single account before forecasting.\n"
        )

    evidence = user_info
    return evidence, user_salary


def _store_budget_evidence(streamlit_mode: bool, evidence: str) -> None:
    if streamlit_mode and st:
        st.session_state.budget_evidence = evidence
    else:
        run_budget_agent_loop.cached_evidence = evidence


def _get_budget_evidence(streamlit_mode: bool) -> str | None:
    if streamlit_mode and st:
        return st.session_state.get("budget_evidence")
    return getattr(run_budget_agent_loop, "cached_evidence", None)


def run_budget_agent_loop(
    transactions_path: str,
    streamlit_mode=False,
    secure_store: SecureFileStore | None = None,
):
    if secure_store is None:
        secure_store = _get_secure_store(streamlit_mode)

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
        user_input = "Please analyze my forecast and offer suggestions."

    evidence = _get_budget_evidence(streamlit_mode)
    if not evidence:
        evidence, _ = _build_budget_evidence(transactions_path, secure_store)
        _store_budget_evidence(streamlit_mode, evidence)

    if not memory:
        result = cot_financial_response(
            evidence=evidence,
            user_query=user_input,
            domain="budget forecasting",
            response_style="budget",
            system_prompt=BUDGET_SYSTEM_PROMPT,
            answer_sentences=(3, 5),
            thinking_lines=(5, 8),
        )
        memory.extend([
            {"role": "system", "content": BUDGET_SYSTEM_PROMPT},
            {"role": "assistant", "content": result.answer},
        ])
        if streamlit_mode:
            return result
        print(f"\n💬 {result.answer}\n")

    elif streamlit_mode:
        result = cot_financial_response(
            evidence=evidence,
            user_query=user_input,
            domain="budget forecasting",
            response_style="budget",
            system_prompt=BUDGET_SYSTEM_PROMPT,
            answer_sentences=(3, 5),
            thinking_lines=(5, 8),
        )
        memory.append({"role": "user", "content": user_input})
        memory.append({"role": "assistant", "content": result.answer})
        return result

    if not streamlit_mode:
        while True:
            user_input = input("BudgetAgent> ").strip()
            if user_input.lower() in ["exit", "quit", "back"]:
                print("↩️ Returning to FinFluent main menu.\n")
                break

            follow_up = cot_financial_response(
                evidence=evidence,
                user_query=user_input,
                domain="budget forecasting",
                response_style="budget",
                system_prompt=BUDGET_SYSTEM_PROMPT,
            )
            memory.append({"role": "user", "content": user_input})
            memory.append({"role": "assistant", "content": follow_up.answer})
            print(f"\n💬 {follow_up.answer}\n")
