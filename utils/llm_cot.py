"""Chain-of-thought prompting via local Ollama — reasoning and answer returned separately."""

from __future__ import annotations

import re
import requests

from utils.agent_response import AgentResponse

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3"

BUDGET_SYSTEM_PROMPT = """
You are an AI-powered Financial Advisor. Your job is to provide accurate, data-driven financial guidance.

The evidence below includes monthly salary, predicted spending for next month (SARIMA forecasts), and per-category amounts.

## Instructions:
1. Be specific and data-driven:
   - When answering questions, refer to the user's salary and predicted spending.
   - Use numbers, percentages, and trends instead of generic advice.
2. Recommend savings strategies:
   - If spending in a category is higher than 30% of salary, suggest ways to reduce it.
   - Recommend savings and investment strategies based on spending habits.
3. Warn about high-risk categories:
   - If spending in a category has increased significantly, explain why it might be a problem.
   - Identify risky spending patterns.
4. Use only numbers from the evidence — do NOT recalculate totals.
5. Use plain text. No bold or italics or any special characters.
""".strip()

ANOMALY_SYSTEM_PROMPT = """
You are a smart financial assistant. The evidence below lists unusual debit transactions detected by an Isolation Forest algorithm.

## Instructions:
1. Summarize the potential concerns in a friendly tone.
2. Mention if these seem risky or need user attention.
3. Suggest follow-up steps or questions to ask the user.
4. Quote exact dates and dollar amounts from the evidence.
5. Refer to the merchant name separately from the statement category label — do not assume the category label is correct.
6. Use only numbers from the evidence.
7. Use plain text. No bold or italics or any special characters.
""".strip()


def ollama_chat(messages: list[dict], model: str = DEFAULT_MODEL) -> str:
    response = requests.post(
        OLLAMA_CHAT_URL,
        json={"model": model, "messages": messages, "stream": False},
        headers={"Content-Type": "application/json"},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def clean_user_text(text: str) -> str:
    text = text.replace("\\", "")
    text = re.sub(r"\*+", "", text)
    text = re.sub(
        r"^(REASONING|CONCLUSION|RECOMMENDATION|Chain-of-thought analysis)\s*:?\s*",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_conclusion(text: str) -> str:
    match = re.search(r"CONCLUSION:\s*(.*)$", text, re.DOTALL | re.IGNORECASE)
    if match:
        return clean_user_text(match.group(1))
    return clean_user_text(text)


def cot_financial_response(
    evidence: str,
    user_query: str,
    domain: str,
    model: str = DEFAULT_MODEL,
    answer_sentences: tuple[int, int] = (3, 5),
    thinking_lines: tuple[int, int] = (5, 8),
    response_style: str = "anomaly",
    system_prompt: str | None = None,
) -> AgentResponse:
    """Step 1: internal reasoning. Step 2: clean final answer for the user."""
    min_lines, max_lines = thinking_lines
    min_sent, max_sent = answer_sentences

    if system_prompt is None:
        system_prompt = (
            BUDGET_SYSTEM_PROMPT if response_style == "budget" else ANOMALY_SYSTEM_PROMPT
        )

    if response_style == "budget":
        thinking_guide = (
            "1. Key forecast amounts per spending category\n"
            "2. Which categories are largest relative to salary (note if any exceed 30%)\n"
            "3. Whether surplus or deficit is expected\n"
            "4. Risky spending patterns or categories that need attention\n"
            "5. What is uncertain in the forecast"
        )
        answer_guide = (
            f"Write the final answer in {min_sent}-{max_sent} sentences.\n"
            "- Follow the system instructions for data-driven, specific advice\n"
            "- Answer the user's question directly (forecasts, savings, affordability)\n"
            "- Name specific categories and dollar amounts with percentages vs salary when relevant\n"
            "- Flag categories above 30% of salary or other risky patterns\n"
            "- End with one practical savings or budgeting step\n"
            "- Plain sentences only: no headings, bullets, asterisks, or repeated lines"
        )
    else:
        thinking_guide = (
            "1. Key numbers and flagged items from the evidence\n"
            "2. Why each item may be risky or unusual\n"
            "3. Patterns across categories or dates\n"
            "4. What context is missing or uncertain"
        )
        answer_guide = (
            f"Write the final answer in {min_sent}-{max_sent} sentences.\n"
            "- Follow the system instructions for tone and follow-up suggestions\n"
            "- Start with a direct yes/no or clear summary when the user asks about red flags or risk\n"
            "- Quote exact dates and dollar amounts from the evidence\n"
            "- Refer to the merchant name (in quotes) separately from the statement category label\n"
            "- Explain why each flagged item matters and suggest follow-up steps\n"
            "- Plain sentences only: no headings, bullets, asterisks, or repeated lines"
        )

    reasoning_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"""Evidence:
{evidence}

User question: {user_query}

Write internal thinking notes ({min_lines}-{max_lines} lines). Cover:
{thinking_guide}

Plain text only. No markdown headings.""",
        },
    ]
    reasoning_text = clean_user_text(ollama_chat(reasoning_messages, model=model))

    advice_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"""Evidence:
{evidence}

Internal analysis:
{reasoning_text}

User question: {user_query}

{answer_guide}""",
        },
    ]
    advice_text = ollama_chat(advice_messages, model=model)
    answer = _parse_conclusion(advice_text)

    return AgentResponse(answer=answer, thinking=reasoning_text or None)
