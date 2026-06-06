"""Chain-of-thought prompting via local Ollama — reasoning and answer returned separately."""

from __future__ import annotations

import re
import requests

from utils.agent_response import AgentResponse

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3"


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
) -> AgentResponse:
    """Step 1: internal reasoning. Step 2: clean final answer for the user."""
    min_lines, max_lines = thinking_lines
    min_sent, max_sent = answer_sentences

    if response_style == "budget":
        thinking_guide = (
            "1. Key forecast amounts per spending category\n"
            "2. Which categories are largest relative to salary\n"
            "3. Whether surplus or deficit is expected\n"
            "4. What is uncertain in the forecast"
        )
        answer_guide = (
            f"Write the final answer in {min_sent}-{max_sent} sentences.\n"
            "- Use only the numbers in the evidence — do NOT recalculate totals\n"
            "- Answer the user's question directly (forecasts, savings, affordability)\n"
            "- Name specific categories and dollar amounts\n"
            "- End with one practical budgeting step\n"
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
            "- Start with a direct yes/no or clear summary when the user asks about red flags or risk\n"
            "- Quote exact dates and dollar amounts from the evidence\n"
            "- Refer to the merchant name (in quotes) separately from the statement category label\n"
            "- Do not assume the category label is correct for the merchant\n"
            "- Explain why each flagged item matters\n"
            "- End with one practical next step\n"
            "- Plain sentences only: no headings, bullets, asterisks, or repeated lines\n"
            "- Use only numbers from the evidence"
        )

    reasoning_messages = [
        {
            "role": "user",
            "content": f"""You are a {domain} specialist. Use ONLY the evidence below.

Evidence:
{evidence}

User question: {user_query}

Write internal thinking notes ({min_lines}-{max_lines} lines). Cover:
{thinking_guide}

Plain text only. No markdown headings.""",
        }
    ]
    reasoning_text = clean_user_text(ollama_chat(reasoning_messages, model=model))

    advice_messages = [
        {
            "role": "user",
            "content": f"""You are a friendly financial assistant.

Evidence:
{evidence}

Internal analysis:
{reasoning_text}

User question: {user_query}

{answer_guide}""",
        }
    ]
    advice_text = ollama_chat(advice_messages, model=model)
    answer = _parse_conclusion(advice_text)

    return AgentResponse(answer=answer, thinking=reasoning_text or None)
