# FinFluent — Your AI Financial Advisor

> *Speak the language of wealth with AI.*

FinFluent is a multi-agent AI financial assistant that routes natural-language queries to specialized agents for budgeting, anomaly detection, stock sentiment, and portfolio analysis. Supports both a Streamlit web UI and a CLI.

---

## Architecture

```
User Query
    │
    ▼
Central Controller (LLaMA 3 via Ollama)
    │  routes to one of:
    ├──▶ Budget Agent       — SARIMA forecasting on transaction history
    ├──▶ Anomaly Agent      — Isolation Forest for unusual spending detection
    ├──▶ Stock Agent        — real-time stock ticker sentiment analysis
    └──▶ Portfolio Agent    — portfolio breakdown and performance insights
```

Each agent maintains its own **conversational memory** across turns, so you can ask follow-up questions within a session.

---

## Agents

| Agent | What it does | Model / Method |
|---|---|---|
| **Budget** | Forecasts future expenses from your transaction CSV | SARIMA (statsmodels) + LLaMA 3 |
| **Anomaly** | Flags outlier transactions and explains them | Isolation Forest (scikit-learn) + LLaMA 3 |
| **Stock** | Fetches live news and gives sentiment analysis for any ticker | LLaMA 3 + live news API |
| **Portfolio** | Analyzes your holdings and surfaces insights | LLaMA 3 |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=flat-square&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)

- **LLM**: LLaMA 3 (local inference via [Ollama](https://ollama.com))
- **Routing**: LLM-based intent classification in `central_controller.py`
- **Forecasting**: SARIMA via `statsmodels`
- **Anomaly detection**: Isolation Forest via `scikit-learn`
- **UI**: Streamlit (wide layout, multi-page chat)
- **API layer**: FastAPI + Uvicorn

---

## Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- LLaMA 3 pulled: `ollama pull llama3`

### Install

```bash
git clone https://github.com/akshatakumble/FinFluent.git
cd FinFluent
pip install -r requirements.txt
```

### Run (Streamlit UI)

```bash
streamlit run finfluent_app.py
```

### Run (CLI)

```bash
python cli/main.py
```

---

## Input Format

Budget and anomaly agents expect a CSV with your transaction history. A sample is included at `data/user_1.csv`.

| Column | Description |
|---|---|
| `date` | Transaction date |
| `amount` | Transaction amount |
| `category` | Spending category |
| `description` | Transaction description |

---

## Project Structure

```
FinFluent/
├── agents/
│   ├── budget_agent.py       # SARIMA forecasting + LLM Q&A
│   ├── anomaly_agent.py      # Isolation Forest + LLM explanation
│   ├── stock_agent.py        # Live ticker sentiment
│   └── portfolio_agent.py    # Portfolio analysis
├── controller/
│   └── central_controller.py # LLaMA 3 query router
├── utils/                    # Ollama API helpers
├── cli/                      # CLI entry point
├── data/                     # Sample transaction CSVs
└── finfluent_app.py          # Streamlit app entry point
```
