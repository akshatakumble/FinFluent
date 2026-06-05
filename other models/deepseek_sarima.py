import requests
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.tseries.offsets import MonthEnd

# 1. Load the CSV file
df = pd.read_csv("C:\\Users\\aksha\\Downloads\\FinFluent-main\\FinFluent-main\\sorted_transactions.csv", parse_dates=["Date"])

# 2. Filter for debit categories
debit_categories = {
    "Shopping", "Entertainment", "Bills", "Restaurants", "Travel", "Mortgage & Rent",
}

df = df[df["Category"].isin(debit_categories)]
df["Amount"] = df["Amount"].abs()
df["Month"] = (df["Date"] + MonthEnd(0)).dt.to_period("M").dt.to_timestamp()
monthly_spending = df.groupby(["Month", "Category"])["Amount"].sum().unstack().fillna(0)

monthly_spending.index = pd.date_range(
    start=monthly_spending.index.min(), periods=len(monthly_spending), freq="MS"
)

# 3. Function to train SARIMA and forecast next month
def forecast_sarima(data, steps=1):
    model = SARIMAX(
        data, order=(3, 0, 0), seasonal_order=(1, 0, 1, 12),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    model_fit = model.fit(disp=False)
    return model_fit.forecast(steps=steps)

# 4. Generate forecast
future_spending = {
    category: forecast_sarima(monthly_spending[category]).iloc[0]
    for category in monthly_spending.columns
}

# 5. Generate system prompt with forecasted spending
forecast_text = ""

# Calculate total predicted spend
total_predicted_spending = 0
category_details = []
for category, amount in future_spending.items():
    percent = (amount / 10000) * 100  # based on $10,000 monthly salary
    category_details.append((category, amount, percent))
    total_predicted_spending += amount
    forecast_text += f"- {category}: ${amount:.2f} ({percent:.1f}% of salary)\n"

leftover = 10000 - total_predicted_spending

system_prompt = f"""
You are an AI-powered Financial Advisor. Your job is to provide **accurate, data-driven financial guidance**.

## User Information:
- **Monthly Salary:** $10,000  
- **Predicted Spending for Next Month:**
{forecast_text}- **Total Predicted Spending:** ${total_predicted_spending:.2f}
- **Estimated Remaining:** ${leftover:.2f}

## Instructions:
1. **Be Specific & Data-Driven:**  
   - When answering questions, refer to the user’s salary and predicted spending.  
   - Use **numbers, percentages, and trends** instead of generic advice.  

2. **Provide Cost-Saving Strategies:**  
   - If spending in a category is **higher than 30% of salary**, suggest ways to reduce it.  
   - Recommend savings and investment strategies based on spending habits.  

3. **Warn About Potential Issues:**  
   - If spending in a category has **increased significantly**, explain why it might be a problem.  
   - Identify risky spending patterns.

4. **Investment Suggestions:**
   - If the user has more than $500 in leftover salary, provide investment options based on current market practices.
   - Match options to risk appetite: low (bonds, savings), medium (ETFs, mutual funds), high (stocks, crypto).
"""

# 6. Chat function using Ollama's local DeepSeek model
def ollama_chat(conversation_history):
    url = "http://localhost:11434/api/chat"
    data = {
        "model": "deepseek-r1",
        "messages": conversation_history,
        "stream": False
    }

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()["message"]["content"]
    except requests.exceptions.RequestException as e:
        print("Ollama API Error:", e)
        return "Sorry, I couldn't process your request due to a server issue."

# 7. Chat loop
conversation_history = [{"role": "system", "content": system_prompt}]

print("Welcome to your AI financial advisor. Type 'exit' to end the conversation.")

while True:
    user_prompt = input("You: ")
    if user_prompt.lower() == "exit":
        print("Goodbye!")
        break

    conversation_history.append({"role": "user", "content": user_prompt})
    response = ollama_chat(conversation_history)
    conversation_history.append({"role": "assistant", "content": response})
    print(f"FinFluent: {response}")
