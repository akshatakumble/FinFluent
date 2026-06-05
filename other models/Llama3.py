import requests

url = "http://localhost:11434/api/chat"


def llama3(system_prompt, conversation_history):
    data = {
        "model": "llama3",
        "messages": conversation_history,
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, json=data)
    return response.json()["message"]["content"]


# Initialize conversation history
conversation_history = [
    {
        "role": "system",
        "content": "You are an AI Financial Advisor assistant providing accurate and concise responses.",
    }
]

print("AI Financial Advisor Chatbot. Type 'exit' to end the conversation.")

while True:
    user_prompt = input("You: ")
    if user_prompt.lower() == "exit":
        print("Goodbye!")
        break

    # Add user message to history
    conversation_history.append({"role": "user", "content": user_prompt})

    # Get AI response
    response = llama3("AI Financial Advisor Assistant", conversation_history)

    # Add AI response to history
    conversation_history.append({"role": "assistant", "content": response})

    print(f"AI: {response}")
