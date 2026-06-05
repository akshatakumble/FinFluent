# utils/llama3_ollama.py
import requests

OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/generate"


def ask_llama3(prompt: str) -> str:
    payload = {"model": "llama3", "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_ENDPOINT, json=payload)
    return response.json().get("response", "")
