import requests

def ask_ollama(messages, model="llama3"):
    context_text = "\\n".join([f"{m['author']}: {m['content']}" for m in messages])
    prompt = f"""Based on the following messages, answer the question:

{context_text}

Question: What happened with the login issue?"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]
