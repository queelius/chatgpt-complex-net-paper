import numpy as np
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration for the Ollama model API
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "nomic-embed-text")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

def get_llm_embedding(text):
    """
    Calls the Ollama model API to get an embedding for the input text.
    Assumes the API endpoint is available at {OLLAMA_HOST}/embed and that it expects
    a JSON payload with "model" and "text" keys, returning a JSON with an "embedding" key.
    """
    endpoint = f"{OLLAMA_HOST}/api/embeddings"
    payload = {
        "model": MODEL_NAME,
        "prompt": text
    }
    response = requests.post(endpoint, json=payload, timeout=OLLAMA_TIMEOUT)
    response.raise_for_status()  # Raise an error for bad status codes.
    data = response.json()
    embedding = data.get("embedding")
    if embedding is None:
        raise ValueError("No 'embedding' key found in the response.")
    return np.array(embedding)

