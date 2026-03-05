import logging
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

logger = logging.getLogger(__name__)


def get_llm_embedding(text):
    """
    Calls the Ollama model API to get an embedding for the input text.
    If the text exceeds the model's context length, truncates and retries.
    """
    endpoint = f"{OLLAMA_HOST}/api/embeddings"
    payload = {
        "model": MODEL_NAME,
        "prompt": text
    }
    response = requests.post(endpoint, json=payload, timeout=OLLAMA_TIMEOUT)

    # Handle context length exceeded by truncating
    if response.status_code == 500:
        error_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        if "context length" in error_data.get("error", ""):
            # Truncate to half and retry (binary search would be slower than just halving)
            words = text.split()
            truncated = " ".join(words[:len(words) // 2])
            logger.warning(f"Context length exceeded ({len(words)} words), truncating to {len(words)//2}")
            payload["prompt"] = truncated
            response = requests.post(endpoint, json=payload, timeout=OLLAMA_TIMEOUT)

    response.raise_for_status()
    data = response.json()
    embedding = data.get("embedding")
    if embedding is None:
        raise ValueError("No 'embedding' key found in the response.")
    return np.array(embedding)

