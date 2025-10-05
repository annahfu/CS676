
"""
Adapter for the /search_and_score endpoint.
Usage in your chatbot backend:
    from templates.search_adapter import search_and_score_for_chat
    results = search_and_score_for_chat("climate change IPCC report", k=5)
    # results => list of dicts with rank, title, url, snippet, score, label, explanation
"""
import os
import requests
from typing import List, Dict, Any

CRED_API = os.getenv("CRED_API_URL", "http://localhost:7860")

def search_and_score_for_chat(query: str, k: int = 5) -> List[Dict[str, Any]]:
    payload = {"query": query, "k": int(k)}
    r = requests.post(f"{CRED_API}/search_and_score", json=payload, timeout=12)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])
    # Minimal normalization for the chat UI
    for item in results:
        # Integer rounding for tidy badge display
        if item.get("score") is not None:
            item["score"] = int(round(item["score"]))
    return results
