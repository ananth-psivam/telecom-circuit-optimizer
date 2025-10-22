# enrich_perplexity.py — optional lookup helper
# If Perplexity API is not configured, this will safely return None.

import os
import requests

def get_context_hint(vendor: str = None, model: str = None,
                     bandwidth: float = None, region: str = None) -> str:
    """
    Optional context enrichment via Perplexity AI API.
    Returns a single-line hint string, or None if not configured.
    """
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return None

    query_parts = []
    if vendor: query_parts.append(vendor)
    if model: query_parts.append(model)
    if bandwidth: query_parts.append(f"{bandwidth} Mbps")
    if region: query_parts.append(region)
    query = " ".join(query_parts) + " telecom network reliability tips"

    try:
        resp = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "sonar-small-chat",
                "messages": [
                    {"role": "system", "content": "Return one concise context hint."},
                    {"role": "user", "content": query}
                ],
                "max_tokens": 60
            },
            timeout=30
        )
        if resp.status_code == 200:
            text = resp.json()["choices"][0]["message"]["content"]
            return text.strip().split("\n")[0][:200]
    except Exception:
        pass

    return None
