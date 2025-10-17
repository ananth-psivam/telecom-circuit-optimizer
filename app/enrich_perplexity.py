"""
enrich_perplexity.py
Optional enrichment via Perplexity's API to add real-world hints.
- get_context_hint(vendor, model, bandwidth=None, region=None) -> str | None
Requires env var: PERPLEXITY_API_KEY
Note: Keep usage light; this is for short hints (1–2 sentences).
"""

import os
import requests
from typing import Optional

PPLX_API_KEY = os.getenv("PERPLEXITY_API_KEY", "").strip()
PPLX_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar").strip()  # 'sonar' or 'sonar-small-online'
PPLX_URL = "https://api.perplexity.ai/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {PPLX_API_KEY}",
    "Content-Type": "application/json"
}

SYSTEM = (
    "You are a concise telecom assistant. "
    "Provide at most 2 short sentences. "
    "Avoid brand marketing language. "
    "If unsure, say you are unsure."
)

def get_context_hint(vendor: Optional[str], model: Optional[str], bandwidth: Optional[int] = None, region: Optional[str] = None) -> Optional[str]:
    if not PPLX_API_KEY:
        return None

    q_parts = []
    if vendor: q_parts.append(f"{vendor}")
    if model: q_parts.append(f"{model}")
    if bandwidth: q_parts.append(f"{bandwidth} Mbps")
    base = "Common causes and quick checks for elevated jitter/CRC on "
    region_part = f" in {region}" if region else ""
    user = base + " ".join(q_parts) + region_part + ". Provide 1–2 practical checks."

    payload = {
        "model": PPLX_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}
        ],
        "max_tokens": 120,
        "temperature": 0.2,
        "top_p": 0.9
    }

    try:
        resp = requests.post(PPLX_URL, headers=HEADERS, json=payload, timeout=45)
        if resp.status_code != 200:
            return None
        data = resp.json()
        # Expected shape similar to OpenAI; adapt if API changes
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content.strip() if content else None
    except Exception:
        return None
