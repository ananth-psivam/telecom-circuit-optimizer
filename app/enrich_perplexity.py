import os, requests
from typing import Optional

PPLX_API_KEY = os.getenv("PERPLEXITY_API_KEY", "").strip()
PPLX_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar").strip()
PPLX_URL = "https://api.perplexity.ai/chat/completions"

HEADERS = {"Authorization": f"Bearer {PPLX_API_KEY}", "Content-Type": "application/json"}

SYSTEM = (
    "You are a concise telecom assistant. Provide at most 2 sentences. "
    "Avoid brand marketing language. If unsure, say you are unsure."
)

def get_context_hint(vendor: Optional[str], model: Optional[str], bandwidth: Optional[int] = None, region: Optional[str] = None) -> Optional[str]:
    if not PPLX_API_KEY:
        return None
    parts = []
    if vendor: parts.append(str(vendor))
    if model: parts.append(str(model))
    if bandwidth: parts.append(f"{bandwidth} Mbps")
    base = "Common causes and quick checks for elevated jitter/CRC on "
    region_part = f" in {region}" if region else ""
    user = base + " ".join(parts) + region_part + ". Provide 1–2 practical checks."

    payload = {
        "model": PPLX_MODEL,
        "messages": [{"role":"system","content":SYSTEM},{"role":"user","content":user}],
        "max_tokens": 120,
        "temperature": 0.2,
        "top_p": 0.9
    }
    try:
        r = requests.post(PPLX_URL, headers=HEADERS, json=payload, timeout=45)
        if r.status_code != 200:
            return None
        data = r.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or None
    except Exception:
        return None
