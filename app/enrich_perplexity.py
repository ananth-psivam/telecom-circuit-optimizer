# app/enrich_perplexity.py
import requests

def get_context_hint(vendor, model, bandwidth_mbps, region):
    """Fetch short external hint using Perplexity API."""
    key = "YOUR_PERPLEXITY_API_KEY"
    if not key:
        return None
    q = f"telecom optimization {vendor} {model} {bandwidth_mbps} Mbps region {region}"
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "sonar-medium-online",
        "messages": [{"role": "user", "content": q}],
        "max_tokens": 120
    }
    r = requests.post(url, headers=headers, json=data, timeout=30)
    if r.status_code == 200:
        text = r.json()["choices"][0]["message"]["content"]
        return text.strip()
    return None
