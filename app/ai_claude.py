import os, json, requests
from typing import Dict, Any, Optional

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "").strip()
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20240229").strip()

API_URL = "https://api.anthropic.com/v1/messages"
API_HEADERS = {
    "x-api-key": CLAUDE_API_KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
}

SYSTEM_PROMPT = (
    "You are a senior telecom network reliability engineer. "
    "Be concise, precise, and actionable. "
    "When fields seem missing, state what's missing instead of guessing."
)

def _safe_json_parse(text: str):
    try:
        return json.loads(text)
    except Exception:
        if "{" in text and "}" in text:
            try:
                chunk = text[text.find('{'): text.rfind('}')+1]
                return json.loads(chunk)
            except Exception:
                return None
        return None

def generate_recommendation(circuit: Dict[str, Any], max_tokens: int = 400) -> Dict[str, Any]:
    if not CLAUDE_API_KEY:
        return {"summary":"Claude API key not configured.", "reasons":[], "actions":[], "confidence":"low", "raw_text":"Set CLAUDE_API_KEY"}

    context = {
        "circuit_id": circuit.get("circuit_id"),
        "region": circuit.get("region"),
        "product": circuit.get("product"),
        "bandwidth_mbps": circuit.get("bandwidth_mbps"),
        "vendor": circuit.get("vendor"),
        "model": circuit.get("model"),
        "sla_tier": circuit.get("sla_tier"),
        "latest_kpis": {
            "utilization_pct": circuit.get("utilization_pct"),
            "latency_ms": circuit.get("latency_ms"),
            "jitter_ms": circuit.get("jitter_ms"),
            "pkt_loss_pct": circuit.get("pkt_loss_pct"),
            "crc_err_rate": circuit.get("crc_err_rate"),
        },
        "risk_score": circuit.get("Risk Score")
    }

    prompt = (
        "Analyze the following circuit context and KPI values, then return an executive-ready recommendation.\n\n"
        f"Context (JSON): {json.dumps(context, ensure_ascii=False)}\n\n"
        "Return ONLY valid JSON with fields: "
        '{"summary": "...", "reasons":["...","...","..."], "actions":["...","...","..."], "confidence":"low|medium|high"} '
        "Keep the total under 120 words."
    )

    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        r = requests.post(API_URL, headers=API_HEADERS, json=payload, timeout=60)
        if r.status_code != 200:
            return {"summary":"Claude API error", "reasons":[], "actions":[], "confidence":"low", "raw_text":r.text}
        data = r.json()
        text = data.get("content", [{}])[0].get("text", json.dumps(data))
        parsed = _safe_json_parse(text) or {}
        return {
            "summary": parsed.get("summary", text),
            "reasons": parsed.get("reasons", []),
            "actions": parsed.get("actions", []),
            "confidence": parsed.get("confidence", "medium"),
            "raw_text": text
        }
    except Exception as e:
        return {"summary":"Exception calling Claude", "reasons":[], "actions":[], "confidence":"low", "raw_text":str(e)}
