# app/ai_claude.py
import os, json, requests
from typing import Dict, Any

API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# ---------- helpers ----------
def _get_secret(name: str, default: str = "") -> str:
    """Read from env first, then Streamlit secrets. Adds fallback to ANTHROPIC_API_KEY if asking for CLAUDE_API_KEY."""
    val = (os.getenv(name) or "").strip()
    if not val and name == "CLAUDE_API_KEY":
        val = (os.getenv("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") or "").strip()
    if val:
        return val
    # Try Streamlit secrets (safe import)
    try:
        import streamlit as st
        v = st.secrets.get(name, default)
        if (not v) and name == "CLAUDE_API_KEY":
            v = st.secrets.get("ANTHROPIC_API_KEY", default)
        return (v or "").strip()
    except Exception:
        return default

def _safe_json_parse(text: str):
    """Try strict JSON; if that fails, try to extract the largest {...} block."""
    try:
        return json.loads(text)
    except Exception:
        if "{" in text and "}" in text:
            try:
                chunk = text[text.find("{"): text.rfind("}") + 1]
                return json.loads(chunk)
            except Exception:
                pass
        return None

def _fallback_extract_lists(text: str):
    """If the model returns bullets instead of JSON, extract a short summary + top 3 reasons/actions."""
    reasons, actions = [], []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    current = None
    for ln in lines:
        low = ln.lower()
        if "reason" in low:
            current = "reasons"; continue
        if "action" in low or "recommend" in low:
            current = "actions"; continue
        if ln.startswith(("-", "*", "•")):
            (reasons if current == "reasons" else actions).append(ln.lstrip("-*• ").strip())
    summary = lines[0] if lines else "Model returned non-JSON text."
    return (summary[:200], reasons[:3], actions[:3])

# ---------- main API ----------
def generate_recommendation(circuit: Dict[str, Any], max_tokens: int = 400) -> Dict[str, Any]:
    """
    Calls Anthropic Claude to generate a compact recommendation JSON.
    Returns a dict with keys: summary, reasons, actions, confidence, raw_text
    """
    api_key = _get_secret("CLAUDE_API_KEY")
    model = _get_secret("CLAUDE_MODEL", "claude-3-haiku-20240307")

    if not api_key:
        return {
            "summary": "Claude API key not configured.",
            "reasons": [],
            "actions": [],
            "confidence": "low",
            "raw_text": "Set CLAUDE_API_KEY or ANTHROPIC_API_KEY in Secrets or env and restart.",
        }

    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

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
        "risk_score": circuit.get("Risk Score"),
    }

    json_instruction = (
        "Respond with ONLY valid JSON (no backticks, no markdown) exactly like:\n"
        '{ "summary": "...", "reasons": ["...","...","..."], '
        '"actions": ["...","...","..."], "confidence": "low|medium|high" }'
    )

    prompt = (
        "Analyze the circuit context and KPI values below and return an executive-ready recommendation.\n\n"
        f"Context (JSON): {json.dumps(context, ensure_ascii=False)}\n\n"
        f"{json_instruction}\n"
        "Keep the total under 120 words. If data is missing, state which fields are missing in the summary."
    )

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": "You are a senior telecom network reliability engineer. Be concise, precise, and actionable.",
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            return {
                "summary": "Claude API error",
                "reasons": [],
                "actions": [],
                "confidence": "low",
                "raw_text": f"HTTP {r.status_code}: {r.text}",
            }

        data = r.json()
        text = data.get("content", [{}])[0].get("text", json.dumps(data))

        # Prefer strict JSON
        parsed = _safe_json_parse(text)
        if parsed:
            return {
                "summary": parsed.get("summary", ""),
                "reasons": (parsed.get("reasons") or [])[:3],
                "actions": (parsed.get("actions") or [])[:3],
                "confidence": parsed.get("confidence", "medium"),
                "raw_text": text,
            }

        # Fallback to heuristic parsing
        summary, reasons, actions = _fallback_extract_lists(text)
        return {
            "summary": summary,
            "reasons": reasons,
            "actions": actions,
            "confidence": "medium" if (reasons or actions) else "low",
            "raw_text": text,
        }

    except Exception as e:
        return {
            "summary": "Exception calling Claude",
            "reasons": [],
            "actions": [],
            "confidence": "low",
            "raw_text": str(e),
        }
