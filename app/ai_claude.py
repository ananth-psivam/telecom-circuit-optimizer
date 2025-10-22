# app/ai_claude.py
import os, json, re, requests
from typing import Dict, Any

API_URL = "https://api.anthropic.com/v1/messages"

def _get_env(name: str, default: str = "") -> str:
    val = os.getenv(name, "").strip()
    if val:
        return val
    try:
        import streamlit as st
        v = st.secrets.get(name, default)
        return (v or "").strip()
    except Exception:
        return default

def _safe_json_parse(text: str):
    try:
        return json.loads(text)
    except Exception:
        if "{" in text and "}" in text:
            try:
                chunk = text[text.find("{"): text.rfind("}")+1]
                return json.loads(chunk)
            except Exception:
                return None
        return None

def _fallback_extract_lists(text: str):
    """
    If model returns plain text or markdown, try to build reasons/actions lists.
    """
    reasons, actions = [], []
    # Pick lines that look like bullets
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # Heuristic: anything under a "reason" heading or first 5 bullets → reasons
    # anything under an "action" heading or next bullets → actions
    current = None
    for ln in lines:
        low = ln.lower()
        if "reason" in low:
            current = "reasons"; continue
        if "action" in low or "recommend" in low:
            current = "actions"; continue
        if ln.startswith(("-", "*", "•")):
            (reasons if current == "reasons" else actions).append(ln.lstrip("-*• ").strip())

    # Trim length
    return reasons[:3], actions[:3]

def generate_recommendation(circuit: Dict[str, Any], max_tokens: int = 400) -> Dict[str, Any]:
    CLAUDE_API_KEY = _get_env("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")
    CLAUDE_MODEL   = _get_env("CLAUDE_MODEL", "claude-3-haiku-20240307")
    if not CLAUDE_API_KEY:
        return {
            "summary": "Claude API key not configured.",
            "reasons": [],
            "actions": [],
            "confidence": "low",
            "raw_text": "Set CLAUDE_API_KEY in Streamlit Secrets or .env and restart."
        }

    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
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

    # Strong JSON instruction (no markdown)
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
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "system": (
            "You are a senior telecom network reliability engineer. "
            "Be concise, precise, and actionable."
        ),
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            return {
                "summary": "Claude API error",
                "reasons": [],
                "actions": [],
                "confidence": "low",
                "raw_text": f"HTTP {r.status_code}: {r.text}"
            }

        data = r.json()
        text = data.get("content", [{}])[0].get("text", json.dumps(data))

        parsed = _safe_json_parse(text)
        if parsed:
            return {
                "summary": parsed.get("summary", ""),
                "reasons": parsed.get("reasons", [])[:3],
                "actions": parsed.get("actions", [])[:3],
                "confidence": parsed.get("confidence", "medium"),
                "raw_text": text
            }

        # Fallback: try to salvage from markdown/plain text
        reasons, actions = _fallback_extract_lists(text)
        summary = text.strip().split("\n")[0][:200]
        return {
            "summary": summary or "Model returned non-JSON text.",
            "reasons": reasons,
            "actions": actions,
            "confidence": "medium" if (reasons or actions) else "low",
            "raw_text": text
        }

    except Exception as e:
        return {
            "summary": "Exception calling Claude",
            "reasons": [],
            "actions": [],
            "confidence": "low",
            "raw_text": str(e)
        }
