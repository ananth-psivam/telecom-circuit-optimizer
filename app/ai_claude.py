"""
ai_claude.py
Thin wrapper for Anthropic Claude messages API.
- generate_recommendation(circuit: dict) -> dict
  Returns structured recommendation with keys: summary, reasons, actions, confidence, raw_text
Requires env var: CLAUDE_API_KEY
"""

import os
import json
import requests
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

JSON_INSTRUCTION = (
    "Return ONLY valid JSON with the following fields:\n"
    "{\n"
    '  "summary": "<one-line summary>",\n'
    '  "reasons": ["<reason1>","<reason2>","<reason3>"],\n'
    '  "actions": ["<action1>","<action2>","<action3>"],\n'
    '  "confidence": "low|medium|high"\n'
    "}\n"
    "Do not include markdown fences or extra commentary."
)

def _safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        # Try to salvage content between braces
        if "{" in text and "}" in text:
            try:
                chunk = text[text.find("{"): text.rfind("}")+1]
                return json.loads(chunk)
            except Exception:
                return None
        return None

def generate_recommendation(circuit: Dict[str, Any], max_tokens: int = 400) -> Dict[str, Any]:
    """
    Calls Claude to generate a structured recommendation.
    circuit: dict with keys like circuit_id, region, product, bandwidth_mbps, utilization_pct, jitter_ms, pkt_loss_pct, latency_ms, crc_err_rate, sla_tier, Risk Score, etc.
    """
    if not CLAUDE_API_KEY:
        return {
            "summary": "Claude API key not configured.",
            "reasons": [],
            "actions": [],
            "confidence": "low",
            "raw_text": "Set CLAUDE_API_KEY environment variable."
        }

    user_context = {
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

    prompt = (
        "Analyze the following circuit context and KPI values, then return an executive-ready recommendation.\n\n"
        f"Context (JSON): {json.dumps(user_context, ensure_ascii=False)}\n\n"
        f"{JSON_INSTRUCTION}\n"
        "Keep the total text under 120 words."
    )

    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        resp = requests.post(API_URL, headers=API_HEADERS, json=payload, timeout=60)
        if resp.status_code != 200:
            return {
                "summary": "Claude API error.",
                "reasons": [],
                "actions": [],
                "confidence": "low",
                "raw_text": resp.text
            }
        data = resp.json()
        # Expected structure: {"content": [{"type":"text","text":"...json..."}], ...}
        text = ""
        try:
            text = data["content"][0]["text"]
        except Exception:
            text = json.dumps(data)

        parsed = _safe_json_parse(text) or {}
        return {
            "summary": parsed.get("summary") or text,
            "reasons": parsed.get("reasons", []),
            "actions": parsed.get("actions", []),
            "confidence": parsed.get("confidence", "medium"),
            "raw_text": text
        }
    except Exception as e:
        return {
            "summary": "Exception calling Claude.",
            "reasons": [],
            "actions": [],
            "confidence": "low",
            "raw_text": str(e)
        }
