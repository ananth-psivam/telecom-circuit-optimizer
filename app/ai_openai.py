# app/ai_openai.py
import os, json, requests
from typing import Dict, Any

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"  # good/cost-effective

# ---------- robust secret fetch ----------
def _get_secret_any(*names, default: str = "") -> str:
    # 1) env
    for n in names:
        v = (os.getenv(n) or "").strip()
        if v:
            return v
    # 2) streamlit secrets
    try:
        import streamlit as st  # lazy import is ok
        for n in names:
            v = (st.secrets.get(n, "") or "").strip()
            if v:
                return v
    except Exception:
        pass
    return default

# ---------- utilities ----------
def _safe_json_parse(text: str):
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

def _fallback_lists(text: str):
    reasons, actions = [], []
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
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

def _heuristic_reco(c: Dict[str, Any]) -> Dict[str, Any]:
    def f(x):
        try: return float(x or 0)
        except: return 0.0
    util, jitter, loss, lat, crc, bw = map(f, [
        c.get("utilization_pct"), c.get("jitter_ms"), c.get("pkt_loss_pct"),
        c.get("latency_ms"), c.get("crc_err_rate"), c.get("bandwidth_mbps")
    ])
    reasons, actions = [], []
    if util > 80:
        reasons.append(f"High utilization {util:.0f}%")
        actions.append("Apply QoS/traffic shaping; plan capacity aug")
    if jitter > 15:
        reasons.append(f"Elevated jitter {jitter:.1f} ms")
        actions.append("Check LLQ/queuing; inspect buffer drops on PE")
    if loss > 0.5:
        reasons.append(f"Packet loss {loss:.2f}%")
        actions.append("Check optics/interfaces; try alternate path")
    if lat > 60:
        reasons.append(f"High latency {lat:.0f} ms")
        actions.append("Reroute to lower RTT; verify MPLS TE constraints")
    if crc > 100:
        reasons.append(f"CRC errors {crc:.0f}")
        actions.append("Replace SFP/patch; clean fiber; loopback test")
    if bw >= 500 and util < 30:
        reasons.append("Under-utilized high-bandwidth circuit")
        actions.append("Right-size plan or consolidate redundant links")
    if not reasons:
        reasons = ["No clear impairment from KPIs"]
        actions = ["Continue monitoring; tighten 24h alerts"]
    return {
        "summary": f"Heuristic plan for {c.get('circuit_id','circuit')}",
        "reasons": reasons[:3],
        "actions": actions[:3],
        "confidence": "low",
        "raw_text": "local_fallback"
    }

# ---------- main entry ----------
def generate_recommendation(circuit: Dict[str, Any], max_tokens: int = 300) -> Dict[str, Any]:
    """
    Returns dict: {summary, reasons[], actions[], confidence, raw_text}
    Never returns None or Ellipsis.
    """
    try:
        api_key = _get_secret_any("OPENAI_API_KEY", "OPENAI_KEY", "OPENAI_API_TOKEN", "OPENAI_TOKEN")
        model = _get_secret_any("OPENAI_MODEL", default=DEFAULT_MODEL)

        if not api_key:
            return {
                "summary": "OpenAI API key not configured.",
                "reasons": [],
                "actions": [],
                "confidence": "low",
                "raw_text": "Set OPENAI_API_KEY in Streamlit Secrets or env.",
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
            "Respond with ONLY valid JSON (no backticks) exactly like:\n"
            '{ "summary": "...", "reasons": ["...","...","..."], '
            '"actions": ["...","...","..."], "confidence": "low|medium|high" }'
        )

        user_prompt = (
            "Analyze the circuit context and KPI values below and return an executive-ready recommendation.\n\n"
            f"Context (JSON): {json.dumps(context, ensure_ascii=False)}\n\n"
            f"{json_instruction}\n"
            "Keep under 120 words. If data is missing, say which fields are missing in the summary."
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You are a senior telecom network reliability engineer. Be concise, precise, and actionable."},
                {"role": "user", "content": user_prompt},
            ],
        }

        resp = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            txt = resp.text
            # Graceful fallback for quota/rate/auth
            if resp.status_code in (429, 400) and ("quota" in txt.lower() or "rate" in txt.lower()):
                lf = _heuristic_reco(circuit)
                lf["summary"] = "OpenAI quota/rate limit — showing heuristic plan."
                return lf
            if resp.status_code in (401, 403):
                return {
                    "summary": "OpenAI authentication/permission error.",
                    "reasons": [],
                    "actions": [],
                    "confidence": "low",
                    "raw_text": f"HTTP {resp.status_code}: {txt}",
                }
            return {
                "summary": "OpenAI API error",
                "reasons": [],
                "actions": [],
                "confidence": "low",
                "raw_text": f"HTTP {resp.status_code}: {txt}",
            }

        data = resp.json()
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        if not text:
            return {
                "summary": "Empty response from OpenAI.",
                "reasons": [],
                "actions": [],
                "confidence": "low",
                "raw_text": json.dumps(data),
            }

        parsed = _safe_json_parse(text)
        if parsed:
            return {
                "summary": parsed.get("summary", ""),
                "reasons": (parsed.get("reasons") or [])[:3],
                "actions": (parsed.get("actions") or [])[:3],
                "confidence": parsed.get("confidence", "medium"),
                "raw_text": text,
            }

        # Heuristic extraction if non-JSON
        summary, reasons, actions = _fallback_lists(text)
        return {
            "summary": summary,
            "reasons": reasons,
            "actions": actions,
            "confidence": "medium" if (reasons or actions) else "low",
            "raw_text": text,
        }

    except Exception as e:
        # FINAL guard — never return None/Ellipsis
        return {
            "summary": "Exception calling OpenAI",
            "reasons": [],
            "actions": [],
            "confidence": "low",
            "raw_text": str(e),
        }
