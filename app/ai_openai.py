# app/ai_openai.py (only showing the parts to replace)
import os, json, requests
from typing import Dict, Any

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"

def _get_secret_any(*names, default: str = "") -> str:
    """Try env first, then Streamlit secrets, across multiple possible key names."""
    # 1) environment
    for n in names:
        v = (os.getenv(n) or "").strip()
        if v:
            return v
    # 2) streamlit secrets
    try:
        import streamlit as st  # safe import
        for n in names:
            v = (st.secrets.get(n, "") or "").strip()
            if v:
                return v
    except Exception:
        pass
    return default

def generate_recommendation(circuit, max_tokens=300):
    try:
        ...  # your existing logic
        return {...}  # every path returns a dict
    except Exception as e:
        return {
            "summary": "Exception calling OpenAI",
            "reasons": [],
            "actions": [],
            "confidence": "low",
            "raw_text": str(e),
        }
