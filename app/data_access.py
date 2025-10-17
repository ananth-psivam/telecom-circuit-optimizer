import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

HEADERS = {
    "apikey": SUPABASE_KEY or "",
    "Authorization": f"Bearer {SUPABASE_KEY}" if SUPABASE_KEY else "",
    "Content-Type": "application/json"
}

def get_circuits():
    # DB path
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            r = requests.get(f"{SUPABASE_URL}/rest/v1/circuits?select=*", headers=HEADERS, timeout=20)
            if r.status_code == 200:
                df = pd.DataFrame(r.json())
                if not df.empty:
                    return df
        except Exception:
            pass
    # Fallback CSV
    return pd.read_csv("data/circuits_sample.csv")

def get_kpis():
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            r = requests.get(f"{SUPABASE_URL}/rest/v1/kpis?select=*", headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return pd.DataFrame(r.json())
        except Exception:
            pass
    return pd.read_csv("data/kpis_sample.csv")

def save_recommendation(circuit_id, summary, actions, confidence="medium", risk_score=None):
    if not (SUPABASE_URL and SUPABASE_KEY):
        return False
    payload = {
        "circuit_id": circuit_id,
        "summary": summary,
        "actions": actions,
        "confidence": confidence,
        "risk_score": risk_score
    }
    try:
        r = requests.post(f"{SUPABASE_URL}/rest/v1/recommendations", headers=HEADERS, json=payload, timeout=20)
        return r.status_code in (200,201)
    except Exception:
        return False
