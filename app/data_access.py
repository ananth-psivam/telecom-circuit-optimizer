import os
import requests
import pandas as pd
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

HEADERS = {
    "apikey": SUPABASE_KEY or "",
    "Authorization": f"Bearer {SUPABASE_KEY}" if SUPABASE_KEY else "",
    "Content-Type": "application/json",
}

def _has_db():
    return bool(SUPABASE_URL and SUPABASE_KEY)

def get_circuits():
    if _has_db():
        try:
            r = requests.get(f"{SUPABASE_URL}/rest/v1/circuits?select=*", headers=HEADERS, timeout=30)
            if r.status_code == 200:
                df = pd.DataFrame(r.json())
                if not df.empty:
                    return df
        except Exception:
            pass
    return pd.read_csv("data/circuits_sample.csv")

def get_kpis():
    if _has_db():
        try:
            r = requests.get(f"{SUPABASE_URL}/rest/v1/kpis?select=*", headers=HEADERS, timeout=60)
            if r.status_code == 200:
                return pd.DataFrame(r.json())
        except Exception:
            pass
    return pd.read_csv("data/kpis_sample.csv")

def save_recommendation(circuit_id: str, summary: str, actions: str, confidence: str="medium", risk_score: Optional[float]=None):
    if not _has_db():
        return False
    payload = {
        "circuit_id": circuit_id,
        "summary": summary,
        "actions": actions,
        "confidence": confidence,
        "risk_score": risk_score,
    }
    try:
        r = requests.post(f"{SUPABASE_URL}/rest/v1/recommendations", headers=HEADERS, json=payload, timeout=20)
        return r.status_code in (200, 201)
    except Exception:
        return False

# -------- Bulk upsert helpers --------
def _chunk(iterable, size: int):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]

def bulk_upsert(table: str, rows: List[Dict[str, Any]], chunk_size: int=500) -> Dict[str, Any]:
    """
    Upsert rows into a Supabase table via REST.
    Requires RLS disabled or appropriate policies + service role key.
    """
    if not _has_db():
        return {"ok": False, "error": "Supabase not configured"}
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = dict(HEADERS)
    headers["Prefer"] = "resolution=merge-duplicates"
    total = 0
    for chunk in _chunk(rows, chunk_size):
        try:
            r = requests.post(url, headers=headers, json=chunk, timeout=60)
            if r.status_code not in (200, 201, 204):
                return {"ok": False, "error": f"HTTP {r.status_code}: {r.text}", "count": total}
            total += len(chunk)
        except Exception as e:
            return {"ok": False, "error": str(e), "count": total}
    return {"ok": True, "count": total}

def load_csv_to_table(csv_path: str, table: str, convert_ts_cols: Optional[list]=None) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    if convert_ts_cols:
        for c in convert_ts_cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = df.to_dict(orient="records")
    return bulk_upsert(table, rows)
