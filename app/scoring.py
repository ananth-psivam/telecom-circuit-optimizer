"""
scoring.py
Risk scoring & redundancy heuristics for the Telecom Circuit Optimizer.
- compute_risk_score(row): returns 0..100 risk score for a circuit row (pd.Series or dict).
- compute_scores(df): vectorized application for a DataFrame.
The scoring is intentionally simple and explainable for workshop/demo use.
"""

from typing import Union, Mapping
import math

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return 0.0

def _get(row: Union[Mapping, "pd.Series"], key: str, default=None):
    try:
        return row.get(key, default) if hasattr(row, "get") else row[key]  # type: ignore
    except Exception:
        return default

def _redundancy_bonus_from_row(row: Union[Mapping, "pd.Series"]) -> float:
    """
    Lightweight redundancy heuristic when we can't compare across rows:
    - If an explicit boolean/flag exists (e.g., 'redundancy', 'redundancy_flag'), use it.
    - Else, if utilization < 30% and bandwidth >= 500 Mbps, consider over-provisioned -> small bonus.
    """
    # explicit flags
    for k in ("redundancy", "redundancy_flag", "is_redundant"):
        val = _get(row, k, None)
        if isinstance(val, bool):
            return 10.0 if val else 0.0
        # some sources use "Y"/"N" or 1/0
        if isinstance(val, (int, float)) and val in (0,1):
            return 10.0 if val == 1 else 0.0
        if isinstance(val, str) and val.strip().lower() in ("y", "yes", "true", "1"):
            return 10.0

    # heuristic
    util = _get(row, "utilization_pct", 0) or 0
    bw   = _get(row, "bandwidth_mbps", 0) or 0
    try:
        util = float(util)
        bw = float(bw)
    except Exception:
        return 0.0

    if util < 30 and bw >= 500:
        return 8.0  # slightly smaller bonus than explicit flag
    return 0.0

def compute_risk_score(row: Union[Mapping, "pd.Series"]) -> float:
    """
    Returns a 0..100 risk score using normalized KPI contributions.
    Uses keys if present, otherwise defaults safely to 0.
    Weights are tuned for demonstration clarity.
    """
    util   = _get(row, "utilization_pct", 0) or 0
    jitter = _get(row, "jitter_ms", 0) or 0
    loss   = _get(row, "pkt_loss_pct", 0) or 0
    lat    = _get(row, "latency_ms", 0) or 0
    crc    = _get(row, "crc_err_rate", 0) or 0

    try:
        util = float(util)
        jitter = float(jitter)
        loss = float(loss)
        lat = float(lat)
        crc = float(crc)
    except Exception:
        util = jitter = loss = lat = crc = 0.0

    # Normalize to 0..1 ranges against reasonable thresholds
    n_util   = _clamp(util / 100.0)         # 100% util => 1.0
    n_jitter = _clamp(jitter / 30.0)        # 30ms jitter ~ high
    n_loss   = _clamp(loss / 2.0)           # 2% loss ~ severe
    n_lat    = _clamp(lat / 100.0)          # 100ms ~ high for metro/agg
    n_crc    = _clamp(crc / 1000.0)         # 1000 errs per interval ~ concerning

    # Weights sum to 1.0
    risk_core = (
        0.35 * n_util +
        0.25 * n_jitter +
        0.20 * n_loss +
        0.10 * n_lat +
        0.10 * n_crc
    )

    bonus = _redundancy_bonus_from_row(row)  # up to +10
    risk = 100.0 * risk_core + bonus

    # keep in 0..100
    return float(max(0.0, min(100.0, risk)))

def compute_scores(df):
    """
    Vectorized helper: adds/returns a 'Risk Score' column for a DataFrame.
    """
    try:
        import pandas as pd  # local import to avoid hard dependency here
        if "Risk Score" not in df.columns:
            df["Risk Score"] = df.apply(compute_risk_score, axis=1)
        else:
            df["Risk Score"] = df.apply(compute_risk_score, axis=1)
        return df
    except Exception:
        # fail quietly for environments without pandas
        return df
