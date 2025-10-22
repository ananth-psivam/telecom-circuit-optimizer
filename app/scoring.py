from typing import Union, Mapping

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
    for k in ("redundancy", "redundancy_flag", "is_redundant"):
        val = _get(row, k, None)
        if isinstance(val, bool):
            return 10.0 if val else 0.0
        if isinstance(val, (int, float)) and val in (0,1):
            return 10.0 if val == 1 else 0.0
        if isinstance(val, str) and val.strip().lower() in ("y","yes","true","1"):
            return 10.0
    util = _get(row, "utilization_pct", 0) or 0
    bw   = _get(row, "bandwidth_mbps", 0) or 0
    try:
        util = float(util); bw = float(bw)
    except Exception:
        return 0.0
    return 8.0 if (util < 30 and bw >= 500) else 0.0

def compute_risk_score(row: Union[Mapping, "pd.Series"]) -> float:
    util   = _get(row, "utilization_pct", 0) or 0
    jitter = _get(row, "jitter_ms", 0) or 0
    loss   = _get(row, "pkt_loss_pct", 0) or 0
    lat    = _get(row, "latency_ms", 0) or 0
    crc    = _get(row, "crc_err_rate", 0) or 0
    try:
        util=float(util); jitter=float(jitter); loss=float(loss); lat=float(lat); crc=float(crc)
    except Exception:
        util=jitter=loss=lat=crc=0.0

    n_util   = _clamp(util/100.0)
    n_jitter = _clamp(jitter/30.0)
    n_loss   = _clamp(loss/2.0)
    n_lat    = _clamp(lat/100.0)
    n_crc    = _clamp(crc/1000.0)

    risk_core = 0.35*n_util + 0.25*n_jitter + 0.20*n_loss + 0.10*n_lat + 0.10*n_crc
    bonus = _redundancy_bonus_from_row(row)
    risk = 100.0 * risk_core + bonus
    return float(max(0.0, min(100.0, risk)))
