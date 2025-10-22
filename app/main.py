# --- robust import setup (works on Streamlit Cloud & local) ---
import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Safe import for dotenv
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return None

import streamlit as st
import pandas as pd
import numpy as np

from data_access import get_circuits, get_kpis, save_recommendation
from scoring import compute_risk_score
from ai_claude import generate_recommendation
from enrich_perplexity import get_context_hint

load_dotenv()

st.set_page_config(page_title="Telecom Circuit Optimizer", page_icon="📡", layout="wide")
st.title("📡 Telecom Circuit Optimization & Predictive Restoration")
st.caption("Portfolio insights + predictive flags + AI recommendations")

# ----------------------- Load data -----------------------
@st.cache_data
def load_portfolio():
    circuits = get_circuits()
    # ensure expected cols exist for scoring
    for col in ["utilization_pct","jitter_ms","pkt_loss_pct","latency_ms","crc_err_rate","bandwidth_mbps","redundancy"]:
        if col not in circuits.columns:
            circuits[col] = 0
    circuits["Risk Score"] = circuits.apply(compute_risk_score, axis=1)

    # KPIs (optional but used for trend-based prediction)
    try:
        kpis = get_kpis()
        kpis["ts"] = pd.to_datetime(kpis["ts"], errors="coerce")
    except Exception:
        kpis = pd.DataFrame()
    return circuits, kpis

circuits_df, kpis_df = load_portfolio()

if circuits_df.empty:
    st.warning("No circuit data found. Populate Supabase tables or place sample CSVs under ./data")
    st.stop()

# ----------------------- Diagnostics -----------------------
with st.expander("🔧 Diagnostics (Claude key visibility)"):
    import os as _os
    import streamlit as _st
    st.write("os.getenv('CLAUDE_API_KEY') present:", bool(_os.getenv("CLAUDE_API_KEY")))
    st.write("'CLAUDE_API_KEY' in st.secrets:", "CLAUDE_API_KEY" in _st.secrets)
    st.write("CLAUDE_MODEL:", _st.secrets.get("CLAUDE_MODEL", "(missing)"))

# ----------------------- Filters (global) -----------------------
colA, colB, colC = st.columns(3)
region = colA.selectbox("Region", ["All"] + sorted(circuits_df["region"].dropna().unique().tolist()), key="flt_region")
sla = colB.selectbox("SLA Tier", ["All"] + sorted(circuits_df["sla_tier"].dropna().unique().tolist()), key="flt_sla")
risk_min = colC.slider("Minimum Risk Score", 0, 100, 60, key="flt_risk_min")

fdf = circuits_df.copy()
if region != "All":
    fdf = fdf[fdf["region"] == region]
if sla != "All":
    fdf = fdf[fdf["sla_tier"] == sla]
if risk_min > 0:
    fdf = fdf[fdf["Risk Score"] >= risk_min]

# ----------------------- Helper: trend slopes over last 24h -----------------------
def compute_trends(kpis: pd.DataFrame) -> pd.DataFrame:
    if kpis.empty:
        return pd.DataFrame()
    # choose last 24 hours
    cutoff = kpis["ts"].max() - pd.Timedelta(hours=24)
    recent = kpis[kpis["ts"] >= cutoff].copy()
    if recent.empty:
        return pd.DataFrame()

    def slope(series):
        # simple linear regression slope over time index
        if len(series) < 3:
            return 0.0
        x = np.arange(len(series))
        y = np.array(series, dtype=float)
        xm = x.mean(); ym = y.mean()
        denom = ((x - xm)**2).sum()
        if denom == 0:
            return 0.0
        return float(((x - xm) * (y - ym)).sum() / denom)

    trends = []
    for cid, grp in recent.sort_values("ts").groupby("circuit_id"):
        trends.append({
            "circuit_id": cid,
            "util_slope": slope(grp["utilization_pct"]),
            "jitter_slope": slope(grp["jitter_ms"]),
            "loss_slope": slope(grp["pkt_loss_pct"]),
            "crc_slope": slope(grp["crc_err_rate"])
        })
    return pd.DataFrame(trends)

trends_df = compute_trends(kpis_df) if not kpis_df.empty else pd.DataFrame()
if not trends_df.empty:
    fdf = fdf.merge(trends_df, on="circuit_id", how="left")
else:
    for c in ["util_slope","jitter_slope","loss_slope","crc_slope"]:
        fdf[c] = 0.0

# Derived flags
fdf["optimize_candidate"] = (fdf["utilization_pct"] < 30) & (fdf["bandwidth_mbps"] >= 500)
fdf["predictive_flag"] = (fdf["util_slope"] > 0.5) | (fdf["jitter_slope"] > 0.15) | (fdf["loss_slope"] > 0.01)

# ----------------------- Tabs -----------------------
tab_opt, tab_pred, tab_assist = st.tabs(["💰 Optimize", "🔮 Predict", "🛠 Assist"])

# -------- 💰 Optimize: under-utilized / redundant / savings --------
with tab_opt:
    st.subheader("Optimization Opportunities")
    opt = fdf[(fdf["optimize_candidate"]) | (fdf["redundancy"] == True)].copy()
    opt["opportunity"] = np.where(
        opt["redundancy"] == True,
        "Redundant path detected — consolidate",
        "Under-utilized high-bandwidth — right-size plan"
    )
    st.dataframe(
        opt[["circuit_id","region","product","bandwidth_mbps","utilization_pct","Risk Score","opportunity"]],
        width="stretch",
    )

    st.markdown("**Top 10 by potential savings (heuristic)**")
    # crude savings heuristic: (bandwidth tier weight) * (1 - util%)
    bw_weight = np.where(opt["bandwidth_mbps"]>=1000, 1.0, np.where(opt["bandwidth_mbps"]>=500, 0.7, 0.4))
    opt = opt.assign(savings_score=(bw_weight * (100 - opt["utilization_pct"]) / 100.0))
    top_savings = opt.sort_values("savings_score", ascending=False).head(10)

    # Inline AI buttons per row
    for _, row in top_savings.iterrows():
        with st.expander(f"🧮 {row['circuit_id']} • {row['region']} • util {row['utilization_pct']}% • {row['bandwidth_mbps']} Mbps"):
            st.write(f"Opportunity: **{row['opportunity']}** | Risk: **{round(row['Risk Score'],1)}**")
            if st.button("AI: Recommend optimization plan", key=f"opt_ai_{row['circuit_id']}"):
                reco = generate_recommendation(row.to_dict())
                if (reco.get("actions") or reco.get("reasons")):
                    st.success("Recommendation ready")
                    st.write(f"**Summary:** {reco.get('summary','')}")
                    if reco.get("reasons"):
                        st.markdown("**Reasons:**")
                        for r in reco["reasons"]: st.markdown(f"- {r}")
                    if reco.get("actions"):
                        st.markdown("**Recommended Actions:**")
                        for a in reco["actions"]: st.markdown(f"- {a}")
                    st.caption(f"Confidence: {reco.get('confidence','medium')}")
                    save_recommendation(
                        circuit_id=row["circuit_id"],
                        summary=reco.get("summary",""),
                        actions="\n".join(reco.get("actions") or []),
                        confidence=reco.get("confidence","medium"),
                        risk_score=row.get("Risk Score"),
                    )
                else:
                    st.error("Claude did not return a recommendation.")
                    with st.expander("Show diagnostic details"):
                        st.code(reco.get("raw_text","No details"), language="json")

# -------- 🔮 Predict: likely-to-breach based on trends --------
with tab_pred:
    st.subheader("Predictive Risk (24h trend)")
    if trends_df.empty:
        st.info("No KPI history available. Add `data/kpis_sample.csv` to enable predictive view.")
    else:
        pred = fdf[
            (fdf["predictive_flag"]) | (fdf["Risk Score"] >= max(60, risk_min))
        ].copy()
        pred["risk_reason"] = np.where(
            fdf["jitter_slope"] > 0.15, "Rising jitter",
            np.where(fdf["loss_slope"] > 0.01, "Rising packet loss",
                     np.where(fdf["util_slope"] > 0.5, "Rising utilization", "Historic risk"))
        )
        st.dataframe(
            pred[["circuit_id","region","product","Risk Score","util_slope","jitter_slope","loss_slope","risk_reason"]],
            width="stretch",
        )

        st.markdown("**Top 10 likely to breach soon**")
        top_pred = pred.sort_values(["Risk Score","jitter_slope","loss_slope","util_slope"], ascending=False).head(10)

        for _, row in top_pred.iterrows():
            with st.expander(f"⚠️ {row['circuit_id']} • {row['region']} • risk {round(row['Risk Score'],1)} • trend: {row['risk_reason']}"):
                st.write(f"Trend slopes — util: {round(row['util_slope'],2)}, jitter: {round(row['jitter_slope'],2)}, loss: {round(row['loss_slope'],3)}")
                if st.button("AI: Preventive action plan", key=f"pred_ai_{row['circuit_id']}"):
                    reco = generate_recommendation(row.to_dict())
                    if (reco.get("actions") or reco.get("reasons")):
                        st.success("Recommendation ready")
                        st.write(f"**Summary:** {reco.get('summary','')}")
                        if reco.get("reasons"):
                            st.markdown("**Reasons:**")
                            for r in reco["reasons"]: st.markdown(f"- {r}")
                        if reco.get("actions"):
                            st.markdown("**Recommended Actions:**")
                            for a in reco["actions"]: st.markdown(f"- {a}")
                        st.caption(f"Confidence: {reco.get('confidence','medium')}")
                        save_recommendation(
                            circuit_id=row["circuit_id"],
                            summary=reco.get("summary",""),
                            actions="\n".join(reco.get("actions") or []),
                            confidence=reco.get("confidence","medium"),
                            risk_score=row.get("Risk Score"),
                        )
                    else:
                        st.error("Claude did not return a recommendation.")
                        with st.expander("Show diagnostic details"):
                            st.code(reco.get("raw_text","No details"), language="json")

# -------- 🛠 Assist: single-circuit deep dive --------
with tab_assist:
    st.subheader("NOC Assist")
    if fdf.empty:
        st.info("No circuits match the current filters.")
    else:
        with st.form(key="assist_form"):
            selected_id = st.selectbox("Choose a circuit", fdf["circuit_id"].tolist(), key="assist_selector")
            submitted = st.form_submit_button("Generate AI Recommendation", key="assist_submit")
            if submitted:
                sel = fdf.loc[fdf["circuit_id"] == selected_id].iloc[0].to_dict()
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Util %", f"{sel.get('utilization_pct',0)}")
                c2.metric("Jitter ms", f"{sel.get('jitter_ms',0)}")
                c3.metric("Loss %", f"{sel.get('pkt_loss_pct',0)}")
                c4.metric("Latency ms", f"{sel.get('latency_ms',0)}")
                c5.metric("CRC errs", f"{sel.get('crc_err_rate',0)}")
                c6.metric("Risk", f"{round(sel.get('Risk Score',0),1)}")

                reco = generate_recommendation(sel)
                actions = reco.get("actions") or []
                reasons = reco.get("reasons") or []

                if actions or reasons:
                    st.success("✅ Recommendation ready")
                    st.write(f"**Summary:** {reco.get('summary','')}")
                    if reasons:
                        st.markdown("**Reasons:**")
                        for r in reasons: st.markdown(f"- {r}")
                    if actions:
                        st.markdown("**Recommended Actions:**")
                        for a in actions: st.markdown(f"- {a}")
                    st.caption(f"Confidence: {reco.get('confidence','medium')}")

                    save_recommendation(
                        circuit_id=sel["circuit_id"],
                        summary=reco.get("summary",""),
                        actions="\n".join(actions),
                        confidence=reco.get("confidence","medium"),
                        risk_score=sel.get("Risk Score"),
                    )

                    hint = get_context_hint(sel.get("vendor"), sel.get("model"), sel.get("bandwidth_mbps"), sel.get("region"))
                    if hint:
                        st.info(f"Context hint: {hint}")
                else:
                    st.error("Claude did not return a recommendation.")
                    with st.expander("Show diagnostic details"):
                        st.code(reco.get("raw_text","No details"), language="json")
