# --- robust import setup (works on Streamlit Cloud & local) ---
import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))     # .../app
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
# ---------------------------------------------------------------

# Safe import for dotenv (provides load_dotenv(); no-op fallback if missing)
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return None

import streamlit as st
import pandas as pd

# Import local modules directly (no "app." prefix)
from data_access import get_circuits, get_kpis, save_recommendation
from scoring import compute_risk_score
from ai_claude import generate_recommendation
from enrich_perplexity import get_context_hint

# Load env vars (only matters if you set Secrets/.env)
load_dotenv()

st.set_page_config(page_title="Telecom Circuit Optimizer", page_icon="📡", layout="wide")
st.title("📡 Telecom Circuit Optimization & Predictive Restoration")
st.caption("Demo: Supabase + Streamlit + Claude (+ Perplexity optional)")

# ----------------------- Load data -----------------------
@st.cache_data
def load_circuits():
    df = get_circuits()
    # ensure expected cols exist for scoring
    for col in ["utilization_pct","jitter_ms","pkt_loss_pct","latency_ms","crc_err_rate","bandwidth_mbps"]:
        if col not in df.columns:
            df[col] = 0
    df["Risk Score"] = df.apply(compute_risk_score, axis=1)
    return df

df = load_circuits()

if df.empty:
    st.warning("No circuit data found. Populate Supabase tables or place sample CSVs under ./data")
    st.stop()

with st.expander("🔧 Diagnostics (Claude key visibility)"):
    import os as _os
    import streamlit as _st
    st.write("os.getenv('CLAUDE_API_KEY') present:", bool(_os.getenv("CLAUDE_API_KEY")))
    st.write("'CLAUDE_API_KEY' in st.secrets:", "CLAUDE_API_KEY" in _st.secrets)
    st.write("CLAUDE_MODEL:", _st.secrets.get("CLAUDE_MODEL", "(missing)"))


# ----------------------- Filters -----------------------
left, mid, right = st.columns(3)
region = left.selectbox("Region", ["All"] + sorted(df["region"].dropna().unique().tolist()))
sla = mid.selectbox("SLA Tier", ["All"] + sorted(df["sla_tier"].dropna().unique().tolist()))
threshold = right.slider("Minimum Risk Score", 0, 100, 60)

filtered = df.copy()
if region != "All":
    filtered = filtered[filtered["region"] == region]
if sla != "All":
    filtered = filtered[filtered["sla_tier"] == sla]
filtered = filtered[filtered["Risk Score"] >= threshold]

st.subheader(f"Circuits meeting criteria: {len(filtered)}")
st.dataframe(filtered[["circuit_id","region","product","bandwidth_mbps","utilization_pct","jitter_ms","pkt_loss_pct","Risk Score"]], use_container_width=True)

# ----------------------- Detail / AI panel -----------------------
st.markdown("---")
st.header("🤖 AI-Powered Recommendation")

if filtered.empty:
    st.info("Adjust filters to view and analyze circuits.")
else:
    selected_id = st.selectbox("Choose a circuit", filtered["circuit_id"].tolist(), key="selected_circuit")

    sel = filtered.loc[filtered["circuit_id"] == selected_id].iloc[0].to_dict()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Util %", f"{sel.get('utilization_pct',0)}")
    c2.metric("Jitter ms", f"{sel.get('jitter_ms',0)}")
    c3.metric("Loss %", f"{sel.get('pkt_loss_pct',0)}")
    c4.metric("Latency ms", f"{sel.get('latency_ms',0)}")
    c5.metric("CRC errs", f"{sel.get('crc_err_rate',0)}")
    c6.metric("Risk", f"{round(sel.get('Risk Score',0),1)}")

    if st.button("Generate AI Recommendation"):
        reco = generate_recommendation(sel)

        st.success("Recommendation ready")
        st.write(f"**Summary:** {reco.get('summary')}")
        reasons = reco.get("reasons", [])
        actions = reco.get("actions", [])

        if reasons:
            st.markdown("**Reasons:**")
            for r in reasons: st.markdown(f"- {r}")
        if actions:
            st.markdown("**Recommended Actions:**")
            for a in actions: st.markdown(f"- {a}")
        st.caption(f"Confidence: {reco.get('confidence','medium')}")

        # Optional: save to DB
        save_recommendation(
            circuit_id=sel["circuit_id"],
            summary=reco.get("summary",""),
            actions="\n".join(actions) if actions else "",
            confidence=reco.get("confidence","medium"),
            risk_score=sel.get("Risk Score")
        )

        # Optional Perplexity enrichment
        hint = get_context_hint(sel.get("vendor"), sel.get("model"), sel.get("bandwidth_mbps"), sel.get("region"))
        if hint:
            st.info(f"Context hint: {hint}")

if st.button("Generate AI Recommendation"):
    reco = generate_recommendation(sel)
    if reco.get("actions") or reco.get("reasons"):
        st.success("✅ Recommendation ready")
        st.write(f"**Summary:** {reco.get('summary')}")
        ...
    else:
        st.error("Claude did not return a recommendation.")
        with st.expander("Show diagnostic details"):
            st.code(reco.get("raw_text", "No details"), language="json")

