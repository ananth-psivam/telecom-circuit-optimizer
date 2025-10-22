# --- robust import setup (works on Streamlit Cloud & local) ---
import os, sys, io
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

# --- Local modules (no "app." prefix) ---
from data_access import get_circuits, get_kpis, save_recommendation
from scoring import compute_risk_score
from ai_openai import generate_recommendation

# Optional helpers — import safely if available
try:
    from enrich_perplexity import get_context_hint
except Exception:
    def get_context_hint(*args, **kwargs):
        return None

try:
    from data_access import load_csv_to_table, bulk_upsert
except Exception:
    load_csv_to_table = None
    bulk_upsert = None

try:
    from ticket_helper import build_servicenow_markdown
except Exception:
    def build_servicenow_markdown(row, reco):
        return f"# Ticket Draft for {row.get('circuit_id','(unknown)')}\n\n" \
               f"**Summary:** {reco.get('summary','')}\n\n" \
               f"**Actions:**\n" + "\n".join(f"- {a}" for a in (reco.get('actions') or []))


# Load env vars
load_dotenv()

st.set_page_config(page_title="Telecom Circuit Optimizer", page_icon="📡", layout="wide")
st.title("📡 Telecom Circuit Optimization & Predictive Restoration")
st.caption("Portfolio insights • predictive flags • AI recommendations (Claude)")

# ----------------------- Load data -----------------------
@st.cache_data
def load_portfolio():
    circuits = get_circuits()
    # Try KPIs (optional, for trends)
    try:
        kpis = get_kpis()
        if "ts" in kpis.columns:
            kpis["ts"] = pd.to_datetime(kpis["ts"], errors="coerce")
    except Exception:
        kpis = pd.DataFrame()
    return circuits, kpis

circuits_df, kpis_df = load_portfolio()
if circuits_df is None or len(circuits_df) == 0:
    st.warning("No circuit data found. Load data from Supabase or replace data/*.csv with real samples.")
    st.stop()

# ---------- BEGIN: defensive schema normalization ----------
required_cols = [
    "circuit_id","region","product","bandwidth_mbps","vendor","model","sla_tier",
    "utilization_pct","jitter_ms","pkt_loss_pct","latency_ms","crc_err_rate","redundancy"
]

# Normalize column names (lowercase + strip)
circuits_df.columns = [c.strip().lower() for c in circuits_df.columns]

# Common aliases -> canonical names
alias_map = {
    "bandwidth": "bandwidth_mbps",
    "bandwidth_mb": "bandwidth_mbps",
    "bw_mbps": "bandwidth_mbps",
    "sla": "sla_tier",
    "loss_pct": "pkt_loss_pct",
    "packet_loss_pct": "pkt_loss_pct",
    "crc": "crc_err_rate",
}
for old, new in alias_map.items():
    if old in circuits_df.columns and new not in circuits_df.columns:
        circuits_df.rename(columns={old: new}, inplace=True)

# Ensure required columns exist with safe defaults
for col in required_cols:
    if col not in circuits_df.columns:
        if col in ("redundancy",):
            circuits_df[col] = False
        elif col in ("circuit_id","region","product","vendor","model","sla_tier"):
            circuits_df[col] = ""
        else:
            circuits_df[col] = 0

# Compute risk (after normalization)
circuits_df["Risk Score"] = circuits_df.apply(compute_risk_score, axis=1)

# Helper to avoid KeyErrors later
def safe_unique(df: pd.DataFrame, col: str):
    return sorted(df[col].dropna().unique().tolist()) if col in df.columns else []

with st.expander("📋 Data schema diagnostics"):
    st.write("Columns:", list(circuits_df.columns))
    st.write("Row count:", len(circuits_df))
    missing_like = [c for c in required_cols if c not in circuits_df.columns]
    if missing_like:
        st.warning(f"Columns missing and defaulted: {missing_like}")
# ---------- END: defensive schema normalization ----------

# ----------------------- Diagnostics (keys) -----------------------
with st.expander("🔧 Diagnostics (OpenAI key visibility)"):
    import os as _os, streamlit as _st
    st.write("os.getenv('OPENAI_API_KEY') present:", bool(_os.getenv("OPENAI_API_KEY")))
    st.write("'OPENAI_API_KEY' in st.secrets:", "OPENAI_API_KEY" in _st.secrets)
    st.write("OPENAI_MODEL:", _st.secrets.get("OPENAI_MODEL", "(missing)"))

# ----------------------- Global filters -----------------------
colA, colB, colC = st.columns(3)
region_choices = ["All"] + safe_unique(circuits_df, "region")
sla_choices = ["All"] + safe_unique(circuits_df, "sla_tier")

region = colA.selectbox("Region", region_choices, key="flt_region")
sla = colB.selectbox("SLA Tier", sla_choices, key="flt_sla")
risk_min = colC.slider("Minimum Risk Score", 0, 100, 60, key="flt_risk_min")

fdf = circuits_df.copy()
if "region" in fdf.columns and region != "All":
    fdf = fdf[fdf["region"] == region]
if "sla_tier" in fdf.columns and sla != "All":
    fdf = fdf[fdf["sla_tier"] == sla]
if "Risk Score" in fdf.columns and risk_min > 0:
    fdf = fdf[fdf["Risk Score"] >= risk_min]

# ----------------------- Helper: trend slopes over last 24h -----------------------
def compute_trends(kpis: pd.DataFrame) -> pd.DataFrame:
    if kpis is None or kpis.empty or "ts" not in kpis.columns or pd.isna(kpis["ts"]).all():
        return pd.DataFrame()
    cutoff = kpis["ts"].max() - pd.Timedelta(hours=24)
    recent = kpis[kpis["ts"] >= cutoff].copy()
    if recent.empty:
        return pd.DataFrame()

    def slope(series):
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
            "util_slope": slope(grp["utilization_pct"]) if "utilization_pct" in grp.columns else 0.0,
            "jitter_slope": slope(grp["jitter_ms"]) if "jitter_ms" in grp.columns else 0.0,
            "loss_slope": slope(grp["pkt_loss_pct"]) if "pkt_loss_pct" in grp.columns else 0.0,
            "crc_slope": slope(grp["crc_err_rate"]) if "crc_err_rate" in grp.columns else 0.0,
        })
    return pd.DataFrame(trends)

trends_df = compute_trends(kpis_df)
if not trends_df.empty:
    fdf = fdf.merge(trends_df, on="circuit_id", how="left")
else:
    for c in ["util_slope","jitter_slope","loss_slope","crc_slope"]:
        fdf[c] = 0.0

# Derived flags (guarded)
fdf["optimize_candidate"] = (fdf.get("utilization_pct", 0) < 30) & (fdf.get("bandwidth_mbps", 0) >= 500)
fdf["predictive_flag"] = (fdf.get("util_slope", 0) > 0.5) | (fdf.get("jitter_slope", 0) > 0.15) | (fdf.get("loss_slope", 0) > 0.01)

# ----------------------- Tabs -----------------------
# If loader/heatmap/ticket helpers are present, we'll show extra tabs
extra_tabs = []
if bulk_upsert or load_csv_to_table:
    extra_tabs.append("⬆️ Load to Supabase")
extra_tabs.append("📊 Vendor Heatmap")
extra_tabs.append("🎫 Ticket Drafts")

tabs = ["💰 Optimize", "🔮 Predict", "🛠 Assist"] + extra_tabs
tab_objs = st.tabs(tabs)

# -------- 💰 Optimize --------
with tab_objs[0]:
    st.subheader("Optimization Opportunities")
    opt = fdf[(fdf["optimize_candidate"]) | (fdf.get("redundancy", False) == True)].copy()
    if not opt.empty:
        opt["opportunity"] = np.where(
            opt.get("redundancy", False) == True,
            "Redundant path — consolidate",
            "Under-utilized high-bandwidth — right-size"
        )
        cols_to_show = [c for c in [
            "circuit_id","region","product","bandwidth_mbps","utilization_pct","Risk Score","opportunity"
        ] if c in opt.columns]
        st.dataframe(opt[cols_to_show], width="stretch")

        st.markdown("**Top 10 by potential savings (heuristic)**")
        bw = opt.get("bandwidth_mbps", pd.Series([0]*len(opt)))
        util = opt.get("utilization_pct", pd.Series([0]*len(opt)))
        bw_weight = np.where(bw>=1000, 1.0, np.where(bw>=500, 0.7, 0.4))
        opt = opt.assign(savings_score=(bw_weight * (100 - util) / 100.0))
        top_savings = opt.sort_values("savings_score", ascending=False).head(10)

        for _, row in top_savings.iterrows():
            title_bits = [row.get("circuit_id","(id)")]
            if "region" in row: title_bits += [row["region"]]
            st_exp = " • ".join([str(x) for x in title_bits if str(x)])
            with st.expander(f"🧮 {st_exp}"):
                st.write(f"Opportunity: **{row.get('opportunity','')}** | Risk: **{round(row.get('Risk Score',0),1)}**")
                if st.button("AI: Recommend optimization plan", key=f"opt_ai_{row.get('circuit_id')}"):
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
                            circuit_id=row.get("circuit_id"),
                            summary=reco.get("summary",""),
                            actions="\n".join(reco.get("actions") or []),
                            confidence=reco.get("confidence","medium"),
                            risk_score=row.get("Risk Score"),
                        )
                    else:
                        st.error("Claude did not return a recommendation.")
                        with st.expander("Show diagnostic details"):
                            st.code(reco.get("raw_text","No details"), language="json")
    else:
        st.info("No optimization candidates under current filters.")

# -------- 🔮 Predict --------
with tab_objs[1]:
    st.subheader("Predictive Risk (24h trend)")
    if trends_df.empty:
        st.info("No KPI history available. Add `data/kpis_sample.csv` or load from Supabase to enable predictive view.")
    else:
        pred = fdf[(fdf["predictive_flag"]) | (fdf["Risk Score"] >= max(60, risk_min))].copy()
        if not pred.empty:
            risk_reason = np.where(pred["jitter_slope"] > 0.15, "Rising jitter",
                            np.where(pred["loss_slope"] > 0.01, "Rising packet loss",
                            np.where(pred["util_slope"] > 0.5, "Rising utilization", "Historic risk")))
            pred["risk_reason"] = risk_reason
            cols_to_show = [c for c in [
                "circuit_id","region","product","Risk Score","util_slope","jitter_slope","loss_slope","risk_reason"
            ] if c in pred.columns]
            st.dataframe(pred[cols_to_show], width="stretch")

            st.markdown("**Top 10 likely to breach soon**")
            sort_cols = [c for c in ["Risk Score","jitter_slope","loss_slope","util_slope"] if c in pred.columns]
            top_pred = pred.sort_values(sort_cols, ascending=False).head(10) if sort_cols else pred.head(10)

            for _, row in top_pred.iterrows():
                exp_title = f"⚠️ {row.get('circuit_id','(id)')} • risk {round(row.get('Risk Score',0),1)} • trend: {row.get('risk_reason','')}"
                with st.expander(exp_title):
                    st.write(f"Trend slopes — util: {round(row.get('util_slope',0),2)}, "
                             f"jitter: {round(row.get('jitter_slope',0),2)}, "
                             f"loss: {round(row.get('loss_slope',0),3)}")
                    if st.button("AI: Preventive action plan", key=f"pred_ai_{row.get('circuit_id')}"):
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
                                circuit_id=row.get("circuit_id"),
                                summary=reco.get("summary",""),
                                actions="\n".join(reco.get("actions") or []),
                                confidence=reco.get("confidence","medium"),
                                risk_score=row.get("Risk Score"),
                            )
                        else:
                            st.error("Claude did not return a recommendation.")
                            with st.expander("Show diagnostic details"):
                                st.code(reco.get("raw_text","No details"), language="json")
        else:
            st.info("No predictive risks detected under the current filters.")

# -------- 🛠 Assist --------
with tab_objs[2]:
    st.subheader("NOC Assist")
    if fdf.empty:
        st.info("No circuits match the current filters.")
    else:
        with st.form(key="assist_form"):
            sel_choices = fdf["circuit_id"].tolist() if "circuit_id" in fdf.columns else []
            selected_id = st.selectbox("Choose a circuit", sel_choices, key="assist_selector")
            submitted = st.form_submit_button("Generate AI Recommendation", key="assist_submit")
            if submitted and selected_id:
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
                        circuit_id=sel.get("circuit_id"),
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

# -------- ⬆️ Load to Supabase (optional) --------
tab_index = 3
if "⬆️ Load to Supabase" in tabs and (bulk_upsert or load_csv_to_table):
    with tab_objs[tab_index]:
        st.subheader("CSV → Supabase Loader")
        st.write("Bulk upsert `data/circuits_sample.csv` and `data/kpis_sample.csv` into your Supabase tables.")
        col1, col2 = st.columns(2)
        with col1:
            if load_csv_to_table and st.button("Load sample circuits → Supabase", key="load_circuits"):
                res = load_csv_to_table("data/circuits_sample.csv", "circuits")
                if res.get("ok"):
                    st.success(f"Loaded {res.get('count',0)} circuits ✅")
                else:
                    st.error(f"Failed: {res.get('error')}")
        with col2:
            if load_csv_to_table and st.button("Load sample KPIs → Supabase", key="load_kpis"):
                res = load_csv_to_table("data/kpis_sample.csv", "kpis", convert_ts_cols=["ts"])
                if res.get("ok"):
                    st.success(f"Loaded {res.get('count',0)} KPI rows ✅")
                else:
                    st.error(f"Failed: {res.get('error')}")

        st.markdown("---")
        st.write("Or upload your own CSV and choose target table:")
        up = st.file_uploader("Upload CSV", type=["csv"], key="uploader_csv")
        table = st.selectbox("Target table", ["circuits","kpis","recommendations"], key="uploader_table")
        if up and bulk_upsert and st.button("Upload to Supabase", key="upload_custom"):
            df = pd.read_csv(up)
            if table == "kpis" and "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            res = bulk_upsert(table, df.to_dict(orient="records"))
            if res.get("ok"):
                st.success(f"Uploaded {res.get('count',0)} rows to {table} ✅")
            else:
                st.error(f"Failed: {res.get('error')}")
    tab_index += 1

# -------- 📊 Vendor Heatmap (always available) --------
with tab_objs[tab_index]:
    st.subheader("Vendor Performance Heatmap (counts by region)")
    if "vendor" in fdf.columns and "region" in fdf.columns:
        pivot = pd.pivot_table(fdf, index="region", columns="vendor", values="circuit_id", aggfunc="count", fill_value=0)
        st.dataframe(pivot, width="stretch")
        st.caption("Counts of circuits by Vendor and Region")
        vendor_counts = fdf.groupby("vendor")["circuit_id"].count().sort_values(ascending=False) if "vendor" in fdf.columns else pd.Series(dtype=int)
        st.bar_chart(vendor_counts, x_label="Vendor", y_label="Circuits")
    else:
        st.info("Vendor or Region columns missing.")
tab_index += 1

# -------- 🎫 Ticket Drafts (optional; uses fallback if helper missing) --------
with tab_objs[tab_index]:
    st.subheader("ServiceNow Ticket Draft Generator")
    sel_choices = fdf["circuit_id"].tolist() if "circuit_id" in fdf.columns else []
    pick = st.selectbox("Select a circuit", sel_choices, key="ticket_pick")
    if pick and st.button("Generate Ticket Draft", key="ticket_btn"):
        row = fdf.loc[fdf["circuit_id"] == pick].iloc[0].to_dict()
        reco = generate_recommendation(row)
        md = build_servicenow_markdown(row, reco)
        st.markdown(md)
        st.download_button("Download as .md", md.encode("utf-8"), file_name=f"{pick}_ticket_draft.md", mime="text/markdown")
