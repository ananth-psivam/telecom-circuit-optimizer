
import datetime as dt

def build_servicenow_markdown(circuit_row: dict, reco: dict) -> str:
    when = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    title = f"[AutoDraft] Circuit {circuit_row.get('circuit_id')} - Risk {round(circuit_row.get('Risk Score',0),1)}"
    lines = [
        f"# {title}",
        f"**Generated:** {when}",
        "",
        "## Summary",
        reco.get("summary","(no summary)"),
        "",
        "## Context",
        f"- Region: {circuit_row.get('region')}",
        f"- Product: {circuit_row.get('product')}",
        f"- Bandwidth: {circuit_row.get('bandwidth_mbps')} Mbps",
        f"- SLA: {circuit_row.get('sla_tier')}",
        f"- KPIs: util {circuit_row.get('utilization_pct')}%, jitter {circuit_row.get('jitter_ms')} ms, loss {circuit_row.get('pkt_loss_pct')}%, latency {circuit_row.get('latency_ms')} ms, CRC {circuit_row.get('crc_err_rate')}",
        f"- Risk Score: {round(circuit_row.get('Risk Score',0),1)}",
        "",
        "## Reasons",
    ]
    for r in (reco.get("reasons") or []):
        lines.append(f"- {r}")
    lines.append("")
    lines.append("## Recommended Actions")
    for a in (reco.get("actions") or []):
        lines.append(f"- {a}")
    lines.append("")
    lines.append(f"**Confidence:** {reco.get('confidence','medium')}")
    return "\n".join(lines)
