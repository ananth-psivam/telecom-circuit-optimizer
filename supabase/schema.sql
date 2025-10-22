-- Supabase schema for telecom-circuit-optimizer
CREATE TABLE IF NOT EXISTS circuits (
    circuit_id TEXT PRIMARY KEY,
    region TEXT,
    product TEXT,
    bandwidth_mbps INTEGER,
    vendor TEXT,
    model TEXT,
    sla_tier TEXT,
    utilization_pct FLOAT,
    jitter_ms FLOAT,
    pkt_loss_pct FLOAT,
    latency_ms FLOAT,
    crc_err_rate FLOAT,
    redundancy BOOLEAN
);

CREATE TABLE IF NOT EXISTS kpis (
    circuit_id TEXT,
    ts TIMESTAMP,
    utilization_pct FLOAT,
    jitter_ms FLOAT,
    pkt_loss_pct FLOAT,
    latency_ms FLOAT,
    crc_err_rate FLOAT
);

CREATE TABLE IF NOT EXISTS recommendations (
    id SERIAL PRIMARY KEY,
    circuit_id TEXT,
    summary TEXT,
    actions TEXT,
    confidence TEXT,
    risk_score FLOAT,
    created_at TIMESTAMP DEFAULT now()
);
