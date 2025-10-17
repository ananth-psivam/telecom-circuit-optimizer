create table if not exists circuits (
  id uuid primary key default gen_random_uuid(),
  circuit_id text unique not null,
  region text not null,
  product text not null,
  site_a text,
  site_b text,
  bandwidth_mbps int not null,
  vendor text,
  model text,
  sla_tier text,
  utilization_pct numeric,
  latency_ms numeric,
  jitter_ms numeric,
  pkt_loss_pct numeric,
  crc_err_rate numeric,
  redundancy boolean default false,
  created_at timestamptz default now()
);

create table if not exists kpis (
  id uuid primary key default gen_random_uuid(),
  circuit_id text references circuits(circuit_id) on delete cascade,
  ts timestamptz not null,
  utilization_pct numeric,
  latency_ms numeric,
  jitter_ms numeric,
  pkt_loss_pct numeric,
  crc_err_rate numeric,
  alarms int,
  unique (circuit_id, ts)
);

create table if not exists recommendations (
  id uuid primary key default gen_random_uuid(),
  circuit_id text references circuits(circuit_id) on delete cascade,
  risk_score numeric,
  risk_factors jsonb,
  summary text,
  actions text,
  confidence text,
  created_at timestamptz default now()
);
