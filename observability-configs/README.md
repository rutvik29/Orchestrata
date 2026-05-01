# Observability Configs

LangSmith, OpenTelemetry, and Grafana configurations for full-stack agent observability.

## Stack

| Tool | Purpose |
|------|---------|
| **LangSmith** | LLM call tracing, prompt versioning, run comparison |
| **OpenTelemetry** | Distributed tracing across agent steps and tool calls |
| **Grafana** | Dashboards for cost, latency, error rates, token usage |
| **Prometheus** | Metrics collection and alerting rules |

## Structure

```
observability-configs/
├── langsmith/        # LangSmith project configs and tracing setup
├── otel/             # OpenTelemetry collector config and instrumentation
├── grafana/          # Dashboard JSON exports and provisioning configs
└── prometheus/       # Scrape configs and alerting rules
```

## Key Metrics to Track

- `llm.tokens.input` / `llm.tokens.output` — per model, per agent
- `agent.step.latency_ms` — P50, P95, P99 per pattern
- `tool.call.error_rate` — by tool name
- `session.total_cost_usd` — per user/tenant
- `eval.score` — tracked as a time series to catch regressions

## Alerting Thresholds (defaults)

| Alert | Threshold |
|-------|-----------|
| High error rate | tool error rate > 5% over 5 min |
| Cost spike | daily spend > 2× 7-day average |
| Latency degradation | P95 > 3× baseline |
| Eval regression | score drop > 5% vs last deploy |
