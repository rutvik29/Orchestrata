# Agent Blueprints

Blueprint implementations organized by agentic design pattern.

## Patterns Included

| Pattern | Use Case | Complexity |
|---------|----------|------------|
| `react-baseline/` | Tool-augmented Q&A, single-step decisions | Low |
| `plan-and-execute/` | Long-horizon tasks, parallelizable sub-tasks | Medium |
| `critic-loop/` | High-stakes outputs requiring verification | Medium |
| `multi-agent-handoff/` | Intent routing to specialist agents | High |
| `hitl-gate/` | Human approval for destructive/external writes | High |

## Usage

Each blueprint is self-contained with its own `docker-compose.yml`, environment config, and eval fixtures.

```bash
cd agent-blueprints/<pattern-name>
cp .env.example .env
docker compose up
```

## Adding a New Blueprint

1. Copy the `_template/` directory
2. Implement the agent logic
3. Add an eval suite under `../../evals/<pattern-name>/`
4. Document trade-offs in an ADR under `../../docs/architecture-decisions/`
