# Docker

Per-agent Dockerfiles and docker-compose orchestration for local development and production deployment.

## Structure

```
docker/
├── base/                  # Base image with shared Python deps and tooling
├── react-agent/           # Dockerfile for the ReAct baseline agent
├── plan-execute-agent/    # Dockerfile for the Plan-and-Execute agent
├── critic-agent/          # Dockerfile for the Critic/Reflection agent
├── orchestrator/          # Dockerfile for the multi-agent orchestrator
└── docker-compose.yml     # Full-stack local dev environment
```

## Quick Start

```bash
# Start the full stack locally
docker compose up

# Start a specific agent only
docker compose up react-agent

# Rebuild after dependency changes
docker compose build --no-cache
```

## Image Strategy

- All agent images extend `docker/base/` to share a common layer cache
- Production images are built with `--target=production` (no dev deps, no test fixtures)
- Images are tagged by commit SHA in CI: `ghcr.io/rutvik29/orchestrata/<agent>:<sha>`

## Environment Variables

Each agent container reads from a `.env` file. Copy `.env.example` and fill in:
- `LLM_API_KEY` — your model provider key
- `LANGSMITH_API_KEY` — for tracing
- `MOCK_LLM=true` — enables deterministic mock responses for local dev
