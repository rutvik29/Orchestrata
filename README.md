# Orchestrata
### A Production-Grade Playbook for Agentic AI Architectures

> For senior engineers who care about latency, token economics, and scalable multi-agent systems.

---

## What This Is

Orchestrata is an opinionated, production-tested technical playbook for building agentic AI systems that actually ship. Not a tutorial. Not a demo repo. A reference architecture built from real deployment experience — covering the hard parts: orchestration topology, token budget management, evaluation infrastructure, observability, and the DevEx layer that keeps teams sane at scale.

If you've already read the LangChain docs and you're past "hello world," this is where you go next.

---

## Core Design Principles

| Principle | Why It Matters |
|-----------|---------------|
| **Latency is a feature** | P95 < 2s for interactive agents; async-first for batch workloads |
| **Token budgets are real money** | Every prompt is a cost center; measure, cache, compress |
| **Agents fail; systems recover** | Idempotent tools, retry envelopes, dead-letter queues |
| **Eval before you ship** | No agent reaches prod without an offline eval harness |
| **Observable by default** | Every span traced, every tool call logged, every anomaly alerted |

---

## Repository Structure

```
Orchestrata/
├── agent-blueprints/       # Pattern implementations: ReAct, Plan-and-Execute, Critic, etc.
├── evals/                  # Offline eval suites, golden sets, regression harnesses
├── observability-configs/  # LangSmith, OpenTelemetry, and Grafana configurations
├── docker/                 # Per-agent Dockerfiles and docker-compose orchestration
├── prompt-registry/        # Versioned prompt library with changelogs and eval results
├── shared-libs/            # Tool wrappers, retry logic, rate limiters, mock LLM utilities
└── docs/
    └── architecture-decisions/  # ADRs for key design choices
```

---

## Agentic Design Patterns

### 1. ReAct (Reason + Act)
The baseline pattern. Works well for tool-augmented Q&A and single-step decision loops. Breaks down with long horizons and compounding errors. Use it where you need fast, interpretable traces.

### 2. Plan-and-Execute
Separate the planning LLM from the execution LLM. The planner produces a DAG of sub-tasks; executors run them in parallel or serial. Key win: you can diff the plan before committing tokens to execution.

### 3. Critic / Reflection Loop
Post-execution verification layer. A secondary LLM checks outputs against a rubric before they surface to users. Adds latency; saves you from hallucinated tool calls reaching production.

### 4. Multi-Agent Handoff (Orchestrator + Specialists)
Orchestrator routes intents to domain specialists (code agent, search agent, summarizer). Each specialist has a constrained tool set and a bounded context window. Critical: define handoff contracts as typed schemas, not prose.

### 5. Human-in-the-Loop (HITL) Gate
Approval checkpoints for high-stakes actions (writes, external API calls with side effects). Implement as async interrupts, not blocking waits. Store checkpoint state in Redis or Postgres — never in-memory.

---

## Token Economics

### Budget Envelope Pattern
```python
class TokenBudget:
    def __init__(self, total: int, reserve_ratio: float = 0.15):
        self.total = total
        self.reserve = int(total * reserve_ratio)
        self.available = total - self.reserve
        self.spent = 0

    def can_afford(self, estimated_tokens: int) -> bool:
        return (self.spent + estimated_tokens) <= self.available

    def charge(self, actual_tokens: int):
        self.spent += actual_tokens
        if self.spent > self.available:
            raise TokenBudgetExceeded(f"Spent {self.spent} / {self.available}")
```

### Context Compression Strategies
- **Sliding window**: Drop oldest messages, keep system prompt + last N turns
- **Summary compression**: Periodically summarize conversation history with a cheap model
- **Retrieval-augmented context**: Replace static context with dynamic RAG hits
- **Semantic deduplication**: Remove near-duplicate tool outputs before re-injection

### Caching
Semantic cache (e.g., GPTCache, Redis + embedding similarity) cuts repeat call costs by 30–60% in production workloads. Cache at the tool layer, not the LLM layer.

---

## Orchestration Topology

### Latency Budget Allocation (example: 2s SLA)

```
User Request
    │
    ▼ ~50ms
Router / Intent Classifier
    │
    ▼ ~200ms
Context Assembly (RAG + memory fetch)
    │
    ▼ ~1200ms
Primary LLM Call (with tools)
    │
    ├──▶ Tool execution: ~300ms (parallel where possible)
    │
    ▼ ~150ms
Output validation + formatting
    │
    ▼ ~100ms
Response + trace flush
```

### Parallelism Rules
- Tool calls with no data dependency: run in parallel (asyncio.gather)
- Dependent tool calls: chain sequentially, minimize round-trips
- Cross-agent calls: use message queues (Kafka, SQS), not synchronous HTTP

---

## Evaluation Infrastructure

### The Eval Stack
1. **Unit evals**: Single-turn, deterministic. Golden input → expected output. Fast, cheap, run in CI.
2. **Trajectory evals**: Multi-turn. Score the reasoning path, not just the final answer.
3. **LLM-as-judge**: For open-ended outputs. Use a stronger model to score a weaker one. Calibrate against human labels.
4. **Red-team evals**: Adversarial inputs, jailbreak attempts, prompt injection. Run pre-deploy.

### Regression Harness
Every prompt change triggers a full eval run against the golden set. If score drops > 2%, block the deploy. Track eval trends in a time-series DB (Prometheus or InfluxDB).

---

## Observability

### What to Instrument
- **Every LLM call**: model, tokens in/out, latency, finish reason, cost estimate
- **Every tool call**: tool name, input hash, output hash, latency, error type
- **Every agent step**: step number, action taken, observation received, cumulative tokens
- **Every session**: session ID, total cost, total latency, user satisfaction signal

### Trace Schema (OpenTelemetry)
```python
with tracer.start_as_current_span("agent.step") as span:
    span.set_attribute("agent.id", agent_id)
    span.set_attribute("step.index", step_index)
    span.set_attribute("llm.model", model_name)
    span.set_attribute("llm.tokens.input", input_tokens)
    span.set_attribute("llm.tokens.output", output_tokens)
    span.set_attribute("tool.name", tool_name)
```

---

## DevEx Layer

Good tooling collapses the feedback loop. The difference between a team that ships agents fast and one that doesn't is usually the DevEx layer, not the model.

### Local Development
- **Mock LLM**: Deterministic response fixtures for unit tests. Never hit production LLMs in CI.
- **Tool sandbox**: Isolated execution environment. Tools get fake credentials, fake DBs.
- **Replay mode**: Record real LLM interactions, replay them locally. Deterministic, free.

### Developer Workflow
```
1. Edit prompt or tool
2. Run unit evals locally (< 30 seconds)
3. PR triggers full eval suite in CI
4. Eval dashboard shows delta vs. baseline
5. Reviewer approves if score delta is acceptable
6. Deploy with automatic rollback if production eval score drops
```

---

## Production Checklist

Before you ship an agent to production:

- [ ] Offline eval harness exists and passes with score ≥ baseline
- [ ] All tools are idempotent or clearly marked as non-idempotent
- [ ] Token budget is bounded and enforced; no unbounded loops possible
- [ ] Retry logic has exponential backoff + jitter + max attempts
- [ ] All LLM calls and tool calls are traced in OTel
- [ ] PII scrubbed from traces before export
- [ ] Rate limits in place for external API tools
- [ ] HITL gate implemented for any destructive or external write operations
- [ ] Graceful degradation path if LLM provider is down
- [ ] Cost alerting configured (e.g., alert if daily spend > $X)
- [ ] Red-team eval run against adversarial inputs

---

## Case Studies

### Case Study 1: Code Review Agent
**Problem**: Engineers spending 3+ hours/day on routine PR reviews.
**Architecture**: ReAct agent with GitHub tools, static analysis tool, style guide RAG.
**Key decisions**: 
- Critic loop to catch hallucinated code suggestions
- Async execution; agent posts comments, doesn't block the reviewer
- Eval set built from 500 historical reviews with expert labels
**Result**: 70% of routine comments automated; P95 latency 4.2s (async, not user-blocking).

### Case Study 2: Data Pipeline Debugger
**Problem**: On-call engineers spending 45 min avg. diagnosing pipeline failures.
**Architecture**: Plan-and-Execute with specialist sub-agents (log analyzer, schema inspector, lineage tracer).
**Key decisions**:
- Planner uses o1-class model; executors use cheaper model
- All tool calls idempotent (read-only); HITL gate before any pipeline restart
- Context compression: logs summarized before injection (50k → 2k tokens)
**Result**: Median diagnosis time dropped to 8 min; 40% cost reduction via model tiering.

---

## Getting Started

```bash
git clone https://github.com/rutvik29/Orchestrata
cd Orchestrata

# Install shared libraries
pip install -e shared-libs/

# Run the eval suite
cd evals && pytest --tb=short

# Start a blueprint locally
cd agent-blueprints/react-baseline
docker compose up
```

---

## Contributing

This is an opinionated playbook. PRs that add patterns should include:
1. A working implementation in `agent-blueprints/`
2. An eval suite in `evals/`
3. An ADR in `docs/architecture-decisions/` explaining the trade-offs

---

## License

MIT
