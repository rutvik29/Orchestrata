# Architecture Decision Records (ADRs)

Architecture Decision Records for key design choices in Orchestrata. Every significant trade-off gets documented here — not just what we decided, but why, and what we considered and rejected.

## What is an ADR?

An ADR is a short document that captures an important architectural decision: the context, the options considered, the decision made, and the consequences. It's a gift to your future self and your teammates.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-001](./ADR-001-orchestration-topology.md) | Orchestration Topology Selection | Accepted |
| [ADR-002](./ADR-002-token-budget-strategy.md) | Token Budget Envelope Pattern | Accepted |
| [ADR-003](./ADR-003-eval-infrastructure.md) | Eval Infrastructure Stack | Accepted |
| [ADR-004](./ADR-004-tool-idempotency.md) | Tool Idempotency Contract | Accepted |
| [ADR-005](./ADR-005-context-compression.md) | Context Compression Strategy | Accepted |

## ADR Template

```markdown
# ADR-XXX: Title

**Date**: YYYY-MM-DD  
**Status**: Proposed | Accepted | Deprecated | Superseded by ADR-XXX

## Context

What is the problem or situation that requires a decision?

## Options Considered

### Option A: ...
Pros: ...  
Cons: ...

### Option B: ...
Pros: ...  
Cons: ...

## Decision

We chose Option X because ...

## Consequences

- What becomes easier?
- What becomes harder?
- What do we need to monitor?
```

## Process

1. Open a PR with a new ADR in draft status
2. Team discussion happens in the PR comments
3. When consensus is reached, status changes to Accepted and PR is merged
4. Superseded ADRs are kept — never deleted. History matters.
