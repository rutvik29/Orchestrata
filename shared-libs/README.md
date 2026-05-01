# Shared Libraries

Common tool wrappers, retry logic, rate limiters, and mock LLM utilities shared across all agent blueprints.

## Packages

```
shared-libs/
├── tool_base/         # Abstract base class for all tools; enforces idempotency contract
├── retry/             # Exponential backoff + jitter + max-attempts wrapper
├── rate_limiter/      # Token bucket rate limiter for external API calls
├── mock_llm/          # Deterministic mock LLM for unit tests and local dev
├── token_budget/      # Token budget envelope and cost tracking utilities
├── context_utils/     # Context compression, sliding window, semantic dedup
└── schemas/           # Shared Pydantic models for tool inputs/outputs and handoff contracts
```

## Installation

```bash
pip install -e shared-libs/
```

All agent blueprints depend on this package. Pin to a specific version in production.

## Key Utilities

### Retry Wrapper
```python
from shared_libs.retry import with_retry

@with_retry(max_attempts=3, base_delay=1.0, jitter=True)
async def call_external_api(payload):
    ...
```

### Mock LLM
```python
from shared_libs.mock_llm import MockLLM

# Returns fixture responses deterministically — no API calls in tests
llm = MockLLM(fixture_path="tests/fixtures/react_responses.jsonl")
```

### Tool Base
```python
from shared_libs.tool_base import BaseTool

class SearchTool(BaseTool):
    name = "search"
    idempotent = True  # Enforce contract: read-only tools must be idempotent

    async def run(self, query: str) -> str:
        ...
```
