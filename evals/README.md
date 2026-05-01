# Evals

Offline evaluation suites, golden sets, and regression harnesses.

## Structure

```
evals/
├── golden-sets/          # Curated input/output pairs per agent pattern
├── unit/                 # Single-turn deterministic tests (fast, run in CI)
├── trajectory/           # Multi-turn reasoning path evaluations
├── llm-as-judge/         # Rubric-based scoring with stronger judge model
└── red-team/             # Adversarial inputs, prompt injection, jailbreak attempts
```

## Running Evals

```bash
# Run all unit evals (< 30s, no LLM calls)
pytest evals/unit/ --tb=short

# Run trajectory evals (requires LLM API key)
pytest evals/trajectory/ -v

# Run full suite with score reporting
python evals/run_suite.py --report --baseline evals/baselines/current.json
```

## Regression Policy

A PR that changes any prompt or tool logic must include eval results. If the eval score drops more than 2% against the golden baseline, the CI pipeline will block the merge.

Baselines are stored in `evals/baselines/` and tagged by date + commit SHA.

## Adding Evals

1. Add golden examples to `golden-sets/<pattern>.jsonl`
2. Write test in the appropriate subfolder
3. Run `python evals/calibrate.py` to update the baseline if the change is intentional
