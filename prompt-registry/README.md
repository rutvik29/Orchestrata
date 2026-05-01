# Prompt Registry

Versioned prompt library with changelogs and eval results.

## Why Version Prompts?

Prompts are code. An unversioned prompt change is a silent regression waiting to happen. Every prompt in production has a version tag, a changelog entry, and an eval score attached to it.

## Structure

```
prompt-registry/
├── system-prompts/        # Agent system prompts, versioned by pattern
├── tool-descriptions/     # Tool docstrings fed to the LLM
├── output-schemas/        # Structured output format instructions
├── compression/           # Summary and context compression prompts
└── CHANGELOG.md           # Human-readable log of all prompt changes
```

## Versioning Convention

```
<pattern>/<prompt-name>/v<major>.<minor>.md
```

Example: `system-prompts/react-baseline/v2.1.md`

Minor bumps: wording tweaks, tone adjustments.  
Major bumps: structural changes, new instructions, role changes.

## Eval Gate

Before promoting a prompt version to `latest`:

```bash
python prompt-registry/eval_prompt.py \
  --prompt system-prompts/react-baseline/v2.1.md \
  --golden evals/golden-sets/react.jsonl \
  --min-score 0.85
```

If the score is below threshold, the promotion is blocked.

## Changelog Format

```markdown
## v2.1 — 2025-04-10
**Changed**: Clarified tool-use instructions for parallel calls  
**Eval score**: 0.91 (baseline: 0.88)  
**Promoted by**: @rutvik29
```
