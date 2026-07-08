# RUN-USE-CASES.md — single file to drive the full POC

One runbook, two use cases, run back-to-back through the **BMO Digital Core — SDLC
Assistant** vsix. Everything you type is in a `>` block — copy it verbatim into the
extension chat. Everything else is what you should see.

---

## 0. One-time setup (5 minutes)

1. Build + install the extension:
   - Windows: `vscode-extension\build-vsix.cmd`
   - then: `code --install-extension vscode-extension\sdlc-assistant-2.0.0.vsix`
2. GitHub Copilot CLI installed and signed in (`copilot` on PATH, or set the
   `sdlc.copilotCliPath` setting).
3. Open **this POC folder** as the VS Code workspace — the project-bound agents are
   already in `.copilot/agents/` inside `vscode-extension/`, or install globally:
   `node vscode-extension/sdlc/scripts/install-agents.mjs`
4. Command Palette → **SDLC: Open Panel** → the BMO Digital Core panel opens.

---

## USE CASE 1 — API generation (short prompt → full company-standard repo)

**Inputs available in the workspace** (the agents' only content sources):
`mortgage-loan-statement/docs/NCP Direct Access B2B Web Services- with GetAccountPDF.docx`
and `mortgage-loan-statement/docs/Statements.xml|.xsd` (WSDL).

**Prompt (this is ALL you type):**

> Build the mortgage statement API end to end from the NCP document and WSDL in docs/.
> BIAN classification: Business Domain "Document Services", Service Domain "Mortgage statement".

**Expected output** — the complete standard repo, generated (not copied):

```
mortgage-loan-statement/
├── build.sh · controller.json · metadata.json · .gitignore · README.md (with "Open Items — require ground truth")
├── swagger/Mortgage-loan-statement-swagger.yaml        <- GENERATED from NCP doc + WSDL
├── infra/  (iam/function/api/usageplan/ssm/operations stacks, swagger-api.ts WITH api_key security,
│            function_stack WITH VPC wiring from local config, config/index.ts.example [NOT IN SOURCE])
└── service/lambda-functions/
    ├── index.js            <- STARTING POINT (flow map: path -> handler -> serviceFunc -> NCP endpoint)
    ├── common/logger/      (framework with 415 + 504 handling)
    ├── service/listMortgageStatement.js · retrieveMortgageStmt.js · lib/(ncpClient|hashUtil|errorMapping|mappers|config)
    └── test/               (Mocha per operation)
```

**Acceptance gates (run in a terminal — ALL must pass):**

```bash
cd mortgage-loan-statement
python3 scripts/validate-generated-repo.py .          # deterministic rules 1-5 -> GENERATION GATE PASSED
cd service/lambda-functions && npm install && npm test # Mocha green
```

**What proves precision:** paths/operationIds/headers/schemas match the original
production interface (see `COMPARISON.md`, 89% with every difference justified), and the
README **Open Items** table lists what was NOT in the sources (hash casing, NFRs, VPC ids,
host, documentID-vs-encryptedKey ruling) instead of assuming them.

---

## USE CASE 2 — UI change (short prompt → UI feature + tests)

**Precondition:** the demo UI runs locally: `cd ui && node mock-server.js` →
open http://localhost:3000, list account `11111111`.

**Prompt (this is ALL you type):**

> Add a hyperlink beside each account number that opens a side panel showing that
> account's full statement history, with View and Download on every statement.

**Expected agent routing (watch the panel phases):**
impact-analysis (classifies: data already available via ListStatement → UI-only change,
lists exact files) → dev (edits `ui/index.html` / webapp repo per its conventions) → qa
(test cases: panel opens with full 36-month history, per-row inline view + download,
auto re-list on expired statementKey (401), generic error rendering, no NCP codes visible).

**Acceptance checks:**

1. Reload http://localhost:3000 → account number is now a link → side panel with full
   history → View opens the PDF inline, Download saves it.
2. Wait 2 minutes (demo key TTL) → click View → toast "session expired — refreshing" →
   list refreshes and the PDF opens. No raw error codes anywhere.
3. QA artifacts written under `qa_artifacts/` with a traceability row per scenario.

**Variations to show flexibility (same pipeline, no agent edits):**

> Add a download-all button to the history side panel.
> Remove the product column from the statements table.

If a request needs data the API does not expose, the Swagger agent proposes an
**additive** contract change first and runs `scripts/validate-openapi.py` — or rejects it
with the exact NCP gap if the backend cannot supply the data.

---

## Combined single-run demo (one prompt, both use cases)

> Build the mortgage statement API end to end from the NCP document and WSDL in docs/
> (BIAN: Document Services / Mortgage statement), then add an account-history hyperlink
> and side panel to the statements UI with view and download per statement, and give me
> the QA artifacts and gate results for both.

Expected: Phase 0-5 for the API, then impact → dev → qa for the UI, ending with a summary
table of artifacts + all gates green.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "No agents found in .copilot/agents" | `node vscode-extension/sdlc/scripts/install-agents.mjs` |
| Copilot CLI not found | set `sdlc.copilotCliPath` or add `copilot` to PATH |
| Gate fails RULE3 (VPC) | copy `infra/config/index.ts.example` → `index.ts`, fill values from platform team (never commit) |
| UI shows generic error | mock server not running: `cd ui && node mock-server.js` |
