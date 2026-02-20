# ClaimClaw

ClaimClaw is a Python-first, multi-agent system to contest rejected Indian health insurance claims and escalate quickly under IRDAI rules.

## What this build includes

1. Sprint 1 - Legal Brain (RAG):
- Local legal corpus ingestion (`FAISS` or `Chroma`).
- Retrieval utilities for IRDAI circulars and grievance material.
- Hard guardrail for the April 2024 5-year moratorium position.

2. Sprint 2 - Evidence Matcher:
- PDF extraction with `PyMuPDF`.
- Vision-ready bill parsing hook for itemized hospital invoices.
- Contestable rejection rules (including non-medical vs surgery consumables mismatch).

3. Sprint 3 - The Claw (Playwright):
- Playwright automation scaffold for Bima Bharosa complaint filing.
- Semantic locator-first strategy (`get_by_role`, `get_by_label`).
- Explicit human-in-the-loop CAPTCHA pause.

4. Sprint 4 - Coordinated Attack (LangGraph):
- State-machine workflow: analyze docs -> draft rebuttal -> wait -> escalate.
- SQLite-backed checkpointing with `SqliteSaver`.
- Resume support after restart via `thread_id`.

## Project structure

```text
ClaimClaw/
  claimclaw/
    agents.py
    cli.py
    config.py
    evidence_matcher.py
    legalbrain/
      rules.py
      scraper.py
      ombudsman_kb.py
    legal_rag.py
    portal_automation.py
    prompts.py
    schemas.py
    whatsapp_interface.py
    workflow.py
  scripts/
    set-secrets.sh
    load-secrets.sh
  data/
    legal/          # put IRDAI circulars and guideline PDFs here
    claims/         # put claim PDFs here
  tests/
  requirements.txt
  .env.example
```

## Setup

```bash
cd "/Users/bimalbairagya/Desktop/ClaimClaw"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

## Secure API Key Setup (Recommended)

Store secrets in macOS Keychain instead of plaintext files.

1. Save keys once:

```bash
cd "/Users/bimalbairagya/Desktop/ClaimClaw"
./scripts/set-secrets.sh
```

2. Load keys for current shell before running ClaimClaw:

```bash
cd "/Users/bimalbairagya/Desktop/ClaimClaw"
source ./scripts/load-secrets.sh
```

3. Run commands normally:

```bash
.venv/bin/python -m claimclaw.cli analyze-docs \
  --policy-document data/claims/Policy_Document.pdf \
  --rejection-letter data/claims/Rejection_Letter.pdf \
  --discharge-summary data/claims/Discharge_Summary.pdf \
  --hospital-bill data/claims/Hospital_Bill.pdf
```

## Alternative Environment File

If you use `.env`, never commit it. This repo includes `.gitignore` rules for `.env` and `*.env`.

Runtime gates:
- `STRICT_LLM_MODE=true` (default): forensic LLM extraction is mandatory.
- `DEV_ALLOW_FALLBACK=false` (default): prevents silent fallback in strict mode.
- Use `DEV_ALLOW_FALLBACK=true` only for local offline debugging.

## Environment Preflight

Run a full environment diagnostic before claim analysis:

```bash
.venv/bin/python -m claimclaw.cli doctor
```

Fail on warnings (strict check):

```bash
.venv/bin/python -m claimclaw.cli doctor --strict
```

## Quickstart

1. Ingest legal corpus:

```bash
.venv/bin/python -m claimclaw.cli ingest-legal \
  --source-dir data/legal \
  --persist-dir storage/legal_index \
  --store chroma
```

2. Validate moratorium logic:

```bash
.venv/bin/python -m claimclaw.cli validate-moratorium \
  --policy-age 6 \
  --rejection-reason "hidden BP issue / non-disclosure"
```

3. Query ingested legal corpus:

```bash
.venv/bin/python -m claimclaw.cli ask-legal \
  --persist-dir storage/legal_index \
  --store chroma \
  --question "A 6-year-old policy was rejected for a hidden BP issue. What clause counters this?"
```

4. Analyze claim documents:

```bash
.venv/bin/python -m claimclaw.cli analyze-docs \
  --policy-document data/claims/Policy_Document.pdf \
  --rejection-letter data/claims/Rejection_Letter.pdf \
  --discharge-summary data/claims/Discharge_Summary.pdf \
  --hospital-bill data/claims/Hospital_Bill.pdf
```

5. Run live LegalBrain sync (IRDAI + Ombudsman, in-memory embeddings):

```bash
.venv/bin/python -m claimclaw.cli live-legal-brain \
  --include-2025-updates true \
  --max-ombudsman-pages 20 \
  --question "Repudiation for pre-existing disease after moratorium - strongest rebuttal path?"
```

6. Run date-aware moratorium override check:

```bash
.venv/bin/python -m claimclaw.cli check-moratorium-date \
  --policy-start-date 2018-01-15 \
  --as-of-date 2026-02-17 \
  --rejection-reason "Repudiation due to non-disclosure of PED"
```

7. Run full workflow (dry-run portal filing):

```bash
.venv/bin/python -m claimclaw.cli run-workflow \
  --claim-id CLM-001 \
  --policy-age 6 \
  --policy-document data/claims/Policy_Document.pdf \
  --rejection-letter data/claims/Rejection_Letter.pdf \
  --discharge-summary data/claims/Discharge_Summary.pdf \
  --hospital-bill data/claims/Hospital_Bill.pdf \
  --dry-run-portal true
```

## Notes

- Keep final legal review by a qualified professional in production.
- CAPTCHA is intentionally not automated.
- Live LegalBrain uses Playwright for IRDAI page rendering and BeautifulSoup for lightweight parsing.
- Live session embeddings are stored in an in-memory Chroma collection for the current process only.
- `/data/legal` may remain empty when operating in Live-Crawl mode.
