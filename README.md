# KoshShield AI

KoshShield AI is a sovereign, air-gapped document intelligence workbench for
Indian public-sector and PSU workflows. It processes confidential documents
locally, masks sensitive information before retrieval, produces cited answers,
gates agent actions behind policy and human approval, and records a complete
audit trail.

## MVP outcome

```text
Upload -> OCR -> PII review -> masked index -> local RAG
       -> approval-gated tool -> verified output -> audit trail
```

## Architecture

- Next.js operational interface
- FastAPI modular backend
- PostgreSQL for metadata, relationships, approvals, and audit events
- Qdrant for masked hybrid retrieval
- Docling and PaddleOCR for document extraction
- Presidio plus Indian-identifier recognizers for PII detection
- BGE-M3 for multilingual retrieval
- Qwen3-VL-4B served by llama.cpp for local multimodal generation
- LangGraph for explicit approval-aware workflows
- Network-disabled Docker containers for restricted tools

The implementation plan and architecture boundaries are documented in
[`docs/architecture.md`](docs/architecture.md) and
[`docs/solo-roadmap.md`](docs/solo-roadmap.md).

## Development status

Milestone 2 is implemented: air-gapped document extraction (PyMuPDF native text
with real local PaddleOCR fallback), deterministic Indian PII recognition (Aadhaar
with Verhoeff checksum, PAN, phone, email, bank accounts, IFSC, passport, government
IDs), encrypted vault artifact storage, and a human-in-the-loop redaction review
workspace in the Next.js console. Indexing is strictly gated behind human
redaction approval. Masked BGE-M3/Qdrant retrieval is the next milestone.

## Local quick start

Prerequisites: Python 3.12, pnpm 11, and Docker Desktop for PostgreSQL and
Qdrant. The API can use SQLite when the container services are not running.

```bash
make bootstrap
cp .env.example .env
make generate-key
```

Copy the generated value into `KOSHSHIELD_MASTER_KEY_BASE64` in `.env`, then
start the infrastructure and applications in separate terminals:

```bash
make infra-up
make dev-api
make dev-web
```

Open `http://localhost:3000`. API documentation is available at
`http://localhost:8000/docs`. A local llama.cpp server is optional in this
milestone; the dashboard reports it as unavailable until it is running.

## Verification

```bash
make test
make lint
pnpm --filter @koshshield/web build
docker compose config --quiet
```

## Agent handoff

When continuing development with Gemini in Antigravity IDE, use
[`GEMINI_BUILD_PROMPT.md`](GEMINI_BUILD_PROMPT.md) as the master prompt.
