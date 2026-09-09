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

Milestones 4 and 5 are currently functional prototypes undergoing active security
boundary hardening. Checkpoint 1 enforces:
- Authoritative non-null tenant ownership on documents and audit records.
- Centralized `RequestContext` dependency that permits header-derived identity strictly in demo mode and fails closed (HTTP 401) in production when verified authentication is absent.
- Strict 404 response on cross-tenant document, extraction, review, indexing, retrieval, visual evidence, audit, and agent operations to prevent resource enumeration.
- Qdrant tenant payloads derived authoritatively from stored document owners.
- Tenant-isolated tamper-evident audit trails.
- Agent `document_report` tool bound strictly to the run tenant.
- Alembic database migration management supporting safe upgrades on existing and fresh schemas.

Graph-assisted retrieval, Qwen3-VL multimodal generation, and gRPC interfaces remain paused
until all security boundaries and hardening checkpoints are completed and verified.

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
make tool-runner-build
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
