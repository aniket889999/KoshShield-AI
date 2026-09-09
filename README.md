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
boundary hardening. Checkpoint 1 and Checkpoint 1.1 enforce:
- Authoritative non-null tenant ownership on documents and audit records (no database or ORM default fallbacks).
- Mandatory `tenant_id` on document ingestion, audit events, and vector store telemetry.
- Safe Alembic database migration management via `make migrate` supporting pre-Alembic database bootstrap and schema validation.
- Startup schema validation in application lifespan that fails closed if the database is uninitialized or behind head.
- Dual-version tamper-evident audit hash algorithm (`v1` for legacy records, `v2` for tenant-aware records) preserving legacy audit chains without recomputing historical hashes.
- Centralized `RequestContext` dependency that permits header-derived identity strictly in demo mode and fails closed (HTTP 401) in production when verified authentication is absent.
- Demo role-based access control (RBAC) enforcing reviewer, approver, executor, auditor, and admin roles, with `X-Roles` permitted in CORS.
- Strict 404 response on cross-tenant document, extraction, review, indexing, retrieval, visual evidence, audit, and agent operations to prevent resource enumeration.
- Tenant-scoped retrieval telemetry (`GET /retrieval/status`).
- Real offline integration tests requiring live Docker Qdrant or local BGE-M3 weights skip cleanly when those optional local services are absent.

Graph-assisted retrieval, Qwen3-VL multimodal generation, and gRPC interfaces remain paused
until all security boundaries and hardening checkpoints are completed and verified.

## Local quick start

Prerequisites: Python 3.12, pnpm 11, and Docker Desktop for PostgreSQL and
Qdrant. The API can use SQLite when the container services are not running.

```bash
make bootstrap
cp .env.example .env
make generate-key
make migrate
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
