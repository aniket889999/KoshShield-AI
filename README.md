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
boundary hardening. Completed checkpoints:
- **Checkpoint 1 & 1.1**: Authoritative non-null tenant ownership on documents and audit records, safe Alembic database migrations, dual-version tamper-evident audit hash algorithm (`v1`/`v2`), centralized `RequestContext` failing closed in production, demo RBAC, cross-tenant isolation (strict 404), and tenant-scoped retrieval telemetry.
- **Checkpoint 1.2**: Strict schema fingerprint validation across all application tables.
- **Checkpoint 2 & 2.1 & 2.2**: Active-index retrieval filtering, crash-safe deferred index cleanup, universal error sanitization, and idempotent reindexing.
- **Checkpoint 3.1 & 3.2: Multimodal Security Closure & Truthful Runtime Contract**:
  - Pinned supported target runtime contract to **llama.cpp release `v0.4.0` (build `b10809`, commit `5266f24`)** with model alias `qwen3-vl-4b-instruct` and local GGUF/mmproj files only.
  - Authoritative fail-closed runtime preflight verifying exact model alias via `GET /v1/models` and native `modalities.vision: true` and `build_info` via `GET /props?model=...`.
  - Strict Pydantic schema validation on complete model completions (`choices`, `message`, `content`, `usage`) with extra fields forbidden and non-negative token counts; maps all validation errors to `LlamaCppResponseInvalidError` and sanitized HTTP 502.
  - Pillow decompression-bomb defense and pre-conversion dimension checks (`max_image_dimension`).
  - Privacy-safe visual evidence: unlocated PII blocks visual evidence (`BLOCKED_UNLOCATED_PII`), and only operator-approved redacted derivatives are served or fed to inference.
  - Universal log and error sanitization: raw exceptions, tracebacks, queries, paths, OCR text, and model prompts/answers are never exposed or logged.
  - **Truthful runtime integration status**: Real Qwen3-VL plus llama.cpp integration remains **NOT EXECUTED** because server processes and multi-gigabyte model weights are intentionally absent in the development repository. Local server startup must use local `--model` and `--mmproj` files only, never `-hf`, external URLs, or remote model APIs.

Graph-assisted retrieval, live model inference execution, and gRPC interfaces remain paused
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
