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

Milestone 4 now has a working multimodal-retrieval MVP slice on top of the
Milestone 3 hardened retrieval base. It features:
- Pre-indexing privacy gate ensuring zero unreviewed PII reaches retrieval stores.
- Deterministic UUIDv5 chunk identities derived from tenant, full document ID, version, and content hash.
- Fail-closed BGE-M3 embedding provider with dynamic dimension verification against Qdrant schema.
- Failure-safe reindexing with authoritative `active_index_version`, point verification, and non-blocking stale point cleanup.
- Reciprocal Rank Fusion (RRF k=60) with mandatory server-enforced tenant filtering.
- Verifiable evidence citations with full 64-character SHA-256 evidence digests.
- Unsalted query privacy auditing (query length and duration only; zero query text or digests stored).
- Separated benchmarks: deterministic synthetic pipeline evaluation vs real-model integration.
- Interactive Intelligence console in Next.js with real-time vector telemetry.
- Encrypted page images captured during local extraction.
- Privacy-masked visual captions and table/form or diagram/map region metadata attached to indexed chunks.
- Tenant-scoped visual evidence endpoint that serves page images only for active retrieved chunks.
- Evidence viewer in the Intelligence console with cited-region highlighting.

Qwen3-VL answer generation over authorized top-ranked images remains the next
generation-layer task.

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
