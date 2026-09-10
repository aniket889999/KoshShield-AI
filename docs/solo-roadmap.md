# Solo development roadmap

## Milestone 0: foundation

- Next.js and FastAPI applications run locally.
- PostgreSQL and Qdrant run through Docker Compose.
- Backend exposes health, readiness, and offline-system status.
- Frontend displays real service state without fake metrics.
- CI-quality local linting and tests are documented.

## Milestone 1: encrypted document intake

- PDF/JPEG/PNG signature and size validation.
- SHA-256 evidence hash and AES-256-GCM vault encryption.
- Document metadata and tamper-evident audit event.
- No plaintext original remains in application storage.

## Milestone 2: extraction and privacy review (completed)

- Native PDF extraction with PyMuPDF and real local PaddleOCR adapter.
- Deterministic Indian PII recognizers (Aadhaar with Verhoeff checksum, PAN, mobile phone, email, bank account, IFSC, passport, employee/gov ID).
- Redaction review workspace in Next.js console with masked context preview.
- Optimistic concurrency locking (version field) and bulk accept high-confidence findings.
- Raw text stored in encrypted vault only; database contains only salted hashes and masked text.
- Indexing remains strictly blocked until all redactions are approved.

## Milestone 3: masked hybrid RAG (completed & hardened)

- Official `qdrant-client` adapter (version-pinned, `prefer_grpc=False`) with typed models and local-only URL enforcement.
- Deterministic token-aware chunking preserving page boundaries and `[REDACTED_*]` tags.
- Deterministic UUIDv5 chunk identities derived from tenant, untruncated doc ID, version, page, sequence, and content hash.
- Pre-indexing privacy gate verifying zero unresolved findings and re-scanning masked text.
- Fail-closed local BGE-M3 embedding provider with dynamic dimension detection and tri-point schema verification.
- Failure-safe reindexing sequence: chunk -> embed -> verify -> upsert -> point verify -> atomic DB activation -> stale point cleanup.
- Authoritative `active_index_version` ensuring old versions remain intact if reindexing fails.
- Reciprocal Rank Fusion (RRF $k=60$) hybrid search with server-enforced tenant isolation on all vector operations.
- Cryptographic evidence citations with full 64-character SHA-256 evidence digests and masked content hashes.
- Unsalted query privacy auditing (recording query character length and execution duration, never query text or SHA-256 digests).
- Accurate benchmark separation: deterministic synthetic pipeline evaluation vs real-model integration (labeled NOT EXECUTED if weights/containers absent).
- Interactive Intelligence console in Next.js with vector store telemetry, corpus indexing, and search.

## Hardening Checkpoints: Tenant Isolation & Migration Correctness (Completed)

- **Checkpoint 1 & 1.1**:
  - Authoritative non-null tenant ownership on documents and audit records (no database or ORM default fallbacks).
  - Mandatory `tenant_id` on document ingestion, audit events, and vector store telemetry.
  - Safe Alembic database migration management via `make migrate` supporting pre-Alembic database bootstrap and schema validation.
  - Startup schema validation in application lifespan that fails closed if the database is uninitialized or behind head.
  - Dual-version tamper-evident audit hash algorithm (`v1` for legacy records, `v2` for tenant-aware records) preserving legacy audit chains without recomputing historical hashes.
  - Centralized `RequestContext` dependency that permits header-derived identity strictly in demo mode and fails closed (HTTP 401) in production when verified authentication is absent.
  - Demo role-based access control (RBAC) enforcing reviewer, approver, executor, auditor, and admin roles, with `X-Roles` permitted in CORS.
  - Strict 404 response on cross-tenant document, extraction, review, indexing, retrieval, visual evidence, audit, and agent operations to prevent resource enumeration.
  - Tenant-scoped retrieval telemetry (`GET /retrieval/status`).
- **Checkpoint 1.2: Strict Schema Fingerprint Validation (Completed)**:
  - Explicit dialect-aware schema contracts defining all 9 application tables, columns, primary keys, nullabilities, foreign keys, unique constraints, and indexes for legacy 0001 and current head schemas.
  - Complete rejection of unmanaged partial databases, missing tables, missing columns, unexpected columns, incompatible constraints, and schema drift.
  - Fail-safe startup and migration validation verifying both Alembic revision and actual physical schema layout without modifying database or creating alembic_version on validation failure.
  - Reused validator across `bootstrap_and_upgrade`, direct Alembic execution (`env.py`), and application startup (`check_schema_at_head`).
- **Checkpoint 2: Active-Index Retrieval & Embedding Runtime Hardening (Completed)**:
  - Strict document-level `active_index_version` filtering during vector search and point retrieval, ensuring stale vector points are never retrieved.
  - Multi-threaded singleton embedding provider initialization protecting local model instances.
  - Fail-closed validation for sparse and dense BGE-M3 representations, rejecting non-integer token keys.
  - Verifiable idempotent same-version reindexing no-op preserving active chunks and vector points.
  - Fail-closed Qdrant telemetry and count tracking.
- **Checkpoint 2.1: Retrieval Lifecycle and Sanitization Closure (Completed)**:
  - Universal error sanitization across retrieval and health routes, preventing leakage of internal filesystem paths, Qdrant URLs, vault keys, queries, and raw exception messages.
  - Irreversible database activation transaction boundary (`DOCUMENT_INDEXED`) guaranteeing active generation points are protected from deletion or rollback.
  - Tenant-isolated point operations in Qdrant adapter using compound `HasIdCondition` + `FieldCondition` filters batched safely in 256-point chunks with defensive payload validation.
  - Strict payload-index schema validation verifying index presence and schema data types.
  - Clarified vector store / indexing service contracts establishing authoritative active-version ownership in the indexing service.
- **Checkpoint 2.2: Crash-Recoverable Deferred Index Cleanup (Completed)**:
  - Crash-safe cleanup marker persistence: atomically commits `index_cleanup_pending=True` alongside the active generation during reindexing.
  - Safe post-activation finalization: deletes stale generations and clears marker in a separate commit; failures preserve `index_cleanup_pending=True` without affecting active searchability.
  - Tenant-scoped reconciliation service (`reconcile_pending_cleanups`) re-reading authoritative active versions, safely skipping invalid records, and isolating per-document failures.
  - Admin-protected operational endpoint (`POST /api/v1/retrieval/cleanup-pending`) enforcing admin RBAC, production authentication barriers, tenant isolation, and bounded limits.
  - Sanitized `INDEX_CLEANUP_COMPLETED` and `INDEX_CLEANUP_FAILED` audit logging storing only tenant ID, document ID, active version, and stable failure codes.

## Milestone 4: multimodal retrieval (implemented and security hardened - live integration not executed)

- Encrypted page images are captured during local extraction for PDFs and image uploads.
- Masked visual region records are generated only after human redaction approval.
- Table/form and diagram/map regions are described through privacy-masked local captions.
- Indexed chunks include visual captions for caption-enriched retrieval without storing raw visual text.
- Page images are served only through a tenant-scoped evidence endpoint tied to an active retrieved chunk.
- Intelligence console can open authorized page evidence and highlight the cited region.
- **Checkpoint 3.1: Multimodal Privacy and Runtime Security Closure**:
  - Encrypted masked-page-image vault derivatives created only upon explicit operator approval.
  - Fail-closed redaction verification: any unlocated PII or post-masking OCR detection blocks visual evidence (`BLOCKED_UNLOCATED_PII`).
  - Evidence images served exclusively as redacted derivatives, never original images.
  - Strict input-size and dimension limits applied before decode and conversion.
- **Checkpoint 3.2: Truthful llama.cpp Runtime Contract & Multimodal Security Closure**:
  - Target contract pinned to **llama.cpp release `v0.4.0` (build `b10809`, commit `5266f24`)** and model alias `qwen3-vl-4b-instruct`.
  - Authoritative fail-closed runtime preflight: checks `GET /v1/models` for exact model alias and `GET /props?model=...` for native `modalities.vision: true` and matching `build_info`.
  - Strict Pydantic models for complete chat completion response (`choices`, `message`, `content`, `usage`) with extra fields forbidden and non-negative token counts; fails closed with sanitized HTTP 502 (`MODEL_RESPONSE_INVALID`).
  - Hardened Pillow image decoding: request byte limits before decode, dimension verification before EXIF transpose and RGB conversion, and `DecompressionBombError` handling.
  - Sanitized log events and API errors: raw exception interpolation, file paths, queries, and model output are completely removed.
  - **Truthful integration notice**: Milestone 4 multimodal code path is fully implemented and tested with contract fixtures. Real Qwen3-VL plus llama.cpp integration remains **NOT EXECUTED** due to the intentional absence of model weights and local inference server processes. Local setup requires `--model` and `--mmproj` flags only; remote downloads, `-hf`, and external APIs are strictly prohibited.

## Milestone 5: policy-gated agent (prototype - security hardening in progress)

- Explicit LangGraph policy states are persisted with every run.
- Independent human approvals use optimistic version checks and prevent self-approval.
- Safe bounded calculator and indexed-document report-generator tools.
- Docker execution disables networking, uses a read-only root filesystem, drops all Linux
  capabilities, applies CPU/memory/PID/time limits, and runs as an unprivileged user.
- Tool arguments use a keyed integrity hash; outputs are schema-validated, size-bounded, and
  cryptographically hashed before storage.
- Shell, filesystem, SQL, browser, network, Python, unknown, and malformed actions are rejected;
  prohibited arguments are withheld from storage while their hash is audited.
- Tenant-scoped agent APIs and an operational approval queue are available in the web console.

## Milestone 6: graph-assisted retrieval

- Entity, relationship, mention, and claim extraction.
- Relationship queries combine graph paths with text evidence.
- Full Microsoft GraphRAG remains an optional later adapter.

## Milestone 7: release hardening

- Offline dependency/model bundle and pinned checksums.
- Security, PII, retrieval, policy, and end-to-end test suites.
- Structured logs, metrics, backup/restore, and operational runbook.
- Demo dataset contains synthetic PII only.

## Demo acceptance path

```text
Disconnect internet
  -> upload a synthetic confidential document
  -> inspect/approve redactions
  -> ask a question and open its page citation
  -> request a calculation
  -> approve execution
  -> reject a prohibited action
  -> verify the audit chain
```
