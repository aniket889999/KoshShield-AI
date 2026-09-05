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

## Milestone 4: multimodal retrieval (MVP slice completed)

- Encrypted page images are captured during local extraction for PDFs and image uploads.
- Masked visual region records are generated only after human redaction approval.
- Table/form and diagram/map regions are described through privacy-masked local captions.
- Indexed chunks include visual captions for caption-enriched retrieval without storing raw visual text.
- Page images are served only through a tenant-scoped evidence endpoint tied to an active retrieved chunk.
- Intelligence console can open authorized page evidence and highlight the cited region.
- Qwen3-VL answer generation remains future work; the implemented boundary ensures it receives only authorized visual evidence when added.

## Milestone 5: policy-gated agent

- Explicit LangGraph states and persisted approvals.
- Safe calculator and report-generator tools.
- Network-disabled Docker execution and output verification.
- Prohibited actions are rejected and audited.

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
