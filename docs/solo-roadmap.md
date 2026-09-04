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

## Milestone 3: masked hybrid RAG (completed)

- Deterministic token-aware chunking preserving page boundaries and `[REDACTED_*]` tags.
- Pre-indexing privacy gate verifying zero unresolved findings and re-scanning masked text.
- BGE-M3 dense (1024-d) and lexical sparse embeddings with offline deterministic fallbacks.
- Qdrant vector store indexing with named vectors (`text_dense`, `text_sparse`) and payload indexes.
- Reciprocal Rank Fusion (RRF $k=60$) hybrid search with tenant isolation and document authorization filters.
- Verifiable citation generation format: `[Document: <id> | Page: <n> | Evidence: <hash:12>]`.
- Privacy-safe query auditing (recording query SHA-256 and result counts, never query text).
- Golden-query retrieval evaluation benchmark achieving Recall@5: 1.0, MRR: 1.0, and 0 cross-tenant leaks.
- Intelligence workspace in Next.js console with vector store telemetry, corpus indexing, and search with citation copy.

## Milestone 4: multimodal retrieval

- Page images and table/diagram crops.
- Local visual descriptions and visual-caption retrieval.
- Qwen3-VL receives only authorized top-ranked images.
- Evidence viewer highlights the cited page region.

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
