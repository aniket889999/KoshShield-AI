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

## Milestone 2: extraction and privacy review

- Native PDF extraction with OCR fallback.
- Aadhaar, PAN, phone, email, account, and employee-ID recognizers.
- Page preview with editable redaction decisions.
- Indexing remains blocked until redactions are approved.

## Milestone 3: masked hybrid RAG

- BGE-M3 dense/sparse embeddings and Qdrant fusion.
- Tenant/document authorization filters.
- Answers include page/chunk/evidence citations.
- Golden-query retrieval and groundedness evaluation.

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
