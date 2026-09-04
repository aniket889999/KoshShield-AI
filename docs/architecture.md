# Architecture decision record

## Decision

KoshShield AI uses a modular monolith for the MVP. The browser communicates with
a FastAPI backend through REST and server-sent events. Long-running ingestion
work executes in a separate worker process from the same Python codebase. Model
inference runs in a separately managed llama.cpp process through its local
OpenAI-compatible HTTP interface.

This shape gives a solo developer clear ownership boundaries without the
operational cost of premature microservices.

## Runtime topology

```text
Next.js UI
   | REST + SSE
FastAPI application
   |-- identity and RBAC
   |-- document intake and redaction review
   |-- retrieval and query routing
   |-- agent policy and approvals
   `-- audit and system status
        |-- PostgreSQL
        |-- Qdrant
        |-- encrypted local vault
        |-- ingestion worker
        |-- llama.cpp / Qwen3-VL-4B
        `-- restricted Docker tool runner
```

## Storage boundaries

### Encrypted original vault

Original document bytes are encrypted with AES-256-GCM before being written to
the vault. PostgreSQL stores metadata and the encrypted object location, not the
encryption key. The MVP reads a master key from a protected environment/file;
production replaces this with envelope encryption backed by an HSM or KMS.

### Masked intelligence store

PostgreSQL and Qdrant may contain only approved, privacy-masked content. Every
chunk carries tenant ID, document ID, page number, section, classification,
redaction version, and evidence hash. Retrieval always applies authorization
filters before similarity ranking.

### Audit store

Audit events contain normalized metadata and hashes rather than raw prompts or
PII. Each event includes the previous event hash to make later modification
detectable. This is tamper-evident, not an immutable-ledger claim.

## Retrieval architecture

The default path is hybrid multimodal RAG:

1. Dense and sparse masked-text retrieval.
2. Reciprocal-rank fusion and optional local reranking.
3. Visual-caption retrieval for tables, diagrams, and scanned pages.
4. Original page/crop loading only after authorization and retrieval.
5. Local multimodal generation with evidence citations.

Graph-assisted retrieval is a secondary path for relationship questions. The
MVP stores entities, relationships, mentions, and claims in PostgreSQL. Full
GraphRAG community extraction and summarization is deferred because it is
indexing-heavy and unnecessary for the core demo.

## API architecture

- REST: uploads, metadata, redaction decisions, queries, approvals, and audit.
- SSE: ingestion progress, query tokens, and agent-run state changes.
- Internal HTTP: FastAPI to llama.cpp.
- Future gRPC: only after a service is independently deployed and requires a
  typed, streaming, cross-language contract.

## Agent safety

The model proposes structured actions but never executes them. A deterministic
policy engine validates the actor, tool, document classification, arguments,
resource limits, and approval requirement. Approved execution occurs in a
network-disabled container with a read-only root filesystem and strict limits.

Document text is always untrusted context. Instructions found inside documents
cannot modify system policy or authorize tools.

## Deployment evolution

1. Docker Compose on one workstation.
2. Hardened departmental node with OIDC, TLS, backups, and monitoring.
3. Multiple inference/worker nodes with gRPC where justified.
4. HA private cluster and SIEM/HSM integration after a successful pilot.
