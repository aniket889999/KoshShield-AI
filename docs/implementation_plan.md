# Implementation Plan: Milestone 3 Security-Hardening Pass

This plan addresses all 11 corrective requirements for Milestone 3 (Masked Hybrid Retrieval with Qdrant and Verifiable Citations), hardening security boundaries, eliminating unverified claims, and integrating the official `qdrant-client`.

## Security & Architecture Specifications

> [!IMPORTANT]
> **Key Architectural & Security Corrections:**
> 1. **Official `qdrant-client` Library**: Replace custom HTTP REST client in `QdrantVectorStore` with official `qdrant_client.QdrantClient` (`prefer_grpc=False`), typed models (`VectorParams`, `Distance`, `SparseVectorParams`, `PointStruct`, `Filter`, `FieldCondition`, `MatchValue`, `MatchAny`, `PayloadSchemaType`), and strict local URL validation (`localhost`/`127.0.0.1`/`::1` or docker container hostname).
> 2. **Removal of Unsalted SHA-256 Query Auditing**: Remove deterministic query SHA-256 digests from audit records to prevent dictionary re-identification of predictable queries. The audit event will store strictly metadata: `actor_id`, `tenant_id`, `query_length`, `top_k`, `permitted_document_ids`, `classification`, `result_chunk_ids`, `result_count`, `duration_ms`, and `policy_result`. `RetrievalResponse` will return `query_length` instead of `query_hash`.
> 3. **Production Embedding Fail-Closed Enforcement**: Prohibit fake or fallback embeddings outside explicit test fixtures. In production, if local BGE-M3 weights are missing or uninitialized, fail closed immediately with HTTP 503 (`ModelUnavailableError`).
> 4. **Dynamic Dense Dimension & Tri-Point Verification**: Do not trust `config.json` alone:
>    - Read expected dimension from local configuration (`config.json` or settings).
>    - Validate it against the first real model embedding output vector shape.
>    - Validate both against the existing Qdrant collection schema (`text_dense` dimension and Cosine distance).
>    - Fail explicitly on any mismatch without silently recreating or deleting the collection.
> 5. **Deterministic UUIDv5 Chunk & Point Identities**: Derive chunk IDs using `uuid.uuid5` with a dedicated namespace from `(tenant_id, document_id, index_version, page_number, chunk_sequence, masked_content_hash)`. Full document ID is preserved without truncation. Reindexing identical content preserves IDs; version or text changes produce new IDs.
> 6. **Unambiguous Reindex Activation & Failure Safety**:
>    - Add authoritative `active_index_version` to `DocumentRecord` and `index_version` to `DocumentChunkRecord`.
>    - Include `index_version` in every Qdrant point payload.
>    - Filter search, count, and scroll operations strictly by active version.
>    - Sequence: State/Gate -> Chunk -> Embed -> Dimension Verification -> Upsert new version -> Verify points -> Activate version in DB atomically -> Delete stale points (`index_version < active_version`).
>    - If stale deletion fails, preserve the valid active index and record cleanup as pending instead of corrupting document state.
>    - If any failure occurs prior to activation, keep previous valid index intact and mark document `INDEX_FAILED`.
> 7. **Mandatory Tenant Authorization**: The `X-Tenant-ID` header is a demo security boundary, not production authentication (documented explicitly in codebase and API schema). Every Qdrant call (search, count, scroll, point retrieval, stale deletion, verification) must include a mandatory tenant filter.
> 8. **Full 64-char Cryptographic Evidence Hash**: Return complete 64-character SHA-256 evidence hash in API responses (`evidence_hash`) along with `masked_content_hash`. The short 12-char string is strictly designated for display formatting (`citation_label`).
> 9. **Benchmark Claims Separation**: Accurately label test suite results as "Deterministic synthetic pipeline evaluation". Real BGE-M3 + Docker Qdrant benchmark will run only if weights and container are available; otherwise explicitly reported as `NOT EXECUTED`.
> 10. **Air-Gap Verification**: Verify `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`, reject URLs or Hugging Face repo IDs, and enforce socket monkeypatch test.

---

## Proposed Changes

### Component 1: Qdrant Vector Store & Dependencies

#### [MODIFY] [pyproject.toml](file:///Users/aniket/Downloads/KoshShield%20AI/apps/api/pyproject.toml)
- Ensure `"qdrant-client>=1.15.0,<1.16.0"` is pinned in dependencies.

#### [MODIFY] [qdrant.py](file:///Users/aniket/Downloads/KoshShield%20AI/apps/api/src/koshshield/services/retrieval/vector_store/qdrant.py)
- Replace custom `httpx` implementation with `qdrant_client.QdrantClient(url=..., timeout=..., prefer_grpc=False)`.
- Use typed models:
  - `VectorParams`, `Distance.COSINE`, `SparseVectorParams` for collections.
  - `PointStruct` for point upserts.
  - `Filter`, `FieldCondition`, `MatchValue`, `MatchAny` for tenant and document filtering.
  - `PayloadSchemaType` for payload indexes.
- Enforce local URL validation (must be localhost / 127.0.0.1 / ::1 or local container hostname; reject external hostnames/IPs).
- Implement `ensure_collection(dense_dim: int)`:
  - Inspect existing collection. If vector name, distance, or dimension differs, raise `VectorStoreError` without deleting or recreating the collection.
- Implement `verify_points(point_ids: list[str], tenant_id: str) -> bool`:
  - Retrieve points and verify they belong to `tenant_id`.
- Implement `delete_stale_chunks(document_id: str, tenant_id: str, active_version: int) -> int`:
  - Delete only points where `document_id == document_id`, `tenant_id == tenant_id`, and `index_version < active_version`.
- Implement `retrieve_points(point_ids: list[str], tenant_id: str) -> list[VectorStoreSearchResult]`.

#### [MODIFY] [interfaces.py](file:///Users/aniket/Downloads/KoshShield%20AI/apps/api/src/koshshield/services/retrieval/vector_store/interfaces.py)
- Update `VectorStore` protocol to specify `verify_points`, `delete_stale_chunks`, and `retrieve_points` with mandatory `tenant_id`.

#### [MODIFY] [in_memory.py](file:///Users/aniket/Downloads/KoshShield%20AI/apps/api/src/koshshield/services/retrieval/vector_store/in_memory.py)
- Update `InMemoryVectorStore` to support `verify_points`, `delete_stale_chunks`, and schema checks matching the protocol.

---

### Component 2: Chunking & Identity Generation

#### [MODIFY] [chunking.py](file:///Users/aniket/Downloads/KoshShield%20AI/apps/api/src/koshshield/services/retrieval/chunking.py)
- Replace arbitrary string hash chunk IDs with deterministic `uuid.uuid5(KOSHSHIELD_CHUNK_NAMESPACE, ...)` derived from:
  `tenant_id`, `document_id` (full, untruncated), `index_version`, `page_number`, `chunk_sequence`, and `masked_content_hash`.
- Guarantee reindexing identical content preserves IDs, while incrementing redaction version or altering text generates new IDs.

---

### Component 3: Metadata Schema, Indexing Service & Failure Safety

#### [MODIFY] [models.py](file:///Users/aniket/Downloads/KoshShield%20AI/apps/api/src/koshshield/models.py)
- Add `active_index_version = Column(Integer, nullable=True)` and `index_cleanup_pending = Column(Boolean, default=False)` to `DocumentRecord`.
- Add `index_version = Column(Integer, nullable=False, default=1)` to `DocumentChunkRecord`.

#### [MODIFY] [indexing_service.py](file:///Users/aniket/Downloads/KoshShield%20AI/apps/api/src/koshshield/services/retrieval/indexing_service.py)
- Reorder and enforce fail-safe indexing lifecycle:
  1. Validate document state (`INDEX_READY` / `REDACTION_APPROVED`) and privacy gate.
  2. Generate all chunks with current target `index_version = doc.version`.
  3. Generate embeddings via `embedding_provider.embed_texts()`.
  4. Validate dimension against first output vector shape and Qdrant collection schema.
  5. Upsert chunks into Qdrant (`vector_store.upsert_chunks()`).
  6. Verify points in Qdrant (`vector_store.verify_points()`).
  7. Atomically mark new version active in DB (`doc.active_index_version = doc.version`, persist chunk records).
  8. Delete stale points (`vector_store.delete_stale_chunks()`). If this fails, set `doc.index_cleanup_pending = True` without failing the indexing.
  9. Transition document to `INDEXED` and record `DOCUMENT_INDEXED` audit event.
- If failure occurs before step 7:
  - Do NOT delete existing points.
  - Mark document `INDEX_FAILED`.
  - Record `DOCUMENT_INDEXING_FAILED` audit event.

---

### Component 4: Embedding Provider & Air-Gap Compliance

#### [MODIFY] [bge_m3.py](file:///Users/aniket/Downloads/KoshShield%20AI/apps/api/src/koshshield/services/retrieval/embeddings/bge_m3.py)
- Remove hardcoded `dense_dim = 1024`. Read dimension dynamically from `config.json` (`hidden_size` or `dim`) and validate against model output vector shape.
- Enforce strict local directory validation: reject URLs (`http://`, `https://`) and Hugging Face repository IDs (strings containing `/` that are not local directories).
- Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_HUB_DISABLE_TELEMETRY=1`.
- Raise `ModelUnavailableError` on missing weights without falling back to any fake or remote provider.

---

### Component 5: Hybrid Retrieval, Citations & Unsalted Privacy Auditing

#### [MODIFY] [hybrid_search.py](file:///Users/aniket/Downloads/KoshShield%20AI/apps/api/src/koshshield/services/retrieval/hybrid_search.py)
- Remove `query_hash` from `EvidenceItem`, `RetrievalEvidencePack`, and audit logs.
- Search audit event `RETRIEVAL_QUERY_EXECUTED` records only:
  - `actor_id`
  - `tenant_id`
  - `query_length`
  - `top_k`
  - `permitted_document_ids`
  - `classification`
  - `result_chunk_ids`
  - `result_count`
  - `duration_ms`
  - `policy_result`
- Include full 64-character `document_evidence_hash` and `masked_content_hash` in `EvidenceItem`.
- Measure query `duration_ms` via `time.perf_counter()`.

#### [MODIFY] [schemas.py](file:///Users/aniket/Downloads/KoshShield%20AI/apps/api/src/koshshield/schemas.py)
- In `RetrievalResponse`, replace `query_hash: str` with `query_length: int` and `duration_ms: float`.
- In `RetrievalEvidenceItem`, add `masked_content_hash: str`.

#### [MODIFY] [retrieval.py](file:///Users/aniket/Downloads/KoshShield%20AI/apps/api/src/koshshield/api/routes/retrieval.py)
- Ensure production dependency `get_embedding_provider` returns only `BgeM3EmbeddingProvider` and raises 503 if unavailable.
- Resolve tenant strictly from `X-Tenant-ID` header (default `default`), never from request body. Document demo header boundary in docstrings.

---

### Component 6: Frontend & Telemetry Updates

#### [MODIFY] [api.ts](file:///Users/aniket/Downloads/KoshShield%20AI/apps/web/lib/api.ts)
- Update `RetrievalResponse` to include `query_length: number` and `duration_ms: number` instead of `query_hash`.
- Update `RetrievalEvidenceItem` to include `masked_content_hash: string`.

#### [MODIFY] [operations-console.tsx](file:///Users/aniket/Downloads/KoshShield%20AI/apps/web/components/operations-console.tsx)
- Replace display of `Query Hash` with `Query Length: {searchMutation.data.query_length} chars` and duration in ms.

---

### Component 7: Tests & Regression Verification

#### [NEW] [test_qdrant_client_adapter.py](file:///Users/aniket/Downloads/KoshShield%20AI/apps/api/tests/test_qdrant_client_adapter.py)
- Unit tests using `qdrant-client` verifying:
  - Local URL enforcement (rejection of external URLs/domains).
  - Proper typed models used for vectors, payload schemas, filters, and points.
  - Dimension mismatch raises `VectorStoreError` without deleting collection.
  - Mandatory tenant filtering in search, count, delete, and point verification.
- Optional Docker Qdrant live integration test (runs if port 6333 is responsive).

#### [NEW] [test_query_privacy_audit.py](file:///Users/aniket/Downloads/KoshShield%20AI/apps/api/tests/test_query_privacy_audit.py)
- Regression test proving neither the raw query nor its SHA-256 digest appears in DB audit records, logs, or serialized responses.

#### [NEW] [test_reindexing_safety.py](file:///Users/aniket/Downloads/KoshShield%20AI/apps/api/tests/test_reindexing_safety.py)
- Simulates failures at:
  1. Embedding generation
  2. Upsertion
  3. Verification
  4. Stale deletion
- Proves previous index and chunks remain intact when reindexing fails.

#### [NEW] [test_airgap_embeddings.py](file:///Users/aniket/Downloads/KoshShield%20AI/apps/api/tests/test_airgap_embeddings.py)
- Monkeypatches network sockets (`socket.socket`, `urllib`, `requests`, `httpx`) and proves BGE-M3 initialization attempts 0 outbound network calls.
- Verifies rejection of Hugging Face repo IDs and non-directory strings.

#### [MODIFY] [test_chunking.py](file:///Users/aniket/Downloads/KoshShield%20AI/apps/api/tests/test_chunking.py)
- Adds tests for UUIDv5 chunk ID stability across reindexing, uniqueness across different versions and tenants, and collision resistance.

#### [MODIFY] [test_retrieval_evaluation.py](file:///Users/aniket/Downloads/KoshShield%20AI/apps/api/tests/test_retrieval_evaluation.py)
- Clearly separates:
  - `test_deterministic_synthetic_pipeline_evaluation`: verifies RRF fusion, tenant isolation, and citation completeness.
  - `test_real_bge_m3_qdrant_integration`: checks local weights and running Qdrant container, reporting metrics or marking `NOT EXECUTED`.
