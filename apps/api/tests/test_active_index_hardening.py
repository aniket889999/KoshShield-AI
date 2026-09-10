import hashlib
import socket
import threading
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client import models
from sqlalchemy import select
from sqlalchemy.orm import Session

from koshshield.config import Settings
from koshshield.database import engine
from koshshield.models import (
    AuditEvent,
    DocumentChunkRecord,
    DocumentPageRecord,
    DocumentRecord,
    DocumentState,
)
from koshshield.services.retrieval.chunking import DeterministicMaskedChunker
from koshshield.services.retrieval.embeddings.bge_m3 import BgeM3EmbeddingProvider
from koshshield.services.retrieval.embeddings.deterministic_fake import (
    DeterministicEmbeddingProvider,
)
from koshshield.services.retrieval.embeddings.interfaces import (
    ModelUnavailableError,
)
from koshshield.services.retrieval.hybrid_search import HybridRetrievalService
from koshshield.services.retrieval.indexing_service import DocumentIndexingService
from koshshield.services.retrieval.privacy_gate import RetrievalPrivacyGate
from koshshield.services.retrieval.provider_registry import (
    get_singleton_embedding_provider,
    reset_provider_registry,
)
from koshshield.services.retrieval.vector_store.in_memory import InMemoryVectorStore
from koshshield.services.retrieval.vector_store.interfaces import (
    VectorStoreChunk,
    VectorStoreError,
    VectorStoreSearchResult,
)
from koshshield.services.retrieval.vector_store.qdrant import QdrantVectorStore


def _sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _create_test_doc(
    session: Session,
    doc_id: str,
    tenant_id: str = "default",
    version: int = 1,
    status: DocumentState = DocumentState.INDEX_READY,
    text: str = "Confidential report for active index hardening test.",
) -> DocumentRecord:
    doc = DocumentRecord(
        id=doc_id,
        tenant_id=tenant_id,
        filename=f"{doc_id}.pdf",
        media_type="application/pdf",
        size_bytes=2048,
        sha256=_sha256_hex(doc_id),
        vault_path=f"vault/{doc_id}.ksh",
        status=status,
        version=version,
    )
    page1 = DocumentPageRecord(
        id=str(uuid.uuid4()),
        document_id=doc.id,
        page_number=1,
        extraction_method="native_pdf",
        text_hash=_sha256_hex(f"text-{doc_id}"),
        encrypted_artifact_path=f"vault/{doc.id}_p1.ksh",
        masked_text=text,
    )
    session.add_all([doc, page1])
    session.commit()
    return doc


def _make_chunk(
    point_id: str,
    chunk_id: str,
    document_id: str,
    tenant_id: str = "default",
    index_version: int = 1,
    masked_text: str = "sample text",
    dense_vector: list[float] | None = None,
    sparse_indices: list[int] | None = None,
    sparse_values: list[float] | None = None,
    masked_content_hash: str = "hash",
    chunk_sequence: int = 0,
) -> VectorStoreChunk:
    return VectorStoreChunk(
        point_id=point_id,
        chunk_id=chunk_id,
        tenant_id=tenant_id,
        document_id=document_id,
        page_number=1,
        redaction_version=index_version,
        index_version=index_version,
        chunk_sequence=chunk_sequence,
        masked_text=masked_text,
        char_start=0,
        char_end=len(masked_text),
        masked_content_hash=masked_content_hash,
        document_evidence_hash=_sha256_hex(document_id),
        classification="INTERNAL",
        document_filename="test.pdf",
        indexed_at="2026-09-10T00:00:00Z",
        dense_vector=dense_vector or [0.1] * 64,
        sparse_indices=sparse_indices or [1],
        sparse_values=sparse_values or [1.0],
    )


def test_scenario_1_stale_vectors_filtered_and_payload_validated() -> None:
    """Stale vectors must never crowd out or appear in active results."""
    vector_store = InMemoryVectorStore()
    emb_provider = DeterministicEmbeddingProvider()
    doc_id = f"doc-stale-{uuid.uuid4().hex[:8]}"

    with Session(bind=engine) as session:
        _create_test_doc(session, doc_id=doc_id, version=2, status=DocumentState.INDEXED)
        doc = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc_id))
        assert doc is not None
        doc.active_index_version = 2
        session.commit()

        stale_point = _make_chunk(
            point_id=f"{doc_id}_v1_c0",
            chunk_id="stale-chunk-id",
            document_id=doc_id,
            tenant_id="default",
            index_version=1,
            masked_text="Stale content from version 1",
            dense_vector=[1.0] * 64,
            sparse_indices=[10],
            sparse_values=[100.0],
            masked_content_hash="hash1",
        )
        active_point = _make_chunk(
            point_id=f"{doc_id}_v2_c0",
            chunk_id="active-chunk-id",
            document_id=doc_id,
            tenant_id="default",
            index_version=2,
            masked_text="Active content from version 2",
            dense_vector=[0.8] * 64,
            sparse_indices=[10],
            sparse_values=[50.0],
            masked_content_hash="hash2",
        )
        vector_store.upsert_chunks([stale_point, active_point])

        search_svc = HybridRetrievalService(
            embedding_provider=emb_provider,
            vector_store=vector_store,
        )

        evidence_pack = search_svc.search(
            query="test query",
            tenant_id="default",
            session=session,
            top_k=5,
        )

        found_chunks = [item.chunk_id for item in evidence_pack.items]
        assert "active-chunk-id" in found_chunks
        assert "stale-chunk-id" not in found_chunks


def test_scenario_2_stale_cleanup_failure_leaves_retrieval_safe() -> None:
    """If stale deletion fails during reindex, retrieval returns only active points."""
    emb_provider = DeterministicEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    privacy_gate = RetrievalPrivacyGate()
    doc_id = f"doc-stale-fail-{uuid.uuid4().hex[:8]}"

    with Session(bind=engine) as session:
        _create_test_doc(session, doc_id=doc_id, version=1)
        service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=vector_store,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )
        service.index_document(session=session, document_id=doc_id)

        doc = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc_id))
        assert doc is not None
        doc.version = 2
        doc.status = DocumentState.INDEX_READY
        session.commit()

        mock_vs = MagicMock(wraps=vector_store)
        mock_vs.delete_stale_chunks.side_effect = RuntimeError("Delete timeout")

        failing_cleanup_service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=mock_vs,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )

        res = failing_cleanup_service.index_document(session=session, document_id=doc_id)
        assert res.status == DocumentState.INDEXED
        assert res.active_index_version == 2
        assert len(vector_store.chunks) == 2

        search_svc = HybridRetrievalService(
            embedding_provider=emb_provider,
            vector_store=vector_store,
        )
        evidence = search_svc.search(
            query="Confidential report",
            tenant_id="default",
            session=session,
        )
        assert len(evidence.items) == 1
        assert evidence.items[0].index_version == 2

        retrieved_chunk = session.scalar(
            select(DocumentChunkRecord).where(
                DocumentChunkRecord.chunk_id == evidence.items[0].chunk_id
            )
        )
        assert retrieved_chunk is not None
        assert retrieved_chunk.index_version == 2


def test_scenario_3_multi_document_multi_version_retrieval() -> None:
    """Documents with different active versions retrieve correctly together."""
    vector_store = InMemoryVectorStore()
    emb_provider = DeterministicEmbeddingProvider()
    doc_a = f"doc-a-{uuid.uuid4().hex[:8]}"
    doc_b = f"doc-b-{uuid.uuid4().hex[:8]}"

    with Session(bind=engine) as session:
        _create_test_doc(session, doc_id=doc_a, version=1, status=DocumentState.INDEXED)
        _create_test_doc(session, doc_id=doc_b, version=3, status=DocumentState.INDEXED)

        doc_a_rec = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc_a))
        assert doc_a_rec is not None
        doc_a_rec.active_index_version = 1

        doc_b_rec = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc_b))
        assert doc_b_rec is not None
        doc_b_rec.active_index_version = 3
        session.commit()

        pt_a = _make_chunk(
            point_id=f"{doc_a}_v1_c0",
            chunk_id="chunk-a-v1",
            document_id=doc_a,
            tenant_id="default",
            index_version=1,
            masked_text="Document A active content v1",
            dense_vector=[0.5] * 64,
            sparse_indices=[1],
            sparse_values=[10.0],
            masked_content_hash="hash_a",
        )
        pt_b = _make_chunk(
            point_id=f"{doc_b}_v3_c0",
            chunk_id="chunk-b-v3",
            document_id=doc_b,
            tenant_id="default",
            index_version=3,
            masked_text="Document B active content v3",
            dense_vector=[0.6] * 64,
            sparse_indices=[1],
            sparse_values=[15.0],
            masked_content_hash="hash_b",
        )
        vector_store.upsert_chunks([pt_a, pt_b])

        search_svc = HybridRetrievalService(
            embedding_provider=emb_provider,
            vector_store=vector_store,
        )

        results = search_svc.search(
            query="content",
            tenant_id="default",
            session=session,
            top_k=5,
        ).items

        retrieved_doc_ids = {r.document_id for r in results}
        assert doc_a in retrieved_doc_ids
        assert doc_b in retrieved_doc_ids


def test_scenario_4_unauthorized_and_malformed_payloads_rejected() -> None:
    """DB payload validation drops points with mismatched tenant or malformed payload."""
    vector_store = InMemoryVectorStore()
    emb_provider = DeterministicEmbeddingProvider()
    doc_id = f"doc-auth-{uuid.uuid4().hex[:8]}"

    with Session(bind=engine) as session:
        _create_test_doc(session, doc_id=doc_id, version=2, status=DocumentState.INDEXED)
        doc = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc_id))
        assert doc is not None
        doc.active_index_version = 2
        session.commit()

        bad_tenant_pt = _make_chunk(
            point_id="bad_tenant",
            chunk_id="chunk-bad-tenant",
            document_id=doc_id,
            tenant_id="other-tenant",
            index_version=2,
            masked_text="Wrong tenant",
            dense_vector=[1.0] * 64,
        )
        bad_ver_pt = _make_chunk(
            point_id="bad_ver",
            chunk_id="chunk-bad-ver",
            document_id=doc_id,
            tenant_id="default",
            index_version=99,
            masked_text="Wrong version",
            dense_vector=[1.0] * 64,
        )
        malformed_result = VectorStoreSearchResult(
            point_id="malformed",
            score=0.8,
            payload="not-a-dict",  # type: ignore[arg-type]
        )

        search_svc = HybridRetrievalService(
            embedding_provider=emb_provider,
            vector_store=vector_store,
        )

        with patch.object(
            vector_store,
            "search_dense",
            return_value=[
                VectorStoreSearchResult(
                    point_id=bad_tenant_pt.point_id,
                    score=1.0,
                    payload=bad_tenant_pt.to_payload(),
                ),
                VectorStoreSearchResult(
                    point_id=bad_ver_pt.point_id,
                    score=0.9,
                    payload=bad_ver_pt.to_payload(),
                ),
                malformed_result,
            ],
        ):
            results = search_svc.search(
                query="query",
                tenant_id="default",
                session=session,
            ).items

            assert len(results) == 0


def test_scenario_5_orphan_cleanup_deletes_only_target_version() -> None:
    """Verification failure removes orphan target points without deleting active generation."""
    emb_provider = DeterministicEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    privacy_gate = RetrievalPrivacyGate()
    doc_id = f"doc-orphan-{uuid.uuid4().hex[:8]}"

    with Session(bind=engine) as session:
        _create_test_doc(session, doc_id=doc_id, version=1)
        service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=vector_store,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )
        service.index_document(session=session, document_id=doc_id)
        v1_point_ids = [c.point_id for c in vector_store.chunks]
        assert len(v1_point_ids) > 0

        doc = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc_id))
        assert doc is not None
        doc.version = 2
        doc.status = DocumentState.INDEX_READY
        session.commit()

        mock_vs = MagicMock(wraps=vector_store)
        mock_vs.verify_points.return_value = False

        fail_service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=mock_vs,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )

        with pytest.raises(VectorStoreError):
            fail_service.index_document(session=session, document_id=doc_id)

        remaining_point_ids = [c.point_id for c in vector_store.chunks]
        for pid in v1_point_ids:
            assert pid in remaining_point_ids
        for pid in remaining_point_ids:
            assert "_v2_" not in pid


def test_scenario_6_reindex_failure_keeps_document_indexed_and_searchable() -> None:
    """Failed reindex with active generation preserves status INDEXED and failure code."""
    emb_provider = DeterministicEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    privacy_gate = RetrievalPrivacyGate()
    doc_id = f"doc-reindex-fail-{uuid.uuid4().hex[:8]}"

    with Session(bind=engine) as session:
        _create_test_doc(session, doc_id=doc_id, version=1)
        service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=vector_store,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )
        service.index_document(session=session, document_id=doc_id)

        doc = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc_id))
        assert doc is not None
        doc.version = 2
        doc.status = DocumentState.INDEX_READY
        session.commit()

        failing_emb = MagicMock()
        failing_emb.dense_dim = 64
        failing_emb.embed_texts.side_effect = RuntimeError("Internal inference crash")

        failing_service = DocumentIndexingService(
            embedding_provider=failing_emb,
            vector_store=vector_store,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )

        with pytest.raises(RuntimeError):
            failing_service.index_document(session=session, document_id=doc_id)

        session.refresh(doc)
        assert doc.status == DocumentState.INDEXED
        assert doc.active_index_version == 1

        audit = session.scalar(
            select(AuditEvent)
            .where(
                AuditEvent.resource_id == doc_id,
                AuditEvent.event_type == "DOCUMENT_REINDEXING_FAILED",
            )
            .order_by(AuditEvent.id.desc())
        )
        assert audit is not None
        assert audit.details.get("error_code") == "EMBEDDING_FAILURE"
        assert audit.details.get("failure_code") == "EMBEDDING_FAILURE"
        assert "Internal inference crash" not in str(audit.details)


def test_scenario_7_same_version_verified_noop() -> None:
    """Same-version reindexing verifies all active points; fails closed if points are missing."""
    emb_provider = DeterministicEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    privacy_gate = RetrievalPrivacyGate()
    doc_id = f"doc-noop-{uuid.uuid4().hex[:8]}"

    with Session(bind=engine) as session:
        _create_test_doc(session, doc_id=doc_id, version=1)
        service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=vector_store,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )
        res1 = service.index_document(session=session, document_id=doc_id)
        assert res1.active_index_version == 1

        res2 = service.index_document(session=session, document_id=doc_id)
        assert res2.active_index_version == 1

        vector_store.chunks.clear()
        with pytest.raises(VectorStoreError) as exc:
            service.index_document(session=session, document_id=doc_id)
        assert "Same-version verification failed" in str(exc.value)


def test_scenario_8_vector_store_upsert_idempotency() -> None:
    """Upserting the same point_id replaces the point instead of creating duplicates."""
    mem_store = InMemoryVectorStore()
    p1 = _make_chunk(
        point_id="pt-1",
        chunk_id="c1",
        document_id="d1",
        masked_text="initial",
        dense_vector=[1.0] * 64,
    )
    p1_updated = _make_chunk(
        point_id="pt-1",
        chunk_id="c1",
        document_id="d1",
        masked_text="updated",
        dense_vector=[2.0] * 64,
    )

    mem_store.upsert_chunks([p1])
    assert len(mem_store.chunks) == 1
    assert mem_store.chunks[0].masked_text == "initial"

    mem_store.upsert_chunks([p1_updated])
    assert len(mem_store.chunks) == 1
    assert mem_store.chunks[0].masked_text == "updated"

    mock_client = MagicMock()
    with patch(
        "koshshield.services.retrieval.vector_store.qdrant.QdrantClient",
        return_value=mock_client,
    ):
        qdrant_store = QdrantVectorStore(
            qdrant_url="http://localhost:6333",
            collection_name="test_col",
        )
        qdrant_store.upsert_chunks([p1, p1_updated])
        assert mock_client.upsert.called
        call_args = mock_client.upsert.call_args[1]
        points = call_args["points"]
        point_ids = [p.id for p in points]
        assert point_ids.count(p1.point_id) == 1


def test_scenario_9_qdrant_count_failures_and_payload_indexes_fail_closed() -> None:
    """Count errors raise VectorStoreError; missing payload index creation fails closed."""
    mock_client = MagicMock()
    with patch(
        "koshshield.services.retrieval.vector_store.qdrant.QdrantClient",
        return_value=mock_client,
    ):
        qdrant_store = QdrantVectorStore(
            qdrant_url="http://localhost:6333",
            collection_name="test_col",
        )

        mock_client.count.side_effect = RuntimeError("Qdrant node unreachable")
        with pytest.raises(VectorStoreError) as exc:
            qdrant_store.count_points(tenant_id="default")
        assert "Failed to count points" in str(exc.value)

        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = MagicMock(
            config=MagicMock(
                params=MagicMock(
                    vectors={"text_dense": MagicMock(size=64, distance="Cosine")},
                    sparse_vectors={"text_sparse": MagicMock()},
                )
            ),
            payload_schema={},
        )
        mock_client.create_payload_index.side_effect = RuntimeError("Index build failure")
        with pytest.raises(VectorStoreError):
            qdrant_store.ensure_collection(64)


def test_scenario_10_sparse_token_keys_validation() -> None:
    """BGE-M3 rejects malformed/negative sparse keys and preserves non-negative integer IDs."""
    indices, values = BgeM3EmbeddingProvider._format_sparse({"101": 0.5, "2054": 1.2})
    assert indices == [101, 2054]
    assert values == [0.5, 1.2]

    with pytest.raises(ValueError) as exc1:
        BgeM3EmbeddingProvider._format_sparse({"word_key": 0.5})
    assert "must be non-negative integer" in str(exc1.value)

    with pytest.raises(ValueError) as exc2:
        BgeM3EmbeddingProvider._format_sparse({"-5": 0.5})
    assert "must be non-negative" in str(exc2.value)


def test_scenario_11_missing_bge_m3_config_fails_without_fallback(tmp_path: Path) -> None:
    """Missing or malformed BGE-M3 config raises ModelUnavailableError instead of fallback."""
    model_dir = tmp_path / "bge_model"
    model_dir.mkdir()
    provider = BgeM3EmbeddingProvider(model_dir=str(model_dir))
    with pytest.raises(ModelUnavailableError) as exc:
        _ = provider.dense_dim
    assert "dense dimension is unknown" in str(exc.value)


def test_scenario_12_thread_safe_singleton_provider_initialization() -> None:
    """Concurrent requests initialize only one shared instance of providers."""
    reset_provider_registry()
    settings = Settings(
        embedding_model_dir="/mock/bge",
        qdrant_url="http://localhost:6333",
    )

    instances = []

    def _worker() -> None:
        with patch(
            "koshshield.services.retrieval.embeddings.bge_m3.BgeM3EmbeddingProvider.is_available",
            return_value=(True, "Ready"),
        ):
            provider = get_singleton_embedding_provider(settings)
            instances.append(provider)

    threads = [threading.Thread(target=_worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(instances) == 10
    first = instances[0]
    for inst in instances[1:]:
        assert inst is first


def test_scenario_13_zero_outbound_network_during_embedding_init(tmp_path: Path) -> None:
    """Embedding provider initialization attempts zero outbound network connections."""

    def guarded_connect(*args: object, **kwargs: object) -> None:
        raise AssertionError("Outbound network connection attempted during local-first execution!")

    model_dir = tmp_path / "bge_offline"
    model_dir.mkdir()
    with patch.object(socket.socket, "connect", side_effect=guarded_connect):
        provider = BgeM3EmbeddingProvider(model_dir=str(model_dir))
        assert provider is not None


def test_scenario_14_audit_records_sanitized_errors_and_full_hashes() -> None:
    """Audit records retain full 64-character SHA-256 evidence hashes and sanitized error codes."""
    emb_provider = DeterministicEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    privacy_gate = RetrievalPrivacyGate()
    doc_id = f"doc-audit-{uuid.uuid4().hex[:8]}"

    with Session(bind=engine) as session:
        _create_test_doc(session, doc_id=doc_id, version=1)
        service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=vector_store,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )
        service.index_document(session=session, document_id=doc_id)

        success_audit = session.scalar(
            select(AuditEvent)
            .where(
                AuditEvent.resource_id == doc_id,
                AuditEvent.event_type == "DOCUMENT_INDEXED",
            )
            .order_by(AuditEvent.id.desc())
        )
        assert success_audit is not None
        evidence_hash = str(success_audit.details.get("evidence_sha256"))
        assert len(evidence_hash) == 64
        assert all(c in "0123456789abcdef" for c in evidence_hash)

        doc = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc_id))
        assert doc is not None
        doc.version = 2
        doc.status = DocumentState.INDEX_READY
        session.commit()

        secret_error = "CRITICAL_SECRET_PATH=/users/secret/key.pem: failed to open"
        fail_emb = MagicMock()
        fail_emb.dense_dim = 64
        fail_emb.embed_texts.side_effect = RuntimeError(secret_error)

        fail_service = DocumentIndexingService(
            embedding_provider=fail_emb,
            vector_store=vector_store,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )

        with pytest.raises(RuntimeError):
            fail_service.index_document(session=session, document_id=doc_id)

        fail_audit = session.scalar(
            select(AuditEvent)
            .where(
                AuditEvent.resource_id == doc_id,
                AuditEvent.event_type == "DOCUMENT_REINDEXING_FAILED",
            )
            .order_by(AuditEvent.id.desc())
        )
        assert fail_audit is not None
        assert fail_audit.details.get("error_code") == "EMBEDDING_FAILURE"
        assert secret_error not in str(fail_audit.details)
        assert "/users/secret" not in str(fail_audit.details)


def test_malformed_evidence_hash_rejected() -> None:
    """Documents with malformed evidence hashes are strictly rejected with ValueError."""
    emb_provider = DeterministicEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    privacy_gate = RetrievalPrivacyGate()
    service = DocumentIndexingService(
        embedding_provider=emb_provider,
        vector_store=vector_store,
        privacy_gate=privacy_gate,
        chunker=DeterministicMaskedChunker(),
    )

    doc_id = f"doc-bad-hash-{uuid.uuid4().hex[:8]}"
    with Session(bind=engine) as session:
        # 1. Non-hex characters
        doc = DocumentRecord(
            id=doc_id,
            tenant_id="default",
            filename="bad_hash.pdf",
            media_type="application/pdf",
            size_bytes=100,
            sha256="not-valid-hex-hash".ljust(64, "0"),
            vault_path="vault/bad.ksh",
            status=DocumentState.INDEX_READY,
            version=1,
        )
        session.add(doc)
        session.commit()

        with pytest.raises(ValueError) as exc1:
            service.index_document(session=session, document_id=doc_id)
        assert "must be a lowercase 64-character hexadecimal" in str(exc1.value)

        # 2. Too short
        doc.sha256 = "abcd1234"
        session.commit()
        with pytest.raises(ValueError) as exc2:
            service.index_document(session=session, document_id=doc_id)
        assert "must be a lowercase 64-character hexadecimal" in str(exc2.value)


def test_qdrant_active_version_filter_contract_and_serialization() -> None:
    """Contract test verifying Qdrant active-version filter serialization with qdrant-client."""
    q_filter = QdrantVectorStore._build_filter(
        tenant_id="tenant-alpha",
        permitted_document_ids=["doc-1", "doc-2"],
        active_document_versions={"doc-1": 1, "doc-2": 3},
        classification="CONFIDENTIAL",
    )
    assert isinstance(q_filter, models.Filter)
    serialized = (
        q_filter.model_dump_json() if hasattr(q_filter, "model_dump_json") else q_filter.json()
    )
    assert "tenant-alpha" in serialized
    assert "doc-1" in serialized
    assert "doc-2" in serialized
    assert "CONFIDENTIAL" in serialized


def test_bge_m3_double_checked_lazy_loading_concurrency() -> None:
    """Concurrent threads must invoke the underlying model initialization exactly once."""
    provider = BgeM3EmbeddingProvider(model_dir="/mock/dir")
    load_count = 0
    lock = threading.Lock()

    def fake_init(*args: object, **kwargs: object) -> MagicMock:
        nonlocal load_count
        with lock:
            load_count += 1
        return MagicMock()

    mock_flag_mod = MagicMock()
    mock_flag_mod.BGEM3FlagModel = fake_init

    with (
        patch.object(provider, "is_available", return_value=(True, "Ready")),
        patch.dict("sys.modules", {"FlagEmbedding": mock_flag_mod}),
    ):
        threads = [threading.Thread(target=provider._ensure_model) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert load_count == 1
    assert provider._model is not None


def test_active_generation_protected_during_cleanup_invariant() -> None:
    """The indexing service must never call delete_version_chunks for the active generation."""
    emb_provider = DeterministicEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    privacy_gate = RetrievalPrivacyGate()
    doc_id = f"doc-protect-active-{uuid.uuid4().hex[:8]}"

    with Session(bind=engine) as session:
        _create_test_doc(session, doc_id=doc_id, version=1)
        service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=vector_store,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )
        service.index_document(session=session, document_id=doc_id)

        mock_vs = MagicMock(wraps=vector_store)
        mock_vs.verify_points.return_value = False

        fail_service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=mock_vs,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )

        with pytest.raises(VectorStoreError):
            fail_service.index_document(session=session, document_id=doc_id)

        # Assert delete_version_chunks was NEVER called for active version 1
        for call in mock_vs.delete_version_chunks.call_args_list:
            called_version = call[1].get("index_version") or (
                call[0][2] if len(call[0]) > 2 else None
            )
            assert called_version != 1, "delete_version_chunks was called for active generation!"
