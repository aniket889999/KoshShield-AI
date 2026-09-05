import uuid
from unittest.mock import MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from koshshield.database import engine
from koshshield.models import DocumentPageRecord, DocumentRecord, DocumentState
from koshshield.services.retrieval.chunking import DeterministicMaskedChunker
from koshshield.services.retrieval.embeddings.deterministic_fake import (
    DeterministicEmbeddingProvider,
)
from koshshield.services.retrieval.indexing_service import DocumentIndexingService
from koshshield.services.retrieval.privacy_gate import RetrievalPrivacyGate
from koshshield.services.retrieval.vector_store import InMemoryVectorStore
from koshshield.services.retrieval.vector_store.interfaces import VectorStoreError


def create_indexed_document(session: Session, doc_id: str) -> DocumentRecord:
    doc = DocumentRecord(
        id=doc_id,
        filename="safety_test.pdf",
        media_type="application/pdf",
        size_bytes=2048,
        sha256="sha256-original-doc-hash-abcdef1234567890abcdef1234567890abcdef123456",
        vault_path=f"vault/{doc_id}.ksh",
        status=DocumentState.INDEX_READY,
        version=1,
    )
    page1 = DocumentPageRecord(
        id=str(uuid.uuid4()),
        document_id=doc.id,
        page_number=1,
        extraction_method="native_pdf",
        text_hash="thash-p1",
        encrypted_artifact_path=f"vault/{doc.id}_p1.ksh",
        masked_text="Initial version 1 approved text for safety testing.",
    )
    session.add_all([doc, page1])
    session.commit()
    return doc


def test_reindexing_failure_at_embedding_preserves_old_index() -> None:
    emb_provider = DeterministicEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    privacy_gate = RetrievalPrivacyGate()
    service = DocumentIndexingService(
        embedding_provider=emb_provider,
        vector_store=vector_store,
        privacy_gate=privacy_gate,
        chunker=DeterministicMaskedChunker(),
    )

    doc_id = f"doc-fail-emb-{uuid.uuid4().hex[:8]}"
    with Session(bind=engine) as session:
        create_indexed_document(session, doc_id)
        # Initial successful indexing at version 1
        res1 = service.index_document(session=session, document_id=doc_id)
        assert res1.status == DocumentState.INDEXED
        assert res1.active_index_version == 1
        assert len(vector_store.chunks) >= 1
        initial_chunk_count = len(vector_store.chunks)
        initial_chunk_ids = [c.point_id for c in vector_store.chunks]

        # Prepare version 2
        doc = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc_id))
        assert doc is not None
        doc.version = 2
        doc.status = DocumentState.INDEX_READY
        session.commit()

        # Mock embedding provider to fail
        failing_emb = MagicMock()
        failing_emb.dense_dim = 1024
        failing_emb.embed_texts.side_effect = RuntimeError("Embedding model GPU OOM")

        failing_service = DocumentIndexingService(
            embedding_provider=failing_emb,
            vector_store=vector_store,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )

        with pytest.raises(RuntimeError) as exc:
            failing_service.index_document(session=session, document_id=doc_id)
        assert "GPU OOM" in str(exc.value)

        # Invariant: Old index points MUST be preserved
        assert len(vector_store.chunks) == initial_chunk_count
        assert [c.point_id for c in vector_store.chunks] == initial_chunk_ids

        # Invariant: Document marked INDEX_FAILED and old active version preserved
        session.refresh(doc)
        assert doc.status == DocumentState.INDEX_FAILED
        assert doc.active_index_version == 1


def test_reindexing_failure_at_upsert_preserves_old_index() -> None:
    emb_provider = DeterministicEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    privacy_gate = RetrievalPrivacyGate()

    doc_id = f"doc-fail-upsert-{uuid.uuid4().hex[:8]}"
    with Session(bind=engine) as session:
        create_indexed_document(session, doc_id)
        service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=vector_store,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )
        service.index_document(session=session, document_id=doc_id)
        initial_chunk_ids = [c.point_id for c in vector_store.chunks]

        # Version 2
        doc = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc_id))
        assert doc is not None
        doc.version = 2
        doc.status = DocumentState.INDEX_READY
        session.commit()

        # Mock vector store upsert to fail
        mock_vs = MagicMock(wraps=vector_store)
        mock_vs.upsert_chunks.side_effect = VectorStoreError("Connection dropped during write")

        failing_service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=mock_vs,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )

        with pytest.raises(VectorStoreError) as exc:
            failing_service.index_document(session=session, document_id=doc_id)
        assert "Connection dropped" in str(exc.value)

        # Invariant: Old points not deleted
        assert [c.point_id for c in vector_store.chunks] == initial_chunk_ids
        session.refresh(doc)
        assert doc.status == DocumentState.INDEX_FAILED
        assert doc.active_index_version == 1


def test_reindexing_failure_at_verification_preserves_old_index() -> None:
    emb_provider = DeterministicEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    privacy_gate = RetrievalPrivacyGate()

    doc_id = f"doc-fail-verify-{uuid.uuid4().hex[:8]}"
    with Session(bind=engine) as session:
        create_indexed_document(session, doc_id)
        service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=vector_store,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )
        service.index_document(session=session, document_id=doc_id)

        # Version 2
        doc = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc_id))
        assert doc is not None
        doc.version = 2
        doc.status = DocumentState.INDEX_READY
        session.commit()

        # Mock verify_points to fail
        mock_vs = MagicMock(wraps=vector_store)
        mock_vs.verify_points.return_value = False

        failing_service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=mock_vs,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )

        with pytest.raises(VectorStoreError) as exc:
            failing_service.index_document(session=session, document_id=doc_id)
        assert "Index verification failed" in str(exc.value)

        session.refresh(doc)
        assert doc.status == DocumentState.INDEX_FAILED
        assert doc.active_index_version == 1


def test_stale_deletion_failure_does_not_corrupt_index() -> None:
    emb_provider = DeterministicEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    privacy_gate = RetrievalPrivacyGate()

    doc_id = f"doc-fail-stale-{uuid.uuid4().hex[:8]}"
    with Session(bind=engine) as session:
        create_indexed_document(session, doc_id)
        service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=vector_store,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )
        service.index_document(session=session, document_id=doc_id)

        # Version 2
        doc = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc_id))
        assert doc is not None
        doc.version = 2
        doc.status = DocumentState.INDEX_READY
        session.commit()

        # Mock delete_stale_chunks to raise error
        mock_vs = MagicMock(wraps=vector_store)
        mock_vs.delete_stale_chunks.side_effect = RuntimeError("Lock timeout during delete")

        resilience_service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=mock_vs,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )

        # Should NOT raise, but mark cleanup as pending while keeping document INDEXED
        result = resilience_service.index_document(session=session, document_id=doc_id)
        assert result.status == DocumentState.INDEXED
        assert result.active_index_version == 2

        session.refresh(doc)
        assert doc.status == DocumentState.INDEXED
        assert doc.active_index_version == 2
        assert doc.index_cleanup_pending is True
