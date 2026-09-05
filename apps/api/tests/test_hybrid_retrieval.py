import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from koshshield.database import engine
from koshshield.models import AuditEvent
from koshshield.services.retrieval.embeddings.deterministic_fake import (
    DeterministicEmbeddingProvider,
)
from koshshield.services.retrieval.hybrid_search import HybridRetrievalService
from koshshield.services.retrieval.vector_store import InMemoryVectorStore
from koshshield.services.retrieval.vector_store.interfaces import (
    VectorStoreChunk,
)


@pytest.fixture
def db_session() -> Session:
    session = Session(bind=engine)
    try:
        yield session
    finally:
        session.rollback()
        session.close()


def test_hybrid_search_rrf_and_citations(db_session: Session) -> None:
    embedding_provider = DeterministicEmbeddingProvider()
    vector_store = InMemoryVectorStore()

    # Create synthetic chunks
    text_1 = "Tender guidelines for public procurement of encrypted hardware"
    text_2 = "Annual budgetary expenditure for state infrastructure projects"

    emb_1 = embedding_provider.embed_query(text_1)
    emb_2 = embedding_provider.embed_query(text_2)

    chunk_1 = VectorStoreChunk(
        point_id="cid-1",
        chunk_id="cid-1",
        tenant_id="dept-procure",
        document_id="doc-procure-1",
        page_number=3,
        redaction_version=2,
        chunk_sequence=0,
        masked_text=text_1,
        char_start=0,
        char_end=len(text_1),
        masked_content_hash="a" * 64,
        document_evidence_hash="b" * 64,
        classification="CONFIDENTIAL",
        document_filename="Tender-2026.pdf",
        indexed_at="2026-09-04T12:00:00Z",
        dense_vector=emb_1.dense,
        sparse_indices=emb_1.sparse_indices,
        sparse_values=emb_1.sparse_values,
    )

    chunk_2 = VectorStoreChunk(
        point_id="cid-2",
        chunk_id="cid-2",
        tenant_id="dept-procure",
        document_id="doc-budget-2",
        page_number=1,
        redaction_version=1,
        chunk_sequence=0,
        masked_text=text_2,
        char_start=0,
        char_end=len(text_2),
        masked_content_hash="c" * 64,
        document_evidence_hash="d" * 64,
        classification="CONFIDENTIAL",
        document_filename="Budget-2026.pdf",
        indexed_at="2026-09-04T12:00:00Z",
        dense_vector=emb_2.dense,
        sparse_indices=emb_2.sparse_indices,
        sparse_values=emb_2.sparse_values,
    )

    vector_store.upsert_chunks([chunk_1, chunk_2])

    retrieval = HybridRetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        rrf_k=60,
    )

    raw_query = "hardware procurement tender requirements"
    pack = retrieval.search(
        query=raw_query,
        tenant_id="dept-procure",
        top_k=5,
        session=db_session,
        actor_id="officer-42",
    )

    assert pack.total_found >= 1
    top_result = pack.items[0]
    assert top_result.document_id == "doc-procure-1"
    assert top_result.page_number == 3
    assert top_result.citation_label.startswith("[Document: Tender-2026.pdf | Page: 3 | Evidence:")
    assert "dense" in top_result.sources
    assert len(top_result.evidence_hash) == 64
    assert len(top_result.masked_content_hash) == 64

    # Check safe audit logging: raw query AND query hash must NOT be present
    import hashlib

    raw_query_digest = hashlib.sha256(raw_query.encode("utf-8")).hexdigest()

    audit = db_session.scalar(
        select(AuditEvent).where(
            AuditEvent.event_type == "RETRIEVAL_QUERY_EXECUTED",
            AuditEvent.actor_id == "officer-42",
        )
    )
    assert audit is not None
    audit_str = str(audit.details)
    assert raw_query not in audit_str
    assert raw_query_digest not in audit_str
    assert "query_hash" not in audit.details
    assert "query_length" in audit.details
    assert audit.details["query_length"] == len(raw_query)
    assert "result_chunk_ids" in audit.details
    assert "duration_ms" in audit.details


def test_cross_tenant_isolation() -> None:
    embedding_provider = DeterministicEmbeddingProvider()
    vector_store = InMemoryVectorStore()

    text = "Restricted military telecommunications frequency bands"
    emb = embedding_provider.embed_query(text)

    chunk = VectorStoreChunk(
        point_id="cid-sec",
        chunk_id="cid-sec",
        tenant_id="tenant-defense",
        document_id="doc-defense-secret",
        page_number=1,
        redaction_version=1,
        chunk_sequence=0,
        masked_text=text,
        char_start=0,
        char_end=len(text),
        masked_content_hash="h1",
        document_evidence_hash="ev1",
        classification="CONFIDENTIAL",
        document_filename="Secret.pdf",
        indexed_at="2026-09-04T12:00:00Z",
        dense_vector=emb.dense,
        sparse_indices=emb.sparse_indices,
        sparse_values=emb.sparse_values,
    )
    vector_store.upsert_chunks([chunk])

    retrieval = HybridRetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    # Query matching exact text from a different tenant
    pack = retrieval.search(
        query="military telecommunications frequency bands",
        tenant_id="tenant-civilian",
        top_k=5,
    )

    # Cross-tenant isolation must guarantee 0 results!
    assert pack.total_found == 0
    assert len(pack.items) == 0
