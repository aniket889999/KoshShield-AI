import hashlib
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from koshshield.api.routes.retrieval import get_indexing_service
from koshshield.config import get_settings
from koshshield.database import engine
from koshshield.main import app
from koshshield.models import (
    AuditEvent,
    DocumentPageRecord,
    DocumentRecord,
    DocumentState,
)
from koshshield.services.retrieval.chunking import DeterministicMaskedChunker
from koshshield.services.retrieval.embeddings.deterministic_fake import (
    DeterministicEmbeddingProvider,
)
from koshshield.services.retrieval.indexing_service import DocumentIndexingService
from koshshield.services.retrieval.privacy_gate import RetrievalPrivacyGate
from koshshield.services.retrieval.vector_store.in_memory import InMemoryVectorStore
from koshshield.services.retrieval.vector_store.interfaces import (
    VectorStoreChunk,
    VectorStoreError,
)


def _valid_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _create_test_doc(
    session: Session,
    doc_id: str,
    tenant_id: str = "tenant-sec",
    version: int = 1,
    status: DocumentState = DocumentState.INDEX_READY,
    text: str = "Sensitive Tender for Ministry of Defence in New Delhi.",
) -> DocumentRecord:
    doc = DocumentRecord(
        id=doc_id,
        tenant_id=tenant_id,
        filename=f"{doc_id}.pdf",
        media_type="application/pdf",
        size_bytes=2048,
        sha256=_valid_hash(f"evidence_{doc_id}"),
        vault_path=f"vault/{doc_id}.ksh",
        status=status,
        version=version,
        active_index_version=1 if status == DocumentState.INDEXED else None,
    )
    session.add(doc)
    session.flush()

    page = DocumentPageRecord(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        page_number=1,
        extraction_method="native_pdf",
        text_hash=_valid_hash(f"text_{doc_id}"),
        encrypted_artifact_path=f"vault/{doc_id}_p1.ksh",
        masked_text=text,
        masked_text_hash=_valid_hash(text),
        page_image_sha256=_valid_hash(f"image_p1_{doc_id}"),
        encrypted_page_image_path=f"vault/{doc_id}_p1.png",
        page_image_media_type="image/png",
    )
    session.add(page)
    session.commit()
    session.refresh(doc)
    return doc


def test_process_interruption_after_activation_leaves_marker_durable() -> None:
    """Simulate process interruption immediately after activation using SystemExit/BaseException.

    Proves the new version is active and index_cleanup_pending=True in a fresh DB session.
    """
    emb_provider = DeterministicEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    privacy_gate = RetrievalPrivacyGate()
    doc_id = f"doc-crash-{uuid.uuid4().hex[:8]}"
    tenant_id = "tenant-crash"

    with Session(bind=engine) as session:
        _create_test_doc(session, doc_id=doc_id, tenant_id=tenant_id, version=1)
        service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=vector_store,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )
        res1 = service.index_document(session=session, document_id=doc_id, tenant_id=tenant_id)
        assert res1.status == DocumentState.INDEXED
        assert res1.active_index_version == 1

        # Prepare reindex to version 2
        doc = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc_id))
        assert doc is not None
        doc.version = 2
        doc.status = DocumentState.INDEX_READY
        session.commit()

        # Simulate SIGKILL crash immediately upon entering stale cleanup
        mock_vs = MagicMock(wraps=vector_store)
        mock_vs.delete_stale_chunks.side_effect = SystemExit("Simulated SIGKILL crash")

        crash_service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=mock_vs,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )

        with pytest.raises(SystemExit):
            crash_service.index_document(session=session, document_id=doc_id, tenant_id=tenant_id)

    # Open fresh DB session and prove crash-safety
    with Session(bind=engine) as fresh_session:
        fresh_doc = fresh_session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc_id))
        assert fresh_doc is not None
        assert fresh_doc.active_index_version == 2
        assert fresh_doc.status == DocumentState.INDEXED
        assert fresh_doc.index_cleanup_pending is True

    # Vector store contains points for both v1 and v2
    v1_points = [c for c in vector_store.chunks if c.document_id == doc_id and c.index_version == 1]
    v2_points = [c for c in vector_store.chunks if c.document_id == doc_id and c.index_version == 2]
    assert len(v1_points) > 0
    assert len(v2_points) > 0


def test_reconciliation_removes_stale_versions_and_clears_marker() -> None:
    """Reconciliation deletes older generations and clears index_cleanup_pending."""
    vector_store = InMemoryVectorStore()
    privacy_gate = RetrievalPrivacyGate()
    emb_provider = DeterministicEmbeddingProvider()
    doc_id = f"doc-reconcile-{uuid.uuid4().hex[:8]}"
    tenant_id = "tenant-recon"

    # Add v1 and v2 chunks to vector store
    for v in [1, 2]:
        chunk = VectorStoreChunk(
            point_id=f"{doc_id}-v{v}-c1",
            chunk_id=f"{doc_id}-v{v}-c1",
            tenant_id=tenant_id,
            document_id=doc_id,
            page_number=1,
            redaction_version=v,
            index_version=v,
            chunk_sequence=0,
            masked_text=f"Text v{v}",
            char_start=0,
            char_end=7,
            masked_content_hash=_valid_hash(f"Text v{v}"),
            document_evidence_hash=_valid_hash(f"evidence_{doc_id}"),
            classification="CONFIDENTIAL",
            document_filename=f"{doc_id}.pdf",
            indexed_at="2026-09-10T12:00:00Z",
            dense_vector=[0.1] * 1024,
            visual_regions=[],
            sparse_indices=[],
            sparse_values=[],
        )
        vector_store.upsert_chunks([chunk])

    with Session(bind=engine) as session:
        doc = _create_test_doc(session, doc_id=doc_id, tenant_id=tenant_id, version=2)
        doc.status = DocumentState.INDEXED
        doc.active_index_version = 2
        doc.index_cleanup_pending = True
        session.commit()

        service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=vector_store,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )

        result = service.reconcile_pending_cleanups(
            session=session,
            tenant_id=tenant_id,
            limit=50,
            actor_id="admin-recon",
        )

        assert result.processed_count == 1
        assert result.succeeded_count == 1
        assert result.failed_count == 0
        assert result.failure_codes == []

    # Verify fresh database state
    with Session(bind=engine) as session:
        doc = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc_id))
        assert doc is not None
        assert doc.index_cleanup_pending is False
        assert doc.active_index_version == 2

        # Verify audit record
        audit = session.scalar(
            select(AuditEvent)
            .where(
                AuditEvent.resource_id == doc_id,
                AuditEvent.event_type == "INDEX_CLEANUP_COMPLETED",
            )
            .order_by(AuditEvent.created_at.desc())
        )
        assert audit is not None
        assert audit.details.get("active_index_version") == 2
        assert audit.details.get("tenant_id") == tenant_id

    # Verify vector store: v1 removed, v2 preserved
    remaining_chunks = [c for c in vector_store.chunks if c.document_id == doc_id]
    assert len(remaining_chunks) == 1
    assert remaining_chunks[0].index_version == 2


def test_cleanup_failure_keeps_marker_true() -> None:
    """When vector store deletion fails during reconciliation, marker remains True."""
    vector_store = InMemoryVectorStore()
    privacy_gate = RetrievalPrivacyGate()
    emb_provider = DeterministicEmbeddingProvider()
    doc_id = f"doc-fail-clean-{uuid.uuid4().hex[:8]}"
    tenant_id = "tenant-fail"

    mock_vs = MagicMock(wraps=vector_store)
    mock_vs.delete_stale_chunks.side_effect = VectorStoreError("Connection to Qdrant lost")

    with Session(bind=engine) as session:
        doc = _create_test_doc(session, doc_id=doc_id, tenant_id=tenant_id, version=2)
        doc.status = DocumentState.INDEXED
        doc.active_index_version = 2
        doc.index_cleanup_pending = True
        session.commit()

        service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=mock_vs,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )

        result = service.reconcile_pending_cleanups(
            session=session,
            tenant_id=tenant_id,
            limit=50,
        )

        assert result.processed_count == 1
        assert result.succeeded_count == 0
        assert result.failed_count == 1
        assert result.failure_codes == ["VECTOR_STORE_ERROR"]

    # Fresh session verify doc remains index_cleanup_pending=True
    with Session(bind=engine) as session:
        doc = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc_id))
        assert doc is not None
        assert doc.index_cleanup_pending is True
        assert doc.active_index_version == 2

        audit = session.scalar(
            select(AuditEvent)
            .where(
                AuditEvent.resource_id == doc_id,
                AuditEvent.event_type == "INDEX_CLEANUP_FAILED",
            )
            .order_by(AuditEvent.created_at.desc())
        )
        assert audit is not None
        assert audit.details.get("failure_code") == "VECTOR_STORE_ERROR"


def test_failure_while_clearing_marker_leaves_it_true_after_rollback() -> None:
    """Database failure during commit when clearing the marker rolls back to True."""
    vector_store = InMemoryVectorStore()
    privacy_gate = RetrievalPrivacyGate()
    emb_provider = DeterministicEmbeddingProvider()
    doc_id = f"doc-commit-fail-{uuid.uuid4().hex[:8]}"
    tenant_id = "tenant-commit-fail"

    with Session(bind=engine) as session:
        doc = _create_test_doc(session, doc_id=doc_id, tenant_id=tenant_id, version=2)
        doc.status = DocumentState.INDEXED
        doc.active_index_version = 2
        doc.index_cleanup_pending = True
        session.commit()

        service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=vector_store,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )

        def failing_commit() -> None:
            raise RuntimeError("Database commit error while clearing cleanup marker")

        with patch.object(session, "commit", side_effect=failing_commit):
            result = service.reconcile_pending_cleanups(
                session=session,
                tenant_id=tenant_id,
            )
            assert result.failed_count == 1

    with Session(bind=engine) as session:
        doc = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc_id))
        assert doc is not None
        assert doc.index_cleanup_pending is True
        assert doc.active_index_version == 2


def test_cross_tenant_pending_records_never_processed() -> None:
    """Reconciliation scopes strictly to tenant_id; cross-tenant records are never processed."""
    vector_store = InMemoryVectorStore()
    privacy_gate = RetrievalPrivacyGate()
    emb_provider = DeterministicEmbeddingProvider()

    doc_a_id = f"doc-a-{uuid.uuid4().hex[:8]}"
    doc_b_id = f"doc-b-{uuid.uuid4().hex[:8]}"

    with Session(bind=engine) as session:
        doc_a = _create_test_doc(session, doc_id=doc_a_id, tenant_id="tenant-A", version=2)
        doc_a.status = DocumentState.INDEXED
        doc_a.active_index_version = 2
        doc_a.index_cleanup_pending = True

        doc_b = _create_test_doc(session, doc_id=doc_b_id, tenant_id="tenant-B", version=2)
        doc_b.status = DocumentState.INDEXED
        doc_b.active_index_version = 2
        doc_b.index_cleanup_pending = True
        session.commit()

        service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=vector_store,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )

        result = service.reconcile_pending_cleanups(
            session=session,
            tenant_id="tenant-A",
        )

        assert result.processed_count == 1
        assert result.succeeded_count == 1

    with Session(bind=engine) as session:
        doc_a = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc_a_id))
        doc_b = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc_b_id))
        assert doc_a is not None and doc_a.index_cleanup_pending is False
        assert doc_b is not None and doc_b.index_cleanup_pending is True


def test_reconciliation_never_deletes_active_generation() -> None:
    """Reconciliation deletes only older generations and skips docs without active version."""
    vector_store = InMemoryVectorStore()
    privacy_gate = RetrievalPrivacyGate()
    emb_provider = DeterministicEmbeddingProvider()
    doc_id = f"doc-active-prot-{uuid.uuid4().hex[:8]}"
    tenant_id = "tenant-prot"

    # Add v1 and v2 chunks
    for v in [1, 2]:
        chunk = VectorStoreChunk(
            point_id=f"{doc_id}-v{v}-c1",
            chunk_id=f"{doc_id}-v{v}-c1",
            tenant_id=tenant_id,
            document_id=doc_id,
            page_number=1,
            redaction_version=v,
            index_version=v,
            chunk_sequence=0,
            masked_text=f"Text v{v}",
            char_start=0,
            char_end=7,
            masked_content_hash=_valid_hash(f"Text v{v}"),
            document_evidence_hash=_valid_hash(f"evidence_{doc_id}"),
            classification="CONFIDENTIAL",
            document_filename=f"{doc_id}.pdf",
            indexed_at="2026-09-10T12:00:00Z",
            dense_vector=[0.1] * 1024,
            visual_regions=[],
            sparse_indices=[],
            sparse_values=[],
        )
        vector_store.upsert_chunks([chunk])

    with Session(bind=engine) as session:
        doc = _create_test_doc(session, doc_id=doc_id, tenant_id=tenant_id, version=2)
        doc.status = DocumentState.INDEXED
        doc.active_index_version = 2
        doc.index_cleanup_pending = True
        session.commit()

        service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=vector_store,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )

        result = service.reconcile_pending_cleanups(
            session=session,
            tenant_id=tenant_id,
        )
        assert result.succeeded_count == 1

    # Active generation v2 remains untouched
    active_points = [
        c for c in vector_store.chunks if c.document_id == doc_id and c.index_version == 2
    ]
    assert len(active_points) == 1

    # Also test record without active version is skipped
    doc_no_active_id = f"doc-no-active-{uuid.uuid4().hex[:8]}"
    with Session(bind=engine) as session:
        doc_na = _create_test_doc(session, doc_id=doc_no_active_id, tenant_id=tenant_id, version=1)
        doc_na.active_index_version = None
        doc_na.index_cleanup_pending = True
        session.commit()

        mock_vs = MagicMock(wraps=vector_store)
        service_na = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=mock_vs,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )

        res_na = service_na.reconcile_pending_cleanups(session=session, tenant_id=tenant_id)
        assert res_na.failed_count == 1
        assert "NO_ACTIVE_VERSION" in res_na.failure_codes
        # delete_stale_chunks was not called
        mock_vs.delete_stale_chunks.assert_not_called()


def test_cleanup_pending_endpoint_rejects_non_admin_and_production_headers() -> None:
    """POST /retrieval/cleanup-pending requires admin role and rejects production headers."""
    client = TestClient(app)

    # User role -> 403 Forbidden
    resp_user = client.post(
        "/api/v1/retrieval/cleanup-pending",
        json={"limit": 10},
        headers={"X-Tenant-ID": "tenant-sec", "X-Roles": "user"},
    )
    assert resp_user.status_code == 403

    # Reviewer role -> 403 Forbidden
    resp_rev = client.post(
        "/api/v1/retrieval/cleanup-pending",
        json={"limit": 10},
        headers={"X-Tenant-ID": "tenant-sec", "X-Roles": "reviewer"},
    )
    assert resp_rev.status_code == 403

    # Admin role -> 200 OK
    resp_admin = client.post(
        "/api/v1/retrieval/cleanup-pending",
        json={"limit": 10},
        headers={"X-Tenant-ID": "tenant-sec", "X-Roles": "admin"},
    )
    assert resp_admin.status_code == 200
    data = resp_admin.json()
    assert data["tenant_id"] == "tenant-sec"
    assert "processed_count" in data

    # Production mode / demo_mode False -> 401 Unauthorized
    settings = get_settings()
    with patch.object(settings, "environment", "production"):
        resp_prod = client.post(
            "/api/v1/retrieval/cleanup-pending",
            json={"limit": 10},
            headers={"X-Tenant-ID": "tenant-sec", "X-Roles": "admin"},
        )
        assert resp_prod.status_code == 401
        assert "production mode" in resp_prod.text


def test_cleanup_api_and_audit_records_do_not_leak_secrets() -> None:
    """Injected secret strings never appear in cleanup API responses or audit records."""
    client = TestClient(app)
    secret_path = "/private/models/bge"
    qdrant_secret_url = "http://internal-host:6333"
    vault_key_path = "/private/key"

    doc_id = f"doc-leak-sec-{uuid.uuid4().hex[:8]}"
    tenant_id = "tenant-leak"

    with Session(bind=engine) as session:
        doc = _create_test_doc(session, doc_id=doc_id, tenant_id=tenant_id, version=2)
        doc.status = DocumentState.INDEXED
        doc.active_index_version = 2
        doc.index_cleanup_pending = True
        session.commit()

    failing_vs = MagicMock(wraps=InMemoryVectorStore())
    failing_vs.delete_stale_chunks.side_effect = VectorStoreError(
        f"CRITICAL_FAILURE at {qdrant_secret_url} with {secret_path} and {vault_key_path}"
    )

    failing_service = DocumentIndexingService(
        embedding_provider=DeterministicEmbeddingProvider(),
        vector_store=failing_vs,
        privacy_gate=RetrievalPrivacyGate(),
        chunker=DeterministicMaskedChunker(),
    )

    app.dependency_overrides[get_indexing_service] = lambda: failing_service

    try:
        response = client.post(
            "/api/v1/retrieval/cleanup-pending",
            json={"limit": 10},
            headers={"X-Tenant-ID": tenant_id, "X-Roles": "admin"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["failed_count"] == 1
        assert data["failure_codes"] == ["VECTOR_STORE_ERROR"]

        # Assert no secrets in response text
        assert secret_path not in response.text
        assert qdrant_secret_url not in response.text
        assert vault_key_path not in response.text

        # Assert no secrets in audit records
        with Session(bind=engine) as session:
            audits = list(
                session.scalars(
                    select(AuditEvent).where(
                        AuditEvent.resource_id == doc_id,
                        AuditEvent.event_type == "INDEX_CLEANUP_FAILED",
                    )
                )
            )
            assert len(audits) > 0
            for a in audits:
                assert secret_path not in str(a.details)
                assert qdrant_secret_url not in str(a.details)
                assert vault_key_path not in str(a.details)
                assert secret_path not in a.event_hash
                assert qdrant_secret_url not in a.event_hash
                assert vault_key_path not in a.event_hash
    finally:
        app.dependency_overrides.pop(get_indexing_service, None)
