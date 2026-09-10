import hashlib
import json
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from qdrant_client import models
from sqlalchemy import select
from sqlalchemy.orm import Session

from koshshield.api.routes.retrieval import (
    get_indexing_service,
    get_retrieval_service,
    get_vector_store,
)
from koshshield.database import engine
from koshshield.main import app
from koshshield.models import (
    AuditEvent,
    DocumentChunkRecord,
    DocumentPageRecord,
    DocumentRecord,
    DocumentState,
)
from koshshield.security.vault import VaultConfigurationError
from koshshield.services.retrieval.chunking import DeterministicMaskedChunker
from koshshield.services.retrieval.embeddings.deterministic_fake import (
    DeterministicEmbeddingProvider,
)
from koshshield.services.retrieval.embeddings.interfaces import ModelUnavailableError
from koshshield.services.retrieval.indexing_service import DocumentIndexingService
from koshshield.services.retrieval.privacy_gate import RetrievalPrivacyGate
from koshshield.services.retrieval.vector_store.in_memory import InMemoryVectorStore
from koshshield.services.retrieval.vector_store.interfaces import (
    VectorStoreError,
    VectorStoreUnavailableError,
)
from koshshield.services.retrieval.vector_store.qdrant import QdrantVectorStore


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


# ==============================================================================
# 1. Lifecycle Regression Tests
# ==============================================================================


def test_activation_audit_failure_preserves_previous_generation() -> None:
    """DOCUMENT_INDEXED audit creation failure before activation commit preserves
    previous generation.
    """
    emb_provider = DeterministicEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    privacy_gate = RetrievalPrivacyGate()
    doc_id = f"doc-act-audit-fail-{uuid.uuid4().hex[:8]}"
    tenant_id = "tenant-sec"

    with Session(bind=engine) as session:
        _create_test_doc(session, doc_id=doc_id, tenant_id=tenant_id, version=1)
        service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=vector_store,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )
        # First index to v1 succeeds
        res1 = service.index_document(session=session, document_id=doc_id, tenant_id=tenant_id)
        assert res1.active_index_version == 1

        # Re-index to v2, but fail during record_audit_event inside the activation transaction
        doc = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc_id))
        assert doc is not None
        doc.version = 2
        session.commit()

        with (
            patch(
                "koshshield.services.retrieval.indexing_service.record_audit_event",
                side_effect=[
                    None,  # privacy gate
                    RuntimeError(
                        "Simulated audit write failure before activation commit"
                    ),  # activation audit
                    None,  # failure audit
                ],
            ),
            pytest.raises(RuntimeError, match="Simulated audit write failure"),
        ):
            service.index_document(session=session, document_id=doc_id, tenant_id=tenant_id)

        # Invariant checks:
        session.expire_all()
        doc_after = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc_id))
        assert doc_after is not None
        assert doc_after.status == DocumentState.INDEXED
        assert doc_after.active_index_version == 1

        # Target v2 chunks were rolled back from DB
        db_chunks = list(
            session.scalars(
                select(DocumentChunkRecord).where(DocumentChunkRecord.document_id == doc_id)
            )
        )
        assert all(c.index_version == 1 for c in db_chunks)

        # Target v2 points were cleaned up from vector store, v1 points remain
        v1_points = [
            c for c in vector_store.chunks if c.document_id == doc_id and c.index_version == 1
        ]
        v2_points = [
            c for c in vector_store.chunks if c.document_id == doc_id and c.index_version == 2
        ]
        assert len(v1_points) > 0
        assert len(v2_points) == 0


def test_failure_after_activation_commit_never_removes_new_active_points() -> None:
    """Failure after activation commit never removes the new active points."""
    emb_provider = DeterministicEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    privacy_gate = RetrievalPrivacyGate()
    doc_id = f"doc-post-act-fail-{uuid.uuid4().hex[:8]}"
    tenant_id = "tenant-sec"

    with Session(bind=engine) as session:
        _create_test_doc(session, doc_id=doc_id, tenant_id=tenant_id, version=1)
        mock_vs = MagicMock(wraps=vector_store)
        mock_vs.delete_stale_chunks.side_effect = VectorStoreError(
            "Stale cleanup failed post-activation"
        )

        service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=mock_vs,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )

        # Indexing succeeds despite post-activation stale cleanup failure
        res = service.index_document(session=session, document_id=doc_id, tenant_id=tenant_id)
        assert res.active_index_version == 1
        assert res.status == DocumentState.INDEXED

        # Assert delete_version_chunks was NEVER called for the new active version 1
        for call in mock_vs.delete_version_chunks.call_args_list:
            called_v = call[1].get("index_version") or (call[0][2] if len(call[0]) > 2 else None)
            assert called_v != 1, "delete_version_chunks was invoked on active generation!"

        # Invariant check: active points remain in vector store
        v1_points = [
            c for c in vector_store.chunks if c.document_id == doc_id and c.index_version == 1
        ]
        assert len(v1_points) > 0


def test_first_index_post_activation_failure_never_leaves_active_pointing_to_deleted_points() -> (
    None
):
    """First-index post-activation failure never leaves active_index_version
    pointing to deleted points.
    """
    emb_provider = DeterministicEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    privacy_gate = RetrievalPrivacyGate()
    doc_id = f"doc-first-post-fail-{uuid.uuid4().hex[:8]}"
    tenant_id = "tenant-sec"

    with Session(bind=engine) as session:
        _create_test_doc(session, doc_id=doc_id, tenant_id=tenant_id, version=1)
        mock_vs = MagicMock(wraps=vector_store)
        mock_vs.delete_stale_chunks.side_effect = RuntimeError("Stale cleanup explosion")

        service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=mock_vs,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )

        res = service.index_document(session=session, document_id=doc_id, tenant_id=tenant_id)
        assert res.active_index_version == 1
        assert res.status == DocumentState.INDEXED

        # Assert points exist and match DB active version
        doc = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc_id))
        assert doc is not None
        assert doc.active_index_version == 1
        active_points = [
            c for c in vector_store.chunks if c.document_id == doc_id and c.index_version == 1
        ]
        assert len(active_points) > 0


def test_cleanup_marker_persistence_failure_does_not_revert_active_generation() -> None:
    """Cleanup-marker persistence failure does not revert the active generation."""
    emb_provider = DeterministicEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    privacy_gate = RetrievalPrivacyGate()
    doc_id = f"doc-cleanup-marker-fail-{uuid.uuid4().hex[:8]}"
    tenant_id = "tenant-sec"

    with Session(bind=engine) as session:
        _create_test_doc(session, doc_id=doc_id, tenant_id=tenant_id, version=1)
        mock_vs = MagicMock(wraps=vector_store)
        mock_vs.delete_stale_chunks.side_effect = VectorStoreError("Stale cleanup failed")

        service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=mock_vs,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )

        # Mock commit failure specifically on the post-activation cleanup flag commit
        original_commit = session.commit
        commit_calls = 0

        def flaky_commit() -> None:
            nonlocal commit_calls
            commit_calls += 1
            if commit_calls == 3:
                raise RuntimeError("Database error persisting index_cleanup_pending marker")
            original_commit()

        with patch.object(session, "commit", side_effect=flaky_commit):
            res = service.index_document(session=session, document_id=doc_id, tenant_id=tenant_id)
            assert res.active_index_version == 1
            assert res.status == DocumentState.INDEXED

        session.expire_all()
        doc = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc_id))
        assert doc is not None
        assert doc.status == DocumentState.INDEXED
        assert doc.active_index_version == 1


def test_db_chunk_metadata_and_vector_points_remain_consistent_in_every_scenario() -> None:
    """Assert DB chunk metadata and vector points remain consistent across operations."""
    emb_provider = DeterministicEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    privacy_gate = RetrievalPrivacyGate()
    doc_id = f"doc-consistency-{uuid.uuid4().hex[:8]}"
    tenant_id = "tenant-sec"

    with Session(bind=engine) as session:
        _create_test_doc(session, doc_id=doc_id, tenant_id=tenant_id, version=1)
        service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=vector_store,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )
        service.index_document(session=session, document_id=doc_id, tenant_id=tenant_id)

        # Check v1 consistency
        db_chunks_v1 = list(
            session.scalars(
                select(DocumentChunkRecord).where(DocumentChunkRecord.document_id == doc_id)
            )
        )
        vs_chunks_v1 = [
            c for c in vector_store.chunks if c.document_id == doc_id and c.index_version == 1
        ]
        assert len(db_chunks_v1) == len(vs_chunks_v1)
        assert {c.chunk_id for c in db_chunks_v1} == {c.point_id for c in vs_chunks_v1}

        # Upgrade doc to v2
        doc = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc_id))
        assert doc is not None
        doc.version = 2
        session.commit()

        service.index_document(session=session, document_id=doc_id, tenant_id=tenant_id)

        # Check v2 consistency: stale v1 chunks deleted, DB matches v2
        db_chunks_v2 = list(
            session.scalars(
                select(DocumentChunkRecord).where(DocumentChunkRecord.document_id == doc_id)
            )
        )
        vs_chunks_v2 = [
            c for c in vector_store.chunks if c.document_id == doc_id and c.index_version == 2
        ]
        vs_chunks_v1_post = [
            c for c in vector_store.chunks if c.document_id == doc_id and c.index_version == 1
        ]
        assert len(db_chunks_v2) == len(vs_chunks_v2)
        assert len(vs_chunks_v1_post) == 0
        assert {c.chunk_id for c in db_chunks_v2} == {c.point_id for c in vs_chunks_v2}


# ==============================================================================
# 2. Qdrant Contract Tests (Tenant Scroll & Payload Index Schema)
# ==============================================================================


def test_qdrant_verify_and_retrieve_points_tenant_scroll_contract() -> None:
    """Qdrant point operations must enforce tenant_id and has_id filter conditions via scroll."""
    mock_client = MagicMock()
    store = QdrantVectorStore(client=mock_client)

    rec = MagicMock()
    rec.id = "pt-1"
    rec.payload = {"tenant_id": "tenant-contract", "document_id": "doc-1"}
    mock_client.scroll.return_value = ([rec], None)

    # 1. verify_points
    verified = store.verify_points(["pt-1"], tenant_id="tenant-contract")
    assert verified is True
    assert mock_client.scroll.called
    call_filter = mock_client.scroll.call_args[1]["scroll_filter"]
    assert isinstance(call_filter, models.Filter)
    cond_has_id = next(c for c in call_filter.must if hasattr(c, "has_id"))
    assert cond_has_id.has_id == ["pt-1"]
    cond_tenant = next(c for c in call_filter.must if getattr(c, "key", None) == "tenant_id")
    assert cond_tenant.match.value == "tenant-contract"

    # 2. retrieve_points
    mock_client.scroll.reset_mock()
    results = store.retrieve_points(["pt-1"], tenant_id="tenant-contract")
    assert len(results) == 1
    assert results[0].point_id == "pt-1"
    assert results[0].payload["tenant_id"] == "tenant-contract"
    call_filter2 = mock_client.scroll.call_args[1]["scroll_filter"]
    cond_tenant2 = next(c for c in call_filter2.must if getattr(c, "key", None) == "tenant_id")
    assert cond_tenant2.match.value == "tenant-contract"


def test_qdrant_point_operations_batch_safely_for_large_sets() -> None:
    """verify_points and retrieve_points must batch safely for large point ID sets."""
    mock_client = MagicMock()
    store = QdrantVectorStore(client=mock_client)
    store.BATCH_SIZE = 100

    # 250 point IDs -> 3 batches of (100, 100, 50)
    point_ids = [f"point_{i}" for i in range(250)]

    def fake_scroll(*args: object, **kwargs: object) -> tuple[list[MagicMock], None]:
        scroll_filter = kwargs.get("scroll_filter")
        assert isinstance(scroll_filter, models.Filter)
        has_id_cond = next(c for c in scroll_filter.must if hasattr(c, "has_id"))
        recs = []
        for pid in has_id_cond.has_id:
            m = MagicMock()
            m.id = pid
            m.payload = {"tenant_id": "tenant-large"}
            recs.append(m)
        return recs, None

    mock_client.scroll.side_effect = fake_scroll

    assert store.verify_points(point_ids, tenant_id="tenant-large") is True
    assert mock_client.scroll.call_count == 3

    mock_client.scroll.reset_mock()
    mock_client.scroll.side_effect = fake_scroll
    retrieved = store.retrieve_points(point_ids, tenant_id="tenant-large")
    assert len(retrieved) == 250
    assert mock_client.scroll.call_count == 3


def test_qdrant_payload_index_schema_validation() -> None:
    """Payload index presence and types must be strictly validated; mismatch fails closed."""
    mock_client = MagicMock()
    store = QdrantVectorStore(client=mock_client)

    collection_info = MagicMock()
    collection_info.config.params.vectors = {
        "text_dense": MagicMock(size=1024, distance=models.Distance.COSINE)
    }
    collection_info.config.params.sparse_vectors = {"text_sparse": MagicMock()}

    # Case A: Correct payload indexes present -> passes
    collection_info.payload_schema = {
        "tenant_id": models.PayloadIndexInfo(data_type=models.PayloadSchemaType.KEYWORD, points=0),
        "document_id": models.PayloadIndexInfo(
            data_type=models.PayloadSchemaType.KEYWORD, points=0
        ),
        "classification": models.PayloadIndexInfo(
            data_type=models.PayloadSchemaType.KEYWORD, points=0
        ),
        "page_number": models.PayloadIndexInfo(
            data_type=models.PayloadSchemaType.INTEGER, points=0
        ),
        "redaction_version": models.PayloadIndexInfo(
            data_type=models.PayloadSchemaType.INTEGER, points=0
        ),
        "index_version": models.PayloadIndexInfo(
            data_type=models.PayloadSchemaType.INTEGER, points=0
        ),
    }
    mock_client.get_collection.return_value = collection_info

    store.ensure_collection(dense_dim=1024)
    assert not mock_client.create_payload_index.called

    # Case B: Wrong-type payload index -> raises VectorStoreError and fails closed
    wrong_type_info = MagicMock()
    wrong_type_info.config = collection_info.config
    wrong_type_info.payload_schema = {
        "tenant_id": models.PayloadIndexInfo(data_type=models.PayloadSchemaType.KEYWORD, points=0),
        "document_id": models.PayloadIndexInfo(
            data_type=models.PayloadSchemaType.KEYWORD, points=0
        ),
        "classification": models.PayloadIndexInfo(
            data_type=models.PayloadSchemaType.KEYWORD, points=0
        ),
        "page_number": models.PayloadIndexInfo(
            data_type=models.PayloadSchemaType.TEXT, points=0
        ),  # Expected INTEGER!
        "redaction_version": models.PayloadIndexInfo(
            data_type=models.PayloadSchemaType.INTEGER, points=0
        ),
        "index_version": models.PayloadIndexInfo(
            data_type=models.PayloadSchemaType.INTEGER, points=0
        ),
    }
    mock_client.get_collection.return_value = wrong_type_info

    with pytest.raises(
        VectorStoreError, match="has schema type 'text', expected 'integer'. Fail closed."
    ):
        store.ensure_collection(dense_dim=1024)

    # Case C: Missing index created and verified by re-fetching metadata
    missing_info = MagicMock()
    missing_info.config = collection_info.config
    missing_info.payload_schema = {
        "tenant_id": models.PayloadIndexInfo(data_type=models.PayloadSchemaType.KEYWORD, points=0),
        # missing document_id, classification, page_number, redaction_version, index_version
    }

    # After create_payload_index, get_collection returns full schema
    mock_client.get_collection.side_effect = [
        missing_info,  # initial inspect
        collection_info,  # re-fetch 1
        collection_info,  # re-fetch 2
        collection_info,  # re-fetch 3
        collection_info,  # re-fetch 4
        collection_info,  # re-fetch 5
    ]

    store.ensure_collection(dense_dim=1024)
    assert mock_client.create_payload_index.call_count == 5


# ==============================================================================
# 3. Guard Contract Test
# ==============================================================================


def test_indexing_service_owns_authoritative_active_version_guard() -> None:
    """The indexing service authoritatively guards against delete_version_chunks
    on active versions.
    """
    emb_provider = DeterministicEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    privacy_gate = RetrievalPrivacyGate()
    doc_id = f"doc-guard-auth-{uuid.uuid4().hex[:8]}"
    tenant_id = "tenant-sec"

    with Session(bind=engine) as session:
        _create_test_doc(session, doc_id=doc_id, tenant_id=tenant_id, version=1)
        mock_vs = MagicMock(wraps=vector_store)

        service = DocumentIndexingService(
            embedding_provider=emb_provider,
            vector_store=mock_vs,
            privacy_gate=privacy_gate,
            chunker=DeterministicMaskedChunker(),
        )

        # Index v1
        service.index_document(session=session, document_id=doc_id, tenant_id=tenant_id)

        # Verification failure on re-attempting same version
        mock_vs.verify_points.return_value = False
        with pytest.raises(VectorStoreError):
            service.index_document(session=session, document_id=doc_id, tenant_id=tenant_id)

        # Assert delete_version_chunks was NEVER called for the active version 1
        for call in mock_vs.delete_version_chunks.call_args_list:
            called_v = call[1].get("index_version") or (call[0][2] if len(call[0]) > 2 else None)
            assert called_v != 1, (
                "delete_version_chunks was called for the authoritative active version!"
            )


# ==============================================================================
# 4. API Leak Regression Tests
# ==============================================================================


def test_api_and_audit_never_leak_injected_secrets(client: TestClient) -> None:
    """Injected secrets in exceptions must NEVER appear in any HTTP response or audit record."""
    secret_path = "/private/models/bge"
    qdrant_secret_url = "http://internal-host:6333"
    vault_key_path = "/private/key"

    # 1. Search endpoint: ModelUnavailableError with SECRET_PATH
    mock_retrieval_service = MagicMock()
    mock_retrieval_service.search.side_effect = ModelUnavailableError(
        f"Failed loading model weights at SECRET_PATH={secret_path}"
    )

    app.dependency_overrides[get_retrieval_service] = lambda: mock_retrieval_service
    try:
        res = client.post(
            "/api/v1/retrieval/search",
            json={"query": "Clearance query", "top_k": 5},
            headers={"X-Actor-ID": "test-user", "X-Tenant-ID": "tenant-sec"},
        )
        assert res.status_code == 503
        assert secret_path not in res.text
        assert "SECRET_PATH" not in res.text
        assert res.json()["detail"] == "Local embedding model unavailable."

        # Search endpoint: VectorStoreUnavailableError with QDRANT_SECRET_URL
        mock_retrieval_service.search.side_effect = VectorStoreUnavailableError(
            f"Cannot connect to QDRANT_SECRET_URL={qdrant_secret_url}"
        )
        res = client.post(
            "/api/v1/retrieval/search",
            json={"query": "Clearance query", "top_k": 5},
            headers={"X-Actor-ID": "test-user", "X-Tenant-ID": "tenant-sec"},
        )
        assert res.status_code == 503
        assert qdrant_secret_url not in res.text
        assert "QDRANT_SECRET_URL" not in res.text
        assert res.json()["detail"] == "Vector store unavailable."

        # Search endpoint: ValueError containing SECRET_PATH
        mock_retrieval_service.search.side_effect = ValueError(
            f"Invalid query near SECRET_PATH={secret_path}"
        )
        res = client.post(
            "/api/v1/retrieval/search",
            json={"query": "Clearance query", "top_k": 5},
            headers={"X-Actor-ID": "test-user", "X-Tenant-ID": "tenant-sec"},
        )
        assert res.status_code == 400
        assert secret_path not in res.text
        assert "SECRET_PATH" not in res.text
        assert res.json()["detail"] == "Invalid retrieval query or parameters."
    finally:
        app.dependency_overrides.pop(get_retrieval_service, None)

    # 2. Evidence page-image endpoint: VaultConfigurationError with VAULT_KEY_PATH
    mock_vs = MagicMock()
    mock_hit = MagicMock()
    mock_hit.payload = {
        "document_id": "doc-leak-1",
        "page_number": 1,
        "index_version": 1,
        "document_evidence_hash": _valid_hash("evidence_doc-leak-1"),
    }
    mock_vs.retrieve_points.return_value = [mock_hit]
    app.dependency_overrides[get_vector_store] = lambda: mock_vs

    with Session(bind=engine) as session:
        _create_test_doc(
            session,
            doc_id="doc-leak-1",
            tenant_id="tenant-sec",
            version=1,
            status=DocumentState.INDEXED,
        )

    try:
        with patch(
            "koshshield.api.routes.retrieval.EncryptedVault",
            side_effect=VaultConfigurationError(
                f"Missing master key at VAULT_KEY_PATH={vault_key_path}"
            ),
        ):
            res = client.get(
                "/api/v1/retrieval/evidence/chunk-123/page-image",
                headers={"X-Actor-ID": "test-user", "X-Tenant-ID": "tenant-sec"},
            )
            assert res.status_code == 503
            assert vault_key_path not in res.text
            assert "VAULT_KEY_PATH" not in res.text
            assert res.json()["detail"] == "Vault storage is unavailable or misconfigured."
    finally:
        app.dependency_overrides.pop(get_vector_store, None)

    # 3. Status endpoints: retrieval/status and system/status
    res_status = client.get(
        "/api/v1/retrieval/status",
        headers={"X-Actor-ID": "test-user", "X-Tenant-ID": "tenant-sec"},
    )
    assert res_status.status_code == 200
    assert secret_path not in res_status.text
    assert qdrant_secret_url not in res_status.text
    assert vault_key_path not in res_status.text

    res_sys = client.get("/api/v1/system/status")
    assert res_sys.status_code == 200
    assert secret_path not in res_sys.text
    assert qdrant_secret_url not in res_sys.text
    assert vault_key_path not in res_sys.text

    # 4. Indexing endpoint: Document indexing failure audit record check
    mock_indexing_service = MagicMock()
    mock_indexing_service.index_document.side_effect = RuntimeError(
        f"Embedding failure for SECRET_PATH={secret_path} and QDRANT_SECRET_URL={qdrant_secret_url}"
    )
    app.dependency_overrides[get_indexing_service] = lambda: mock_indexing_service

    try:
        res_idx = client.post(
            "/api/v1/documents/doc-leak-1/index",
            headers={"X-Actor-ID": "test-user", "X-Tenant-ID": "tenant-sec", "X-Roles": "admin"},
        )
        assert secret_path not in res_idx.text
        assert qdrant_secret_url not in res_idx.text
        assert vault_key_path not in res_idx.text
    finally:
        app.dependency_overrides.pop(get_indexing_service, None)

    # 5. Database audit record inspection: verify NO secret strings exist in audit_events
    with Session(bind=engine) as session:
        events = list(session.scalars(select(AuditEvent)))
        for ev in events:
            details_str = json.dumps(ev.details or {})
            assert secret_path not in details_str, (
                f"Found secret_path in AuditEvent: {ev.event_type}"
            )
            assert qdrant_secret_url not in details_str, (
                f"Found qdrant_secret_url in AuditEvent: {ev.event_type}"
            )
            assert vault_key_path not in details_str, (
                f"Found vault_key_path in AuditEvent: {ev.event_type}"
            )
            assert secret_path not in str(ev.event_type)
            assert qdrant_secret_url not in str(ev.event_type)
            assert vault_key_path not in str(ev.event_type)
