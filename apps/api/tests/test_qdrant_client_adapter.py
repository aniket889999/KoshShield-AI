from unittest.mock import MagicMock

import pytest
from qdrant_client import models

from koshshield.services.retrieval.vector_store.interfaces import (
    VectorStoreChunk,
    VectorStoreError,
)
from koshshield.services.retrieval.vector_store.qdrant import QdrantVectorStore


def test_qdrant_dimension_mismatch_fails_closed() -> None:
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True

    # Existing collection with 512 dimensions
    mock_collection_info = MagicMock()
    mock_dense_cfg = MagicMock()
    mock_dense_cfg.size = 512
    mock_dense_cfg.distance = models.Distance.COSINE
    mock_collection_info.config.params.vectors = {"text_dense": mock_dense_cfg}
    mock_collection_info.config.params.sparse_vectors = {"text_sparse": MagicMock()}
    mock_client.get_collection.return_value = mock_collection_info

    store = QdrantVectorStore(
        qdrant_url="http://localhost:6333",
        collection_name="test_collection",
        client=mock_client,
    )

    # Calling ensure_collection with 1024 dims must fail closed without deleting collection
    with pytest.raises(VectorStoreError) as exc_info:
        store.ensure_collection(dense_dim=1024)

    assert "text_dense dim 512, expected 1024" in str(exc_info.value)
    assert "Silently recreating collection is prohibited" in str(exc_info.value)
    # Ensure delete_collection was NEVER called!
    assert not mock_client.delete_collection.called


def test_qdrant_distance_mismatch_fails_closed() -> None:
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True

    # Existing collection with Euclidean distance instead of Cosine
    mock_collection_info = MagicMock()
    mock_dense_cfg = MagicMock()
    mock_dense_cfg.size = 1024
    mock_dense_cfg.distance = models.Distance.EUCLID
    mock_collection_info.config.params.vectors = {"text_dense": mock_dense_cfg}
    mock_collection_info.config.params.sparse_vectors = {"text_sparse": MagicMock()}
    mock_client.get_collection.return_value = mock_collection_info

    store = QdrantVectorStore(
        qdrant_url="http://localhost:6333",
        collection_name="test_collection",
        client=mock_client,
    )

    with pytest.raises(VectorStoreError) as exc_info:
        store.ensure_collection(dense_dim=1024)

    assert "expected Cosine" in str(exc_info.value)
    assert not mock_client.delete_collection.called


def test_qdrant_missing_sparse_vector_fails_closed() -> None:
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True

    mock_collection_info = MagicMock()
    mock_dense_cfg = MagicMock()
    mock_dense_cfg.size = 1024
    mock_dense_cfg.distance = models.Distance.COSINE
    mock_collection_info.config.params.vectors = {"text_dense": mock_dense_cfg}
    # Missing text_sparse
    mock_collection_info.config.params.sparse_vectors = {}
    mock_client.get_collection.return_value = mock_collection_info

    store = QdrantVectorStore(
        qdrant_url="http://localhost:6333",
        collection_name="test_collection",
        client=mock_client,
    )

    with pytest.raises(VectorStoreError) as exc_info:
        store.ensure_collection(dense_dim=1024)

    assert "missing sparse vector 'text_sparse'" in str(exc_info.value)


def test_qdrant_upsert_typed_point_struct() -> None:
    mock_client = MagicMock()
    store = QdrantVectorStore(
        qdrant_url="http://localhost:6333",
        collection_name="test_collection",
        client=mock_client,
    )

    chunk = VectorStoreChunk(
        point_id="00000000-0000-0000-0000-000000000001",
        chunk_id="00000000-0000-0000-0000-000000000001",
        tenant_id="dept-law",
        document_id="doc-99",
        page_number=1,
        redaction_version=2,
        index_version=2,
        chunk_sequence=0,
        masked_text="Masked text [REDACTED_AADHAAR]",
        char_start=0,
        char_end=30,
        masked_content_hash="hash1",
        document_evidence_hash="ev1",
        classification="CONFIDENTIAL",
        document_filename="doc.pdf",
        indexed_at="2026-09-04T12:00:00Z",
        dense_vector=[0.1] * 1024,
        sparse_indices=[12, 45],
        sparse_values=[0.5, 0.9],
    )

    count = store.upsert_chunks([chunk])
    assert count == 1
    assert mock_client.upsert.called
    kwargs = mock_client.upsert.call_args[1]
    assert kwargs["collection_name"] == "test_collection"
    points = kwargs["points"]
    assert len(points) == 1
    point = points[0]
    assert isinstance(point, models.PointStruct)
    assert point.id == "00000000-0000-0000-0000-000000000001"
    assert point.payload["index_version"] == 2
    assert point.payload["tenant_id"] == "dept-law"


def test_qdrant_verify_points_tenant_isolation() -> None:
    mock_client = MagicMock()
    store = QdrantVectorStore(
        qdrant_url="http://localhost:6333",
        collection_name="test_collection",
        client=mock_client,
    )

    # Point belonging to tenant-alpha
    mock_rec1 = MagicMock()
    mock_rec1.id = "p1"
    mock_rec1.payload = {"tenant_id": "tenant-alpha"}

    mock_client.retrieve.return_value = [mock_rec1]

    # Verify matching tenant returns True
    assert store.verify_points(["p1"], tenant_id="tenant-alpha") is True

    # Verify different tenant returns False (isolation violation)
    assert store.verify_points(["p1"], tenant_id="tenant-beta") is False


def test_qdrant_stale_deletion_version_filter() -> None:
    mock_client = MagicMock()
    store = QdrantVectorStore(
        qdrant_url="http://localhost:6333",
        collection_name="test_collection",
        client=mock_client,
    )

    store.delete_stale_chunks(
        document_id="doc-123",
        tenant_id="tenant-gamma",
        active_version=3,
    )

    assert mock_client.delete.called
    del_filter = mock_client.delete.call_args[1]["points_selector"].filter
    assert isinstance(del_filter, models.Filter)
    cond_keys = {c.key: c for c in del_filter.must}
    assert "document_id" in cond_keys
    assert cond_keys["document_id"].match.value == "doc-123"
    assert "tenant_id" in cond_keys
    assert cond_keys["tenant_id"].match.value == "tenant-gamma"
    assert "index_version" in cond_keys
    assert cond_keys["index_version"].range.lt == 3


def test_real_docker_qdrant_integration_if_running() -> None:
    """Live integration test against Docker Qdrant (127.0.0.1:6333) if running."""
    import httpx

    try:
        r = httpx.get("http://127.0.0.1:6333/healthz", timeout=1.0)
        if r.status_code != 200:
            pytest.skip("Docker Qdrant is not responsive on 127.0.0.1:6333")
    except Exception:
        pytest.skip("Docker Qdrant is not running")

    test_col = "koshshield_test_integration"
    store = QdrantVectorStore(
        qdrant_url="http://127.0.0.1:6333",
        collection_name=test_col,
    )

    try:
        # Create test collection
        store.ensure_collection(dense_dim=8)

        # Upsert point
        chunk = VectorStoreChunk(
            point_id="11111111-1111-1111-1111-111111111111",
            chunk_id="11111111-1111-1111-1111-111111111111",
            tenant_id="test-tenant",
            document_id="doc-test-1",
            page_number=1,
            redaction_version=1,
            index_version=1,
            chunk_sequence=0,
            masked_text="Integrated test text",
            char_start=0,
            char_end=20,
            masked_content_hash="chash",
            document_evidence_hash="ehash",
            classification="CONFIDENTIAL",
            document_filename="test.pdf",
            indexed_at="2026-09-04T12:00:00Z",
            dense_vector=[0.1] * 8,
            sparse_indices=[1, 2],
            sparse_values=[0.5, 0.8],
        )
        store.upsert_chunks([chunk])

        # Verify points
        pid = "11111111-1111-1111-1111-111111111111"
        assert store.verify_points([pid], "test-tenant") is True
        assert store.verify_points([pid], "other-tenant") is False

        # Search dense
        hits = store.search_dense(
            query_vector=[0.1] * 8,
            tenant_id="test-tenant",
            limit=5,
        )
        assert len(hits) == 1
        assert hits[0].payload["tenant_id"] == "test-tenant"

        # Search cross-tenant returns empty
        hits_other = store.search_dense(
            query_vector=[0.1] * 8,
            tenant_id="other-tenant",
            limit=5,
        )
        assert len(hits_other) == 0

    finally:
        import contextlib

        with contextlib.suppress(Exception):
            store._client.delete_collection(test_col)
