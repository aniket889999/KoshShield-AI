import pytest
from qdrant_client import models

from koshshield.services.retrieval.vector_store.qdrant import QdrantVectorStore


def test_qdrant_url_validation() -> None:
    # Valid local URLs
    QdrantVectorStore(qdrant_url="http://localhost:6333")
    QdrantVectorStore(qdrant_url="http://127.0.0.1:6333")
    QdrantVectorStore(qdrant_url="http://qdrant:6333")

    # Disallowed external / cloud URLs
    with pytest.raises(ValueError) as exc_info:
        QdrantVectorStore(qdrant_url="https://cloud.qdrant.io:6333")
    err_msg = str(exc_info.value).lower()
    assert "target localhost" in err_msg or "private container name" in err_msg

    # Disallowed external IP
    with pytest.raises(ValueError) as exc_info2:
        QdrantVectorStore(qdrant_url="http://192.168.1.50:6333")
    assert "target localhost" in str(exc_info2.value).lower()


def test_qdrant_build_filter_mandatory_tenant() -> None:
    # Must enforce tenant
    f1 = QdrantVectorStore._build_filter(tenant_id="dept-finance")
    assert isinstance(f1, models.Filter)
    assert len(f1.must) == 1
    assert f1.must[0].key == "tenant_id"
    assert f1.must[0].match.value == "dept-finance"

    # Enforce permitted documents
    f2 = QdrantVectorStore._build_filter(
        tenant_id="dept-finance",
        permitted_document_ids=["doc-1", "doc-2"],
    )
    assert len(f2.must) == 2
    assert f2.must[0].key == "tenant_id"
    assert f2.must[1].key == "document_id"
    assert f2.must[1].match.any == ["doc-1", "doc-2"]

    # If permitted_document_ids is empty, must forbid all access
    f3 = QdrantVectorStore._build_filter(
        tenant_id="dept-finance",
        permitted_document_ids=[],
    )
    assert len(f3.must) == 2
    assert f3.must[1].key == "document_id"
    assert f3.must[1].match.value == "__NO_ACCESS__"
