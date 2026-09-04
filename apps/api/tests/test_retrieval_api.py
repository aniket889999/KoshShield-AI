import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from koshshield.api.routes.retrieval import get_embedding_provider, get_vector_store
from koshshield.database import engine
from koshshield.main import app
from koshshield.models import DocumentPageRecord, DocumentRecord, DocumentState
from koshshield.services.retrieval.embeddings.deterministic_fake import (
    DeterministicEmbeddingProvider,
)
from koshshield.services.retrieval.vector_store import InMemoryVectorStore


@pytest.fixture
def mock_dependencies() -> tuple[DeterministicEmbeddingProvider, InMemoryVectorStore]:
    fake_emb = DeterministicEmbeddingProvider()
    fake_store = InMemoryVectorStore()
    app.dependency_overrides[get_embedding_provider] = lambda: fake_emb
    app.dependency_overrides[get_vector_store] = lambda: fake_store
    yield fake_emb, fake_store
    app.dependency_overrides.pop(get_embedding_provider, None)
    app.dependency_overrides.pop(get_vector_store, None)


def test_retrieval_status_endpoint(
    client: TestClient,
    mock_dependencies: tuple[DeterministicEmbeddingProvider, InMemoryVectorStore],
) -> None:
    res = client.get("/api/v1/retrieval/status")
    assert res.status_code == 200
    data = res.json()
    assert data["vector_store_status"] == "ready"
    assert data["embedding_model_status"] == "ready"
    assert "collection_name" in data


def test_document_indexing_lifecycle_endpoint(
    client: TestClient,
    mock_dependencies: tuple[DeterministicEmbeddingProvider, InMemoryVectorStore],
) -> None:
    with Session(bind=engine) as session:
        doc = DocumentRecord(
            id=str(uuid.uuid4()),
            filename="tender_approved.pdf",
            media_type="application/pdf",
            size_bytes=4096,
            sha256="abc1234567890def1234567890abcdef1234567890abcdef1234567890abcdef",
            vault_path="vault/tender.ksh",
            status=DocumentState.INDEX_READY,
        )
        page1 = DocumentPageRecord(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            page_number=1,
            extraction_method="native_pdf",
            text_hash="thash-1",
            encrypted_artifact_path="vault/p1.ksh",
            masked_text="The Ministry tender deadline is set for 30th September 2026.",
        )
        session.add_all([doc, page1])
        session.commit()
        doc_id = doc.id

    # 1. Index document
    index_res = client.post(f"/api/v1/documents/{doc_id}/index")
    assert index_res.status_code == 200
    idx_data = index_res.json()
    assert idx_data["document_id"] == doc_id
    assert idx_data["status"] == "INDEXED"
    assert idx_data["chunk_count"] >= 1

    # 2. Check indexing status
    status_res = client.get(f"/api/v1/documents/{doc_id}/indexing")
    assert status_res.status_code == 200
    assert status_res.json()["status"] == "INDEXED"
    assert status_res.json()["chunk_count"] >= 1

    # 3. Search for evidence
    search_res = client.post(
        "/api/v1/retrieval/search",
        json={"query": "tender deadline", "top_k": 3},
    )
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert search_data["total_found"] >= 1
    evidence = search_data["results"][0]
    assert evidence["document_id"] == doc_id
    assert "tender_approved.pdf" in evidence["citation_label"]
    assert "September" in evidence["masked_snippet"]
