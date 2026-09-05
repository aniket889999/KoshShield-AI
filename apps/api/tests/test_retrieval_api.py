import hashlib
import os
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from koshshield.api.routes.retrieval import get_embedding_provider, get_vector_store
from koshshield.database import engine
from koshshield.main import app
from koshshield.models import DocumentPageRecord, DocumentRecord, DocumentState
from koshshield.security.vault import EncryptedVault
from koshshield.services.retrieval.embeddings.deterministic_fake import (
    DeterministicEmbeddingProvider,
)
from koshshield.services.retrieval.vector_store import InMemoryVectorStore
from koshshield.services.retrieval.vector_store.interfaces import VectorStoreChunk


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
    assert idx_data["active_index_version"] == 1

    # 2. Check indexing status
    status_res = client.get(f"/api/v1/documents/{doc_id}/indexing")
    assert status_res.status_code == 200
    assert status_res.json()["status"] == "INDEXED"
    assert status_res.json()["chunk_count"] >= 1
    assert status_res.json()["active_index_version"] == 1

    # 3. Search for evidence
    search_res = client.post(
        "/api/v1/retrieval/search",
        json={"query": "tender deadline", "top_k": 3},
    )
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert search_data["total_found"] >= 1
    assert search_data["query_length"] == len("tender deadline")
    assert "duration_ms" in search_data
    assert "query_hash" not in search_data

    evidence = search_data["results"][0]
    assert evidence["document_id"] == doc_id
    assert "tender_approved.pdf" in evidence["citation_label"]
    assert "September" in evidence["masked_snippet"]
    assert len(evidence["evidence_hash"]) == 64
    assert len(evidence["masked_content_hash"]) == 64
    assert evidence["index_version"] == 1

    # 4. Prove body tenant spoofing is ignored / impossible
    spoofed_res = client.post(
        "/api/v1/retrieval/search",
        headers={"X-Tenant-ID": "tenant-authorized"},
        json={"query": "tender deadline", "tenant_id": "tenant-spoofed"},
    )
    assert spoofed_res.status_code == 200
    assert spoofed_res.json()["tenant_id"] == "tenant-authorized"


def test_visual_evidence_page_image_requires_tenant_scoped_chunk(
    client: TestClient,
    mock_dependencies: tuple[DeterministicEmbeddingProvider, InMemoryVectorStore],
) -> None:
    embedding_provider, vector_store = mock_dependencies
    doc_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())
    evidence_hash = "e" * 64
    image_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?"
        b"\x00\x05\xfe\x02\xfeA\xde\xfc\x9b\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    vault = EncryptedVault(
        Path(os.environ["KOSHSHIELD_VAULT_DIR"]),
        os.environ["KOSHSHIELD_MASTER_KEY_BASE64"],
    )
    image_path = vault.encrypt(
        document_id=f"{doc_id}_p1_image",
        evidence_hash=image_hash,
        plaintext=image_bytes,
    )

    with Session(bind=engine) as session:
        doc = DocumentRecord(
            id=doc_id,
            filename="visual-evidence.pdf",
            media_type="application/pdf",
            size_bytes=1024,
            sha256=evidence_hash,
            vault_path="vault/doc.ksh",
            status=DocumentState.INDEXED,
            active_index_version=2,
        )
        page = DocumentPageRecord(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            page_number=1,
            width=612,
            height=792,
            extraction_method="native_pdf",
            text_hash="text-hash",
            encrypted_artifact_path="vault/page-text.ksh",
            page_image_sha256=image_hash,
            page_image_media_type="image/png",
            encrypted_page_image_path=str(image_path),
            masked_text="Masked table content",
        )
        session.add_all([doc, page])
        session.commit()

    emb = embedding_provider.embed_query("Masked table content")
    vector_store.upsert_chunks(
        [
            VectorStoreChunk(
                point_id=chunk_id,
                chunk_id=chunk_id,
                tenant_id="default",
                document_id=doc_id,
                page_number=1,
                redaction_version=2,
                index_version=2,
                chunk_sequence=0,
                masked_text="Masked table content",
                char_start=0,
                char_end=20,
                masked_content_hash="m" * 64,
                document_evidence_hash=evidence_hash,
                classification="CONFIDENTIAL",
                document_filename="visual-evidence.pdf",
                indexed_at="2026-09-06T12:00:00Z",
                dense_vector=emb.dense,
                sparse_indices=emb.sparse_indices,
                sparse_values=emb.sparse_values,
            )
        ]
    )

    allowed = client.get(f"/api/v1/retrieval/evidence/{chunk_id}/page-image")
    assert allowed.status_code == 200
    assert allowed.headers["content-type"] == "image/png"
    assert allowed.content == image_bytes

    blocked = client.get(
        f"/api/v1/retrieval/evidence/{chunk_id}/page-image",
        headers={"X-Tenant-ID": "other-tenant"},
    )
    assert blocked.status_code == 404
