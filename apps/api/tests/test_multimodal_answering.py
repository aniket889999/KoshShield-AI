import uuid
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from koshshield.api.routes.retrieval import (
    get_embedding_provider,
    get_llama_client,
    get_vector_store,
)
from koshshield.config import get_settings
from koshshield.database import engine
from koshshield.main import app
from koshshield.models import (
    AuditEvent,
    DocumentChunkRecord,
    DocumentRecord,
    DocumentState,
)
from koshshield.services.retrieval.embeddings.deterministic_fake import (
    DeterministicEmbeddingProvider,
)
from koshshield.services.retrieval.llama_cpp_client import (
    LlamaCppIncapableError,
    LlamaCppMultimodalClient,
    LlamaCppSecurityError,
    LlamaCppUnavailableError,
)
from koshshield.services.retrieval.vector_store import InMemoryVectorStore
from koshshield.services.retrieval.vector_store.interfaces import VectorStoreChunk


@pytest.fixture
def mock_multimodal_setup() -> tuple[
    DeterministicEmbeddingProvider, InMemoryVectorStore, MagicMock
]:
    fake_emb = DeterministicEmbeddingProvider()
    fake_store = InMemoryVectorStore()
    mock_client = MagicMock(spec=LlamaCppMultimodalClient)
    mock_client.model_id = "Qwen3VL-4B-Instruct"

    app.dependency_overrides[get_embedding_provider] = lambda: fake_emb
    app.dependency_overrides[get_vector_store] = lambda: fake_store
    app.dependency_overrides[get_llama_client] = lambda: mock_client

    yield fake_emb, fake_store, mock_client

    app.dependency_overrides.pop(get_embedding_provider, None)
    app.dependency_overrides.pop(get_vector_store, None)
    app.dependency_overrides.pop(get_llama_client, None)


def test_ssrf_and_host_validation_rejections() -> None:
    # 1. Reject remote URL
    with pytest.raises(LlamaCppSecurityError, match="prohibited"):
        LlamaCppMultimodalClient(base_url="http://remote-inference.ai/v1")

    # 2. Reject arbitrary single-label host
    with pytest.raises(LlamaCppSecurityError, match="prohibited"):
        LlamaCppMultimodalClient(base_url="http://internal-victim/v1")

    # 3. Reject credentials in URL
    with pytest.raises(LlamaCppSecurityError, match="Credentials"):
        LlamaCppMultimodalClient(base_url="http://user:pass@localhost:8080/v1")

    # 4. Reject invalid schemes
    with pytest.raises(LlamaCppSecurityError, match="Prohibited scheme"):
        LlamaCppMultimodalClient(base_url="ftp://localhost:8080/v1")

    # 5. Permitted hosts: localhost, 127.0.0.1, ::1, and explicit service name
    c1 = LlamaCppMultimodalClient(base_url="http://127.0.0.1:8080/v1")
    assert c1.base_url == "http://127.0.0.1:8080/v1"

    c2 = LlamaCppMultimodalClient(
        base_url="http://llama-server:8080/v1", service_name="llama-server"
    )
    assert c2.base_url == "http://llama-server:8080/v1"


def test_trust_env_false_and_no_redirects() -> None:
    # Verify that client initialization and call configuration forbids proxies and redirects
    client = LlamaCppMultimodalClient(base_url="http://localhost:8080/v1")

    with patch("httpx.Client") as mock_httpx:
        mock_ctx = MagicMock()
        mock_httpx.return_value.__enter__.return_value = mock_ctx
        mock_ctx.get.return_value.status_code = 200
        mock_ctx.get.return_value.json.return_value = {
            "data": [{"id": "Qwen3VL-4B-Instruct", "meta": {"has_vision": True}}]
        }

        client.check_health_and_capability()

        # Assert trust_env is False and follow_redirects is False
        mock_httpx.assert_called_with(
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(connect=3.0, read=5.0, write=3.0, pool=3.0),
        )


def test_capability_absence_raises_incapable_error() -> None:
    client = LlamaCppMultimodalClient(
        base_url="http://localhost:8080/v1", model_id="Qwen3VL-4B-Instruct"
    )

    with patch("httpx.Client") as mock_httpx:
        mock_ctx = MagicMock()
        mock_httpx.return_value.__enter__.return_value = mock_ctx
        # Return model without vision capability
        mock_ctx.get.return_value.status_code = 200
        mock_ctx.get.return_value.json.return_value = {
            "data": [{"id": "Qwen-Text-Only-Model", "meta": {"has_vision": False}}]
        }

        with pytest.raises(LlamaCppIncapableError):
            client.check_health_and_capability()


def test_feature_flag_disabled_returns_503(
    client: TestClient,
    mock_multimodal_setup: tuple[DeterministicEmbeddingProvider, InMemoryVectorStore, MagicMock],
) -> None:
    # Ensure flag is False
    settings = get_settings()
    settings.enable_multimodal_answering = False

    res = client.post(
        "/api/v1/retrieval/answer",
        json={"query": "What is the budget?", "top_k": 3},
        headers={"X-Tenant-ID": "default", "X-Actor-ID": "test-user"},
    )
    assert res.status_code == 503
    assert "disabled" in res.json()["detail"].lower()


def test_grounded_answer_generation_and_citation_validation(
    client: TestClient,
    mock_multimodal_setup: tuple[DeterministicEmbeddingProvider, InMemoryVectorStore, MagicMock],
) -> None:
    embedding_provider, vector_store, mock_llama = mock_multimodal_setup
    settings = get_settings()
    settings.enable_multimodal_answering = True

    doc_id = str(uuid.uuid4())
    chunk_id_1 = str(uuid.uuid4())
    chunk_id_2 = str(uuid.uuid4())
    fake_chunk_id = str(uuid.uuid4())
    evidence_hash = "e" * 64

    # Seed Document and Chunks in DB
    with Session(bind=engine) as session:
        doc = DocumentRecord(
            id=doc_id,
            tenant_id="default",
            filename="procurement_rules.pdf",
            media_type="application/pdf",
            size_bytes=4096,
            sha256=evidence_hash,
            vault_path=f"vault/{doc_id}.ksh",
            status=DocumentState.INDEXED,
            active_index_version=1,
        )
        c1 = DocumentChunkRecord(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            page_number=1,
            chunk_sequence=0,
            index_version=1,
            chunk_id=chunk_id_1,
            char_start=0,
            char_end=50,
            masked_content_hash="c1" + "0" * 62,
        )
        c2 = DocumentChunkRecord(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            page_number=2,
            chunk_sequence=1,
            index_version=1,
            chunk_id=chunk_id_2,
            char_start=0,
            char_end=50,
            masked_content_hash="c2" + "0" * 62,
        )
        session.add_all([doc, c1, c2])
        session.commit()

    # Seed in vector store
    emb = embedding_provider.embed_query("Budget allocation limit")
    vector_store.upsert_chunks(
        [
            VectorStoreChunk(
                point_id=chunk_id_1,
                chunk_id=chunk_id_1,
                tenant_id="default",
                document_id=doc_id,
                page_number=1,
                redaction_version=1,
                index_version=1,
                chunk_sequence=0,
                masked_text="Annual procurement budget limit is fixed at INR 50,00,000.",
                char_start=0,
                char_end=50,
                masked_content_hash="c1" + "0" * 62,
                document_evidence_hash=evidence_hash,
                classification="CONFIDENTIAL",
                document_filename="procurement_rules.pdf",
                indexed_at="2026-09-06T12:00:00Z",
                dense_vector=emb.dense,
                sparse_indices=emb.sparse_indices,
                sparse_values=emb.sparse_values,
            ),
            VectorStoreChunk(
                point_id=chunk_id_2,
                chunk_id=chunk_id_2,
                tenant_id="default",
                document_id=doc_id,
                page_number=2,
                redaction_version=1,
                index_version=1,
                chunk_sequence=1,
                masked_text="Approval requires General Manager sanction.",
                char_start=0,
                char_end=50,
                masked_content_hash="c2" + "0" * 62,
                document_evidence_hash=evidence_hash,
                classification="CONFIDENTIAL",
                document_filename="procurement_rules.pdf",
                indexed_at="2026-09-06T12:00:00Z",
                dense_vector=emb.dense,
                sparse_indices=emb.sparse_indices,
                sparse_values=emb.sparse_values,
            ),
        ]
    )

    # Configure mock llama client: returns valid citation for chunk_id_1 AND a hallucinated chunk_id
    mock_llama.generate_grounded_answer.return_value = {
        "answer": "The budget is capped at INR 50,00,000 per annual procurement rules.",
        "cited_chunk_ids": [chunk_id_1, fake_chunk_id],  # fake_chunk_id must be rejected
        "insufficient_evidence": False,
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
    }

    res = client.post(
        "/api/v1/retrieval/answer",
        json={"query": "What is the budget limit?", "top_k": 3},
        headers={"X-Tenant-ID": "default", "X-Actor-ID": "analyst-1"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["insufficient_evidence"] is False
    assert "50,00,000" in data["answer"]
    # Only valid retrieved chunk_id is kept; hallucinated fake_chunk_id is rejected
    assert data["cited_chunk_ids"] == [chunk_id_1]
    assert len(data["citations"]) == 1
    cit = data["citations"][0]
    assert cit["chunk_id"] == chunk_id_1
    assert cit["document_filename"] == "procurement_rules.pdf"
    assert cit["page_number"] == 1


def test_insufficient_evidence_when_no_valid_citations(
    client: TestClient,
    mock_multimodal_setup: tuple[DeterministicEmbeddingProvider, InMemoryVectorStore, MagicMock],
) -> None:
    embedding_provider, vector_store, mock_llama = mock_multimodal_setup
    settings = get_settings()
    settings.enable_multimodal_answering = True

    # Empty vector store / no matching chunks
    res = client.post(
        "/api/v1/retrieval/answer",
        json={"query": "Unmatched query topic?", "top_k": 3},
        headers={"X-Tenant-ID": "default", "X-Actor-ID": "analyst-1"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["insufficient_evidence"] is True
    assert data["cited_chunk_ids"] == []
    assert data["citations"] == []


def test_output_pii_masked_before_returning(
    client: TestClient,
    mock_multimodal_setup: tuple[DeterministicEmbeddingProvider, InMemoryVectorStore, MagicMock],
) -> None:
    embedding_provider, vector_store, mock_llama = mock_multimodal_setup
    settings = get_settings()
    settings.enable_multimodal_answering = True

    doc_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())

    with Session(bind=engine) as session:
        doc = DocumentRecord(
            id=doc_id,
            tenant_id="default",
            filename="id_card.pdf",
            media_type="application/pdf",
            size_bytes=1024,
            sha256="d" * 64,
            vault_path=f"vault/{doc_id}.ksh",
            status=DocumentState.INDEXED,
            active_index_version=1,
        )
        c = DocumentChunkRecord(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            page_number=1,
            chunk_sequence=0,
            index_version=1,
            chunk_id=chunk_id,
            char_start=0,
            char_end=20,
            masked_content_hash="c" * 64,
        )
        session.add_all([doc, c])
        session.commit()

    emb = embedding_provider.embed_query("tax id")
    vector_store.upsert_chunks(
        [
            VectorStoreChunk(
                point_id=chunk_id,
                chunk_id=chunk_id,
                tenant_id="default",
                document_id=doc_id,
                page_number=1,
                redaction_version=1,
                index_version=1,
                chunk_sequence=0,
                masked_text="Tax record",
                char_start=0,
                char_end=10,
                masked_content_hash="c" * 64,
                document_evidence_hash="d" * 64,
                classification="CONFIDENTIAL",
                document_filename="id_card.pdf",
                indexed_at="2026-09-06T12:00:00Z",
                dense_vector=emb.dense,
                sparse_indices=emb.sparse_indices,
                sparse_values=emb.sparse_values,
            )
        ]
    )

    # Model hallucinates a valid PAN in output
    mock_llama.generate_grounded_answer.return_value = {
        "answer": "Officer tax PAN is ABCPE1234F according to records.",
        "cited_chunk_ids": [chunk_id],
        "insufficient_evidence": False,
    }

    res = client.post(
        "/api/v1/retrieval/answer",
        json={"query": "What is the tax id?", "top_k": 3},
        headers={"X-Tenant-ID": "default", "X-Actor-ID": "analyst-1"},
    )
    assert res.status_code == 200
    data = res.json()
    # PAN must be masked to placeholder
    assert "ABCPE1234F" not in data["answer"]
    assert "[PAN_REDACTED]" in data["answer"]


def test_privacy_safe_audit_logs_metadata_only(
    client: TestClient,
    mock_multimodal_setup: tuple[DeterministicEmbeddingProvider, InMemoryVectorStore, MagicMock],
) -> None:
    embedding_provider, vector_store, mock_llama = mock_multimodal_setup
    settings = get_settings()
    settings.enable_multimodal_answering = True

    secret_query = "Sensitive clearance authorization code ALPHA-99"
    res = client.post(
        "/api/v1/retrieval/answer",
        json={"query": secret_query, "top_k": 3},
        headers={"X-Tenant-ID": "default", "X-Actor-ID": "auditor-1"},
    )
    assert res.status_code == 200

    # Inspect the most recent audit event
    with Session(bind=engine) as session:
        event = session.scalar(
            select(AuditEvent)
            .where(AuditEvent.event_type == "MULTIMODAL_ANSWER_GENERATED")
            .order_by(AuditEvent.id.desc())
        )
        assert event is not None
        details = event.details
        assert details["query_length"] == len(secret_query)
        assert secret_query not in str(details)
        assert "ALPHA-99" not in str(details)
        assert "prompt" not in details
        assert "answer" not in details


def test_real_llama_cpp_integration_skips_truthfully_when_unavailable() -> None:
    """Truthfully tests against local llama.cpp server if running, or skips truthfully."""
    client = LlamaCppMultimodalClient(base_url="http://localhost:8080/v1")
    try:
        model_info = client.check_health_and_capability()
        assert model_info is not None
    except (LlamaCppUnavailableError, LlamaCppIncapableError) as err:
        pytest.skip(
            f"Local llama.cpp server with Qwen3-VL is offline or incapable on localhost:8080: {err}"
        )
