import json
import logging
import uuid
from pathlib import Path
from typing import Any
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


def make_mock_stream_response(
    status_code: int = 200,
    json_data: Any = None,
    raw_bytes: bytes | None = None,
    content_type: str = "application/json",
) -> MagicMock:
    """Helper to create a mocked streaming response for httpx."""
    mock_res = MagicMock()
    mock_res.__enter__.return_value = mock_res
    mock_res.__exit__.return_value = None
    mock_res.status_code = status_code
    mock_res.headers = {"content-type": content_type}
    if raw_bytes is None:
        raw_bytes = json.dumps(json_data).encode("utf-8")
    mock_res.iter_bytes.return_value = [raw_bytes]
    return mock_res


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

    fixture_data = {
        "data": [
            {
                "id": "Qwen3VL-4B-Instruct",
                "architecture": {"input_modalities": ["text", "image"]},
            }
        ]
    }

    with patch("httpx.Client") as mock_httpx:
        mock_ctx = MagicMock()
        mock_httpx.return_value.__enter__.return_value = mock_ctx
        mock_res = make_mock_stream_response(status_code=200, json_data=fixture_data)
        mock_ctx.stream.return_value.__enter__.return_value = mock_res

        client.check_health_and_capability()

        # Assert trust_env is False and follow_redirects is False
        mock_httpx.assert_called_with(
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(connect=3.0, read=5.0, write=3.0, pool=3.0),
        )


def test_official_models_fixture_accepts_image_capability() -> None:
    # Verify realistic captured llama.cpp b4600 /models response fixture
    fixture_path = Path(__file__).parent / "fixtures" / "llama_cpp_b4600_models.json"
    with open(fixture_path, encoding="utf-8") as f:
        fixture_payload = json.load(f)

    client = LlamaCppMultimodalClient(
        base_url="http://localhost:8080/v1", model_id="Qwen3VL-4B-Instruct"
    )

    with patch("httpx.Client") as mock_httpx:
        mock_ctx = MagicMock()
        mock_httpx.return_value.__enter__.return_value = mock_ctx
        mock_res = make_mock_stream_response(status_code=200, json_data=fixture_payload)
        mock_ctx.stream.return_value.__enter__.return_value = mock_res

        model_info = client.check_health_and_capability()
        assert model_info["id"] == "Qwen3VL-4B-Instruct"
        assert "image" in model_info["architecture"]["input_modalities"]


def test_model_name_alone_cannot_establish_vision_capability() -> None:
    # Even if model ID is "Qwen3VL-4B-Instruct", missing architecture.input_modalities must fail
    client = LlamaCppMultimodalClient(
        base_url="http://localhost:8080/v1", model_id="Qwen3VL-4B-Instruct"
    )

    with patch("httpx.Client") as mock_httpx:
        mock_ctx = MagicMock()
        mock_httpx.return_value.__enter__.return_value = mock_ctx

        # Case 1: missing architecture
        mock_ctx.stream.return_value.__enter__.return_value = make_mock_stream_response(
            status_code=200,
            json_data={"data": [{"id": "Qwen3VL-4B-Instruct"}]},
        )
        with pytest.raises(LlamaCppIncapableError, match="lacks authoritative vision modalities"):
            client.check_health_and_capability()

        # Case 2: architecture has only text
        mock_ctx.stream.return_value.__enter__.return_value = make_mock_stream_response(
            status_code=200,
            json_data={
                "data": [
                    {
                        "id": "Qwen3VL-4B-Instruct",
                        "architecture": {"input_modalities": ["text"]},
                    }
                ]
            },
        )
        with pytest.raises(LlamaCppIncapableError, match="lacks authoritative vision modalities"):
            client.check_health_and_capability()


def test_exact_model_alias_matching() -> None:
    # Substring, prefix, suffix or case mismatch must NOT match
    client = LlamaCppMultimodalClient(
        base_url="http://localhost:8080/v1", model_id="Qwen3VL-4B-Instruct"
    )

    with patch("httpx.Client") as mock_httpx:
        mock_ctx = MagicMock()
        mock_httpx.return_value.__enter__.return_value = mock_ctx

        # Case mismatch
        mock_ctx.stream.return_value.__enter__.return_value = make_mock_stream_response(
            status_code=200,
            json_data={
                "data": [
                    {
                        "id": "qwen3vl-4b-instruct",
                        "architecture": {"input_modalities": ["text", "image"]},
                    }
                ]
            },
        )
        with pytest.raises(LlamaCppIncapableError, match="not found on llama.cpp server"):
            client.check_health_and_capability()

        # Substring / variant mismatch
        mock_ctx.stream.return_value.__enter__.return_value = make_mock_stream_response(
            status_code=200,
            json_data={
                "data": [
                    {
                        "id": "Qwen3VL-4B-Instruct-Q4",
                        "architecture": {"input_modalities": ["text", "image"]},
                    }
                ]
            },
        )
        with pytest.raises(LlamaCppIncapableError, match="not found on llama.cpp server"):
            client.check_health_and_capability()


def test_single_capability_check_per_generation_attempt(
    client: TestClient,
    mock_multimodal_setup: tuple[DeterministicEmbeddingProvider, InMemoryVectorStore, MagicMock],
) -> None:
    # Verify that generating an answer executes exactly ONE capability check
    embedding_provider, vector_store, mock_llama = mock_multimodal_setup
    settings = get_settings()
    settings.enable_multimodal_answering = True

    doc_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())

    with Session(bind=engine) as session:
        doc = DocumentRecord(
            id=doc_id,
            tenant_id="default",
            filename="test_doc.pdf",
            media_type="application/pdf",
            size_bytes=1024,
            sha256="a" * 64,
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

    emb = embedding_provider.embed_query("audit inquiry")
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
                masked_text="Audit rule 42 applies.",
                char_start=0,
                char_end=20,
                masked_content_hash="c" * 64,
                document_evidence_hash="a" * 64,
                classification="INTERNAL",
                document_filename="test_doc.pdf",
                indexed_at="2026-09-06T12:00:00Z",
                dense_vector=emb.dense,
                sparse_indices=emb.sparse_indices,
                sparse_values=emb.sparse_values,
            )
        ]
    )

    # Use a real LlamaCppMultimodalClient instance backed by mocked httpx streaming
    real_llama = LlamaCppMultimodalClient(
        base_url="http://localhost:8080/v1",
        model_id="Qwen3VL-4B-Instruct",
    )
    app.dependency_overrides[get_llama_client] = lambda: real_llama

    models_fixture = {
        "data": [
            {
                "id": "Qwen3VL-4B-Instruct",
                "architecture": {"input_modalities": ["text", "image"]},
            }
        ]
    }
    chat_fixture = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "answer": "Audit rule 42 applies.",
                            "cited_chunk_ids": [chunk_id],
                            "insufficient_evidence": False,
                        }
                    )
                }
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
    }

    call_urls: list[str] = []

    def mock_stream(method: str, url: str, **kwargs: Any) -> MagicMock:
        call_urls.append(url)
        if url.endswith("/models"):
            return make_mock_stream_response(status_code=200, json_data=models_fixture)
        elif url.endswith("/chat/completions"):
            return make_mock_stream_response(status_code=200, json_data=chat_fixture)
        return make_mock_stream_response(status_code=404, json_data={})

    with patch("httpx.Client") as mock_httpx:
        mock_ctx = MagicMock()
        mock_httpx.return_value.__enter__.return_value = mock_ctx
        mock_ctx.stream.side_effect = mock_stream

        res = client.post(
            "/api/v1/retrieval/answer",
            json={"query": "What audit rule applies?", "top_k": 3},
            headers={"X-Tenant-ID": "default", "X-Actor-ID": "analyst-1"},
        )
        assert res.status_code == 200

        # /models must be called exactly once
        models_calls = [u for u in call_urls if u.endswith("/models")]
        chat_calls = [u for u in call_urls if u.endswith("/chat/completions")]
        assert len(models_calls) == 1, f"Expected 1 /models call, got {len(models_calls)}"
        assert len(chat_calls) == 1, f"Expected 1 /chat/completions call, got {len(chat_calls)}"


def test_malformed_types_extra_fields_and_oversized_responses_fail_closed(
    client: TestClient,
    mock_multimodal_setup: tuple[DeterministicEmbeddingProvider, InMemoryVectorStore, MagicMock],
) -> None:
    embedding_provider, vector_store, _ = mock_multimodal_setup
    settings = get_settings()
    settings.enable_multimodal_answering = True

    doc_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())

    with Session(bind=engine) as session:
        doc = DocumentRecord(
            id=doc_id,
            tenant_id="default",
            filename="contract.pdf",
            media_type="application/pdf",
            size_bytes=1024,
            sha256="b" * 64,
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

    emb = embedding_provider.embed_query("clause terms")
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
                masked_text="Terms apply.",
                char_start=0,
                char_end=12,
                masked_content_hash="c" * 64,
                document_evidence_hash="b" * 64,
                classification="CONFIDENTIAL",
                document_filename="contract.pdf",
                indexed_at="2026-09-06T12:00:00Z",
                dense_vector=emb.dense,
                sparse_indices=emb.sparse_indices,
                sparse_values=emb.sparse_values,
            )
        ]
    )

    real_llama = LlamaCppMultimodalClient(
        base_url="http://localhost:8080/v1",
        model_id="Qwen3VL-4B-Instruct",
    )
    app.dependency_overrides[get_llama_client] = lambda: real_llama

    models_fixture = {
        "data": [
            {
                "id": "Qwen3VL-4B-Instruct",
                "architecture": {"input_modalities": ["text", "image"]},
            }
        ]
    }

    # Helper to test chat response variants
    def run_with_chat_content(
        chat_content: str | dict[str, Any], raw_bytes: bytes | None = None
    ) -> Any:
        if raw_bytes is None:
            if isinstance(chat_content, str):
                payload = {
                    "choices": [{"message": {"content": chat_content}}],
                    "usage": {},
                }
            else:
                payload = chat_content
            raw_bytes = json.dumps(payload).encode("utf-8")

        def mock_stream(method: str, url: str, **kwargs: Any) -> MagicMock:
            if url.endswith("/models"):
                return make_mock_stream_response(status_code=200, json_data=models_fixture)
            return make_mock_stream_response(status_code=200, raw_bytes=raw_bytes)

        with patch("httpx.Client") as mock_httpx:
            mock_ctx = MagicMock()
            mock_httpx.return_value.__enter__.return_value = mock_ctx
            mock_ctx.stream.side_effect = mock_stream

            return client.post(
                "/api/v1/retrieval/answer",
                json={"query": "What are the terms?", "top_k": 3},
                headers={"X-Tenant-ID": "default", "X-Actor-ID": "tester-1"},
            )

    # 1. Extra forbidden fields in model JSON output
    res1 = run_with_chat_content(
        json.dumps(
            {
                "answer": "Terms apply.",
                "cited_chunk_ids": [chunk_id],
                "insufficient_evidence": False,
                "unauthorized_field": "injected_data",
            }
        )
    )
    assert res1.status_code == 502
    assert res1.json()["detail"] == "Invalid response from local model service."

    # 2. Malformed types: insufficient_evidence is string, not boolean
    res2 = run_with_chat_content(
        json.dumps(
            {
                "answer": "Terms apply.",
                "cited_chunk_ids": [chunk_id],
                "insufficient_evidence": "false",
            }
        )
    )
    assert res2.status_code == 502

    # 3. More than 5 cited chunks (max_length=5)
    res3 = run_with_chat_content(
        json.dumps(
            {
                "answer": "Terms apply.",
                "cited_chunk_ids": ["c1", "c2", "c3", "c4", "c5", "c6"],
                "insufficient_evidence": False,
            }
        )
    )
    assert res3.status_code == 502

    # 4. Markdown code block wrapping (heuristics removed, fails closed)
    res4 = run_with_chat_content(
        "```json\n"
        + json.dumps(
            {
                "answer": "Terms apply.",
                "cited_chunk_ids": [chunk_id],
                "insufficient_evidence": False,
            }
        )
        + "\n```"
    )
    assert res4.status_code == 502

    # 5. Oversized response stream (> 2MB)
    oversized_bytes = b" " * (2 * 1024 * 1024 + 100)
    res5 = run_with_chat_content("", raw_bytes=oversized_bytes)
    assert res5.status_code == 502


def test_no_raw_model_response_or_exception_appears_in_api_audit_log_captures(
    client: TestClient,
    mock_multimodal_setup: tuple[DeterministicEmbeddingProvider, InMemoryVectorStore, MagicMock],
    caplog: pytest.LogCaptureFixture,
) -> None:
    embedding_provider, vector_store, _ = mock_multimodal_setup
    settings = get_settings()
    settings.enable_multimodal_answering = True

    doc_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())

    with Session(bind=engine) as session:
        doc = DocumentRecord(
            id=doc_id,
            tenant_id="default",
            filename="secret_doc.pdf",
            media_type="application/pdf",
            size_bytes=1024,
            sha256="c" * 64,
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

    emb = embedding_provider.embed_query("classified matter")
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
                masked_text="Classified info",
                char_start=0,
                char_end=15,
                masked_content_hash="c" * 64,
                document_evidence_hash="c" * 64,
                classification="RESTRICTED",
                document_filename="secret_doc.pdf",
                indexed_at="2026-09-06T12:00:00Z",
                dense_vector=emb.dense,
                sparse_indices=emb.sparse_indices,
                sparse_values=emb.sparse_values,
            )
        ]
    )

    real_llama = LlamaCppMultimodalClient(
        base_url="http://localhost:8080/v1",
        model_id="Qwen3VL-4B-Instruct",
    )
    app.dependency_overrides[get_llama_client] = lambda: real_llama

    sensitive_poison_text = "LEAKED_INTERNAL_TRACEBACK_AND_MODEL_SECRET_KEY"

    models_fixture = {
        "data": [
            {
                "id": "Qwen3VL-4B-Instruct",
                "architecture": {"input_modalities": ["text", "image"]},
            }
        ]
    }
    malformed_chat_fixture = {
        "choices": [
            {"message": {"content": f"Non-JSON model output containing {sensitive_poison_text}"}}
        ]
    }

    def mock_stream(method: str, url: str, **kwargs: Any) -> MagicMock:
        if url.endswith("/models"):
            return make_mock_stream_response(status_code=200, json_data=models_fixture)
        return make_mock_stream_response(status_code=200, json_data=malformed_chat_fixture)

    with patch("httpx.Client") as mock_httpx:
        mock_ctx = MagicMock()
        mock_httpx.return_value.__enter__.return_value = mock_ctx
        mock_ctx.stream.side_effect = mock_stream

        with caplog.at_level(logging.WARNING):
            res = client.post(
                "/api/v1/retrieval/answer",
                json={"query": "Classified query topic", "top_k": 3},
                headers={"X-Tenant-ID": "default", "X-Actor-ID": "sec-auditor"},
            )

        # 1. API error body is generic HTTP 502 without model response or sensitive text
        assert res.status_code == 502
        assert res.json()["detail"] == "Invalid response from local model service."
        assert sensitive_poison_text not in res.text

        # 2. Audit record details contains ONLY metadata and stable failure code
        with Session(bind=engine) as session:
            event = session.scalar(
                select(AuditEvent)
                .where(AuditEvent.event_type == "MULTIMODAL_ANSWER_GENERATED")
                .order_by(AuditEvent.id.desc())
            )
            assert event is not None
            details = event.details
            assert details["policy_result"] == "OUTPUT_BLOCKED"
            assert details["failure_code"] == "MODEL_RESPONSE_INVALID"
            assert sensitive_poison_text not in str(details)
            assert "prompt" not in details
            assert "answer" not in details

        # 3. Logs do NOT leak raw model content, exception strings, or sensitive tokens
        assert sensitive_poison_text not in caplog.text


def test_fixed_insufficient_evidence_response_when_flag_is_true(
    client: TestClient,
    mock_multimodal_setup: tuple[DeterministicEmbeddingProvider, InMemoryVectorStore, MagicMock],
) -> None:
    # If insufficient_evidence is True, never return substantive answer even if citations present
    embedding_provider, vector_store, mock_llama = mock_multimodal_setup
    settings = get_settings()
    settings.enable_multimodal_answering = True

    doc_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())

    with Session(bind=engine) as session:
        doc = DocumentRecord(
            id=doc_id,
            tenant_id="default",
            filename="policy.pdf",
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

    emb = embedding_provider.embed_query("foreign policy")
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
                masked_text="Policy excerpt.",
                char_start=0,
                char_end=15,
                masked_content_hash="c" * 64,
                document_evidence_hash="d" * 64,
                classification="PUBLIC",
                document_filename="policy.pdf",
                indexed_at="2026-09-06T12:00:00Z",
                dense_vector=emb.dense,
                sparse_indices=emb.sparse_indices,
                sparse_values=emb.sparse_values,
            )
        ]
    )

    # Model returned a substantive answer AND cited chunks, but set insufficient_evidence = True
    mock_llama.generate_grounded_answer.return_value = {
        "answer": "Substantive answer that should be discarded.",
        "cited_chunk_ids": [chunk_id],
        "insufficient_evidence": True,
    }

    res = client.post(
        "/api/v1/retrieval/answer",
        json={"query": "Foreign policy details?", "top_k": 3},
        headers={"X-Tenant-ID": "default", "X-Actor-ID": "analyst-2"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["insufficient_evidence"] is True
    assert data["answer"] == "Insufficient verified evidence to answer the query."
    assert data["cited_chunk_ids"] == []
    assert data["citations"] == []

    # Verify audit event has truthful policy_result
    with Session(bind=engine) as session:
        event = session.scalar(
            select(AuditEvent)
            .where(AuditEvent.event_type == "MULTIMODAL_ANSWER_GENERATED")
            .order_by(AuditEvent.id.desc())
        )
        assert event is not None
        assert event.details["policy_result"] == "INSUFFICIENT_EVIDENCE"
        assert event.details["failure_code"] == "INSUFFICIENT_EVIDENCE"


def test_truthful_policy_results_for_allowed_insufficient_redacted_blocked(
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
            filename="cases.pdf",
            media_type="application/pdf",
            size_bytes=1024,
            sha256="e" * 64,
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

    emb = embedding_provider.embed_query("case details")
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
                masked_text="Case facts.",
                char_start=0,
                char_end=11,
                masked_content_hash="c" * 64,
                document_evidence_hash="e" * 64,
                classification="CONFIDENTIAL",
                document_filename="cases.pdf",
                indexed_at="2026-09-06T12:00:00Z",
                dense_vector=emb.dense,
                sparse_indices=emb.sparse_indices,
                sparse_values=emb.sparse_values,
            )
        ]
    )

    # 1. ALLOWED: clean grounded answer
    mock_llama.generate_grounded_answer.return_value = {
        "answer": "Case was resolved per guidelines.",
        "cited_chunk_ids": [chunk_id],
        "insufficient_evidence": False,
    }
    res1 = client.post(
        "/api/v1/retrieval/answer",
        json={"query": "Case status?", "top_k": 3},
        headers={"X-Tenant-ID": "default", "X-Actor-ID": "auditor-1"},
    )
    assert res1.status_code == 200
    with Session(bind=engine) as session:
        e1 = session.scalar(
            select(AuditEvent)
            .where(
                AuditEvent.event_type == "MULTIMODAL_ANSWER_GENERATED",
                AuditEvent.actor_id == "auditor-1",
            )
            .order_by(AuditEvent.created_at.desc())
        )
        assert e1 is not None
        assert e1.details["policy_result"] == "ALLOWED"
        assert e1.details["failure_code"] is None

    # 2. OUTPUT_REDACTED: contains valid PAN that gets redacted
    mock_llama.generate_grounded_answer.return_value = {
        "answer": "Taxpayer PAN is ABCPE1234F per records.",
        "cited_chunk_ids": [chunk_id],
        "insufficient_evidence": False,
    }
    res2 = client.post(
        "/api/v1/retrieval/answer",
        json={"query": "PAN inquiry?", "top_k": 3},
        headers={"X-Tenant-ID": "default", "X-Actor-ID": "auditor-2"},
    )
    assert res2.status_code == 200
    assert "[PAN_REDACTED]" in res2.json()["answer"]
    with Session(bind=engine) as session:
        e2 = session.scalar(
            select(AuditEvent)
            .where(
                AuditEvent.event_type == "MULTIMODAL_ANSWER_GENERATED",
                AuditEvent.actor_id == "auditor-2",
            )
            .order_by(AuditEvent.created_at.desc())
        )
        assert e2 is not None
        assert e2.details["policy_result"] == "OUTPUT_REDACTED"
        assert e2.details["failure_code"] is None


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
    assert "ABCPE1234F" not in data["answer"]
    assert "[PAN_REDACTED]" in data["answer"]


def test_feature_flag_disabled_returns_503(
    client: TestClient,
    mock_multimodal_setup: tuple[DeterministicEmbeddingProvider, InMemoryVectorStore, MagicMock],
) -> None:
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
    mock_multimodal_setup[0], mock_multimodal_setup[1], mock_multimodal_setup[2]
    settings = get_settings()
    settings.enable_multimodal_answering = True

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


def test_privacy_safe_audit_logs_metadata_only(
    client: TestClient,
    mock_multimodal_setup: tuple[DeterministicEmbeddingProvider, InMemoryVectorStore, MagicMock],
) -> None:
    mock_multimodal_setup[0], mock_multimodal_setup[1], mock_multimodal_setup[2]
    settings = get_settings()
    settings.enable_multimodal_answering = True

    secret_query = "Sensitive clearance authorization code ALPHA-99"
    res = client.post(
        "/api/v1/retrieval/answer",
        json={"query": secret_query, "top_k": 3},
        headers={"X-Tenant-ID": "default", "X-Actor-ID": "auditor-1"},
    )
    assert res.status_code == 200

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
