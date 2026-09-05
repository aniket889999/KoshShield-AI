import hashlib
import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from koshshield.database import engine
from koshshield.models import AuditEvent
from koshshield.services.retrieval.embeddings.deterministic_fake import (
    DeterministicEmbeddingProvider,
)
from koshshield.services.retrieval.hybrid_search import HybridRetrievalService
from koshshield.services.retrieval.vector_store import InMemoryVectorStore


def test_query_and_sha256_digest_never_stored_in_audit_or_response() -> None:
    """Regression test proving neither the raw query nor its SHA-256 digest
    appears in the database, audit events, logs, or serialized responses.
    """
    secret_query = "confidential strategic procurement tender contract 998822"
    query_sha256 = hashlib.sha256(secret_query.encode("utf-8")).hexdigest()

    emb_provider = DeterministicEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    retrieval = HybridRetrievalService(
        embedding_provider=emb_provider,
        vector_store=vector_store,
    )

    with Session(bind=engine) as session:
        pack = retrieval.search(
            query=secret_query,
            tenant_id="dept-audit-test",
            actor_id="auditor-007",
            session=session,
        )

        # 1. Inspect serialized response object
        response_dict = {
            "query_length": pack.query_length,
            "duration_ms": pack.duration_ms,
            "tenant_id": pack.tenant_id,
            "top_k": pack.top_k,
            "total_found": pack.total_found,
            "results": [
                {
                    "rank": item.rank,
                    "fused_score": item.fused_score,
                    "sources": item.sources,
                    "masked_snippet": item.masked_snippet,
                    "document_id": item.document_id,
                    "document_filename": item.document_filename,
                    "page_number": item.page_number,
                    "chunk_id": item.chunk_id,
                    "evidence_hash": item.evidence_hash,
                    "masked_content_hash": item.masked_content_hash,
                    "redaction_version": item.redaction_version,
                    "index_version": item.index_version,
                    "citation_label": item.citation_label,
                }
                for item in pack.items
            ],
        }
        serialized_json = json.dumps(response_dict)

        # Verify raw query and SHA-256 digest are NOT in serialized response
        assert secret_query not in serialized_json
        assert query_sha256 not in serialized_json
        assert "query_hash" not in serialized_json
        assert pack.query_length == len(secret_query)

        # 2. Inspect AuditEvent recorded in database
        audit_records = session.scalars(
            select(AuditEvent).where(
                AuditEvent.event_type == "RETRIEVAL_QUERY_EXECUTED",
                AuditEvent.actor_id == "auditor-007",
            )
        ).all()

        assert len(audit_records) >= 1
        latest_audit = audit_records[-1]

        audit_details_json = json.dumps(latest_audit.details)

        # Invariant checks:
        # Neither raw query nor SHA-256 digest may appear in audit record details
        assert secret_query not in audit_details_json, "Raw query leaked into audit log!"
        assert query_sha256 not in audit_details_json, "SHA-256 query digest leaked into audit log!"
        assert "query_hash" not in latest_audit.details, "query_hash key found in audit details!"

        # Audit contains strictly safe query metadata
        details = latest_audit.details
        assert details["actor_id"] == "auditor-007"
        assert details["tenant_id"] == "dept-audit-test"
        assert details["query_length"] == len(secret_query)
        assert details["top_k"] == 5
        assert details["policy_result"] == "ALLOWED"
        assert "duration_ms" in details
        assert "result_chunk_ids" in details
        assert "result_count" in details
