import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from koshshield.services.audit import record_audit_event
from koshshield.services.retrieval.embeddings.interfaces import EmbeddingProvider
from koshshield.services.retrieval.vector_store.interfaces import (
    VectorStore,
    VectorStoreSearchResult,
)

logger = logging.getLogger(__name__)


@dataclass
class EvidenceItem:
    rank: int
    fused_score: float
    sources: list[str]
    masked_snippet: str
    document_id: str
    document_filename: str
    page_number: int
    chunk_id: str
    evidence_hash: str
    redaction_version: int
    citation_label: str


@dataclass
class RetrievalEvidencePack:
    query_hash: str
    tenant_id: str
    top_k: int
    total_found: int
    items: list[EvidenceItem]


class HybridRetrievalService:
    """Local hybrid retrieval service combining dense and sparse BGE-M3 representations

    with Reciprocal Rank Fusion (RRF) and verifiable citations.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        rrf_k: int = 60,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        tenant_id: str = "default",
        permitted_document_ids: list[str] | None = None,
        classification: str | None = None,
        top_k: int = 5,
        session: Session | None = None,
        actor_id: str = "demo-user",
    ) -> RetrievalEvidencePack:
        clean_query = query.strip()
        if len(clean_query) < 2:
            raise ValueError("Search query must be at least 2 characters long")
        if len(clean_query) > 500:
            raise ValueError("Search query cannot exceed 500 characters")

        top_k = max(1, min(top_k, 50))
        query_hash = hashlib.sha256(clean_query.encode("utf-8")).hexdigest()

        # 1. Embed query
        query_embedding = self.embedding_provider.embed_query(clean_query)

        # 2. Retrieve dense candidates
        dense_hits = self.vector_store.search_dense(
            query_vector=query_embedding.dense,
            tenant_id=tenant_id,
            permitted_document_ids=permitted_document_ids,
            classification=classification,
            limit=top_k * 3,
        )

        # 3. Retrieve sparse candidates
        sparse_hits = self.vector_store.search_sparse(
            indices=query_embedding.sparse_indices,
            values=query_embedding.sparse_values,
            tenant_id=tenant_id,
            permitted_document_ids=permitted_document_ids,
            classification=classification,
            limit=top_k * 3,
        )

        # 4. Reciprocal Rank Fusion (RRF)
        fused_scores: dict[str, float] = {}
        hit_sources: dict[str, set[str]] = {}
        payload_map: dict[str, dict[str, Any]] = {}

        def process_ranked_list(hits: list[VectorStoreSearchResult], source_name: str) -> None:
            for rank_idx, hit in enumerate(hits, start=1):
                chunk_id = str(hit.payload.get("chunk_id") or hit.point_id)
                rrf_increment = 1.0 / (self.rrf_k + rank_idx)
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + rrf_increment

                if chunk_id not in hit_sources:
                    hit_sources[chunk_id] = set()
                hit_sources[chunk_id].add(source_name)

                if chunk_id not in payload_map:
                    payload_map[chunk_id] = hit.payload

        process_ranked_list(dense_hits, "dense")
        process_ranked_list(sparse_hits, "sparse")

        # 5. Sort by fused score descending
        sorted_chunk_ids = sorted(
            fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True
        )[:top_k]

        # 6. Build EvidenceItems with verifiable citations
        evidence_items: list[EvidenceItem] = []
        for rank, cid in enumerate(sorted_chunk_ids, start=1):
            payload = payload_map[cid]
            doc_filename = payload.get("document_filename", "Document")
            page_num = int(payload.get("page_number", 1))
            ev_hash = str(payload.get("document_evidence_hash", ""))
            short_ev = ev_hash[:8] if ev_hash else "verified"
            red_version = int(payload.get("redaction_version", 1))

            citation = f"[Document: {doc_filename} | Page: {page_num} | Evidence: {short_ev}]"

            sources_list = sorted(hit_sources.get(cid, set()))

            evidence_items.append(
                EvidenceItem(
                    rank=rank,
                    fused_score=round(fused_scores[cid], 6),
                    sources=sources_list,
                    masked_snippet=str(payload.get("masked_text", "")),
                    document_id=str(payload.get("document_id", "")),
                    document_filename=doc_filename,
                    page_number=page_num,
                    chunk_id=cid,
                    evidence_hash=ev_hash,
                    redaction_version=red_version,
                    citation_label=citation,
                )
            )

        # 7. Privacy-safe audit (strictly no raw query string stored)
        if session is not None:
            top_score = evidence_items[0].fused_score if evidence_items else 0.0
            record_audit_event(
                session=session,
                actor_id=actor_id,
                event_type="RETRIEVAL_QUERY_EXECUTED",
                resource_type="retrieval",
                resource_id=None,
                details={
                    "tenant_id": tenant_id,
                    "query_hash": query_hash,
                    "query_length": len(clean_query),
                    "result_count": len(evidence_items),
                    "top_fused_score": top_score,
                },
            )
            session.commit()

        return RetrievalEvidencePack(
            query_hash=query_hash,
            tenant_id=tenant_id,
            top_k=top_k,
            total_found=len(evidence_items),
            items=evidence_items,
        )
