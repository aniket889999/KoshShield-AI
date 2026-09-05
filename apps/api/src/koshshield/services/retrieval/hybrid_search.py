import logging
import time
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
class VisualEvidenceRegion:
    region_id: str
    region_type: str
    page_number: int
    bbox: list[float] | None
    page_width: float | None
    page_height: float | None
    caption: str
    caption_hash: str
    image_sha256: str | None
    image_available: bool
    source: str


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
    masked_content_hash: str
    redaction_version: int
    index_version: int
    citation_label: str
    visual_regions: list[VisualEvidenceRegion]


@dataclass
class RetrievalEvidencePack:
    query_length: int
    duration_ms: float
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
        start_time = time.perf_counter()

        clean_query = query.strip()
        if len(clean_query) < 2:
            raise ValueError("Search query must be at least 2 characters long")
        if len(clean_query) > 500:
            raise ValueError("Search query cannot exceed 500 characters")

        top_k = max(1, min(top_k, 50))

        # 1. Embed query
        query_embedding = self.embedding_provider.embed_query(clean_query)

        # 2. Retrieve dense candidates with mandatory tenant filter
        dense_hits = self.vector_store.search_dense(
            query_vector=query_embedding.dense,
            tenant_id=tenant_id,
            permitted_document_ids=permitted_document_ids,
            classification=classification,
            limit=top_k * 3,
        )

        # 3. Retrieve sparse candidates with mandatory tenant filter
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

        # 6. Build EvidenceItems with complete SHA-256 evidence hashes and citations
        evidence_items: list[EvidenceItem] = []
        for rank, cid in enumerate(sorted_chunk_ids, start=1):
            payload = payload_map[cid]
            doc_filename = payload.get("document_filename", "Document")
            page_num = int(payload.get("page_number", 1))
            ev_hash = str(payload.get("document_evidence_hash", ""))
            masked_hash = str(payload.get("masked_content_hash", ""))
            short_ev = ev_hash[:12] if ev_hash else "verified"
            red_version = int(payload.get("redaction_version", 1))
            idx_version = int(payload.get("index_version", red_version))

            # Display label uses shortened hash for UI readability; full hash is in evidence_hash
            citation = f"[Document: {doc_filename} | Page: {page_num} | Evidence: {short_ev}]"
            sources_list = sorted(hit_sources.get(cid, set()))
            visual_regions = _parse_visual_regions(payload)
            if visual_regions:
                sources_list = sorted({*sources_list, "visual-caption"})

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
                    masked_content_hash=masked_hash,
                    redaction_version=red_version,
                    index_version=idx_version,
                    citation_label=citation,
                    visual_regions=visual_regions,
                )
            )

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # 7. Privacy-safe audit: strictly metadata only; no raw query, no query hash, no text
        if session is not None:
            record_audit_event(
                session=session,
                actor_id=actor_id,
                event_type="RETRIEVAL_QUERY_EXECUTED",
                resource_type="retrieval",
                resource_id=None,
                details={
                    "actor_id": actor_id,
                    "tenant_id": tenant_id,
                    "query_length": len(clean_query),
                    "top_k": top_k,
                    "permitted_document_ids": permitted_document_ids,
                    "classification": classification,
                    "result_chunk_ids": [item.chunk_id for item in evidence_items],
                    "result_count": len(evidence_items),
                    "duration_ms": round(duration_ms, 2),
                    "policy_result": "ALLOWED",
                },
            )
            session.commit()

        return RetrievalEvidencePack(
            query_length=len(clean_query),
            duration_ms=round(duration_ms, 2),
            tenant_id=tenant_id,
            top_k=top_k,
            total_found=len(evidence_items),
            items=evidence_items,
        )


def _parse_visual_regions(payload: dict[str, Any]) -> list[VisualEvidenceRegion]:
    raw_regions = payload.get("visual_regions", [])
    if not isinstance(raw_regions, list):
        return []

    parsed: list[VisualEvidenceRegion] = []
    for raw in raw_regions:
        if not isinstance(raw, dict):
            continue
        bbox = raw.get("bbox")
        try:
            safe_bbox = (
                [float(v) for v in bbox[:4]] if isinstance(bbox, list) and len(bbox) >= 4 else None
            )
        except (TypeError, ValueError):
            safe_bbox = None
        page_width = raw.get("page_width")
        page_height = raw.get("page_height")
        try:
            page_number = int(raw.get("page_number", payload.get("page_number", 1)))
        except (TypeError, ValueError):
            page_number = int(payload.get("page_number", 1))
        parsed.append(
            VisualEvidenceRegion(
                region_id=str(raw.get("region_id", "")),
                region_type=str(raw.get("region_type", "PAGE_IMAGE")),
                page_number=page_number,
                bbox=safe_bbox,
                page_width=float(page_width) if page_width else None,
                page_height=float(page_height) if page_height else None,
                caption=str(raw.get("caption", "")),
                caption_hash=str(raw.get("caption_hash", "")),
                image_sha256=(
                    str(raw.get("image_sha256")) if raw.get("image_sha256") is not None else None
                ),
                image_available=bool(raw.get("image_available", False)),
                source=str(raw.get("source", "unknown")),
            )
        )
    return parsed
