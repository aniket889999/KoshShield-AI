"""Secure evidence and citation catalog service for approved indexed documents.

Provides authorization-safe inspection of indexed document chunks, citation labels,
masked snippets, and visual region metadata.
Guarantees:
1. Originals remain strictly in the encrypted vault; no vault decryption is performed.
2. Only approved, privacy-masked content is returned.
3. Residual PII check verifies snippets before returning.
4. Tenant isolation is strictly enforced.
"""

from __future__ import annotations

from fastapi import HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from koshshield.models import (
    DocumentChunkRecord,
    DocumentPageRecord,
    DocumentRecord,
    DocumentState,
    DocumentVisualRegionRecord,
)
from koshshield.schemas import RetrievalVisualRegion
from koshshield.security.pii.indian_pii import IndianPiiDetector


class DocumentEvidenceChunk(BaseModel):
    chunk_id: str
    chunk_sequence: int
    page_number: int
    char_start: int
    char_end: int
    masked_snippet: str
    masked_content_hash: str
    citation_label: str
    visual_regions: list[RetrievalVisualRegion] = Field(default_factory=list)


class DocumentEvidenceCatalog(BaseModel):
    document_id: str
    tenant_id: str
    filename: str
    current_status: str
    index_version: int | None
    chunk_count: int
    chunks: list[DocumentEvidenceChunk]
    residual_pii_checked: bool = True
    privacy_gate_verdict: str = "APPROVED_MASKED_ONLY"
    disclaimer: str = (
        "Secure evidence catalog. Returns privacy-masked derivatives only. Original unredacted "
        "bytes remain in the encrypted vault and cannot be retrieved through this interface."
    )


class EvidenceSecurityError(Exception):
    """Raised when an unapproved document or residual PII is detected."""


def build_document_evidence_catalog(
    session: Session,
    document: DocumentRecord,
    tenant_id: str,
) -> DocumentEvidenceCatalog:
    """Builds the secure evidence and citation catalog for an indexed document."""
    if document.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or access denied for tenant",
        )

    # Privacy Gate: Document must be approved and indexed
    if document.status not in {DocumentState.INDEXED, DocumentState.INDEX_READY}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Evidence catalog restricted: document has not completed governance approval "
                f"and indexing (current state: {document.status})."
            ),
        )

    doc_id = document.id

    # Load pages with masked text
    pages = list(
        session.scalars(
            select(DocumentPageRecord)
            .where(DocumentPageRecord.document_id == doc_id)
            .order_by(DocumentPageRecord.page_number.asc())
        )
    )
    pages_by_number: dict[int, DocumentPageRecord] = {p.page_number: p for p in pages}

    # Load visual regions
    regions = list(
        session.scalars(
            select(DocumentVisualRegionRecord)
            .where(
                DocumentVisualRegionRecord.document_id == doc_id,
                DocumentVisualRegionRecord.tenant_id == tenant_id,
            )
            .order_by(
                DocumentVisualRegionRecord.page_number.asc(),
                DocumentVisualRegionRecord.region_sequence.asc(),
            )
        )
    )
    regions_by_page: dict[int, list[DocumentVisualRegionRecord]] = {}
    for r in regions:
        regions_by_page.setdefault(r.page_number, []).append(r)

    # Load chunks
    chunks = list(
        session.scalars(
            select(DocumentChunkRecord)
            .where(DocumentChunkRecord.document_id == doc_id)
            .order_by(
                DocumentChunkRecord.page_number.asc(),
                DocumentChunkRecord.chunk_sequence.asc(),
            )
        )
    )

    detector = IndianPiiDetector()
    evidence_chunks: list[DocumentEvidenceChunk] = []

    for c in chunks:
        page = pages_by_number.get(c.page_number)
        page_masked_text = page.masked_text if page and page.masked_text else ""

        # Extract bounded snippet
        if page_masked_text and 0 <= c.char_start < len(page_masked_text):
            char_end = min(c.char_end, len(page_masked_text))
            snippet = page_masked_text[c.char_start : char_end].strip()
        else:
            snippet = "[MASKED_EVIDENCE_SNIPPET]"

        # Strict residual PII guard on the snippet
        pii_findings = detector.detect(snippet)
        if pii_findings:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Security gate violation: residual PII pattern detected in indexed snippet.",
            )

        # Build visual regions for this page
        page_regions = regions_by_page.get(c.page_number, [])
        retrieval_regions = [
            RetrievalVisualRegion(
                region_id=reg.id,
                region_type=reg.region_type,
                page_number=reg.page_number,
                bbox=reg.bbox_json.get("bbox") if isinstance(reg.bbox_json, dict) else None,
                caption=reg.caption_text,
                caption_hash=reg.caption_hash,
                image_sha256=reg.masked_image_sha256 or reg.image_sha256,
                image_available=bool(reg.masked_image_sha256 or reg.image_sha256),
                source=reg.source,
            )
            for reg in page_regions
        ]

        # Computed citation label
        citation_label = f"[{document.filename} p.{c.page_number} #{c.chunk_sequence}]"

        evidence_chunks.append(
            DocumentEvidenceChunk(
                chunk_id=c.chunk_id,
                chunk_sequence=c.chunk_sequence,
                page_number=c.page_number,
                char_start=c.char_start,
                char_end=c.char_end,
                masked_snippet=snippet,
                masked_content_hash=c.masked_content_hash,
                citation_label=citation_label,
                visual_regions=retrieval_regions,
            )
        )

    return DocumentEvidenceCatalog(
        document_id=doc_id,
        tenant_id=tenant_id,
        filename=document.filename,
        current_status=document.status,
        index_version=document.active_index_version or document.version,
        chunk_count=len(evidence_chunks),
        chunks=evidence_chunks,
        residual_pii_checked=True,
        privacy_gate_verdict="APPROVED_MASKED_ONLY",
    )
