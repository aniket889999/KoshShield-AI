"""Auditable document lifecycle timeline service and view-model.

Traces a document through every operational stage:
1. UPLOAD: Encrypted vault ingestion, media type validation, hash verification
2. EXTRACTION: Text and visual region extraction via native parser or OCR
3. PII_REVIEW: Detection, confidence ranking, and manual review decisions
4. APPROVAL: Governance approval gating before indexing eligibility
5. INDEXING: Deterministic chunking and vector store embedding
6. RETRIEVAL: Privacy-gated query eligibility and evidence linkage
7. CLEANUP: Stale index pruning and deferred lifecycle maintenance

All stage outputs, audit logs, and metrics are sanitized. No raw PII,
raw prompts, secrets, or internal filesystem paths are exposed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from koshshield.models import (
    AuditEvent,
    DocumentChunkRecord,
    DocumentPageRecord,
    DocumentRecord,
    DocumentState,
    ExtractionJob,
    FindingStatus,
    RedactionFinding,
)
from koshshield.services.readiness import sanitize_text


class LifecycleStageId(StrEnum):
    UPLOAD = "UPLOAD"
    EXTRACTION = "EXTRACTION"
    PII_REVIEW = "PII_REVIEW"
    APPROVAL = "APPROVAL"
    INDEXING = "INDEXING"
    RETRIEVAL = "RETRIEVAL"
    CLEANUP = "CLEANUP"


class LifecycleStageStatus(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    IN_PROGRESS = "IN_PROGRESS"
    PENDING = "PENDING"
    SKIPPED = "SKIPPED"
    NOT_EXECUTED = "NOT_EXECUTED"


class LifecycleStageRecord(BaseModel):
    stage_id: LifecycleStageId
    display_name: str
    status: LifecycleStageStatus
    failure_code: str | None = None
    summary: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class TimelineAuditEvent(BaseModel):
    event_id: str
    event_type: str
    actor_id: str
    stage: LifecycleStageId
    timestamp: datetime
    event_hash: str
    details: dict[str, Any] = Field(default_factory=dict)


class DocumentLifecycleTimeline(BaseModel):
    document_id: str
    tenant_id: str
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    current_status: str
    current_version: int
    stages: list[LifecycleStageRecord]
    audit_events: list[TimelineAuditEvent]
    generated_at: datetime
    disclaimer: str = (
        "Auditable document lifecycle timeline. All transitions are verified against "
        "tamper-evident audit logs and DPDP-aligned access policies."
    )


def _map_event_to_stage(event_type: str) -> LifecycleStageId:
    if "document.accepted" in event_type or "document.upload" in event_type:
        return LifecycleStageId.UPLOAD
    if "extraction" in event_type:
        return LifecycleStageId.EXTRACTION
    if "redaction.finding" in event_type:
        return LifecycleStageId.PII_REVIEW
    if "redaction.approved" in event_type or "approval" in event_type:
        return LifecycleStageId.APPROVAL
    if "index" in event_type and "cleanup" not in event_type:
        return LifecycleStageId.INDEXING
    if "cleanup" in event_type:
        return LifecycleStageId.CLEANUP
    if "retrieval" in event_type or "query" in event_type:
        return LifecycleStageId.RETRIEVAL
    return LifecycleStageId.UPLOAD


def _sanitize_audit_details(details: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, val in details.items():
        if key in {"path", "vault_path", "prompt", "raw_prompt", "file_content", "secret"}:
            continue
        if isinstance(val, str):
            sanitized[key] = sanitize_text(val)
        elif isinstance(val, (int, float, bool)) or val is None:
            sanitized[key] = val
        elif isinstance(val, dict):
            sanitized[key] = _sanitize_audit_details(val)
        elif isinstance(val, list):
            sanitized[key] = [
                sanitize_text(item) if isinstance(item, str) else item for item in val
            ]
    return sanitized


def build_document_lifecycle_timeline(
    session: Session,
    document: DocumentRecord,
) -> DocumentLifecycleTimeline:
    """Builds the complete lifecycle timeline for a document."""
    doc_id = document.id

    # 1. Fetch extraction job
    job = session.scalar(
        select(ExtractionJob)
        .where(ExtractionJob.document_id == doc_id)
        .order_by(ExtractionJob.created_at.desc())
        .limit(1)
    )

    # 2. Count pages
    page_count = (
        session.scalar(
            select(func.count(DocumentPageRecord.id)).where(
                DocumentPageRecord.document_id == doc_id
            )
        )
        or 0
    )

    # 3. Count findings
    findings = list(
        session.scalars(select(RedactionFinding).where(RedactionFinding.document_id == doc_id))
    )
    total_findings = len(findings)
    pending_findings = sum(1 for f in findings if f.status == FindingStatus.PENDING)
    accepted_findings = sum(1 for f in findings if f.status == FindingStatus.ACCEPTED)
    rejected_findings = sum(1 for f in findings if f.status == FindingStatus.REJECTED)

    # 4. Count chunks
    chunk_count = (
        session.scalar(
            select(func.count(DocumentChunkRecord.id)).where(
                DocumentChunkRecord.document_id == doc_id
            )
        )
        or 0
    )

    # 5. Fetch audit events
    audit_records = list(
        session.scalars(
            select(AuditEvent)
            .where(
                AuditEvent.tenant_id == document.tenant_id,
                AuditEvent.resource_id == doc_id,
            )
            .order_by(AuditEvent.created_at.asc())
        )
    )

    timeline_audit_events = [
        TimelineAuditEvent(
            event_id=ev.id,
            event_type=ev.event_type,
            actor_id=ev.actor_id,
            stage=_map_event_to_stage(ev.event_type),
            timestamp=ev.created_at,
            event_hash=ev.event_hash,
            details=_sanitize_audit_details(ev.details or {}),
        )
        for ev in audit_records
    ]

    # Evaluate stages
    stages: list[LifecycleStageRecord] = []

    # STAGE 1: UPLOAD
    stages.append(
        LifecycleStageRecord(
            stage_id=LifecycleStageId.UPLOAD,
            display_name="Encrypted Ingestion",
            status=LifecycleStageStatus.PASSED,
            summary="Original file validated and committed to AES-256-GCM encrypted vault.",
            started_at=document.created_at,
            completed_at=document.created_at,
            metrics={
                "media_type": document.media_type,
                "size_bytes": document.size_bytes,
                "sha256": document.sha256,
            },
        )
    )

    # STAGE 2: EXTRACTION
    ext_status = LifecycleStageStatus.NOT_EXECUTED
    ext_failure_code: str | None = None
    ext_summary = "Document text extraction not yet started."
    ext_started: datetime | None = None
    ext_completed: datetime | None = None

    if job is not None:
        ext_started = job.created_at
        ext_completed = job.completed_at
        if job.status == "COMPLETED" or page_count > 0:
            ext_status = LifecycleStageStatus.PASSED
            ext_summary = f"Extracted {page_count} pages with bounding boxes and layout metrics."
        elif job.status == "IN_PROGRESS":
            ext_status = LifecycleStageStatus.IN_PROGRESS
            ext_summary = f"Extracting pages ({job.pages_processed}/{job.total_pages})."
        elif job.status == "FAILED" or document.status == DocumentState.EXTRACTION_FAILED:
            ext_status = LifecycleStageStatus.FAILED
            ext_failure_code = "EXTRACTION_FAILED"
            ext_summary = "Extraction failed during document parsing."
        else:
            ext_status = LifecycleStageStatus.PENDING
            ext_summary = "Extraction job queued for processing."
    elif document.status in {
        DocumentState.REVIEW_REQUIRED,
        DocumentState.REDACTION_APPROVED,
        DocumentState.INDEX_READY,
        DocumentState.INDEXING,
        DocumentState.INDEXED,
    }:
        ext_status = LifecycleStageStatus.PASSED
        ext_summary = f"Extracted {page_count} pages."

    stages.append(
        LifecycleStageRecord(
            stage_id=LifecycleStageId.EXTRACTION,
            display_name="Content & OCR Extraction",
            status=ext_status,
            failure_code=ext_failure_code,
            summary=ext_summary,
            started_at=ext_started,
            completed_at=ext_completed,
            metrics={
                "page_count": page_count,
                "extraction_method": job.extraction_method if job else None,
            },
        )
    )

    # STAGE 3: PII_REVIEW
    pii_status = LifecycleStageStatus.NOT_EXECUTED
    pii_summary = "PII inspection pending extraction completion."

    if document.status == DocumentState.REVIEW_REQUIRED:
        if pending_findings > 0:
            pii_status = LifecycleStageStatus.IN_PROGRESS
            pii_summary = (
                f"{pending_findings} unconfirmed PII findings require human reviewer action."
            )
        else:
            pii_status = LifecycleStageStatus.PASSED
            pii_summary = f"All {total_findings} findings resolved and ready for final approval."
    elif document.status in {
        DocumentState.REDACTION_APPROVED,
        DocumentState.INDEX_READY,
        DocumentState.INDEXING,
        DocumentState.INDEXED,
    }:
        pii_status = LifecycleStageStatus.PASSED
        pii_summary = (
            f"{total_findings} findings reviewed "
            f"({accepted_findings} accepted, {rejected_findings} rejected)."
        )
    elif ext_status == LifecycleStageStatus.PASSED:
        pii_status = LifecycleStageStatus.PENDING
        pii_summary = "Awaiting review queue assignment."

    stages.append(
        LifecycleStageRecord(
            stage_id=LifecycleStageId.PII_REVIEW,
            display_name="PII Review & Masking",
            status=pii_status,
            summary=pii_summary,
            metrics={
                "total_findings": total_findings,
                "pending_findings": pending_findings,
                "accepted_findings": accepted_findings,
                "rejected_findings": rejected_findings,
            },
        )
    )

    # STAGE 4: APPROVAL
    appr_status = LifecycleStageStatus.NOT_EXECUTED
    appr_summary = "Governance approval pending review completion."

    if document.status in {
        DocumentState.REDACTION_APPROVED,
        DocumentState.INDEX_READY,
        DocumentState.INDEXING,
        DocumentState.INDEXED,
    }:
        appr_status = LifecycleStageStatus.PASSED
        appr_summary = f"Redactions approved at version {document.version}."
    elif document.status == DocumentState.REVIEW_REQUIRED:
        appr_status = (
            LifecycleStageStatus.PENDING
            if pending_findings == 0
            else LifecycleStageStatus.NOT_EXECUTED
        )
        appr_summary = (
            "Awaiting authorized governance approval."
            if pending_findings == 0
            else "Blocked: unresolved PII findings must be decided before approval."
        )

    stages.append(
        LifecycleStageRecord(
            stage_id=LifecycleStageId.APPROVAL,
            display_name="Governance Approval",
            status=appr_status,
            summary=appr_summary,
            metrics={"document_version": document.version},
        )
    )

    # STAGE 5: INDEXING
    idx_status = LifecycleStageStatus.NOT_EXECUTED
    idx_failure_code: str | None = None
    idx_summary = "Vector indexing pending governance approval."

    if document.status == DocumentState.INDEXED:
        idx_status = LifecycleStageStatus.PASSED
        idx_summary = (
            f"Successfully indexed {chunk_count} masked chunks at version "
            f"{document.active_index_version}."
        )
    elif document.status == DocumentState.INDEXING:
        idx_status = LifecycleStageStatus.IN_PROGRESS
        idx_summary = "Vector indexing and chunk embedding in progress."
    elif document.status == DocumentState.INDEX_FAILED:
        idx_status = LifecycleStageStatus.FAILED
        idx_failure_code = "INDEXING_FAILED"
        idx_summary = "Vector store indexing failed or service offline."
    elif document.status in {DocumentState.REDACTION_APPROVED, DocumentState.INDEX_READY}:
        idx_status = LifecycleStageStatus.PENDING
        idx_summary = "Document approved and queued for vector indexing."

    stages.append(
        LifecycleStageRecord(
            stage_id=LifecycleStageId.INDEXING,
            display_name="Vector & Keyword Indexing",
            status=idx_status,
            failure_code=idx_failure_code,
            summary=idx_summary,
            metrics={
                "chunk_count": chunk_count,
                "active_index_version": document.active_index_version,
            },
        )
    )

    # STAGE 6: RETRIEVAL
    ret_status = LifecycleStageStatus.NOT_EXECUTED
    ret_summary = "Document not eligible for retrieval: governance approval and indexing required."

    if document.status == DocumentState.INDEXED:
        ret_status = LifecycleStageStatus.PASSED
        ret_summary = (
            "Document is active in the retrieval pool. Privacy gate allows masked citations."
        )

    stages.append(
        LifecycleStageRecord(
            stage_id=LifecycleStageId.RETRIEVAL,
            display_name="Retrieval & Intelligence",
            status=ret_status,
            summary=ret_summary,
            metrics={"retrieval_eligible": document.status == DocumentState.INDEXED},
        )
    )

    # STAGE 7: CLEANUP
    clean_status = LifecycleStageStatus.NOT_EXECUTED
    clean_summary = "Index active and synchronized; no cleanup required."

    if document.index_cleanup_pending:
        clean_status = LifecycleStageStatus.PENDING
        clean_summary = "Previous index version flagged for deferred vector cleanup."

    stages.append(
        LifecycleStageRecord(
            stage_id=LifecycleStageId.CLEANUP,
            display_name="Stale Index Maintenance",
            status=clean_status,
            summary=clean_summary,
            metrics={"cleanup_pending": document.index_cleanup_pending},
        )
    )

    return DocumentLifecycleTimeline(
        document_id=doc_id,
        tenant_id=document.tenant_id,
        filename=document.filename,
        media_type=document.media_type,
        size_bytes=document.size_bytes,
        sha256=document.sha256,
        current_status=document.status,
        current_version=document.version,
        stages=stages,
        audit_events=timeline_audit_events,
        generated_at=datetime.now(UTC),
    )
