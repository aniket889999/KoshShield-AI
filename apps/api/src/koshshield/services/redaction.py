import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pymupdf
from sqlalchemy import select
from sqlalchemy.orm import Session

from koshshield.config import Settings
from koshshield.models import (
    DocumentPageRecord,
    DocumentRecord,
    DocumentState,
    DocumentVisualRegionRecord,
    ExtractionJob,
    FindingStatus,
    RedactionFinding,
    validate_transition,
)
from koshshield.security.pii import (
    REDACTION_PLACEHOLDERS,
    IndianPiiDetector,
    generate_safe_contexts_for_page,
    hash_pii_value,
)
from koshshield.security.vault import EncryptedVault
from koshshield.services.audit import append_audit_event
from koshshield.services.extraction.interfaces import ExtractionError
from koshshield.services.extraction.service import UnifiedDocumentExtractor
from koshshield.services.retrieval.visuals import build_visual_region_drafts


class RedactionError(Exception):
    """Base error for redaction review operations."""


class ConcurrencyConflictError(RedactionError):
    """Raised when an update conflicts with the current finding version."""


class UnresolvedFindingsError(RedactionError):
    """Raised when approval is attempted with pending findings."""


def utc_now() -> datetime:
    return datetime.now(UTC)


def start_document_extraction(
    session: Session,
    *,
    document_id: str,
    actor_id: str,
    settings: Settings,
    vault: EncryptedVault,
) -> ExtractionJob:
    """Queues and executes local extraction and PII detection for an encrypted document."""
    document = session.get(DocumentRecord, document_id)
    if not document:
        raise ValueError(f"Document not found: {document_id}")

    # Validate state transition
    validate_transition(document.status, DocumentState.EXTRACTION_QUEUED)
    document.status = DocumentState.EXTRACTION_QUEUED

    job = ExtractionJob(
        id=str(uuid4()),
        document_id=document_id,
        status="QUEUED",
    )
    session.add(job)

    append_audit_event(
        session,
        actor_id=actor_id,
        event_type="document.extraction_started",
        resource_type="document",
        resource_id=document_id,
        details={"job_id": job.id, "initial_status": "QUEUED"},
    )
    session.commit()
    session.refresh(job)
    session.refresh(document)

    # Begin synchronous extraction execution
    try:
        validate_transition(document.status, DocumentState.EXTRACTING)
        document.status = DocumentState.EXTRACTING
        job.status = "RUNNING"
        session.commit()

        # Decrypt original document bytes from vault
        original_bytes = vault.decrypt(
            document_id=document.id,
            evidence_hash=document.sha256,
            path=Path(document.vault_path),
        )

        extractor = UnifiedDocumentExtractor(settings)
        result = extractor.extract(
            filename=document.filename,
            content=original_bytes,
            media_type=document.media_type,
        )
        page_images = _extract_page_images(
            content=original_bytes,
            media_type=document.media_type,
            max_pages=settings.max_extraction_pages,
        )

        pii_detector = IndianPiiDetector(salt=settings.pii_salt)
        total_findings = 0

        # Remove any previous pages or findings if retrying extraction
        existing_pages = list(
            session.scalars(
                select(DocumentPageRecord).where(DocumentPageRecord.document_id == document_id)
            )
        )
        for p in existing_pages:
            session.delete(p)
        existing_findings = list(
            session.scalars(
                select(RedactionFinding).where(RedactionFinding.document_id == document_id)
            )
        )
        for f in existing_findings:
            session.delete(f)
        existing_regions = list(
            session.scalars(
                select(DocumentVisualRegionRecord).where(
                    DocumentVisualRegionRecord.document_id == document_id
                )
            )
        )
        for region in existing_regions:
            session.delete(region)

        for page in result.pages:
            # 1. Store raw extracted text in encrypted vault only
            page_text_bytes = page.text.encode("utf-8")
            page_text_hash = hashlib.sha256(page_text_bytes).hexdigest()
            page_artifact_id = f"{document_id}_p{page.page_number}_raw"
            encrypted_page_path = vault.encrypt(
                document_id=page_artifact_id,
                evidence_hash=page_text_hash,
                plaintext=page_text_bytes,
            )

            page_image = page_images.get(page.page_number)
            encrypted_page_image_path: Path | None = None
            page_image_hash: str | None = None
            page_image_media_type: str | None = None
            if page_image is not None:
                page_image_bytes, page_image_media_type = page_image
                page_image_hash = hashlib.sha256(page_image_bytes).hexdigest()
                page_image_artifact_id = f"{document_id}_p{page.page_number}_image"
                encrypted_page_image_path = vault.encrypt(
                    document_id=page_image_artifact_id,
                    evidence_hash=page_image_hash,
                    plaintext=page_image_bytes,
                )

            # 2. Record page metadata in database (no raw text)
            page_record = DocumentPageRecord(
                id=str(uuid4()),
                document_id=document_id,
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                extraction_method=page.extraction_method,
                text_hash=page_text_hash,
                encrypted_artifact_path=str(encrypted_page_path),
                page_image_sha256=page_image_hash,
                page_image_media_type=page_image_media_type,
                encrypted_page_image_path=(
                    str(encrypted_page_image_path) if encrypted_page_image_path else None
                ),
                masked_text=None,
                masked_text_hash=None,
            )
            session.add(page_record)

            # 3. Detect PII
            blocks_for_detection = [
                (b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3], b.text) for b in page.blocks
            ]
            detected = pii_detector.detect(
                text=page.text,
                page_number=page.page_number,
                blocks=blocks_for_detection,
            )
            safe_contexts = generate_safe_contexts_for_page(page.text, detected)

            for idx, item in enumerate(detected):
                total_findings += 1
                masked_ctx = safe_contexts[idx]
                salted_hash = hash_pii_value(item.value, settings.pii_salt)

                finding_record = RedactionFinding(
                    id=str(uuid4()),
                    document_id=document_id,
                    page_number=page.page_number,
                    finding_type=item.finding_type,
                    confidence=item.confidence,
                    detection_source=item.detection_source,
                    start_offset=item.start,
                    end_offset=item.end,
                    bbox_json={"bbox": item.bbox} if item.bbox else None,
                    salted_value_hash=salted_hash,
                    masked_context=masked_ctx,
                    status=FindingStatus.PENDING,
                    version=1,
                )
                session.add(finding_record)

        job.pages_processed = len(result.pages)
        job.total_pages = result.total_pages
        job.extraction_method = result.extraction_method
        job.status = "COMPLETED"
        job.completed_at = utc_now()

        validate_transition(document.status, DocumentState.REVIEW_REQUIRED)
        document.status = DocumentState.REVIEW_REQUIRED
        document.version += 1

        append_audit_event(
            session,
            actor_id=actor_id,
            event_type="document.extraction_completed",
            resource_type="document",
            resource_id=document_id,
            details={
                "job_id": job.id,
                "pages_processed": len(result.pages),
                "total_pages": result.total_pages,
                "findings_count": total_findings,
                "extraction_method": result.extraction_method,
            },
        )
        session.commit()
        session.refresh(job)
        session.refresh(document)
        return job

    except Exception as exc:
        session.rollback()
        # Mark failure states
        failed_doc = session.get(DocumentRecord, document_id)
        if failed_doc:
            failed_doc.status = DocumentState.EXTRACTION_FAILED
            failed_doc.version += 1
        failed_job = session.get(ExtractionJob, job.id)
        if failed_job:
            failed_job.status = "FAILED"
            failed_job.error_message = str(exc)
            failed_job.completed_at = utc_now()

        append_audit_event(
            session,
            actor_id=actor_id,
            event_type="document.extraction_failed",
            resource_type="document",
            resource_id=document_id,
            details={"job_id": job.id, "error": str(exc)},
        )
        session.commit()
        raise


def update_finding_decision(
    session: Session,
    *,
    document_id: str,
    finding_id: str,
    decision: str,
    expected_version: int,
    actor_id: str,
) -> RedactionFinding:
    """Applies a human review decision to an individual PII finding with version locking."""
    finding = session.scalar(
        select(RedactionFinding).where(
            RedactionFinding.id == finding_id,
            RedactionFinding.document_id == document_id,
        )
    )
    if not finding:
        raise ValueError(f"Finding {finding_id} not found for document {document_id}")

    if decision not in {FindingStatus.ACCEPTED, FindingStatus.REJECTED}:
        raise ValueError(f"Invalid decision: {decision}. Must be ACCEPTED or REJECTED")

    if finding.version != expected_version:
        raise ConcurrencyConflictError(
            f"Finding has been modified concurrently. Expected version {expected_version}, "
            f"found {finding.version}"
        )

    finding.status = decision
    finding.reviewer_id = actor_id
    finding.version += 1
    finding.updated_at = utc_now()

    append_audit_event(
        session,
        actor_id=actor_id,
        event_type="redaction.finding_updated",
        resource_type="document",
        resource_id=document_id,
        details={
            "finding_id": finding_id,
            "finding_type": finding.finding_type,
            "decision": decision,
            "version": finding.version,
        },
    )
    session.commit()
    session.refresh(finding)
    return finding


def accept_all_high_confidence(
    session: Session,
    *,
    document_id: str,
    actor_id: str,
    threshold: float = 0.85,
) -> int:
    """Accepts all pending findings whose confidence meets or exceeds the threshold."""
    findings = list(
        session.scalars(
            select(RedactionFinding).where(
                RedactionFinding.document_id == document_id,
                RedactionFinding.status == FindingStatus.PENDING,
                RedactionFinding.confidence >= threshold,
            )
        )
    )

    for finding in findings:
        finding.status = FindingStatus.ACCEPTED
        finding.reviewer_id = actor_id
        finding.version += 1
        finding.updated_at = utc_now()

    if findings:
        append_audit_event(
            session,
            actor_id=actor_id,
            event_type="redaction.high_confidence_accepted",
            resource_type="document",
            resource_id=document_id,
            details={
                "accepted_count": len(findings),
                "threshold": threshold,
            },
        )
        session.commit()

    return len(findings)


def approve_redactions(
    session: Session,
    *,
    document_id: str,
    actor_id: str,
    vault: EncryptedVault,
) -> DocumentRecord:
    """Finalizes redactions, generates deterministic masked text, and marks document INDEX_READY."""
    document = session.get(DocumentRecord, document_id)
    if not document:
        raise ValueError(f"Document not found: {document_id}")

    if document.status != DocumentState.REVIEW_REQUIRED:
        raise RedactionError(
            f"Document must be in {DocumentState.REVIEW_REQUIRED} state to approve redactions; "
            f"current: {document.status}"
        )

    # Check for unresolved findings
    unresolved_count = len(
        list(
            session.scalars(
                select(RedactionFinding).where(
                    RedactionFinding.document_id == document_id,
                    RedactionFinding.status == FindingStatus.PENDING,
                )
            )
        )
    )
    if unresolved_count > 0:
        raise UnresolvedFindingsError(
            f"Cannot approve redactions: {unresolved_count} unresolved finding(s) remain"
        )

    validate_transition(document.status, DocumentState.REDACTION_APPROVED)
    document.status = DocumentState.REDACTION_APPROVED

    # Generate deterministic masked text for each page
    pages = list(
        session.scalars(
            select(DocumentPageRecord)
            .where(DocumentPageRecord.document_id == document_id)
            .order_by(DocumentPageRecord.page_number.asc())
        )
    )

    accepted_findings = list(
        session.scalars(
            select(RedactionFinding).where(
                RedactionFinding.document_id == document_id,
                RedactionFinding.status == FindingStatus.ACCEPTED,
            )
        )
    )

    findings_by_page: dict[int, list[RedactionFinding]] = {}
    for f in accepted_findings:
        findings_by_page.setdefault(f.page_number, []).append(f)

    for page in pages:
        # Decrypt raw text from vault
        page_artifact_id = f"{document_id}_p{page.page_number}_raw"
        raw_text_bytes = vault.decrypt(
            document_id=page_artifact_id,
            evidence_hash=page.text_hash,
            path=Path(page.encrypted_artifact_path),
        )
        page_text = raw_text_bytes.decode("utf-8")

        # Deterministically apply accepted redactions in reverse character order
        page_findings = sorted(
            findings_by_page.get(page.page_number, []),
            key=lambda item: item.start_offset,
            reverse=True,
        )

        masked_text = page_text
        for finding in page_findings:
            placeholder = REDACTION_PLACEHOLDERS.get(finding.finding_type, "[REDACTED]")
            masked_text = (
                masked_text[: finding.start_offset]
                + placeholder
                + masked_text[finding.end_offset :]
            )

        masked_hash = hashlib.sha256(masked_text.encode("utf-8")).hexdigest()
        page.masked_text = masked_text
        page.masked_text_hash = masked_hash

    existing_regions = list(
        session.scalars(
            select(DocumentVisualRegionRecord).where(
                DocumentVisualRegionRecord.document_id == document_id
            )
        )
    )
    for region in existing_regions:
        session.delete(region)

    for page in pages:
        if not page.masked_text:
            continue
        for draft in build_visual_region_drafts(
            masked_text=page.masked_text,
            page_number=page.page_number,
            width=page.width,
            height=page.height,
            image_sha256=page.page_image_sha256,
        ):
            session.add(
                DocumentVisualRegionRecord(
                    id=str(uuid4()),
                    document_id=document_id,
                    page_number=page.page_number,
                    region_sequence=draft.region_sequence,
                    region_type=draft.region_type,
                    source=draft.source,
                    bbox_json=draft.bbox_json,
                    caption_text=draft.caption_text,
                    caption_hash=draft.caption_hash,
                    image_sha256=draft.image_sha256,
                )
            )

    # Transition to INDEX_READY
    validate_transition(document.status, DocumentState.INDEX_READY)
    document.status = DocumentState.INDEX_READY
    document.version += 1

    rejected_count = len(
        list(
            session.scalars(
                select(RedactionFinding).where(
                    RedactionFinding.document_id == document_id,
                    RedactionFinding.status == FindingStatus.REJECTED,
                )
            )
        )
    )

    append_audit_event(
        session,
        actor_id=actor_id,
        event_type="redaction.approved",
        resource_type="document",
        resource_id=document_id,
        details={
            "accepted_count": len(accepted_findings),
            "rejected_count": rejected_count,
            "total_pages": len(pages),
            "next_state": DocumentState.INDEX_READY,
        },
    )
    session.commit()
    session.refresh(document)
    return document


def _extract_page_images(
    *,
    content: bytes,
    media_type: str,
    max_pages: int,
) -> dict[int, tuple[bytes, str]]:
    if media_type == "application/pdf":
        doc = pymupdf.open(stream=content, filetype="pdf")
        if len(doc) > max_pages:
            raise ExtractionError(
                f"Document page count ({len(doc)}) exceeds configured limit ({max_pages})"
            )
        rendered: dict[int, tuple[bytes, str]] = {}
        for idx in range(len(doc)):
            pix = doc[idx].get_pixmap(dpi=120, alpha=False)
            rendered[idx + 1] = (pix.tobytes("png"), "image/png")
        return rendered

    if media_type in {"image/png", "image/jpeg"}:
        return {1: (content, media_type)}

    return {}
