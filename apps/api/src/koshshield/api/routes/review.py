from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from koshshield.config import Settings, get_settings
from koshshield.database import get_db
from koshshield.models import (
    DocumentPageRecord,
    DocumentRecord,
    DocumentState,
    ExtractionJob,
    FindingStatus,
    InvalidStateTransitionError,
    RedactionFinding,
)
from koshshield.schemas import (
    DocumentPagePreviewResponse,
    DocumentRedactionsResponse,
    DocumentResponse,
    ExtractionJobResponse,
    RedactionDecisionRequest,
    RedactionFindingResponse,
    ReviewQueueItemResponse,
)
from koshshield.security.vault import EncryptedVault, VaultConfigurationError
from koshshield.services.extraction.interfaces import ExtractionError, OcrUnavailableError
from koshshield.services.redaction import (
    ConcurrencyConflictError,
    RedactionError,
    UnresolvedFindingsError,
    accept_all_high_confidence,
    approve_redactions,
    start_document_extraction,
    update_finding_decision,
)

router = APIRouter()
SessionDependency = Annotated[Session, Depends(get_db)]
SettingsDependency = Annotated[Settings, Depends(get_settings)]


@router.post(
    "/documents/{document_id}/extraction",
    response_model=ExtractionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def trigger_extraction(
    document_id: str,
    session: SessionDependency,
    settings: SettingsDependency,
    actor_id: Annotated[str, Header(alias="X-Actor-ID", max_length=120)] = "local-demo-user",
) -> ExtractionJob:
    try:
        vault = EncryptedVault(settings.vault_dir, settings.master_key_base64)
        job = start_document_extraction(
            session=session,
            document_id=document_id,
            actor_id=actor_id,
            settings=settings,
            vault=vault,
        )
        return job
    except VaultConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except InvalidStateTransitionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except OcrUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except ExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/documents/{document_id}/extraction",
    response_model=ExtractionJobResponse,
)
def get_extraction_status(
    document_id: str,
    session: SessionDependency,
) -> ExtractionJob:
    job = session.scalar(
        select(ExtractionJob)
        .where(ExtractionJob.document_id == document_id)
        .order_by(ExtractionJob.created_at.desc())
        .limit(1)
    )
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No extraction job found for document {document_id}",
        )
    return job


@router.get("/review", response_model=list[ReviewQueueItemResponse])
def get_review_queue(session: SessionDependency) -> list[ReviewQueueItemResponse]:
    """Lists documents that require redaction review or have findings."""
    documents = list(
        session.scalars(
            select(DocumentRecord)
            .where(
                DocumentRecord.status.in_(
                    [
                        DocumentState.REVIEW_REQUIRED,
                        DocumentState.REDACTION_APPROVED,
                        DocumentState.INDEX_READY,
                    ]
                )
            )
            .order_by(DocumentRecord.created_at.desc())
        )
    )

    items: list[ReviewQueueItemResponse] = []
    for doc in documents:
        total_pages = (
            session.scalar(
                select(func.count(DocumentPageRecord.id)).where(
                    DocumentPageRecord.document_id == doc.id
                )
            )
            or 0
        )
        total_findings = (
            session.scalar(
                select(func.count(RedactionFinding.id)).where(
                    RedactionFinding.document_id == doc.id
                )
            )
            or 0
        )
        pending = (
            session.scalar(
                select(func.count(RedactionFinding.id)).where(
                    RedactionFinding.document_id == doc.id,
                    RedactionFinding.status == FindingStatus.PENDING,
                )
            )
            or 0
        )
        accepted = (
            session.scalar(
                select(func.count(RedactionFinding.id)).where(
                    RedactionFinding.document_id == doc.id,
                    RedactionFinding.status == FindingStatus.ACCEPTED,
                )
            )
            or 0
        )
        rejected = (
            session.scalar(
                select(func.count(RedactionFinding.id)).where(
                    RedactionFinding.document_id == doc.id,
                    RedactionFinding.status == FindingStatus.REJECTED,
                )
            )
            or 0
        )

        items.append(
            ReviewQueueItemResponse(
                document_id=doc.id,
                filename=doc.filename,
                status=doc.status,
                total_pages=total_pages,
                total_findings=total_findings,
                pending_findings=pending,
                accepted_findings=accepted,
                rejected_findings=rejected,
                created_at=doc.created_at,
            )
        )

    return items


@router.get(
    "/documents/{document_id}/redactions",
    response_model=DocumentRedactionsResponse,
)
def get_document_redactions(
    document_id: str,
    session: SessionDependency,
) -> DocumentRedactionsResponse:
    document = session.get(DocumentRecord, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Document {document_id} not found"
        )

    findings = list(
        session.scalars(
            select(RedactionFinding)
            .where(RedactionFinding.document_id == document_id)
            .order_by(RedactionFinding.page_number.asc(), RedactionFinding.start_offset.asc())
        )
    )

    pages = list(
        session.scalars(
            select(DocumentPageRecord)
            .where(DocumentPageRecord.document_id == document_id)
            .order_by(DocumentPageRecord.page_number.asc())
        )
    )

    unresolved_count = sum(1 for f in findings if f.status == FindingStatus.PENDING)

    findings_by_page: dict[int, list[RedactionFinding]] = {}
    for f in findings:
        findings_by_page.setdefault(f.page_number, []).append(f)

    page_previews = [
        DocumentPagePreviewResponse(
            page_number=p.page_number,
            width=p.width,
            height=p.height,
            extraction_method=p.extraction_method,
            masked_text=p.masked_text,
            findings=[
                RedactionFindingResponse.model_validate(f)
                for f in findings_by_page.get(p.page_number, [])
            ],
        )
        for p in pages
    ]

    return DocumentRedactionsResponse(
        document_id=document.id,
        status=document.status,
        total_pages=len(pages),
        total_findings=len(findings),
        unresolved_count=unresolved_count,
        findings=[RedactionFindingResponse.model_validate(f) for f in findings],
        pages=page_previews,
    )


@router.patch(
    "/documents/{document_id}/redactions/{finding_id}",
    response_model=RedactionFindingResponse,
)
def update_redaction(
    document_id: str,
    finding_id: str,
    request: RedactionDecisionRequest,
    session: SessionDependency,
    actor_id: Annotated[str, Header(alias="X-Actor-ID", max_length=120)] = "local-demo-user",
) -> RedactionFinding:
    try:
        return update_finding_decision(
            session=session,
            document_id=document_id,
            finding_id=finding_id,
            decision=request.decision,
            expected_version=request.version,
            actor_id=actor_id,
        )
    except ConcurrencyConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RedactionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/documents/{document_id}/redactions/accept-high-confidence",
    response_model=dict[str, int],
)
def accept_high_confidence(
    document_id: str,
    session: SessionDependency,
    settings: SettingsDependency,
    actor_id: Annotated[str, Header(alias="X-Actor-ID", max_length=120)] = "local-demo-user",
) -> dict[str, int]:
    document = session.get(DocumentRecord, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Document {document_id} not found"
        )
    count = accept_all_high_confidence(
        session=session,
        document_id=document_id,
        actor_id=actor_id,
        threshold=settings.high_confidence_threshold,
    )
    return {"accepted_count": count}


@router.post(
    "/documents/{document_id}/redactions/approve",
    response_model=DocumentResponse,
)
def approve_document_redactions(
    document_id: str,
    session: SessionDependency,
    settings: SettingsDependency,
    actor_id: Annotated[str, Header(alias="X-Actor-ID", max_length=120)] = "local-demo-user",
) -> DocumentRecord:
    try:
        vault = EncryptedVault(settings.vault_dir, settings.master_key_base64)
        return approve_redactions(
            session=session,
            document_id=document_id,
            actor_id=actor_id,
            vault=vault,
        )
    except VaultConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except UnresolvedFindingsError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except InvalidStateTransitionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except RedactionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
