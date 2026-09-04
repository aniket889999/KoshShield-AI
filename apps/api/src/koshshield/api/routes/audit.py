from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from koshshield.database import get_db
from koshshield.models import AuditEvent
from koshshield.schemas import AuditEventResponse, AuditIntegrityResponse
from koshshield.services.audit import verify_audit_chain

router = APIRouter()
SessionDependency = Annotated[Session, Depends(get_db)]


@router.get("/events", response_model=list[AuditEventResponse])
def list_audit_events(
    session: SessionDependency,
    limit: int = 50,
) -> list[AuditEvent]:
    safe_limit = max(1, min(limit, 200))
    return list(
        session.scalars(select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(safe_limit))
    )


@router.get("/integrity", response_model=AuditIntegrityResponse)
def audit_integrity(session: SessionDependency) -> AuditIntegrityResponse:
    valid, events, invalid_event_id = verify_audit_chain(session)
    return AuditIntegrityResponse(
        valid=valid,
        event_count=len(events),
        head_hash=events[-1].event_hash if events else None,
        first_invalid_event_id=invalid_event_id,
    )
