import hashlib
import json
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from koshshield.models import AuditEvent


def canonical_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def calculate_event_hash(
    *,
    event_id: str,
    actor_id: str,
    event_type: str,
    resource_type: str,
    resource_id: str | None,
    details: dict[str, object],
    previous_hash: str | None,
    created_at: datetime,
) -> str:
    canonical = json.dumps(
        {
            "id": event_id,
            "actor_id": actor_id,
            "event_type": event_type,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "details": details,
            "previous_hash": previous_hash,
            "created_at": canonical_timestamp(created_at),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def append_audit_event(
    session: Session,
    *,
    actor_id: str,
    event_type: str,
    resource_type: str,
    resource_id: str | None,
    details: dict[str, object],
) -> AuditEvent:
    previous = session.scalar(select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(1))
    created_at = datetime.now(UTC)
    event_id = str(uuid4())
    previous_hash = previous.event_hash if previous else None
    event = AuditEvent(
        id=event_id,
        actor_id=actor_id,
        event_type=event_type,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
        previous_hash=previous_hash,
        event_hash=calculate_event_hash(
            event_id=event_id,
            actor_id=actor_id,
            event_type=event_type,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            previous_hash=previous_hash,
            created_at=created_at,
        ),
        created_at=created_at,
    )
    session.add(event)
    return event


def verify_audit_chain(session: Session) -> tuple[bool, list[AuditEvent], str | None]:
    events = list(session.scalars(select(AuditEvent).order_by(AuditEvent.created_at.asc())))
    expected_previous_hash: str | None = None

    for event in events:
        expected_hash = calculate_event_hash(
            event_id=event.id,
            actor_id=event.actor_id,
            event_type=event.event_type,
            resource_type=event.resource_type,
            resource_id=event.resource_id,
            details=event.details,
            previous_hash=event.previous_hash,
            created_at=event.created_at,
        )
        if event.previous_hash != expected_previous_hash or event.event_hash != expected_hash:
            return False, events, event.id
        expected_previous_hash = event.event_hash

    return True, events, None
