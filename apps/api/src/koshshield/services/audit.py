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


def calculate_event_hash_v1(
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
    """Calculates the legacy pre-tenant audit hash (v1)."""
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


def calculate_event_hash_v2(
    *,
    event_id: str,
    tenant_id: str,
    actor_id: str,
    event_type: str,
    resource_type: str,
    resource_id: str | None,
    details: dict[str, object],
    previous_hash: str | None,
    created_at: datetime,
) -> str:
    """Calculates the tenant-aware audit hash (v2)."""
    canonical = json.dumps(
        {
            "id": event_id,
            "tenant_id": tenant_id,
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
    tenant_id: str | None = None,
    hash_version: str = "v2",
) -> str:
    if hash_version == "v1":
        return calculate_event_hash_v1(
            event_id=event_id,
            actor_id=actor_id,
            event_type=event_type,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            previous_hash=previous_hash,
            created_at=created_at,
        )
    if tenant_id is None:
        raise ValueError("tenant_id is mandatory for v2 audit hash calculation")
    return calculate_event_hash_v2(
        event_id=event_id,
        tenant_id=tenant_id,
        actor_id=actor_id,
        event_type=event_type,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
        previous_hash=previous_hash,
        created_at=created_at,
    )


def append_audit_event(
    session: Session,
    *,
    tenant_id: str,
    actor_id: str,
    event_type: str,
    resource_type: str,
    resource_id: str | None,
    details: dict[str, object],
) -> AuditEvent:
    if not tenant_id:
        raise ValueError("tenant_id is required to record an audit event")

    previous = session.scalar(
        select(AuditEvent)
        .where(AuditEvent.tenant_id == tenant_id)
        .order_by(AuditEvent.created_at.desc(), AuditEvent.id.desc())
        .limit(1)
    )
    created_at = datetime.now(UTC)
    event_id = str(uuid4())
    previous_hash = previous.event_hash if previous else None
    event_hash = calculate_event_hash_v2(
        event_id=event_id,
        tenant_id=tenant_id,
        actor_id=actor_id,
        event_type=event_type,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
        previous_hash=previous_hash,
        created_at=created_at,
    )
    event = AuditEvent(
        id=event_id,
        tenant_id=tenant_id,
        actor_id=actor_id,
        event_type=event_type,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
        previous_hash=previous_hash,
        event_hash=event_hash,
        hash_version="v2",
        created_at=created_at,
    )
    session.add(event)
    return event


record_audit_event = append_audit_event


def verify_audit_chain(
    session: Session,
    tenant_id: str | None = None,
) -> tuple[bool, list[AuditEvent], str | None]:
    query = select(AuditEvent).order_by(AuditEvent.created_at.asc())
    if tenant_id is not None:
        query = query.where(AuditEvent.tenant_id == tenant_id)
    events = list(session.scalars(query))
    expected_previous_hash: str | None = None

    for event in events:
        event_version = getattr(event, "hash_version", "v1") or "v1"
        event_tenant = getattr(event, "tenant_id", None)
        expected_hash = calculate_event_hash(
            event_id=event.id,
            tenant_id=event_tenant,
            actor_id=event.actor_id,
            event_type=event.event_type,
            resource_type=event.resource_type,
            resource_id=event.resource_id,
            details=event.details,
            previous_hash=event.previous_hash,
            created_at=event.created_at,
            hash_version=event_version,
        )
        if event.previous_hash != expected_previous_hash or event.event_hash != expected_hash:
            return False, events, event.id
        expected_previous_hash = event.event_hash

    return True, events, None
