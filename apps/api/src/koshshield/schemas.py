from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    status: str
    created_at: datetime


class AuditEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    actor_id: str
    event_type: str
    resource_type: str
    resource_id: str | None
    details: dict[str, object]
    previous_hash: str | None
    event_hash: str
    created_at: datetime


class AuditIntegrityResponse(BaseModel):
    valid: bool
    event_count: int
    head_hash: str | None
    first_invalid_event_id: str | None = None
