from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    status: str
    version: int = 1
    created_at: datetime


class ExtractionJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    status: str
    pages_processed: int
    total_pages: int
    extraction_method: str | None = None
    error_message: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class RedactionFindingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    page_number: int
    finding_type: str
    confidence: float
    detection_source: str
    start_offset: int
    end_offset: int
    bbox_json: Any | None = None
    salted_value_hash: str
    masked_context: str
    status: str
    reviewer_id: str | None = None
    version: int
    created_at: datetime
    updated_at: datetime


class RedactionDecisionRequest(BaseModel):
    decision: Literal["ACCEPTED", "REJECTED"]
    version: int


class ReviewQueueItemResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    total_pages: int
    total_findings: int
    pending_findings: int
    accepted_findings: int
    rejected_findings: int
    created_at: datetime


class DocumentPagePreviewResponse(BaseModel):
    page_number: int
    width: float
    height: float
    extraction_method: str
    masked_text: str | None = None
    findings: list[RedactionFindingResponse] = []


class DocumentRedactionsResponse(BaseModel):
    document_id: str
    status: str
    total_pages: int
    total_findings: int
    unresolved_count: int
    findings: list[RedactionFindingResponse]
    pages: list[DocumentPagePreviewResponse] = []


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
