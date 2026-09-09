from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    tenant_id: str
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
    tenant_id: str
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
    tenant_id: str
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


class IndexingStatusResponse(BaseModel):
    document_id: str
    status: str
    chunk_count: int
    redaction_version: int
    active_index_version: int | None = None
    tenant_id: str
    collection_name: str
    completed_at: str | None = None


class RetrievalSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    permitted_document_ids: list[str] | None = None
    classification: str | None = None


class RetrievalVisualRegion(BaseModel):
    region_id: str
    region_type: str
    page_number: int
    bbox: list[float] | None = None
    page_width: float | None = None
    page_height: float | None = None
    caption: str
    caption_hash: str
    image_sha256: str | None = None
    image_available: bool
    source: str


class RetrievalEvidenceItem(BaseModel):
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
    visual_regions: list[RetrievalVisualRegion] = []


class RetrievalResponse(BaseModel):
    query_length: int
    duration_ms: float
    tenant_id: str
    top_k: int
    total_found: int
    results: list[RetrievalEvidenceItem]


class RetrievalStatusResponse(BaseModel):
    collection_name: str
    vector_store_status: str
    embedding_model_status: str
    embedding_model_reason: str
    total_chunks: int
    indexed_documents_count: int


class AgentActionRequest(BaseModel):
    tool_name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    classification: Literal["INTERNAL", "CONFIDENTIAL", "RESTRICTED"] = "CONFIDENTIAL"
    arguments: dict[str, object]

    @field_validator("arguments")
    @classmethod
    def limit_arguments(cls, value: dict[str, object]) -> dict[str, object]:
        import json

        if len(json.dumps(value, sort_keys=True, separators=(",", ":"))) > 4096:
            raise ValueError("tool arguments exceed the 4096-byte policy limit")
        return value


class AgentApprovalResponse(BaseModel):
    decision: str
    reviewer_id: str | None
    version: int
    decided_at: datetime | None


class AgentRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    tenant_id: str
    actor_id: str
    tool_name: str
    classification: str
    arguments_hash: str
    argument_summary: str
    status: str
    state_history: list[str]
    policy_decision: str
    policy_reason: str
    approval_required: bool
    approval: AgentApprovalResponse | None = None
    result: dict[str, object] | None = None
    result_hash: str | None = None
    failure_code: str | None = None
    version: int
    created_at: datetime
    updated_at: datetime


class AgentApprovalDecisionRequest(BaseModel):
    decision: Literal["APPROVED", "REJECTED"]
    version: int = Field(ge=1)


class AgentExecuteRequest(BaseModel):
    version: int = Field(ge=1)
