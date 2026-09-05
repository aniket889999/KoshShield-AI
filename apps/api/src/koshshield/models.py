from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from koshshield.database import Base


def utc_now() -> datetime:
    return datetime.now(UTC)


class DocumentState:
    ENCRYPTED = "ENCRYPTED"
    EXTRACTION_QUEUED = "EXTRACTION_QUEUED"
    EXTRACTING = "EXTRACTING"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    REDACTION_APPROVED = "REDACTION_APPROVED"
    INDEX_READY = "INDEX_READY"
    INDEXING = "INDEXING"
    INDEXED = "INDEXED"
    INDEX_FAILED = "INDEX_FAILED"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"


class FindingStatus:
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


VALID_DOCUMENT_TRANSITIONS: dict[str, set[str]] = {
    DocumentState.ENCRYPTED: {DocumentState.EXTRACTION_QUEUED},
    DocumentState.EXTRACTION_QUEUED: {DocumentState.EXTRACTING, DocumentState.EXTRACTION_FAILED},
    DocumentState.EXTRACTING: {
        DocumentState.REVIEW_REQUIRED,
        DocumentState.INDEX_READY,
        DocumentState.EXTRACTION_FAILED,
    },
    DocumentState.REVIEW_REQUIRED: {
        DocumentState.REDACTION_APPROVED,
        DocumentState.EXTRACTION_FAILED,
    },
    DocumentState.REDACTION_APPROVED: {DocumentState.INDEX_READY},
    DocumentState.INDEX_READY: {DocumentState.INDEXING},
    DocumentState.INDEXING: {DocumentState.INDEXED, DocumentState.INDEX_FAILED},
    DocumentState.INDEXED: {DocumentState.INDEXING},
    DocumentState.INDEX_FAILED: {DocumentState.INDEXING, DocumentState.INDEX_READY},
    DocumentState.EXTRACTION_FAILED: {DocumentState.EXTRACTION_QUEUED},
}


class InvalidStateTransitionError(ValueError):
    """Raised when an illegal document state transition is attempted."""


def validate_transition(current_state: str, target_state: str) -> None:
    current = current_state.upper()
    target = target_state.upper()
    allowed = VALID_DOCUMENT_TRANSITIONS.get(current, set())
    if target not in allowed:
        raise InvalidStateTransitionError(
            f"Invalid transition from {current} to {target}. Allowed transitions: {sorted(allowed)}"
        )


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    media_type: Mapped[str] = mapped_column(String(80))
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    vault_path: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(40), default=DocumentState.ENCRYPTED)
    version: Mapped[int] = mapped_column(Integer, default=1)
    active_index_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    index_cleanup_pending: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class ExtractionJob(Base):
    __tablename__ = "extraction_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String(40), default="QUEUED")
    pages_processed: Mapped[int] = mapped_column(Integer, default=0)
    total_pages: Mapped[int] = mapped_column(Integer, default=0)
    extraction_method: Mapped[str | None] = mapped_column(String(40), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class DocumentPageRecord(Base):
    __tablename__ = "document_pages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    page_number: Mapped[int] = mapped_column(Integer)
    width: Mapped[float] = mapped_column(Float, default=0.0)
    height: Mapped[float] = mapped_column(Float, default=0.0)
    extraction_method: Mapped[str] = mapped_column(String(40))
    text_hash: Mapped[str] = mapped_column(String(64))
    encrypted_artifact_path: Mapped[str] = mapped_column(Text)
    page_image_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    page_image_media_type: Mapped[str | None] = mapped_column(String(80), nullable=True)
    encrypted_page_image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    masked_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    masked_text_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class DocumentVisualRegionRecord(Base):
    __tablename__ = "document_visual_regions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    page_number: Mapped[int] = mapped_column(Integer)
    region_sequence: Mapped[int] = mapped_column(Integer)
    region_type: Mapped[str] = mapped_column(String(40))
    source: Mapped[str] = mapped_column(String(80))
    bbox_json: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    caption_text: Mapped[str] = mapped_column(Text)
    caption_hash: Mapped[str] = mapped_column(String(64))
    image_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class RedactionFinding(Base):
    __tablename__ = "redaction_findings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    page_number: Mapped[int] = mapped_column(Integer)
    finding_type: Mapped[str] = mapped_column(String(60), index=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    detection_source: Mapped[str] = mapped_column(String(80))
    start_offset: Mapped[int] = mapped_column(Integer)
    end_offset: Mapped[int] = mapped_column(Integer)
    bbox_json: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    salted_value_hash: Mapped[str] = mapped_column(String(64))
    masked_context: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(40), default=FindingStatus.PENDING)
    reviewer_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    actor_id: Mapped[str] = mapped_column(String(120))
    event_type: Mapped[str] = mapped_column(String(120), index=True)
    resource_type: Mapped[str] = mapped_column(String(80))
    resource_id: Mapped[str | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    details: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    previous_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    event_hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class DocumentChunkRecord(Base):
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    page_number: Mapped[int] = mapped_column(Integer)
    chunk_sequence: Mapped[int] = mapped_column(Integer)
    index_version: Mapped[int] = mapped_column(Integer, default=1)
    chunk_id: Mapped[str] = mapped_column(String(64), index=True)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    masked_content_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
