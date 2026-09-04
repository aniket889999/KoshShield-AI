import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from koshshield.database import engine
from koshshield.models import (
    AuditEvent,
    DocumentPageRecord,
    DocumentRecord,
    DocumentState,
    FindingStatus,
    RedactionFinding,
)
from koshshield.services.retrieval.privacy_gate import (
    DocumentNotApprovedError,
    ResidualPiiDetectedError,
    RetrievalPrivacyGate,
    UnresolvedFindingsError,
)


@pytest.fixture
def db_session() -> Session:
    session = Session(bind=engine)
    try:
        yield session
    finally:
        session.rollback()
        session.close()


def test_privacy_gate_rejects_unapproved_document(db_session: Session) -> None:
    gate = RetrievalPrivacyGate()
    doc = DocumentRecord(
        id=str(uuid.uuid4()),
        filename="draft.pdf",
        media_type="application/pdf",
        size_bytes=1024,
        sha256="fake-hash",
        vault_path="vault/draft.ksh",
        status=DocumentState.REVIEW_REQUIRED,
    )
    db_session.add(doc)
    db_session.commit()

    with pytest.raises(DocumentNotApprovedError):
        gate.validate_document_for_indexing(db_session, doc)

    # Verify audit event was logged
    audit = db_session.scalar(
        select(AuditEvent).where(
            AuditEvent.event_type == "INDEXING_PRIVACY_GATE_BLOCKED",
            AuditEvent.resource_id == doc.id,
        )
    )
    assert audit is not None
    assert audit.details.get("reason") == "ILLEGAL_STATE"


def test_privacy_gate_rejects_unresolved_findings(db_session: Session) -> None:
    gate = RetrievalPrivacyGate()
    doc = DocumentRecord(
        id=str(uuid.uuid4()),
        filename="approved_with_pending.pdf",
        media_type="application/pdf",
        size_bytes=1024,
        sha256="fake-hash-2",
        vault_path="vault/draft.ksh",
        status=DocumentState.INDEX_READY,
    )
    finding = RedactionFinding(
        id=str(uuid.uuid4()),
        document_id=doc.id,
        page_number=1,
        finding_type="AADHAAR",
        confidence=0.99,
        detection_source="verhoeff",
        start_offset=10,
        end_offset=24,
        salted_value_hash="salted-hash",
        masked_context="Context preview",
        status=FindingStatus.PENDING,
    )
    db_session.add_all([doc, finding])
    db_session.commit()

    with pytest.raises(UnresolvedFindingsError):
        gate.validate_document_for_indexing(db_session, doc)


def test_privacy_gate_detects_residual_raw_pii(db_session: Session) -> None:
    gate = RetrievalPrivacyGate()
    doc = DocumentRecord(
        id=str(uuid.uuid4()),
        filename="leaky.pdf",
        media_type="application/pdf",
        size_bytes=1024,
        sha256="fake-hash-3",
        vault_path="vault/draft.ksh",
        status=DocumentState.INDEX_READY,
    )
    # Valid Indian PAN: ABCPE1234F
    raw_pan = "ABCPE1234F"
    page = DocumentPageRecord(
        id=str(uuid.uuid4()),
        document_id=doc.id,
        page_number=1,
        extraction_method="native_pdf",
        text_hash="hash",
        encrypted_artifact_path="vault/p1.ksh",
        masked_text=f"Official statement containing raw PAN: {raw_pan}",
    )
    db_session.add_all([doc, page])
    db_session.commit()

    with pytest.raises(ResidualPiiDetectedError) as exc_info:
        gate.validate_document_for_indexing(db_session, doc)

    assert "Residual raw PII detected" in str(exc_info.value)

    # Verify audit event was logged without leaking raw PAN
    audit = db_session.scalar(
        select(AuditEvent).where(
            AuditEvent.event_type == "INDEXING_PRIVACY_GATE_BLOCKED",
            AuditEvent.resource_id == doc.id,
        )
    )
    assert audit is not None
    assert raw_pan not in str(audit.details)


def test_privacy_gate_accepts_clean_masked_document(db_session: Session) -> None:
    gate = RetrievalPrivacyGate()
    doc = DocumentRecord(
        id=str(uuid.uuid4()),
        filename="clean.pdf",
        media_type="application/pdf",
        size_bytes=1024,
        sha256="fake-hash-clean",
        vault_path="vault/clean.ksh",
        status=DocumentState.INDEX_READY,
    )
    page = DocumentPageRecord(
        id=str(uuid.uuid4()),
        document_id=doc.id,
        page_number=1,
        extraction_method="native_pdf",
        text_hash="hash",
        encrypted_artifact_path="vault/p1.ksh",
        masked_text="Official statement for tender. Contractor identity is [REDACTED_AADHAAR].",
    )
    db_session.add_all([doc, page])
    db_session.commit()

    pages = gate.validate_document_for_indexing(db_session, doc)
    assert len(pages) == 1
    assert pages[0].id == page.id
