import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from koshshield.models import (
    DocumentPageRecord,
    DocumentRecord,
    DocumentState,
    FindingStatus,
    RedactionFinding,
)
from koshshield.security.pii.indian_pii import IndianPiiDetector
from koshshield.services.audit import record_audit_event

logger = logging.getLogger(__name__)


class PrivacyGateError(ValueError):
    """Base error for indexing privacy gate violations."""


class DocumentNotApprovedError(PrivacyGateError):
    """Raised when indexing is attempted on a document that has not been approved."""


class UnresolvedFindingsError(PrivacyGateError):
    """Raised when unresolved redaction findings remain on the document."""


class ResidualPiiDetectedError(PrivacyGateError):
    """Raised when raw PII is detected in approved masked text before indexing."""


class RetrievalPrivacyGate:
    """Enforces privacy invariants prior to chunking and vector indexing.

    Invariants:
    1. Document must be in REDACTION_APPROVED or INDEX_READY state.
    2. Zero findings may be in PENDING status.
    3. Approved masked text must contain NO detectable raw Indian PII.
    4. Violations trigger privacy-safe audit logging and block indexing.
    """

    def __init__(self, pii_salt: str = "koshshield-default-dev-salt") -> None:
        self.detector = IndianPiiDetector(salt=pii_salt)

    def validate_document_for_indexing(
        self,
        session: Session,
        document: DocumentRecord,
        actor_id: str = "system",
    ) -> list[DocumentPageRecord]:
        # 1. State check
        if document.status not in (
            DocumentState.REDACTION_APPROVED,
            DocumentState.INDEX_READY,
            DocumentState.INDEXED,
            DocumentState.INDEXING,
        ):
            msg = (
                f"Document '{document.id}' in state '{document.status}' cannot be indexed. "
                "Must be REDACTION_APPROVED or INDEX_READY."
            )
            self._audit_block(session, document.id, actor_id, "ILLEGAL_STATE", msg)
            raise DocumentNotApprovedError(msg)

        # 2. Unresolved findings check
        pending_count = session.scalar(
            select(RedactionFinding).where(
                RedactionFinding.document_id == document.id,
                RedactionFinding.status == FindingStatus.PENDING,
            )
        )
        if pending_count is not None:
            msg = f"Document '{document.id}' contains unresolved redaction findings."
            self._audit_block(session, document.id, actor_id, "UNRESOLVED_FINDINGS", msg)
            raise UnresolvedFindingsError(msg)

        # 3. Retrieve approved masked pages
        pages = list(
            session.scalars(
                select(DocumentPageRecord)
                .where(DocumentPageRecord.document_id == document.id)
                .order_by(DocumentPageRecord.page_number)
            )
        )
        if not pages:
            msg = f"Document '{document.id}' has no extracted pages to index."
            self._audit_block(session, document.id, actor_id, "NO_PAGES", msg)
            raise PrivacyGateError(msg)

        # 4. Scan masked text on all pages with IndianPiiDetector
        for page in pages:
            text_to_check = page.masked_text or ""
            if not text_to_check:
                continue

            findings = self.detector.detect(text_to_check, page_number=page.page_number)
            if findings:
                types = sorted({f.finding_type for f in findings})
                msg = (
                    f"Privacy Gate rejected document '{document.id}' on page {page.page_number}: "
                    f"Residual raw PII detected ({types}) in masked text."
                )
                self._audit_block(
                    session,
                    document.id,
                    actor_id,
                    "RESIDUAL_PII_DETECTED",
                    f"Residual PII types detected: {types}",
                    extra={"page_number": page.page_number, "types": types},
                )
                logger.error(msg)
                raise ResidualPiiDetectedError(msg)

        return pages

    @staticmethod
    def _audit_block(
        session: Session,
        document_id: str,
        actor_id: str,
        reason: str,
        description: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        details: dict[str, Any] = {
            "reason": reason,
            "description": description,
        }
        if extra:
            details.update(extra)
        record_audit_event(
            session=session,
            actor_id=actor_id,
            event_type="INDEXING_PRIVACY_GATE_BLOCKED",
            resource_type="document",
            resource_id=document_id,
            details=details,
        )
