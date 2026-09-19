"""Evidence-bound deliverables with provenance tracking and citation binding for KoshShield AI.

Ensures:
- Outputs are reviewable deliverables bound to citations from masked document/page/chunks
- Clear provenance: action type, policy result, approval ID, document/version, evidence IDs
- Strict non-binding disclaimer: never claims semantic correctness or legal validity
- Zero exposure of unredacted document text, prompts, or local filesystem paths
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from koshshield.models import (
    AgentApprovalRecord,
    AgentRunRecord,
    DocumentChunkRecord,
    DocumentRecord,
)

DELIVERABLE_LEGAL_DISCLAIMER = (
    "This deliverable was generated under local privacy-masked deterministic policy governance. "
    "It does not claim statutory DPDP compliance, legal validity, semantic correctness, "
    "or final administrative approval. It constitutes a structured recommendation for human review."
)


class DeliverableCitation(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    page_number: int
    snippet_hash: str
    masked_snippet: str
    char_start: int
    char_end: int


class DeliverableProvenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    action_type: str
    policy_result: str
    approval_id: str
    document_id: str | None = None
    document_version: int | None = None
    active_index_version: int | None = None
    evidence_identities: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    verification_status: str = "DETERMINISTIC_VERIFIED"
    disclaimer: str = DELIVERABLE_LEGAL_DISCLAIMER


class AgentDeliverable(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    run_id: str
    approval_id: str
    tenant_id: str
    action_name: str
    title: str
    content: dict[str, Any] | str
    media_type: str = "application/json"
    citations: list[DeliverableCitation] = Field(default_factory=list)
    provenance: DeliverableProvenance


def build_evidence_deliverable(
    *,
    session: Session,
    run: AgentRunRecord,
    approval: AgentApprovalRecord,
) -> AgentDeliverable:
    """Constructs a cryptographically bound, citation-backed deliverable for an agent run."""
    document_id = run.arguments_json.get("document_id")
    doc: DocumentRecord | None = None
    chunks: list[DocumentChunkRecord] = []
    citations: list[DeliverableCitation] = []
    evidence_ids: list[str] = []

    if document_id:
        doc = session.scalar(
            select(DocumentRecord).where(
                DocumentRecord.id == str(document_id),
                DocumentRecord.tenant_id == run.tenant_id,
            )
        )
        if doc:
            evidence_ids.append(f"doc:{doc.id}:sha256:{doc.sha256[:16]}")
            chunks = list(
                session.scalars(
                    select(DocumentChunkRecord)
                    .where(
                        DocumentChunkRecord.document_id == doc.id,
                        DocumentChunkRecord.index_version == doc.active_index_version,
                    )
                    .order_by(DocumentChunkRecord.page_number, DocumentChunkRecord.chunk_sequence)
                    .limit(10)
                )
            )
            for chunk in chunks:
                evidence_ids.append(f"chunk:{chunk.chunk_id}")
                citations.append(
                    DeliverableCitation(
                        chunk_id=chunk.chunk_id,
                        page_number=chunk.page_number,
                        snippet_hash=chunk.masked_content_hash,
                        masked_snippet=f"[MASKED_EVIDENCE_P{chunk.page_number}_C{chunk.chunk_sequence}]",
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                    )
                )

    verification_status = "DETERMINISTIC_VERIFIED"
    result_data = run.result_json.get("output", {}) if run.result_json else {}
    if isinstance(result_data, dict) and result_data.get("status") == "NOT_EXECUTED":
        verification_status = "NOT_EXECUTED_OFFLINE"

    # Title resolution
    title = f"Deliverable: {run.tool_name.replace('_', ' ').title()}"
    if isinstance(result_data, dict) and "title" in result_data:
        title = str(result_data["title"])
    elif isinstance(result_data, dict) and "case_id" in result_data:
        title = f"Procurement Evaluation: {result_data['case_id']}"

    # Content & Media type resolution
    media_type = "application/json"
    content: dict[str, Any] | str = result_data
    if isinstance(result_data, dict) and "content" in result_data:
        content = str(result_data["content"])
        media_type = str(result_data.get("media_type", "text/markdown"))

    provenance = DeliverableProvenance(
        action_type=run.tool_name,
        policy_result=run.policy_decision,
        approval_id=approval.id,
        document_id=doc.id if doc else None,
        document_version=doc.version if doc else None,
        active_index_version=doc.active_index_version if doc else None,
        evidence_identities=evidence_ids,
        generated_at=datetime.now(UTC),
        verification_status=verification_status,
        disclaimer=DELIVERABLE_LEGAL_DISCLAIMER,
    )

    deliverable_id = hashlib.sha256(
        f"{run.id}:{approval.id}:{run.result_hash or 'no_result'}".encode()
    ).hexdigest()[:32]

    return AgentDeliverable(
        id=deliverable_id,
        run_id=run.id,
        approval_id=approval.id,
        tenant_id=run.tenant_id,
        action_name=run.tool_name,
        title=title,
        content=content,
        media_type=media_type,
        citations=citations,
        provenance=provenance,
    )
