from uuid import uuid4

from koshshield.models import DocumentState
from koshshield.services.agent.policy import (
    AgentPolicyEngine,
    PolicyDecision,
    PolicyReasonCode,
)


def test_policy_engine_incomplete_context_fails_closed() -> None:
    engine = AgentPolicyEngine()

    # Missing tool name
    res1 = engine.evaluate(
        tool_name=None,
        arguments={"expression": "1 + 1"},
        classification="INTERNAL",
    )
    assert res1.verdict == PolicyDecision.DENY
    assert res1.decision == "REJECTED"
    assert res1.reason_code == PolicyReasonCode.INCOMPLETE_CONTEXT

    # Missing arguments
    res2 = engine.evaluate(
        tool_name="calculator",
        arguments=None,
        classification="INTERNAL",
    )
    assert res2.verdict == PolicyDecision.DENY
    assert res2.decision == "REJECTED"
    assert res2.reason_code == PolicyReasonCode.INCOMPLETE_CONTEXT

    # Missing classification
    res3 = engine.evaluate(
        tool_name="calculator",
        arguments={"expression": "1 + 1"},
        classification=None,
    )
    assert res3.verdict == PolicyDecision.DENY
    assert res3.decision == "REJECTED"
    assert res3.reason_code == PolicyReasonCode.INCOMPLETE_CONTEXT


def test_policy_engine_role_gating() -> None:
    engine = AgentPolicyEngine()
    doc_id = str(uuid4())

    # Empty roles denied
    res_empty = engine.evaluate(
        tool_name="draft_approval_note",
        arguments={
            "document_id": doc_id,
            "note_subject": "Official sanctioned procurement note",
            "recommended_action": "Sanction the procurement order",
        },
        classification="CONFIDENTIAL",
        roles=[],
    )
    assert res_empty.verdict == PolicyDecision.DENY
    assert res_empty.reason_code == PolicyReasonCode.INSUFFICIENT_ROLE_PERMISSIONS

    # Auditor-only denied
    res_auditor = engine.evaluate(
        tool_name="draft_approval_note",
        arguments={
            "document_id": doc_id,
            "note_subject": "Official sanctioned procurement note",
            "recommended_action": "Sanction the procurement order",
        },
        classification="CONFIDENTIAL",
        roles=["auditor"],
    )
    assert res_auditor.verdict == PolicyDecision.DENY
    assert res_auditor.reason_code == PolicyReasonCode.INSUFFICIENT_ROLE_PERMISSIONS

    # Valid reviewer/approver role allowed
    res_valid = engine.evaluate(
        tool_name="draft_approval_note",
        arguments={
            "document_id": doc_id,
            "note_subject": "Official sanctioned procurement note",
            "recommended_action": "Sanction the procurement order",
        },
        classification="CONFIDENTIAL",
        roles=["reviewer", "approver"],
        document_status=DocumentState.INDEXED,
    )
    assert res_valid.verdict == PolicyDecision.REQUIRE_APPROVAL
    assert res_valid.decision == "ALLOWED"
    assert res_valid.approval_required is True


def test_policy_engine_tenant_isolation_gating() -> None:
    engine = AgentPolicyEngine()
    doc_id = str(uuid4())

    cross_tenant = engine.evaluate(
        tool_name="summarize_document",
        arguments={"document_id": doc_id},
        classification="INTERNAL",
        tenant_id="tenant-a",
        document_tenant_id="tenant-b",
    )
    assert cross_tenant.verdict == PolicyDecision.DENY
    assert cross_tenant.reason_code == PolicyReasonCode.CROSS_TENANT_ACCESS_DENIED


def test_policy_engine_document_approval_state_and_evidence() -> None:
    engine = AgentPolicyEngine()
    doc_id = str(uuid4())

    # Unindexed document denied
    unindexed = engine.evaluate(
        tool_name="summarize_document",
        arguments={"document_id": doc_id},
        classification="INTERNAL",
        document_status=DocumentState.REVIEW_REQUIRED,
    )
    assert unindexed.verdict == PolicyDecision.DENY
    assert unindexed.reason_code == PolicyReasonCode.DOCUMENT_NOT_APPROVED

    # Missing evidence chunks denied
    missing_evidence = engine.evaluate(
        tool_name="summarize_document",
        arguments={"document_id": doc_id},
        classification="INTERNAL",
        document_status=DocumentState.INDEXED,
        has_evidence=False,
    )
    assert missing_evidence.verdict == PolicyDecision.DENY
    assert missing_evidence.reason_code == PolicyReasonCode.EVIDENCE_UNAVAILABLE


def test_policy_engine_restricted_classification_requires_approval() -> None:
    engine = AgentPolicyEngine()
    doc_id = str(uuid4())

    res = engine.evaluate(
        tool_name="summarize_document",
        arguments={"document_id": doc_id},
        classification="RESTRICTED",
        document_status=DocumentState.INDEXED,
    )
    assert res.verdict == PolicyDecision.REQUIRE_APPROVAL
    assert res.decision == "ALLOWED"
    assert res.reason_code == PolicyReasonCode.RESTRICTED_DATA_APPROVAL_REQUIRED
    assert res.approval_required is True
