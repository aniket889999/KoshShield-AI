from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from koshshield.models import AgentApprovalRecord, AgentRunRecord, ApprovalDecision
from koshshield.services.agent.lifecycle import (
    ApprovalExpiredError,
    ApprovalReplayError,
    ApprovalTransitionError,
    StaleDocumentVersionError,
    compute_safe_input_digest,
    is_approval_expired,
    validate_approval_transition,
    validate_execution_prerequisites,
)


def test_valid_approval_lifecycle_transitions() -> None:
    # PENDING -> APPROVED
    validate_approval_transition(
        current_state=ApprovalDecision.PENDING,
        target_state=ApprovalDecision.APPROVED,
    )

    # PENDING -> REJECTED
    validate_approval_transition(
        current_state=ApprovalDecision.PENDING,
        target_state=ApprovalDecision.REJECTED,
    )

    # APPROVED -> EXECUTED
    validate_approval_transition(
        current_state=ApprovalDecision.APPROVED,
        target_state=ApprovalDecision.EXECUTED,
    )

    # APPROVED -> FAILED
    validate_approval_transition(
        current_state=ApprovalDecision.APPROVED,
        target_state=ApprovalDecision.FAILED,
    )


def test_invalid_transitions_rejected() -> None:
    # Cannot jump PENDING directly to EXECUTED
    with pytest.raises(ApprovalTransitionError):
        validate_approval_transition(
            current_state=ApprovalDecision.PENDING,
            target_state=ApprovalDecision.EXECUTED,
        )

    # Cannot transition out of terminal EXECUTED
    with pytest.raises(ApprovalReplayError):
        validate_approval_transition(
            current_state=ApprovalDecision.EXECUTED,
            target_state=ApprovalDecision.APPROVED,
        )

    # Cannot transition out of terminal REJECTED
    with pytest.raises(ApprovalReplayError):
        validate_approval_transition(
            current_state=ApprovalDecision.REJECTED,
            target_state=ApprovalDecision.APPROVED,
        )


def test_approval_expiration_detection() -> None:
    now = datetime.now(UTC)
    old_time = now - timedelta(hours=25)

    assert is_approval_expired(old_time, ttl_seconds=86400, now=now) is True
    assert is_approval_expired(now - timedelta(hours=1), ttl_seconds=86400, now=now) is False

    with pytest.raises(ApprovalExpiredError):
        validate_approval_transition(
            current_state=ApprovalDecision.PENDING,
            target_state=ApprovalDecision.APPROVED,
            created_at=old_time,
            now=now,
        )


def test_validate_execution_prerequisites_guards() -> None:
    secret = "test-integrity-secret"
    args = {"document_id": str(uuid4()), "document_version": 2}
    valid_hash = compute_safe_input_digest(args, secret)

    run = AgentRunRecord(
        id=str(uuid4()),
        tenant_id="dept-a",
        actor_id="officer-1",
        tool_name="summarize_document",
        classification="CONFIDENTIAL",
        arguments_json=args,
        arguments_hash=valid_hash,
        argument_summary="Summary",
        status="APPROVED",
        state_history=["REQUESTED", "POLICY_EVALUATED", "APPROVAL_PENDING", "APPROVED"],
        policy_decision="ALLOWED",
        policy_reason="Policy passed",
        approval_required=True,
    )

    approval = AgentApprovalRecord(
        id=str(uuid4()),
        agent_run_id=run.id,
        decision=ApprovalDecision.APPROVED,
        reviewer_id="supervisor-1",
        created_at=datetime.now(UTC),
    )

    # 1. Valid execution passes
    validate_execution_prerequisites(
        approval=approval,
        run=run,
        integrity_secret=secret,
        active_document_version=2,
    )

    # 2. Tampered arguments fail
    run.arguments_hash = "tampered_hash_000000000000000000000000000000000000000000000000000000"
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_execution_prerequisites(
            approval=approval,
            run=run,
            integrity_secret=secret,
            active_document_version=2,
        )
    run.arguments_hash = valid_hash

    # 3. Stale document version fails
    with pytest.raises(StaleDocumentVersionError):
        validate_execution_prerequisites(
            approval=approval,
            run=run,
            integrity_secret=secret,
            active_document_version=3,  # Newer version on server
        )

    # 4. Replay execution rejected
    approval.decision = ApprovalDecision.EXECUTED
    with pytest.raises(ApprovalReplayError):
        validate_execution_prerequisites(
            approval=approval,
            run=run,
            integrity_secret=secret,
            active_document_version=2,
        )
