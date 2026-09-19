"""Approval-request lifecycle management for KoshShield AI agent actions.

Defines:
- Lifecycle states: PENDING, APPROVED, REJECTED, EXPIRED, EXECUTED, FAILED
- Transition guards and expiration validation
- Provenance binding: tenant, actor, action, safe input digest, document/version, policy decision
- Prevention of double-execution, replay, and stale document version execution
- Zero raw prompts, zero source document text, zero PII, and zero local paths
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from typing import Any

from koshshield.models import AgentApprovalRecord, AgentRunRecord, ApprovalDecision


class ApprovalTransitionError(ValueError):
    """Raised when an invalid approval lifecycle transition is attempted."""


class ApprovalExpiredError(ValueError):
    """Raised when attempting an action on an expired approval request."""


class StaleDocumentVersionError(ValueError):
    """Raised when the document version has drifted since the approval was granted."""


class ApprovalReplayError(ValueError):
    """Raised when attempting to re-approve or re-execute an already final approval."""


DEFAULT_APPROVAL_TTL_SECONDS = 86400  # 24 hours

VALID_APPROVAL_TRANSITIONS: dict[str, frozenset[str]] = {
    ApprovalDecision.PENDING: frozenset(
        {
            ApprovalDecision.APPROVED,
            ApprovalDecision.REJECTED,
            ApprovalDecision.EXPIRED,
        }
    ),
    ApprovalDecision.APPROVED: frozenset(
        {
            ApprovalDecision.EXECUTED,
            ApprovalDecision.FAILED,
            ApprovalDecision.EXPIRED,
        }
    ),
    ApprovalDecision.REJECTED: frozenset(),
    ApprovalDecision.EXPIRED: frozenset(),
    ApprovalDecision.EXECUTED: frozenset(),
    ApprovalDecision.FAILED: frozenset(),
}


def compute_safe_input_digest(arguments: dict[str, Any], secret: str) -> str:
    """Computes a cryptographically bound HMAC-SHA256 digest of canonical normalized arguments."""
    canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hmac.new(secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()


def is_approval_expired(
    created_at: datetime,
    ttl_seconds: int = DEFAULT_APPROVAL_TTL_SECONDS,
    now: datetime | None = None,
) -> bool:
    """Checks whether an approval request has exceeded its time-to-live window."""
    current_time = now or datetime.now(UTC)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return current_time > (created_at + timedelta(seconds=ttl_seconds))


def validate_approval_transition(
    *,
    current_state: str,
    target_state: str,
    created_at: datetime | None = None,
    ttl_seconds: int = DEFAULT_APPROVAL_TTL_SECONDS,
    now: datetime | None = None,
) -> None:
    """Validates that a transition between approval states is legitimate and unexpired."""
    if current_state in {
        ApprovalDecision.REJECTED,
        ApprovalDecision.EXPIRED,
        ApprovalDecision.EXECUTED,
        ApprovalDecision.FAILED,
    }:
        raise ApprovalReplayError(
            f"Cannot transition from terminal approval state '{current_state}' to '{target_state}'."
        )

    if (
        created_at
        and is_approval_expired(created_at, ttl_seconds=ttl_seconds, now=now)
        and target_state != ApprovalDecision.EXPIRED
    ):
        raise ApprovalExpiredError(
            f"Approval request created at {created_at.isoformat()} has expired."
        )

    allowed = VALID_APPROVAL_TRANSITIONS.get(current_state, frozenset())
    if target_state not in allowed:
        raise ApprovalTransitionError(
            f"Invalid approval state transition: '{current_state}' -> '{target_state}'."
        )


def validate_execution_prerequisites(
    *,
    approval: AgentApprovalRecord,
    run: AgentRunRecord,
    integrity_secret: str,
    active_document_version: int | None = None,
    now: datetime | None = None,
) -> None:
    """Strictly validates that an approval request is eligible for execution.

    Guards against:
    - Unapproved requests
    - Expired approvals
    - Replay / double execution
    - Stale document index/redaction versions
    - Tampered arguments
    """
    if approval.decision == ApprovalDecision.EXECUTED:
        raise ApprovalReplayError("Approval request has already been executed (replay prevented).")

    if approval.decision != ApprovalDecision.APPROVED:
        raise ApprovalTransitionError(
            f"Approval decision is '{approval.decision}', required '{ApprovalDecision.APPROVED}'."
        )

    if is_approval_expired(approval.created_at, now=now):
        raise ApprovalExpiredError("Approval request has expired and cannot be executed.")

    expected_hash = compute_safe_input_digest(run.arguments_json, integrity_secret)
    if not hmac.compare_digest(expected_hash, run.arguments_hash):
        raise ValueError("Safe input digest mismatch: arguments have been modified.")

    if active_document_version is not None:
        run_doc_version = run.arguments_json.get("document_version")
        if run_doc_version is not None and int(run_doc_version) != active_document_version:
            raise StaleDocumentVersionError(
                f"Document version mismatch: approved version {run_doc_version} "
                f"!= current {active_document_version}."
            )
