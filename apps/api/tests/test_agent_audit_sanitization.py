"""Audit integrity and sanitization verification for policy-gated agent operations."""

from __future__ import annotations

import json
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from koshshield.database import engine
from koshshield.models import AuditEvent, DocumentRecord
from koshshield.services.agent.policy import AgentPolicyEngine
from koshshield.services.agent.runner import RestrictedDeterministicToolRunner
from koshshield.services.agent.service import AgentRunService


def test_agent_audit_events_metadata_only_and_sanitized() -> None:
    """Proves agent lifecycle audit logs contain strictly metadata with zero raw inputs or PII."""
    policy_engine = AgentPolicyEngine()
    tool_runner = RestrictedDeterministicToolRunner()
    service = AgentRunService(
        policy_engine=policy_engine,
        tool_runner=tool_runner,
        integrity_secret="test-audit-integrity-secret",
    )

    tenant_id = f"tenant-audit-{uuid4().hex[:8]}"
    requester_id = f"requester-{uuid4().hex[:8]}"
    reviewer_id = f"reviewer-{uuid4().hex[:8]}"
    executor_id = f"executor-{uuid4().hex[:8]}"

    with Session(bind=engine) as session:
        # Create an approved indexed document for testing
        doc = DocumentRecord(
            id=str(uuid4()),
            tenant_id=tenant_id,
            filename="procurement_sanction.pdf",
            media_type="application/pdf",
            size_bytes=1024,
            sha256="0" * 64,
            vault_path="/vault/doc.enc",
            status="INDEXED",
            version=1,
        )
        session.add(doc)
        session.commit()

        # 1. Propose action
        run = service.propose_action(
            session=session,
            tenant_id=tenant_id,
            actor_id=requester_id,
            tool_name="calculator",
            classification="CONFIDENTIAL",
            arguments={"expression": "250 * 4"},
        )
        run_id = run.id

        # 2. Decide approval
        run = service.decide_approval(
            session=session,
            run_id=run_id,
            tenant_id=tenant_id,
            reviewer_id=reviewer_id,
            decision="APPROVED",
            version=run.version,
        )

        # 3. Execute action
        run = service.execute_action(
            session=session,
            run_id=run_id,
            tenant_id=tenant_id,
            executor_id=executor_id,
            version=run.version,
        )
        assert run.status == "COMPLETED"

        # Query all audit events for this run
        events = list(
            session.scalars(
                select(AuditEvent)
                .where(
                    AuditEvent.tenant_id == tenant_id,
                    AuditEvent.resource_type == "agent_run",
                    AuditEvent.resource_id == run_id,
                )
                .order_by(AuditEvent.created_at.asc())
            )
        )

        event_types = [e.event_type for e in events]
        assert "AGENT_POLICY_EVALUATED" in event_types
        assert "AGENT_ACTION_PROPOSED" in event_types
        assert "AGENT_ACTION_APPROVED" in event_types
        assert "AGENT_EXECUTION_ATTEMPTED" in event_types
        assert "AGENT_ACTION_COMPLETED" in event_types

        # Verify each event details is strictly metadata
        for event in events:
            details_str = json.dumps(event.details)
            # Must not contain raw prompt tokens, local paths, or secrets
            assert "/Users/" not in details_str
            assert "/home/" not in details_str
            assert "test-audit-integrity-secret" not in details_str
            assert "raw_prompt" not in details_str
            assert "system_prompt" not in details_str

            # Check specific event details
            if event.event_type == "AGENT_POLICY_EVALUATED":
                assert event.details["tool_name"] == "calculator"
                assert "arguments_hash" in event.details
                assert event.details["policy_decision"] in {"ALLOWED", "REQUIRE_APPROVAL", "DENY"}
                assert "approval_required" in event.details

            elif event.event_type == "AGENT_ACTION_PROPOSED":
                assert event.details["run_id"] == run_id
                assert "arguments_hash" in event.details

            elif event.event_type == "AGENT_ACTION_APPROVED":
                assert event.details["decision"] == "APPROVED"
                assert event.details["requester_id"] == requester_id
                assert event.actor_id == reviewer_id

            elif event.event_type == "AGENT_EXECUTION_ATTEMPTED":
                assert event.details["executor_id"] == executor_id
                assert "approval_id" in event.details

            elif event.event_type == "AGENT_ACTION_COMPLETED":
                assert "result_hash" in event.details
                assert "sandbox_output_hash" in event.details
                assert event.details["network_disabled"] is True


def test_agent_rejected_action_audit_sanitization() -> None:
    """Verifies that rejected or blocked action audit events do not leak forbidden arguments."""
    policy_engine = AgentPolicyEngine()
    tool_runner = RestrictedDeterministicToolRunner()
    service = AgentRunService(
        policy_engine=policy_engine,
        tool_runner=tool_runner,
        integrity_secret="test-audit-integrity-secret",
    )

    tenant_id = f"tenant-reject-{uuid4().hex[:8]}"
    requester_id = f"requester-{uuid4().hex[:8]}"

    with Session(bind=engine) as session:
        # Propose blocked tool
        run = service.propose_action(
            session=session,
            tenant_id=tenant_id,
            actor_id=requester_id,
            tool_name="network_request",
            classification="RESTRICTED",
            arguments={"destination": "https://external.leak.example"},
        )
        assert run.status == "REJECTED"

        events = list(
            session.scalars(
                select(AuditEvent).where(
                    AuditEvent.tenant_id == tenant_id,
                    AuditEvent.resource_type == "agent_run",
                    AuditEvent.resource_id == run.id,
                )
            )
        )

        assert len(events) >= 1
        for event in events:
            details_str = json.dumps(event.details)
            # The URL destination must NOT be leaked in the audit event details
            assert "https://external.leak.example" not in details_str
            assert "leak.example" not in details_str
            assert "arguments_hash" in event.details
            assert event.details["policy_decision"] == "REJECTED"
