import hashlib
import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from koshshield.api.routes.agent import get_tool_runner
from koshshield.database import SessionLocal
from koshshield.main import app
from koshshield.models import AgentRunRecord, AuditEvent, DocumentRecord, DocumentState
from koshshield.services.agent.tool_runner import ToolRunnerResult


class FakeRestrictedRunner:
    def execute(self, *, tool_name: str, payload: dict[str, object]) -> ToolRunnerResult:
        assert tool_name == "calculator"
        assert payload == {"expression": "1250 * 18 / 100"}
        output = {"ok": True, "tool": tool_name, "result": {"value": "225"}}
        canonical = json.dumps(output, sort_keys=True, separators=(",", ":"))
        return ToolRunnerResult(
            payload=output,
            output_hash=hashlib.sha256(canonical.encode()).hexdigest(),
            sandbox={
                "network_disabled": True,
                "read_only_root": True,
                "capabilities_dropped": True,
                "image": "test-runner:0.1.0",
                "timeout_seconds": 1,
            },
        )


@pytest.fixture
def agent_client(client: TestClient) -> TestClient:
    app.dependency_overrides[get_tool_runner] = lambda: FakeRestrictedRunner()
    try:
        yield client
    finally:
        app.dependency_overrides.pop(get_tool_runner, None)


def test_agent_action_requires_independent_approval_before_execution(
    agent_client: TestClient,
) -> None:
    expression = "1250 * 18 / 100"
    headers = {"X-Actor-ID": "finance-requester", "X-Tenant-ID": "finance"}
    proposed = agent_client.post(
        "/api/v1/agent/runs",
        headers=headers,
        json={
            "tool_name": "calculator",
            "classification": "CONFIDENTIAL",
            "arguments": {"expression": expression},
        },
    )

    assert proposed.status_code == 201
    pending = proposed.json()
    assert pending["status"] == "APPROVAL_PENDING"
    assert pending["approval"]["decision"] == "PENDING"
    assert pending["result"] is None

    self_approval = agent_client.post(
        f"/api/v1/agent/runs/{pending['id']}/approval",
        headers=headers,
        json={"decision": "APPROVED", "version": pending["version"]},
    )
    assert self_approval.status_code == 403

    approved = agent_client.post(
        f"/api/v1/agent/runs/{pending['id']}/approval",
        headers={"X-Actor-ID": "finance-supervisor", "X-Tenant-ID": "finance"},
        json={"decision": "APPROVED", "version": pending["version"]},
    )
    assert approved.status_code == 200
    approved_run = approved.json()
    assert approved_run["status"] == "APPROVED"
    assert approved_run["approval"]["reviewer_id"] == "finance-supervisor"

    executed = agent_client.post(
        f"/api/v1/agent/runs/{pending['id']}/execute",
        headers={"X-Actor-ID": "tool-executor", "X-Tenant-ID": "finance"},
        json={"version": approved_run["version"]},
    )
    assert executed.status_code == 200
    completed = executed.json()
    assert completed["status"] == "COMPLETED"
    assert completed["result"]["output"] == {"value": "225"}
    assert completed["result"]["sandbox"]["network_disabled"] is True
    assert completed["result_hash"]
    assert completed["state_history"][-3:] == ["EXECUTING", "VERIFYING", "COMPLETED"]

    audit_payload = agent_client.get(
        "/api/v1/audit/events?limit=20", headers={"X-Tenant-ID": "finance"}
    ).text
    assert expression not in audit_payload
    assert "AGENT_ACTION_PROPOSED" in audit_payload
    assert "AGENT_ACTION_APPROVED" in audit_payload
    assert "AGENT_ACTION_COMPLETED" in audit_payload


def test_prohibited_action_is_rejected_audited_and_arguments_are_withheld(
    agent_client: TestClient,
) -> None:
    command = "curl https://outside.example/private"
    response = agent_client.post(
        "/api/v1/agent/runs",
        headers={"X-Actor-ID": "malicious-document", "X-Tenant-ID": "dept-a"},
        json={
            "tool_name": "network_request",
            "classification": "RESTRICTED",
            "arguments": {"command": command},
        },
    )

    assert response.status_code == 201
    rejected = response.json()
    assert rejected["status"] == "REJECTED"
    assert rejected["policy_decision"] == "REJECTED"
    assert rejected["approval"] is None

    with SessionLocal() as session:
        run = session.scalar(select(AgentRunRecord).where(AgentRunRecord.id == rejected["id"]))
        assert run is not None
        assert run.arguments_json == {}
        events = list(
            session.scalars(
                select(AuditEvent).where(AuditEvent.event_type == "AGENT_ACTION_REJECTED")
            )
        )
        serialized_events = json.dumps([event.details for event in events])
        assert command not in serialized_events
        assert rejected["arguments_hash"] in serialized_events


def test_agent_runs_are_tenant_scoped(agent_client: TestClient) -> None:
    created = agent_client.post(
        "/api/v1/agent/runs",
        headers={"X-Tenant-ID": "dept-a"},
        json={"tool_name": "calculator", "arguments": {"expression": "2 + 2"}},
    ).json()

    assert (
        len(agent_client.get("/api/v1/agent/runs", headers={"X-Tenant-ID": "dept-a"}).json()) == 1
    )
    assert (
        agent_client.get(
            f"/api/v1/agent/runs/{created['id']}", headers={"X-Tenant-ID": "dept-b"}
        ).status_code
        == 404
    )
    assert agent_client.get("/api/v1/agent/runs", headers={"X-Tenant-ID": "dept-b"}).json() == []


def test_document_report_requires_a_securely_indexed_document(agent_client: TestClient) -> None:
    document_id = str(uuid4())
    with SessionLocal() as session:
        session.add(
            DocumentRecord(
                id=document_id,
                filename="pending-report.pdf",
                media_type="application/pdf",
                size_bytes=128,
                sha256="b" * 64,
                vault_path="/encrypted/pending.ksh",
                status=DocumentState.INDEX_READY,
            )
        )
        session.commit()

    rejected = agent_client.post(
        "/api/v1/agent/runs",
        json={"tool_name": "document_report", "arguments": {"document_id": document_id}},
    )
    assert rejected.status_code == 201
    assert rejected.json()["status"] == "REJECTED"

    with SessionLocal() as session:
        document = session.get(DocumentRecord, document_id)
        assert document is not None
        document.status = DocumentState.INDEXED
        document.active_index_version = 1
        session.commit()

    pending = agent_client.post(
        "/api/v1/agent/runs",
        json={"tool_name": "document_report", "arguments": {"document_id": document_id}},
    )
    assert pending.status_code == 201
    assert pending.json()["status"] == "APPROVAL_PENDING"
