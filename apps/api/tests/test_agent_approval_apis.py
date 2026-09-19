import hashlib
import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from koshshield.api.routes.agent import get_tool_runner
from koshshield.database import SessionLocal
from koshshield.main import app
from koshshield.models import DocumentRecord, DocumentState
from koshshield.services.agent.tool_runner import ToolRunnerResult


class DummyRunner:
    def execute(self, *, tool_name: str, payload: dict[str, object]) -> ToolRunnerResult:
        res = {"value": "mock-verified-calc"}
        canonical = json.dumps(res, sort_keys=True, separators=(",", ":"))
        return ToolRunnerResult(
            payload={"result": res},
            output_hash=hashlib.sha256(canonical.encode()).hexdigest(),
            sandbox={"network_disabled": True},
        )


@pytest.fixture
def rbac_client(client: TestClient) -> TestClient:
    app.dependency_overrides[get_tool_runner] = lambda: DummyRunner()
    try:
        yield client
    finally:
        app.dependency_overrides.pop(get_tool_runner, None)


def test_rbac_propose_inspect_approve_and_execute_lifecycle(rbac_client: TestClient) -> None:
    # 1. Proposer creates run with 'requester' role
    req_headers = {
        "X-Actor-ID": "officer-requester",
        "X-Tenant-ID": "dept-alpha",
        "X-Roles": "requester",
    }
    create_res = rbac_client.post(
        "/api/v1/agent/runs",
        headers=req_headers,
        json={
            "tool_name": "calculator",
            "classification": "CONFIDENTIAL",
            "arguments": {"expression": "500 + 25"},
        },
    )
    assert create_res.status_code == 201
    run_data = create_res.json()
    run_id = run_data["id"]
    version = run_data["version"]
    assert run_data["status"] == "APPROVAL_PENDING"

    # 2. Inspect approval request detail via GET /runs/{run_id}/approval
    inspect_res = rbac_client.get(
        f"/api/v1/agent/runs/{run_id}/approval",
        headers={"X-Tenant-ID": "dept-alpha", "X-Roles": "auditor"},
    )
    assert inspect_res.status_code == 200
    approval_detail = inspect_res.json()
    assert approval_detail["decision"] == "PENDING"
    assert approval_detail["tenant_id"] == "dept-alpha"
    assert approval_detail["arguments_hash"] == run_data["arguments_hash"]
    assert approval_detail["is_expired"] is False

    # 3. Cross-tenant access is rejected with 404
    cross_res = rbac_client.get(
        f"/api/v1/agent/runs/{run_id}/approval",
        headers={"X-Tenant-ID": "dept-beta", "X-Roles": "auditor"},
    )
    assert cross_res.status_code == 404

    # 4. Self-approval is rejected with 403
    self_approve = rbac_client.post(
        f"/api/v1/agent/runs/{run_id}/approval",
        headers={
            "X-Actor-ID": "officer-requester",
            "X-Tenant-ID": "dept-alpha",
            "X-Roles": "approver",
        },
        json={"decision": "APPROVED", "version": version},
    )
    assert self_approve.status_code == 403

    # 5. Non-approver role is rejected with 403
    auditor_approve = rbac_client.post(
        f"/api/v1/agent/runs/{run_id}/approval",
        headers={
            "X-Actor-ID": "auditor-1",
            "X-Tenant-ID": "dept-alpha",
            "X-Roles": "auditor",
        },
        json={"decision": "APPROVED", "version": version},
    )
    assert auditor_approve.status_code == 403

    # 6. Valid supervisor approves
    sup_headers = {
        "X-Actor-ID": "supervisor-1",
        "X-Tenant-ID": "dept-alpha",
        "X-Roles": "approver",
    }
    approve_res = rbac_client.post(
        f"/api/v1/agent/runs/{run_id}/approval",
        headers=sup_headers,
        json={"decision": "APPROVED", "version": version},
    )
    assert approve_res.status_code == 200
    approved_run = approve_res.json()
    assert approved_run["status"] == "APPROVED"
    assert approved_run["approval"]["decision"] == "APPROVED"
    new_version = approved_run["version"]

    # 7. Approval replay / re-approval rejected with 409 Conflict
    replay_approve = rbac_client.post(
        f"/api/v1/agent/runs/{run_id}/approval",
        headers=sup_headers,
        json={"decision": "APPROVED", "version": new_version},
    )
    assert replay_approve.status_code == 409

    # 8. Execution without executor role rejected with 403
    non_exec = rbac_client.post(
        f"/api/v1/agent/runs/{run_id}/execute",
        headers={
            "X-Actor-ID": "supervisor-1",
            "X-Tenant-ID": "dept-alpha",
            "X-Roles": "approver",
        },
        json={"version": new_version},
    )
    assert non_exec.status_code == 403

    # 9. Valid executor executes
    exec_headers = {
        "X-Actor-ID": "exec-1",
        "X-Tenant-ID": "dept-alpha",
        "X-Roles": "executor",
    }
    exec_res = rbac_client.post(
        f"/api/v1/agent/runs/{run_id}/execute",
        headers=exec_headers,
        json={"version": new_version},
    )
    assert exec_res.status_code == 200
    completed = exec_res.json()
    assert completed["status"] == "COMPLETED"
    assert completed["approval"]["decision"] == "EXECUTED"

    # 10. Double execution rejected with 409
    double_exec = rbac_client.post(
        f"/api/v1/agent/runs/{run_id}/execute",
        headers=exec_headers,
        json={"version": completed["version"]},
    )
    assert double_exec.status_code == 409


def test_stale_document_version_execution_rejected(rbac_client: TestClient) -> None:
    doc_id = str(uuid4())
    with SessionLocal() as session:
        session.add(
            DocumentRecord(
                id=doc_id,
                tenant_id="tenant-stale",
                filename="spec.pdf",
                media_type="application/pdf",
                size_bytes=100,
                sha256="c" * 64,
                vault_path="/vault/spec.ksh",
                status=DocumentState.INDEXED,
                version=1,
                active_index_version=1,
            )
        )
        session.commit()

    # Propose action bound to version 1
    prop = rbac_client.post(
        "/api/v1/agent/runs",
        headers={"X-Tenant-ID": "tenant-stale", "X-Roles": "requester"},
        json={
            "tool_name": "document_report",
            "classification": "CONFIDENTIAL",
            "arguments": {"document_id": doc_id},
        },
    ).json()

    # Approve
    approved = rbac_client.post(
        f"/api/v1/agent/runs/{prop['id']}/approval",
        headers={
            "X-Actor-ID": "approver-2",
            "X-Tenant-ID": "tenant-stale",
            "X-Roles": "approver",
        },
        json={"decision": "APPROVED", "version": prop["version"]},
    ).json()

    # Now document version changes (e.g. re-indexed / edited)
    with SessionLocal() as session:
        doc = session.get(DocumentRecord, doc_id)
        assert doc is not None
        doc.version = 2
        session.commit()

    # Execution should fail with 409 or conflict
    exec_res = rbac_client.post(
        f"/api/v1/agent/runs/{prop['id']}/execute",
        headers={"X-Tenant-ID": "tenant-stale", "X-Roles": "executor"},
        json={"version": approved["version"]},
    )
    assert exec_res.status_code in {400, 409}
