"""End-to-end integration test for the policy-gated local agent and deliverable workflow.

Exercises the full 7-stage governance workflow:
1. Select approved document
2. Request action (typed contract validation)
3. Policy decision (fail-closed, role and risk gating)
4. Approver review (separation of duties, replay prevention)
5. Execute (restricted deterministic tools vs unexecuted offline neural actions)
6. Cited deliverable (masked chunk citations, provenance, legal disclaimer)
7. Audit timeline (tamper-evident, metadata-only)
"""

from __future__ import annotations

import json
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from koshshield.database import engine
from koshshield.main import app
from koshshield.models import (
    AuditEvent,
    DocumentChunkRecord,
    DocumentRecord,
    DocumentState,
)

client = TestClient(app)


def test_full_agent_governance_workflow_end_to_end() -> None:
    """Tests the full end-to-end policy-gated agent workflow with both deterministic
    and offline model-backed actions.
    """
    tenant_id = f"dept-gov-{uuid4().hex[:8]}"
    requester_id = f"requester-{uuid4().hex[:8]}"
    approver_id = f"approver-{uuid4().hex[:8]}"
    executor_id = f"executor-{uuid4().hex[:8]}"

    # Headers for different roles
    requester_headers = {
        "X-Tenant-ID": tenant_id,
        "X-Actor-ID": requester_id,
        "X-Roles": "requester",
    }
    approver_headers = {
        "X-Tenant-ID": tenant_id,
        "X-Actor-ID": approver_id,
        "X-Roles": "approver",
    }
    executor_headers = {
        "X-Tenant-ID": tenant_id,
        "X-Actor-ID": executor_id,
        "X-Roles": "executor",
    }

    # 1. Setup approved indexed document with masked chunks
    doc_id = str(uuid4())
    chunk_id = str(uuid4())
    masked_snippet = (
        "Procurement of 50 workstations approved for [REDACTED_DEPT] at INR 50000 each."
    )

    with Session(bind=engine) as session:
        doc = DocumentRecord(
            id=doc_id,
            tenant_id=tenant_id,
            filename="procurement_sanction_2026.pdf",
            media_type="application/pdf",
            size_bytes=4096,
            sha256="a" * 64,
            vault_path="/vault/doc.enc",
            status=DocumentState.INDEXED,
            version=1,
            active_index_version=1,
        )
        chunk = DocumentChunkRecord(
            id=chunk_id,
            chunk_id=chunk_id,
            document_id=doc_id,
            page_number=1,
            chunk_sequence=0,
            masked_content_hash="b" * 64,
            char_start=0,
            char_end=len(masked_snippet),
            index_version=1,
        )
        session.add(doc)
        session.add(chunk)
        session.commit()

    # 2. Stage 1 & 2: Request Action (Contract validation)
    # Propose deterministic procurement comparison
    propose_resp = client.post(
        "/api/v1/agent/runs",
        headers=requester_headers,
        json={
            "tool_name": "calculate_procurement_comparison",
            "classification": "CONFIDENTIAL",
            "arguments": {
                "case_id": "case-proc-2026-workstations",
                "document_id": doc_id,
                "comparison_criteria": ["flow_rate", "pressure"],
            },
        },
    )
    assert propose_resp.status_code == 201, propose_resp.text
    run_data = propose_resp.json()
    run_id = run_data["id"]
    assert run_data["status"] == "APPROVAL_PENDING"
    assert run_data["approval_required"] is True
    assert run_data["policy_decision"] == "ALLOWED"

    # 3. Stage 3 & 4: Policy & Human Approval Lifecycle
    # Verify inspect approval detail endpoint
    approval_resp = client.get(
        f"/api/v1/agent/runs/{run_id}/approval",
        headers=approver_headers,
    )
    assert approval_resp.status_code == 200
    approval_detail = approval_resp.json()
    assert approval_detail["decision"] == "PENDING"
    assert approval_detail["document_id"] == doc_id
    assert approval_detail["is_expired"] is False

    # Separation of duties: Requester attempts to self-approve
    self_approve_resp = client.post(
        f"/api/v1/agent/runs/{run_id}/approval",
        headers=requester_headers,
        json={"decision": "APPROVED", "version": run_data["version"]},
    )
    assert self_approve_resp.status_code in {403, 409}, self_approve_resp.text

    # Independent approver reviews and approves
    approve_resp = client.post(
        f"/api/v1/agent/runs/{run_id}/approval",
        headers=approver_headers,
        json={"decision": "APPROVED", "version": run_data["version"]},
    )
    assert approve_resp.status_code == 200
    approved_run = approve_resp.json()
    assert approved_run["status"] == "APPROVED"
    assert approved_run["approval"]["decision"] == "APPROVED"

    # Replay protection: Attempt double-approval
    replay_resp = client.post(
        f"/api/v1/agent/runs/{run_id}/approval",
        headers=approver_headers,
        json={"decision": "APPROVED", "version": approved_run["version"]},
    )
    assert replay_resp.status_code == 409

    # 4. Stage 5: Execution in Restricted Tool Runner
    # Unauthorized role cannot execute
    unauthorized_exec = client.post(
        f"/api/v1/agent/runs/{run_id}/execute",
        headers={"X-Tenant-ID": tenant_id, "X-Actor-ID": "random", "X-Roles": "requester"},
        json={"version": approved_run["version"]},
    )
    assert unauthorized_exec.status_code == 403

    # Authorized executor runs the approved action
    exec_resp = client.post(
        f"/api/v1/agent/runs/{run_id}/execute",
        headers=executor_headers,
        json={"version": approved_run["version"]},
    )
    assert exec_resp.status_code == 200
    exec_data = exec_resp.json()
    assert exec_data["status"] == "COMPLETED"
    assert exec_data["result"] is not None
    assert exec_data["result"]["output"]["case_id"] == "case-proc-2026-workstations"
    assert exec_data["result"]["output"]["comparison_verdict"] == "COMPLIANT_SPECIFICATION"
    assert exec_data["result"]["sandbox"]["network_disabled"] is True

    # 5. Stage 6: Evidence-Bound Deliverable Inspection
    deliverable_resp = client.get(
        f"/api/v1/agent/runs/{run_id}/deliverable",
        headers=requester_headers,
    )
    assert deliverable_resp.status_code == 200
    deliverable = deliverable_resp.json()
    assert deliverable["run_id"] == run_id
    assert deliverable["provenance"]["verification_status"] == "DETERMINISTIC_VERIFIED"
    assert "DPDP" in deliverable["provenance"]["disclaimer"]
    # Verify citations bound to document chunk
    assert len(deliverable["citations"]) == 1
    assert deliverable["citations"][0]["chunk_id"] == chunk_id
    assert deliverable["citations"][0]["masked_snippet"].startswith("[MASKED_EVIDENCE_P1_C0")

    # 6. Truthful Offline Neural Handling
    # Propose model-backed action: summarize_document
    model_run_resp = client.post(
        "/api/v1/agent/runs",
        headers=requester_headers,
        json={
            "tool_name": "summarize_document",
            "classification": "CONFIDENTIAL",
            "arguments": {
                "document_id": doc_id,
                "focus": "procurement",
                "max_length": 250,
            },
        },
    )
    assert model_run_resp.status_code == 201, model_run_resp.text
    model_run = model_run_resp.json()
    model_run_id = model_run["id"]

    # Approve the summary request
    client.post(
        f"/api/v1/agent/runs/{model_run_id}/approval",
        headers=approver_headers,
        json={"decision": "APPROVED", "version": model_run["version"]},
    )

    # Re-fetch approved run to get current version
    current_model_run = client.get(
        f"/api/v1/agent/runs/{model_run_id}",
        headers=requester_headers,
    ).json()

    # Execute the summary request -> Truthfully fails closed with NOT_EXECUTED code
    exec_model_resp = client.post(
        f"/api/v1/agent/runs/{model_run_id}/execute",
        headers=executor_headers,
        json={"version": current_model_run["version"]},
    )
    assert exec_model_resp.status_code == 200
    failed_model_run = exec_model_resp.json()
    assert failed_model_run["status"] == "FAILED"
    assert failed_model_run["failure_code"] == "MODEL_RUNTIME_UNAVAILABLE"
    assert failed_model_run["result"]["output"]["status"] == "NOT_EXECUTED"
    assert failed_model_run["result"]["output"]["code"] == "MODEL_RUNTIME_UNAVAILABLE"

    # Deliverable inspection for unexecuted model action
    failed_deliv_resp = client.get(
        f"/api/v1/agent/runs/{model_run_id}/deliverable",
        headers=requester_headers,
    )
    assert failed_deliv_resp.status_code == 200
    failed_deliv = failed_deliv_resp.json()
    assert "NOT_EXECUTED" in failed_deliv["provenance"]["verification_status"]

    # 7. Stage 7: Metadata-Only Audit Trail Verification
    with Session(bind=engine) as session:
        audit_events = list(
            session.scalars(
                select(AuditEvent).where(
                    AuditEvent.tenant_id == tenant_id,
                    AuditEvent.resource_type == "agent_run",
                )
            )
        )

        assert len(audit_events) >= 6
        for ev in audit_events:
            ev_str = json.dumps(ev.details)
            # Ensure zero raw prompt text, no PII, no unredacted text, no internal paths
            assert "raw_prompt" not in ev_str
            assert "/Users/" not in ev_str
            assert "traceback" not in ev_str
            assert "Aadhaar" not in ev_str
            assert "PAN" not in ev_str
