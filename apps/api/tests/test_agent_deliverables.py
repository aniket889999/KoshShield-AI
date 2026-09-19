from uuid import uuid4

from fastapi.testclient import TestClient

from koshshield.database import SessionLocal
from koshshield.models import DocumentChunkRecord, DocumentRecord, DocumentState


def test_deliverable_generated_with_provenance_and_citations(client: TestClient) -> None:
    doc_id = str(uuid4())
    chunk_id = "chk-" + str(uuid4())[:12]

    with SessionLocal() as session:
        session.add(
            DocumentRecord(
                id=doc_id,
                tenant_id="dept-gov",
                filename="procurement_sanction.pdf",
                media_type="application/pdf",
                size_bytes=2048,
                sha256="d" * 64,
                vault_path="/vault/procurement.ksh",
                status=DocumentState.INDEXED,
                version=1,
                active_index_version=1,
            )
        )
        session.add(
            DocumentChunkRecord(
                id=str(uuid4()),
                document_id=doc_id,
                page_number=1,
                chunk_sequence=0,
                index_version=1,
                chunk_id=chunk_id,
                char_start=0,
                char_end=150,
                masked_content_hash="e" * 64,
            )
        )
        session.commit()

    # 1. Propose draft_approval_note
    prop_res = client.post(
        "/api/v1/agent/runs",
        headers={"X-Tenant-ID": "dept-gov", "X-Roles": "requester"},
        json={
            "tool_name": "draft_approval_note",
            "classification": "CONFIDENTIAL",
            "arguments": {
                "document_id": doc_id,
                "note_subject": "Sanction of procurement order",
                "recommended_action": "Approve compliant bid",
            },
        },
    )
    assert prop_res.status_code == 201
    run_id = prop_res.json()["id"]
    version = prop_res.json()["version"]

    # 2. Approve
    app_res = client.post(
        f"/api/v1/agent/runs/{run_id}/approval",
        headers={"X-Actor-ID": "supervisor-1", "X-Tenant-ID": "dept-gov", "X-Roles": "approver"},
        json={"decision": "APPROVED", "version": version},
    )
    assert app_res.status_code == 200
    app_version = app_res.json()["version"]

    # 3. Execute
    exec_res = client.post(
        f"/api/v1/agent/runs/{run_id}/execute",
        headers={"X-Actor-ID": "exec-1", "X-Tenant-ID": "dept-gov", "X-Roles": "executor"},
        json={"version": app_version},
    )
    assert exec_res.status_code == 200

    # 4. Fetch deliverable via GET /runs/{run_id}/deliverable
    get_res = client.get(
        f"/api/v1/agent/runs/{run_id}/deliverable",
        headers={"X-Tenant-ID": "dept-gov"},
    )
    assert get_res.status_code == 200
    deliverable = get_res.json()

    assert deliverable["run_id"] == run_id
    assert deliverable["tenant_id"] == "dept-gov"
    assert deliverable["action_name"] == "draft_approval_note"
    assert "Sanction of procurement order" in str(deliverable["content"])
    assert deliverable["media_type"] == "text/markdown"

    # Verify citations bound to masked chunks
    citations = deliverable["citations"]
    assert len(citations) >= 1
    assert citations[0]["chunk_id"] == chunk_id
    assert citations[0]["page_number"] == 1
    assert citations[0]["snippet_hash"] == "e" * 64

    # Verify provenance
    prov = deliverable["provenance"]
    assert prov["action_type"] == "draft_approval_note"
    assert prov["policy_result"] == "ALLOWED"
    assert prov["document_id"] == doc_id
    assert prov["document_version"] == 1
    assert prov["active_index_version"] == 1
    assert prov["verification_status"] == "DETERMINISTIC_VERIFIED"
    assert "statutory DPDP compliance" in prov["disclaimer"]
    assert "legal validity" in prov["disclaimer"]

    # 5. Cross-tenant access fails with 404
    cross_res = client.get(
        f"/api/v1/agent/runs/{run_id}/deliverable",
        headers={"X-Tenant-ID": "foreign-dept"},
    )
    assert cross_res.status_code == 404
