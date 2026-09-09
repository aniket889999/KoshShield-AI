import uuid

import pymupdf
import pytest
from fastapi.testclient import TestClient

from koshshield.api.routes.agent import get_tool_runner
from koshshield.api.routes.retrieval import get_embedding_provider, get_vector_store
from koshshield.config import get_settings
from koshshield.database import SessionLocal
from koshshield.main import app
from koshshield.models import DocumentRecord, DocumentState
from koshshield.services.agent.tool_runner import ToolRunnerResult
from koshshield.services.retrieval.embeddings.deterministic_fake import (
    DeterministicEmbeddingProvider,
)
from koshshield.services.retrieval.vector_store import InMemoryVectorStore


class FakeRunner:
    def execute(self, *, tool_name: str, payload: dict[str, object]) -> ToolRunnerResult:
        return ToolRunnerResult(
            payload={"ok": True, "tool": tool_name, "result": {"content": "Report generated"}},
            output_hash="d" * 64,
            sandbox={
                "network_disabled": True,
                "read_only_root": True,
                "capabilities_dropped": True,
                "image": "img",
                "timeout_seconds": 5,
            },
        )


@pytest.fixture
def isolated_env(client: TestClient):
    fake_emb = DeterministicEmbeddingProvider()
    fake_store = InMemoryVectorStore()
    fake_runner = FakeRunner()
    app.dependency_overrides[get_embedding_provider] = lambda: fake_emb
    app.dependency_overrides[get_vector_store] = lambda: fake_store
    app.dependency_overrides[get_tool_runner] = lambda: fake_runner
    client.headers["X-Roles"] = "user,reviewer,approver,executor,auditor,admin"
    try:
        yield client, fake_store
    finally:
        client.headers.pop("X-Roles", None)
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)
        app.dependency_overrides.pop(get_tool_runner, None)


def create_synthetic_pdf(text: str = "CONFIDENTIAL TENDER NOTE") -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    content = f"GOVERNMENT OF INDIA\n{text}\nAadhaar: 3675 9834 1238\nPAN: ABCPR1234F"
    page.insert_text((50, 72), content)
    return doc.tobytes()


def test_upload_and_list_documents_tenant_isolation(isolated_env) -> None:
    client, _ = isolated_env

    # Tenant A uploads Doc A
    res_a = client.post(
        "/api/v1/documents",
        files={"file": ("doc_a.pdf", create_synthetic_pdf("Doc A content"), "application/pdf")},
        headers={"X-Tenant-ID": "tenant-a", "X-Actor-ID": "user-a"},
    )
    assert res_a.status_code == 201
    doc_a_id = res_a.json()["id"]
    assert res_a.json()["tenant_id"] == "tenant-a"

    # Tenant B uploads Doc B
    res_b = client.post(
        "/api/v1/documents",
        files={"file": ("doc_b.pdf", create_synthetic_pdf("Doc B content"), "application/pdf")},
        headers={"X-Tenant-ID": "tenant-b", "X-Actor-ID": "user-b"},
    )
    assert res_b.status_code == 201
    doc_b_id = res_b.json()["id"]
    assert res_b.json()["tenant_id"] == "tenant-b"

    # Tenant A list
    list_a = client.get("/api/v1/documents", headers={"X-Tenant-ID": "tenant-a"}).json()
    ids_a = [d["id"] for d in list_a]
    assert doc_a_id in ids_a
    assert doc_b_id not in ids_a

    # Tenant B list
    list_b = client.get("/api/v1/documents", headers={"X-Tenant-ID": "tenant-b"}).json()
    ids_b = [d["id"] for d in list_b]
    assert doc_b_id in ids_b
    assert doc_a_id not in ids_b


def test_extraction_cross_tenant_access_returns_404(isolated_env) -> None:
    client, _ = isolated_env

    res_a = client.post(
        "/api/v1/documents",
        files={"file": ("doc_a.pdf", create_synthetic_pdf("Extraction test"), "application/pdf")},
        headers={"X-Tenant-ID": "tenant-a", "X-Actor-ID": "user-a"},
    )
    doc_a_id = res_a.json()["id"]

    # Tenant B tries to trigger extraction for Tenant A's document
    ext_b = client.post(
        f"/api/v1/documents/{doc_a_id}/extraction",
        headers={"X-Tenant-ID": "tenant-b", "X-Actor-ID": "user-b"},
    )
    assert ext_b.status_code == 404

    # Tenant B tries to inspect extraction status for Tenant A's document
    status_b = client.get(
        f"/api/v1/documents/{doc_a_id}/extraction",
        headers={"X-Tenant-ID": "tenant-b"},
    )
    assert status_b.status_code == 404

    # Tenant A triggers extraction successfully
    ext_a = client.post(
        f"/api/v1/documents/{doc_a_id}/extraction",
        headers={"X-Tenant-ID": "tenant-a", "X-Actor-ID": "user-a"},
    )
    assert ext_a.status_code == 202
    assert ext_a.json()["status"] == "COMPLETED"


def test_review_and_redaction_cross_tenant_isolation(isolated_env) -> None:
    client, _ = isolated_env

    res_a = client.post(
        "/api/v1/documents",
        files={"file": ("doc_a.pdf", create_synthetic_pdf("Redaction test"), "application/pdf")},
        headers={"X-Tenant-ID": "tenant-a", "X-Actor-ID": "user-a"},
    )
    doc_a_id = res_a.json()["id"]

    client.post(
        f"/api/v1/documents/{doc_a_id}/extraction",
        headers={"X-Tenant-ID": "tenant-a", "X-Actor-ID": "user-a"},
    )

    # 1. Review queue isolation
    queue_b = client.get("/api/v1/review", headers={"X-Tenant-ID": "tenant-b"}).json()
    assert not any(item["document_id"] == doc_a_id for item in queue_b)

    queue_a = client.get("/api/v1/review", headers={"X-Tenant-ID": "tenant-a"}).json()
    assert any(item["document_id"] == doc_a_id for item in queue_a)

    # 2. Get redactions cross-tenant returns 404
    redactions_b = client.get(
        f"/api/v1/documents/{doc_a_id}/redactions",
        headers={"X-Tenant-ID": "tenant-b"},
    )
    assert redactions_b.status_code == 404

    redactions_a = client.get(
        f"/api/v1/documents/{doc_a_id}/redactions",
        headers={"X-Tenant-ID": "tenant-a"},
    )
    assert redactions_a.status_code == 200
    findings = redactions_a.json()["findings"]
    assert len(findings) > 0
    finding_id = findings[0]["id"]

    # 3. Update decision cross-tenant returns 404
    patch_b = client.patch(
        f"/api/v1/documents/{doc_a_id}/redactions/{finding_id}",
        headers={"X-Tenant-ID": "tenant-b", "X-Actor-ID": "user-b"},
        json={"decision": "ACCEPTED", "version": 1},
    )
    assert patch_b.status_code == 404

    # 4. Accept high-confidence cross-tenant returns 404
    bulk_b = client.post(
        f"/api/v1/documents/{doc_a_id}/redactions/accept-high-confidence",
        headers={"X-Tenant-ID": "tenant-b", "X-Actor-ID": "user-b"},
    )
    assert bulk_b.status_code == 404

    # 5. Approve redactions cross-tenant returns 404
    approve_b = client.post(
        f"/api/v1/documents/{doc_a_id}/redactions/approve",
        headers={"X-Tenant-ID": "tenant-b", "X-Actor-ID": "user-b"},
    )
    assert approve_b.status_code == 404


def test_indexing_cross_tenant_returns_404_and_payload_derived_from_doc_owner(
    isolated_env,
) -> None:
    client, fake_store = isolated_env

    res_a = client.post(
        "/api/v1/documents",
        files={
            "file": (
                "tender_a.pdf",
                create_synthetic_pdf("Naval procurement specifications"),
                "application/pdf",
            )
        },
        headers={"X-Tenant-ID": "tenant-a", "X-Actor-ID": "user-a"},
    )
    doc_a_id = res_a.json()["id"]

    client.post(
        f"/api/v1/documents/{doc_a_id}/extraction",
        headers={"X-Tenant-ID": "tenant-a", "X-Actor-ID": "user-a"},
    )
    client.post(
        f"/api/v1/documents/{doc_a_id}/redactions/accept-high-confidence",
        headers={"X-Tenant-ID": "tenant-a", "X-Actor-ID": "user-a"},
    )
    approve_res = client.post(
        f"/api/v1/documents/{doc_a_id}/redactions/approve",
        headers={"X-Tenant-ID": "tenant-a", "X-Actor-ID": "user-a"},
    )
    assert approve_res.status_code == 200

    # Tenant B tries to index Tenant A's document -> 404
    index_b = client.post(
        f"/api/v1/documents/{doc_a_id}/index",
        headers={"X-Tenant-ID": "tenant-b", "X-Actor-ID": "user-b"},
    )
    assert index_b.status_code == 404

    # Tenant B tries to view indexing status of Tenant A's document -> 404
    status_b = client.get(
        f"/api/v1/documents/{doc_a_id}/indexing",
        headers={"X-Tenant-ID": "tenant-b"},
    )
    assert status_b.status_code == 404

    # Tenant A indexes document
    index_a = client.post(
        f"/api/v1/documents/{doc_a_id}/index",
        headers={"X-Tenant-ID": "tenant-a", "X-Actor-ID": "user-a"},
    )
    assert index_a.status_code == 200
    assert index_a.json()["status"] == "INDEXED"
    assert index_a.json()["tenant_id"] == "tenant-a"

    # Verify vector store points are authoritatively tagged with tenant-a
    assert len(fake_store.chunks) > 0
    for chunk in fake_store.chunks:
        assert chunk.tenant_id == "tenant-a"

    # Tenant B retrieves points from store -> 0 points
    points_b = fake_store.retrieve_points(
        point_ids=[c.point_id for c in fake_store.chunks], tenant_id="tenant-b"
    )
    assert len(points_b) == 0


def test_retrieval_and_visual_evidence_cross_tenant_isolation(isolated_env) -> None:
    client, fake_store = isolated_env

    # Setup indexed document for tenant-a
    res_a = client.post(
        "/api/v1/documents",
        files={
            "file": (
                "classified_infra.pdf",
                create_synthetic_pdf("Nuclear power grid telemetry encrypted"),
                "application/pdf",
            )
        },
        headers={"X-Tenant-ID": "tenant-a", "X-Actor-ID": "user-a"},
    )
    doc_a_id = res_a.json()["id"]

    client.post(
        f"/api/v1/documents/{doc_a_id}/extraction",
        headers={"X-Tenant-ID": "tenant-a", "X-Actor-ID": "user-a"},
    )
    client.post(
        f"/api/v1/documents/{doc_a_id}/redactions/accept-high-confidence",
        headers={"X-Tenant-ID": "tenant-a", "X-Actor-ID": "user-a"},
    )
    client.post(
        f"/api/v1/documents/{doc_a_id}/redactions/approve",
        headers={"X-Tenant-ID": "tenant-a", "X-Actor-ID": "user-a"},
    )
    client.post(
        f"/api/v1/documents/{doc_a_id}/index",
        headers={"X-Tenant-ID": "tenant-a", "X-Actor-ID": "user-a"},
    )

    # Tenant A searches -> receives hits
    search_a = client.post(
        "/api/v1/retrieval/search",
        headers={"X-Tenant-ID": "tenant-a", "X-Actor-ID": "user-a"},
        json={"query": "power grid telemetry"},
    )
    assert search_a.status_code == 200
    results_a = search_a.json()["results"]
    assert len(results_a) > 0
    chunk_id = results_a[0]["chunk_id"]

    # Tenant B searches for the exact same query -> receives 0 results
    search_b = client.post(
        "/api/v1/retrieval/search",
        headers={"X-Tenant-ID": "tenant-b", "X-Actor-ID": "user-b"},
        json={"query": "power grid telemetry"},
    )
    assert search_b.status_code == 200
    assert search_b.json()["total_found"] == 0
    assert len(search_b.json()["results"]) == 0

    # Tenant B tries to retrieve visual evidence page image for Tenant A's chunk -> 404
    img_b = client.get(
        f"/api/v1/retrieval/evidence/{chunk_id}/page-image",
        headers={"X-Tenant-ID": "tenant-b"},
    )
    assert img_b.status_code == 404

    # Tenant A retrieves visual evidence page image -> 200
    img_a = client.get(
        f"/api/v1/retrieval/evidence/{chunk_id}/page-image",
        headers={"X-Tenant-ID": "tenant-a"},
    )
    assert img_a.status_code == 200
    assert img_a.headers["content-type"] == "image/png"


def test_audit_events_and_integrity_tenant_isolation(isolated_env) -> None:
    client, _ = isolated_env

    # Tenant A creates an event
    client.post(
        "/api/v1/documents",
        files={"file": ("doc_a.pdf", create_synthetic_pdf("Audit doc A"), "application/pdf")},
        headers={"X-Tenant-ID": "tenant-a", "X-Actor-ID": "officer-a"},
    )

    # Tenant B creates an event
    client.post(
        "/api/v1/documents",
        files={"file": ("doc_b.pdf", create_synthetic_pdf("Audit doc B"), "application/pdf")},
        headers={"X-Tenant-ID": "tenant-b", "X-Actor-ID": "officer-b"},
    )

    # Tenant A audit list
    events_a = client.get("/api/v1/audit/events", headers={"X-Tenant-ID": "tenant-a"}).json()
    assert len(events_a) >= 1
    for ev in events_a:
        assert ev["tenant_id"] == "tenant-a"
        assert ev["actor_id"] == "officer-a"

    # Tenant B audit list
    events_b = client.get("/api/v1/audit/events", headers={"X-Tenant-ID": "tenant-b"}).json()
    assert len(events_b) >= 1
    for ev in events_b:
        assert ev["tenant_id"] == "tenant-b"
        assert ev["actor_id"] == "officer-b"

    # Integrity verification
    integ_a = client.get("/api/v1/audit/integrity", headers={"X-Tenant-ID": "tenant-a"}).json()
    assert integ_a["valid"] is True
    assert integ_a["event_count"] == len(events_a)

    integ_b = client.get("/api/v1/audit/integrity", headers={"X-Tenant-ID": "tenant-b"}).json()
    assert integ_b["valid"] is True
    assert integ_b["event_count"] == len(events_b)


def test_agent_document_report_verifies_run_tenant_equals_document_tenant(isolated_env) -> None:
    client, _ = isolated_env

    # Setup indexed document for tenant-a in DB
    doc_id = str(uuid.uuid4())
    with SessionLocal() as session:
        doc = DocumentRecord(
            id=doc_id,
            tenant_id="tenant-a",
            filename="secure-indexed-doc.pdf",
            media_type="application/pdf",
            size_bytes=1024,
            sha256="c" * 64,
            vault_path="/vault/doc.ksh",
            status=DocumentState.INDEXED,
            active_index_version=1,
        )
        session.add(doc)
        session.commit()

    # Tenant B proposes document_report on Tenant A's document -> REJECTED by policy
    res_b = client.post(
        "/api/v1/agent/runs",
        headers={"X-Tenant-ID": "tenant-b", "X-Actor-ID": "user-b"},
        json={
            "tool_name": "document_report",
            "arguments": {"document_id": doc_id},
        },
    )
    assert res_b.status_code == 201
    assert res_b.json()["status"] == "REJECTED"
    assert res_b.json()["policy_decision"] == "REJECTED"

    # Tenant A proposes document_report on Tenant A's document -> APPROVAL_PENDING
    res_a = client.post(
        "/api/v1/agent/runs",
        headers={"X-Tenant-ID": "tenant-a", "X-Actor-ID": "user-a"},
        json={
            "tool_name": "document_report",
            "arguments": {"document_id": doc_id},
        },
    )
    assert res_a.status_code == 201
    assert res_a.json()["status"] == "APPROVAL_PENDING"
    assert res_a.json()["policy_decision"] == "ALLOWED"


def test_fail_closed_in_production_when_unverified(isolated_env) -> None:
    client, _ = isolated_env
    settings = get_settings()
    original_demo_mode = settings.demo_mode
    original_env = settings.environment

    try:
        # Simulate production mode
        settings.demo_mode = False
        settings.environment = "production"

        res = client.get("/api/v1/documents", headers={"X-Tenant-ID": "tenant-a"})
        assert res.status_code == 401
        assert "Verified authentication required" in res.json()["detail"]

        res_upload = client.post("/api/v1/documents")
        assert res_upload.status_code == 401
    finally:
        settings.demo_mode = original_demo_mode
        settings.environment = original_env


def test_cross_tenant_retrieval_telemetry_isolation(isolated_env) -> None:
    client, fake_store = isolated_env

    # Tenant A uploads, extracts, approves, and indexes a document
    res_a = client.post(
        "/api/v1/documents",
        files={
            "file": (
                "telemetry_doc.pdf",
                create_synthetic_pdf("Telemetry isolation confidential content"),
                "application/pdf",
            )
        },
        headers={"X-Tenant-ID": "tenant-a", "X-Actor-ID": "officer-a"},
    )
    doc_a_id = res_a.json()["id"]

    client.post(
        f"/api/v1/documents/{doc_a_id}/extraction",
        headers={"X-Tenant-ID": "tenant-a", "X-Actor-ID": "officer-a"},
    )
    client.post(
        f"/api/v1/documents/{doc_a_id}/redactions/accept-high-confidence",
        headers={"X-Tenant-ID": "tenant-a", "X-Actor-ID": "officer-a"},
    )
    client.post(
        f"/api/v1/documents/{doc_a_id}/redactions/approve",
        headers={"X-Tenant-ID": "tenant-a", "X-Actor-ID": "officer-a"},
    )
    client.post(
        f"/api/v1/documents/{doc_a_id}/index",
        headers={"X-Tenant-ID": "tenant-a", "X-Actor-ID": "officer-a"},
    )

    # Tenant A status: 1 document and indexed vector points
    status_a = client.get("/api/v1/retrieval/status", headers={"X-Tenant-ID": "tenant-a"}).json()
    assert status_a["indexed_documents_count"] == 1
    assert status_a["total_chunks"] >= 1

    # Tenant B status: 0 documents and 0 vector points
    status_b = client.get("/api/v1/retrieval/status", headers={"X-Tenant-ID": "tenant-b"}).json()
    assert status_b["indexed_documents_count"] == 0
    assert status_b["total_chunks"] == 0

    # Default tenant status: 0 documents and 0 vector points
    status_default = client.get("/api/v1/retrieval/status").json()
    assert status_default["indexed_documents_count"] == 0
    assert status_default["total_chunks"] == 0


def test_rbac_unauthorized_returns_403_and_authorized_succeeds(isolated_env) -> None:
    client, _ = isolated_env

    # 1. Create a document for testing
    res = client.post(
        "/api/v1/documents",
        files={"file": ("rbac_doc.pdf", create_synthetic_pdf("RBAC test"), "application/pdf")},
        headers={"X-Tenant-ID": "tenant-rbac", "X-Actor-ID": "user-1", "X-Roles": "user"},
    )
    assert res.status_code == 201
    doc_id = res.json()["id"]

    client.post(
        f"/api/v1/documents/{doc_id}/extraction",
        headers={"X-Tenant-ID": "tenant-rbac", "X-Actor-ID": "user-1", "X-Roles": "user"},
    )

    # 2. Reviewer endpoint: GET /documents/{id}/redactions
    res_unauth = client.get(
        f"/api/v1/documents/{doc_id}/redactions",
        headers={"X-Tenant-ID": "tenant-rbac", "X-Roles": "user"},
    )
    assert res_unauth.status_code == 403
    assert "requires one of the following roles" in res_unauth.json()["detail"]

    res_auth = client.get(
        f"/api/v1/documents/{doc_id}/redactions",
        headers={"X-Tenant-ID": "tenant-rbac", "X-Roles": "reviewer"},
    )
    assert res_auth.status_code == 200

    # 3. Approver endpoint: POST /documents/{id}/redactions/approve
    res_approve_unauth = client.post(
        f"/api/v1/documents/{doc_id}/redactions/approve",
        headers={"X-Tenant-ID": "tenant-rbac", "X-Roles": "user"},
    )
    assert res_approve_unauth.status_code == 403

    # 4. Admin endpoint: POST /documents/{id}/index
    res_index_unauth = client.post(
        f"/api/v1/documents/{doc_id}/index",
        headers={"X-Tenant-ID": "tenant-rbac", "X-Roles": "reviewer"},
    )
    assert res_index_unauth.status_code == 403

    # 5. Auditor endpoint: GET /audit/events and GET /audit/integrity
    audit_unauth = client.get(
        "/api/v1/audit/events",
        headers={"X-Tenant-ID": "tenant-rbac", "X-Roles": "user"},
    )
    assert audit_unauth.status_code == 403

    audit_auth = client.get(
        "/api/v1/audit/events",
        headers={"X-Tenant-ID": "tenant-rbac", "X-Roles": "auditor"},
    )
    assert audit_auth.status_code == 200

    integ_unauth = client.get(
        "/api/v1/audit/integrity",
        headers={"X-Tenant-ID": "tenant-rbac", "X-Roles": "user"},
    )
    assert integ_unauth.status_code == 403

    integ_auth = client.get(
        "/api/v1/audit/integrity",
        headers={"X-Tenant-ID": "tenant-rbac", "X-Roles": "auditor"},
    )
    assert integ_auth.status_code == 200

    # 6. Executor endpoint: POST /agent/runs
    agent_unauth = client.post(
        "/api/v1/agent/runs",
        headers={"X-Tenant-ID": "tenant-rbac", "X-Actor-ID": "u1", "X-Roles": "user"},
        json={"tool_name": "calculator", "arguments": {"expression": "1 + 1"}},
    )
    assert agent_unauth.status_code == 403

    agent_auth = client.post(
        "/api/v1/agent/runs",
        headers={"X-Tenant-ID": "tenant-rbac", "X-Actor-ID": "u1", "X-Roles": "executor"},
        json={"tool_name": "calculator", "arguments": {"expression": "1 + 1"}},
    )
    assert agent_auth.status_code == 201
    run_id = agent_auth.json()["id"]

    # 7. Approver endpoint: POST /agent/runs/{id}/approval
    agent_appr_unauth = client.post(
        f"/api/v1/agent/runs/{run_id}/approval",
        headers={"X-Tenant-ID": "tenant-rbac", "X-Actor-ID": "u2", "X-Roles": "executor"},
        json={"decision": "APPROVED", "version": 1},
    )
    assert agent_appr_unauth.status_code == 403

    agent_appr_auth = client.post(
        f"/api/v1/agent/runs/{run_id}/approval",
        headers={"X-Tenant-ID": "tenant-rbac", "X-Actor-ID": "u2", "X-Roles": "approver"},
        json={"decision": "APPROVED", "version": 1},
    )
    assert agent_appr_auth.status_code == 200
