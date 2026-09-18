import io

import pytest
from fastapi.testclient import TestClient

from koshshield.config import Settings, get_settings
from koshshield.main import app


@pytest.fixture
def api_client():
    return TestClient(app)


def test_production_mode_strictly_rejects_header_auth(api_client):
    """In production mode, header-derived identities must fail closed with 401."""
    try:
        app.dependency_overrides[get_settings] = lambda: Settings(
            demo_mode=False,
            environment="production",
        )
        headers = {
            "X-Tenant-ID": "prod-tenant",
            "X-Actor-ID": "attacker",
            "X-Roles": "admin",
        }

        # 1. Readiness endpoint
        resp1 = api_client.get("/api/v1/system/readiness", headers=headers)
        assert resp1.status_code == 401
        assert "Verified authentication required" in resp1.json()["detail"]

        # 2. Documents timeline endpoint
        resp2 = api_client.get("/api/v1/documents/any-id/timeline", headers=headers)
        assert resp2.status_code == 401

        # 3. Documents evidence endpoint
        resp3 = api_client.get("/api/v1/documents/any-id/evidence", headers=headers)
        assert resp3.status_code == 401
    finally:
        app.dependency_overrides.pop(get_settings, None)


def test_readiness_probe_rbac_enforcement(api_client):
    """Privileged roles (admin, auditor, operator, reviewer) may trigger live service probes."""
    standard_headers = {
        "X-Tenant-ID": "tenant-test",
        "X-Actor-ID": "user-1",
        "X-Roles": "user",
    }
    # Unprivileged user trying to trigger probe
    resp = api_client.get("/api/v1/system/readiness?probe_services=true", headers=standard_headers)
    assert resp.status_code == 403
    assert "Forbidden" in resp.json()["detail"]

    # Read-only readiness without probe succeeds for standard user
    resp_ok = api_client.get(
        "/api/v1/system/readiness?probe_services=false", headers=standard_headers
    )
    assert resp_ok.status_code == 200
    assert resp_ok.json()["tenant_id"] == "tenant-test"


def test_sanitization_guarantees_no_internal_paths_or_secrets(api_client):
    """Asserts that readiness and timeline payloads never leak system paths or secrets."""
    headers = {
        "X-Tenant-ID": "tenant-audit",
        "X-Actor-ID": "auditor-1",
        "X-Roles": "admin",
    }

    # Upload document
    pdf_data = b"%PDF-1.4 private test document for sanitization validation"
    upload_resp = api_client.post(
        "/api/v1/documents",
        files={"file": ("confidential_spec.pdf", io.BytesIO(pdf_data), "application/pdf")},
        headers=headers,
    )
    assert upload_resp.status_code == 201
    doc_id = upload_resp.json()["id"]

    # 1. Inspect timeline response text
    timeline_resp = api_client.get(f"/api/v1/documents/{doc_id}/timeline", headers=headers)
    assert timeline_resp.status_code == 200
    timeline_text = timeline_resp.text

    assert "/Users/" not in timeline_text
    assert "/home/" not in timeline_text
    assert "/var/" not in timeline_text
    assert "/tmp/" not in timeline_text
    assert "vault_path" not in timeline_text
    assert "master_key" not in timeline_text

    # 2. Inspect readiness response text
    readiness_resp = api_client.get(
        "/api/v1/system/readiness?probe_services=false", headers=headers
    )
    assert readiness_resp.status_code == 200
    readiness_text = readiness_resp.text

    assert "/Users/" not in readiness_text
    assert "/home/" not in readiness_text
    assert "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=" not in readiness_text


def test_cross_tenant_isolation_returns_404_not_found(api_client):
    """Cross-tenant requests must return 404 to avoid revealing resource existence."""
    tenant_red_headers = {"X-Tenant-ID": "tenant-red", "X-Actor-ID": "user-red"}
    tenant_blue_headers = {"X-Tenant-ID": "tenant-blue", "X-Actor-ID": "user-blue"}

    # Upload document in tenant-red
    pdf_bytes = b"%PDF-1.4 red tenant confidential document"
    res = api_client.post(
        "/api/v1/documents",
        files={"file": ("red_doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        headers=tenant_red_headers,
    )
    doc_id = res.json()["id"]

    # Tenant blue queries timeline -> 404
    blue_timeline = api_client.get(
        f"/api/v1/documents/{doc_id}/timeline", headers=tenant_blue_headers
    )
    assert blue_timeline.status_code == 404

    # Tenant blue queries evidence -> 404
    blue_evidence = api_client.get(
        f"/api/v1/documents/{doc_id}/evidence", headers=tenant_blue_headers
    )
    assert blue_evidence.status_code == 404


def test_unavailable_runtime_fail_closed_behavior(api_client):
    """When models or services are unavailable, endpoints fail closed without simulated AI."""
    headers = {
        "X-Tenant-ID": "tenant-offline",
        "X-Actor-ID": "operator",
        "X-Roles": "admin",
    }

    resp = api_client.get("/api/v1/system/readiness?probe_services=true", headers=headers)
    readiness = resp.json()

    # The report truthfully states demo restriction
    assert readiness["overall_status"] in {"DEMO_RESTRICTED", "READY"}
    assert readiness["can_run_offline_demo"] is True
    assert "disclaimer" in readiness
    assert "DPDP-aligned" in readiness["disclaimer"]

    # Components report truthful status
    comps = {c["component_id"]: c for c in readiness["components"]}
    assert comps["relational_db"]["status"] == "READY"
    assert comps["encrypted_vault"]["status"] == "READY"

    # Offline components must not be marked executable
    for cid in ["bge_m3", "qwen_gguf", "llama_cpp", "qdrant"]:
        if comps[cid]["status"] != "READY":
            assert comps[cid]["executable"] is False
            assert comps[cid]["failure_code"] is not None
