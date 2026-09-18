from fastapi.testclient import TestClient

from koshshield.config import Settings, get_settings
from koshshield.main import app


def test_readiness_endpoint_success():
    client = TestClient(app)
    headers = {
        "X-Tenant-ID": "tenant-delta",
        "X-Actor-ID": "operator-bob",
        "X-Roles": "user",
    }
    response = client.get("/api/v1/system/readiness", headers=headers)
    assert response.status_code == 200
    data = response.json()

    assert data["report_type"] == "demo_readiness"
    assert data["tenant_id"] == "tenant-delta"
    assert "overall_status" in data
    assert "can_run_offline_demo" in data
    assert "can_run_live_inference" in data
    assert len(data["components"]) == 7

    # Ensure no absolute paths or secret keys leak into any field
    raw_text = response.text
    assert "/Users/" not in raw_text
    assert "/home/" not in raw_text
    assert "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=" not in raw_text
    assert "DPDP-aligned" in data["disclaimer"]


def test_readiness_probe_requires_privileged_role():
    client = TestClient(app)
    user_headers = {
        "X-Tenant-ID": "tenant-delta",
        "X-Actor-ID": "standard-user",
        "X-Roles": "user",
    }
    resp = client.get("/api/v1/system/readiness?probe_services=true", headers=user_headers)
    assert resp.status_code == 403
    assert "Forbidden" in resp.json()["detail"]

    admin_headers = {
        "X-Tenant-ID": "tenant-delta",
        "X-Actor-ID": "admin-alice",
        "X-Roles": "admin",
    }
    resp_admin = client.get("/api/v1/system/readiness?probe_services=true", headers=admin_headers)
    assert resp_admin.status_code == 200


def test_readiness_fails_closed_in_production():
    try:
        app.dependency_overrides[get_settings] = lambda: Settings(
            demo_mode=False,
            environment="production",
        )
        prod_client = TestClient(app)

        headers = {
            "X-Tenant-ID": "tenant-delta",
            "X-Actor-ID": "operator-bob",
            "X-Roles": "admin",
        }
        resp = prod_client.get("/api/v1/system/readiness", headers=headers)
        assert resp.status_code == 401
        assert "Verified authentication required" in resp.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_settings, None)
