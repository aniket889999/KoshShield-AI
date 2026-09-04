from fastapi.testclient import TestClient


def test_liveness(client: TestClient) -> None:
    response = client.get("/api/v1/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_system_status_never_enables_external_ai(client: TestClient) -> None:
    response = client.get("/api/v1/system/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["processing_boundary"] == "local-only"
    assert payload["external_ai_enabled"] is False
    assert payload["metadata_backend"] == "SQLite development"
    assert payload["vault"]["status"] == "ready"
