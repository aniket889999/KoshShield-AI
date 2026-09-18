import pytest
from fastapi.testclient import TestClient

from koshshield.main import app
from koshshield.services.local_health import is_safe_local_endpoint, probe_local_http


def test_is_safe_local_endpoint():
    assert is_safe_local_endpoint("http://localhost:8000/models") is True
    assert is_safe_local_endpoint("http://127.0.0.1:6333") is True
    assert is_safe_local_endpoint("http://qdrant:6333") is True
    assert is_safe_local_endpoint("http://llama-server:8080/v1") is True

    # External, cloud, or remote addresses must be strictly rejected
    assert is_safe_local_endpoint("https://api.openai.com/v1") is False
    assert is_safe_local_endpoint("http://8.8.8.8") is False
    assert is_safe_local_endpoint("https://huggingface.co/models") is False
    assert is_safe_local_endpoint("http://evil.corp/leak") is False
    assert is_safe_local_endpoint("not-a-url") is False


@pytest.mark.anyio
async def test_probe_non_local_endpoint_rejected_without_network_call():
    probe = await probe_local_http(
        service_id="external_leak",
        display_name="External Service",
        url="https://api.openai.com/v1/models",
    )
    assert probe.status == "unavailable"
    assert probe.failure_code == "NON_LOCAL_DESTINATION_REJECTED"
    assert "outside the local-only boundary" in probe.details


@pytest.mark.anyio
async def test_probe_offline_local_service_fails_closed():
    probe = await probe_local_http(
        service_id="offline_llama",
        display_name="Local LLM",
        url="http://127.0.0.1:54329/models",
        timeout_seconds=0.1,
    )
    assert probe.status == "unavailable"
    assert probe.failure_code == "SERVICE_OFFLINE"
    assert probe.local_only is True


def test_local_health_endpoint_response():
    client = TestClient(app)
    resp = client.get("/api/v1/health/local")
    assert resp.status_code == 200
    data = resp.json()

    assert data["boundary"] == "local-only"
    assert data["network_isolated"] is True
    assert data["external_calls_blocked"] is True
    assert "safe_mode_active" in data
    assert "services" in data

    services = data["services"]
    assert "database" in services
    assert "vault" in services
    assert "qdrant" in services
    assert "llama_cpp" in services

    assert services["database"]["status"] == "ready"
    assert services["vault"]["status"] == "ready"
    # In test environment, qdrant and llama_cpp are offline and fail closed
    assert services["qdrant"]["status"] in {"ready", "unavailable"}
    assert services["llama_cpp"]["status"] in {"ready", "unavailable"}

    assert "Strictly local-only" in data["disclaimer"]
