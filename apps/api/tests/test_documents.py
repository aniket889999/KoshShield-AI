from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select

from koshshield.database import SessionLocal
from koshshield.models import AuditEvent


def test_accepts_and_encrypts_pdf(client: TestClient) -> None:
    content = b"%PDF-1.4\nSynthetic confidential tender\n%%EOF"

    response = client.post(
        "/api/v1/documents",
        files={"file": ("tender.pdf", content, "application/pdf")},
        headers={"X-Actor-ID": "procurement-officer"},
    )

    assert response.status_code == 201
    document = response.json()
    assert document["filename"] == "tender.pdf"
    assert document["status"].upper() == "ENCRYPTED"
    assert document["media_type"] == "application/pdf"

    vault_objects = list(Path("/tmp/koshshield-test-vault").glob("*.ksh"))
    assert len(vault_objects) == 1
    assert content not in vault_objects[0].read_bytes()

    events = client.get("/api/v1/audit/events").json()
    assert len(events) == 1
    assert events[0]["event_type"] == "document.accepted"
    assert events[0]["actor_id"] == "procurement-officer"
    assert "filename" not in events[0]["details"]

    integrity = client.get("/api/v1/audit/integrity").json()
    assert integrity["valid"] is True
    assert integrity["event_count"] == 1
    assert integrity["head_hash"] == events[0]["event_hash"]
    assert integrity["first_invalid_event_id"] is None


def test_rejects_extension_mismatch(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents",
        files={"file": ("image.png", b"%PDF-1.4\n%%EOF", "image/png")},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "file extension does not match its content"


def test_detects_audit_event_tampering(client: TestClient) -> None:
    content = b"%PDF-1.4\nAudit evidence\n%%EOF"
    response = client.post(
        "/api/v1/documents",
        files={"file": ("evidence.pdf", content, "application/pdf")},
    )
    assert response.status_code == 201

    with SessionLocal() as session:
        event = session.scalar(select(AuditEvent))
        assert event is not None
        event.actor_id = "tampered-user"
        session.commit()

    integrity = client.get("/api/v1/audit/integrity").json()
    assert integrity["valid"] is False
    assert integrity["event_count"] == 1
    assert integrity["first_invalid_event_id"] is not None
