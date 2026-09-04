from fastapi.testclient import TestClient
from sqlalchemy import select
from test_document_lifecycle import create_synthetic_pdf_with_pii

from koshshield.database import SessionLocal
from koshshield.models import DocumentPageRecord


def test_redaction_review_workflow_and_approval(client: TestClient) -> None:
    pdf_bytes = create_synthetic_pdf_with_pii()

    # Upload & extract
    upload_res = client.post(
        "/api/v1/documents",
        files={"file": ("report.pdf", pdf_bytes, "application/pdf")},
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["id"]

    extract_res = client.post(f"/api/v1/documents/{doc_id}/extraction")
    assert extract_res.status_code == 202

    # Get findings
    redactions_res = client.get(f"/api/v1/documents/{doc_id}/redactions")
    assert redactions_res.status_code == 200
    redactions = redactions_res.json()
    assert redactions["status"] == "REVIEW_REQUIRED"
    assert redactions["unresolved_count"] > 0
    findings = redactions["findings"]
    first_finding = findings[0]

    # Test optimistic concurrency: wrong version returns 409
    bad_version_res = client.patch(
        f"/api/v1/documents/{doc_id}/redactions/{first_finding['id']}",
        json={"decision": "ACCEPTED", "version": 999},
    )
    assert bad_version_res.status_code == 409
    detail_lower = bad_version_res.json()["detail"].lower()
    assert "concurrency" in detail_lower or "version" in detail_lower

    # Test individual decision: accept first finding with correct version
    update_res = client.patch(
        f"/api/v1/documents/{doc_id}/redactions/{first_finding['id']}",
        json={"decision": "ACCEPTED", "version": first_finding["version"]},
        headers={"X-Actor-ID": "reviewer-1"},
    )
    assert update_res.status_code == 200
    updated_finding = update_res.json()
    assert updated_finding["status"] == "ACCEPTED"
    assert updated_finding["version"] == first_finding["version"] + 1
    assert updated_finding["reviewer_id"] == "reviewer-1"

    # Attempting to approve while unresolved findings exist must fail
    premature_approve = client.post(f"/api/v1/documents/{doc_id}/redactions/approve")
    assert premature_approve.status_code == 400
    assert "unresolved finding" in premature_approve.json()["detail"]

    # Use accept-high-confidence to resolve remaining high confidence findings
    bulk_res = client.post(f"/api/v1/documents/{doc_id}/redactions/accept-high-confidence")
    assert bulk_res.status_code == 200
    assert bulk_res.json()["accepted_count"] > 0

    # Check if any unresolved findings remain; reject them
    redactions_res2 = client.get(f"/api/v1/documents/{doc_id}/redactions")
    findings2 = redactions_res2.json()["findings"]
    for f in findings2:
        if f["status"] == "PENDING":
            patch_res = client.patch(
                f"/api/v1/documents/{doc_id}/redactions/{f['id']}",
                json={"decision": "REJECTED", "version": f["version"]},
            )
            assert patch_res.status_code == 200

    # Now approval must succeed!
    approve_res = client.post(
        f"/api/v1/documents/{doc_id}/redactions/approve",
        headers={"X-Actor-ID": "lead-reviewer"},
    )
    assert approve_res.status_code == 200
    final_doc = approve_res.json()
    assert final_doc["status"] == "INDEX_READY"

    # Verify masked text is now populated and deterministic
    with SessionLocal() as session:
        page = session.scalar(
            select(DocumentPageRecord).where(DocumentPageRecord.document_id == doc_id)
        )
        assert page is not None
        assert page.masked_text is not None
        assert "[AADHAAR_REDACTED]" in page.masked_text
        assert "[PAN_REDACTED]" in page.masked_text
        assert "[PHONE_REDACTED]" in page.masked_text

    # Verify audit integrity remains valid
    integrity_res = client.get("/api/v1/audit/integrity")
    assert integrity_res.status_code == 200
    assert integrity_res.json()["valid"] is True
