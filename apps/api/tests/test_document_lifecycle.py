import pymupdf
from fastapi.testclient import TestClient

from koshshield.security.pii.verhoeff import generate_verhoeff_check_digit


def create_synthetic_pdf_with_pii() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()

    base_aadhaar = "36759834123"
    aadhaar = f"{base_aadhaar}{generate_verhoeff_check_digit(base_aadhaar)}"

    content = f"""
    GOVERNMENT OF INDIA
    CONFIDENTIAL DEPARTMENTAL VERIFICATION REPORT

    Officer Name: Shri Rajesh Kumar
    Designation: Senior Procurement Specialist
    Aadhaar Number: {aadhaar[:4]} {aadhaar[4:8]} {aadhaar[8:]}
    PAN: ABCPR1234F
    Mobile: 9876543210
    Official Email: rajesh.kumar@nic.in
    Salary Account: 123456789012
    IFSC: SBIN0001234
    Passport: Z1234567
    Employee ID: EMP-89412
    """
    page.insert_text((50, 72), content)
    return doc.tobytes()


def test_document_lifecycle_state_transitions(client: TestClient) -> None:
    pdf_bytes = create_synthetic_pdf_with_pii()

    # 1. Upload document -> ENCRYPTED
    upload_res = client.post(
        "/api/v1/documents",
        files={"file": ("report.pdf", pdf_bytes, "application/pdf")},
        headers={"X-Actor-ID": "intake-officer"},
    )
    assert upload_res.status_code == 201
    doc = upload_res.json()
    doc_id = doc["id"]
    assert doc["status"] == "ENCRYPTED"

    # 2. Prevent premature approval before review is required
    approve_res = client.post(
        f"/api/v1/documents/{doc_id}/redactions/approve",
        headers={"X-Actor-ID": "compliance-officer"},
    )
    assert approve_res.status_code == 400
    assert "must be in REVIEW_REQUIRED state" in approve_res.json()["detail"]

    # 3. Start extraction -> moves to REVIEW_REQUIRED
    extract_res = client.post(
        f"/api/v1/documents/{doc_id}/extraction",
        headers={"X-Actor-ID": "extractor-service"},
    )
    assert extract_res.status_code == 202
    job = extract_res.json()
    assert job["status"] == "COMPLETED"
    assert job["pages_processed"] == 1

    # 4. Check review queue
    queue_res = client.get("/api/v1/review")
    assert queue_res.status_code == 200
    queue = queue_res.json()
    queue_item = next((item for item in queue if item["document_id"] == doc_id), None)
    assert queue_item is not None
    assert queue_item["status"] == "REVIEW_REQUIRED"
    assert queue_item["total_findings"] > 0
    assert queue_item["pending_findings"] > 0

    # 5. Prevent starting extraction again from REVIEW_REQUIRED (must be explicit)
    conflict_res = client.post(
        f"/api/v1/documents/{doc_id}/extraction",
        headers={"X-Actor-ID": "extractor-service"},
    )
    assert conflict_res.status_code == 409
    assert "Invalid transition" in conflict_res.json()["detail"]
