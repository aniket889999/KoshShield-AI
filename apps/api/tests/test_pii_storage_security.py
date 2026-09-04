import json
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select
from test_document_lifecycle import create_synthetic_pdf_with_pii

from koshshield.database import SessionLocal
from koshshield.models import (
    AuditEvent,
    DocumentPageRecord,
    DocumentRecord,
    ExtractionJob,
    RedactionFinding,
)


def test_raw_pii_never_stored_in_plaintext(client: TestClient) -> None:
    pdf_bytes = create_synthetic_pdf_with_pii()

    # The known synthetic sensitive identifiers in the test document
    sensitive_values = [
        "36759834123",
        "ABCPR1234F",
        "9876543210",
        "rajesh.kumar@nic.in",
        "123456789012",
        "SBIN0001234",
        "Z1234567",
    ]

    # 1. Upload document
    upload_res = client.post(
        "/api/v1/documents",
        files={"file": ("tender_sensitive.pdf", pdf_bytes, "application/pdf")},
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["id"]

    # 2. Extract document
    extract_res = client.post(f"/api/v1/documents/{doc_id}/extraction")
    assert extract_res.status_code == 202

    # 3. Deep-scan all database rows before approval
    with SessionLocal() as session:
        # Scan documents table
        for doc in session.scalars(select(DocumentRecord)):
            for val in sensitive_values:
                assert val not in doc.filename
                assert val not in doc.status

        # Scan document_pages table: masked_text must be None before approval
        for page in session.scalars(select(DocumentPageRecord)):
            assert page.masked_text is None
            for val in sensitive_values:
                assert val not in page.text_hash
                assert val not in page.encrypted_artifact_path

        # Scan redaction_findings: only salted_value_hash and masked_context exist
        for finding in session.scalars(select(RedactionFinding)):
            for val in sensitive_values:
                assert val not in finding.salted_value_hash, "PII leaked into salted_value_hash"
                assert val not in finding.masked_context, (
                    f"Sensitive value {val} leaked into masked_context {finding.masked_context}"
                )

        # Scan extraction_jobs table
        for job in session.scalars(select(ExtractionJob)):
            if job.error_message:
                for val in sensitive_values:
                    assert val not in job.error_message

        # Scan audit_events table
        for event in session.scalars(select(AuditEvent)):
            event_json = json.dumps(event.details)
            for val in sensitive_values:
                assert val not in event_json, (
                    f"Sensitive value {val} leaked into audit event {event.event_type}"
                )

    # 4. Resolve findings and approve
    client.post(f"/api/v1/documents/{doc_id}/redactions/accept-high-confidence")
    redactions_res = client.get(f"/api/v1/documents/{doc_id}/redactions")
    for f in redactions_res.json()["findings"]:
        if f["status"] == "PENDING":
            client.patch(
                f"/api/v1/documents/{doc_id}/redactions/{f['id']}",
                json={"decision": "ACCEPTED", "version": f["version"]},
            )

    approve_res = client.post(f"/api/v1/documents/{doc_id}/redactions/approve")
    assert approve_res.status_code == 200

    # 5. Deep-scan all database rows after approval
    with SessionLocal() as session:
        for page in session.scalars(select(DocumentPageRecord)):
            assert page.masked_text is not None
            for val in sensitive_values:
                assert val not in page.masked_text, (
                    f"Sensitive value {val} found in approved masked text"
                )

        for event in session.scalars(select(AuditEvent)):
            event_json = json.dumps(event.details)
            for val in sensitive_values:
                assert val not in event_json, f"Sensitive value {val} found in audit event details"

    # 6. Verify that raw vault files are encrypted and do not contain plaintext PII
    vault_files = list(Path("/tmp/koshshield-test-vault").glob("*.ksh"))
    assert len(vault_files) >= 2  # Original document + raw page artifact
    for vfile in vault_files:
        payload = vfile.read_bytes()
        assert payload.startswith(b"KSH1")
        for val in sensitive_values:
            assert val.encode() not in payload, (
                f"Plaintext value {val} found in encrypted vault file {vfile}"
            )
