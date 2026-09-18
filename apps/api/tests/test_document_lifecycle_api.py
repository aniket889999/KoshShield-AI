import io

from fastapi.testclient import TestClient

from koshshield.database import get_db
from koshshield.main import app
from koshshield.models import (
    DocumentChunkRecord,
    DocumentPageRecord,
    DocumentRecord,
    DocumentState,
    ExtractionJob,
    FindingStatus,
    RedactionFinding,
)
from koshshield.services.lifecycle import LifecycleStageId, LifecycleStageStatus


def test_document_lifecycle_initial_upload():
    client = TestClient(app)
    headers = {
        "X-Tenant-ID": "tenant-alpha",
        "X-Actor-ID": "analyst-1",
        "X-Roles": "user,reviewer",
    }

    pdf_content = b"%PDF-1.4 sample document bytes for testing lifecycle timeline"
    response = client.post(
        "/api/v1/documents",
        files={"file": ("report.pdf", io.BytesIO(pdf_content), "application/pdf")},
        headers=headers,
    )
    assert response.status_code == 201
    doc_id = response.json()["id"]

    # Now retrieve timeline
    timeline_resp = client.get(f"/api/v1/documents/{doc_id}/timeline", headers=headers)
    assert timeline_resp.status_code == 200
    timeline = timeline_resp.json()

    assert timeline["document_id"] == doc_id
    assert timeline["tenant_id"] == "tenant-alpha"
    assert timeline["filename"] == "report.pdf"
    assert timeline["current_status"] == DocumentState.ENCRYPTED

    stages = {s["stage_id"]: s for s in timeline["stages"]}
    assert stages[LifecycleStageId.UPLOAD]["status"] == LifecycleStageStatus.PASSED
    assert stages[LifecycleStageId.EXTRACTION]["status"] == LifecycleStageStatus.NOT_EXECUTED
    assert stages[LifecycleStageId.PII_REVIEW]["status"] == LifecycleStageStatus.NOT_EXECUTED
    assert stages[LifecycleStageId.APPROVAL]["status"] == LifecycleStageStatus.NOT_EXECUTED
    assert stages[LifecycleStageId.INDEXING]["status"] == LifecycleStageStatus.NOT_EXECUTED
    assert stages[LifecycleStageId.RETRIEVAL]["status"] == LifecycleStageStatus.NOT_EXECUTED

    # Audit events
    assert len(timeline["audit_events"]) >= 1
    assert timeline["audit_events"][0]["event_type"] == "document.accepted"
    # Ensure no vault path leaks into audit details
    details = timeline["audit_events"][0]["details"]
    assert "vault_path" not in details
    assert "path" not in details


def test_document_lifecycle_tenant_isolation():
    client = TestClient(app)
    headers_alpha = {
        "X-Tenant-ID": "tenant-alpha",
        "X-Actor-ID": "analyst-1",
    }
    headers_beta = {
        "X-Tenant-ID": "tenant-beta",
        "X-Actor-ID": "analyst-2",
    }

    pdf_content = b"%PDF-1.4 tenant isolation sample"
    upload_resp = client.post(
        "/api/v1/documents",
        files={"file": ("confidential.pdf", io.BytesIO(pdf_content), "application/pdf")},
        headers=headers_alpha,
    )
    doc_id = upload_resp.json()["id"]

    # Cross-tenant access must return 404
    cross_resp = client.get(f"/api/v1/documents/{doc_id}/timeline", headers=headers_beta)
    assert cross_resp.status_code == 404
    assert "Document not found" in cross_resp.json()["detail"]


def test_document_lifecycle_full_progression():
    client = TestClient(app)
    headers = {
        "X-Tenant-ID": "tenant-gamma",
        "X-Actor-ID": "analyst-3",
        "X-Roles": "admin,reviewer,approver",
    }

    # Simulate advanced document state in DB
    db = next(app.dependency_overrides.get(get_db, get_db)())
    try:
        doc = DocumentRecord(
            id="doc-lifecycle-indexed",
            tenant_id="tenant-gamma",
            filename="approved_tender.pdf",
            media_type="application/pdf",
            size_bytes=1024,
            sha256="abc123hash",
            vault_path="encrypted/fake",
            status=DocumentState.INDEXED,
            version=2,
            active_index_version=1,
            index_cleanup_pending=True,
        )
        db.add(doc)

        job = ExtractionJob(
            id="job-1",
            document_id=doc.id,
            status="COMPLETED",
            pages_processed=3,
            total_pages=3,
            extraction_method="native_pdf",
        )
        db.add(job)

        page = DocumentPageRecord(
            id="page-1",
            document_id=doc.id,
            page_number=1,
            width=612.0,
            height=792.0,
            extraction_method="native_pdf",
            text_hash="texthash1",
            encrypted_artifact_path="vault/art1",
            masked_text="Public tender content [AADHAAR_REDACTED]",
        )
        db.add(page)

        finding = RedactionFinding(
            id="finding-1",
            document_id=doc.id,
            page_number=1,
            finding_type="AADHAAR",
            confidence=0.98,
            detection_source="regex",
            start_offset=22,
            end_offset=34,
            salted_value_hash="saltedhash123",
            masked_context="content [AADHAAR_REDACTED]",
            status=FindingStatus.ACCEPTED,
        )
        db.add(finding)

        chunk = DocumentChunkRecord(
            id="chunk-1",
            document_id=doc.id,
            page_number=1,
            chunk_sequence=0,
            index_version=1,
            chunk_id="chk-001",
            char_start=0,
            char_end=40,
            masked_content_hash="chunkhash123",
        )
        db.add(chunk)
        db.commit()

        resp = client.get(f"/api/v1/documents/{doc.id}/timeline", headers=headers)
        assert resp.status_code == 200
        data = resp.json()

        stages = {s["stage_id"]: s for s in data["stages"]}
        assert stages[LifecycleStageId.UPLOAD]["status"] == LifecycleStageStatus.PASSED
        assert stages[LifecycleStageId.EXTRACTION]["status"] == LifecycleStageStatus.PASSED
        assert stages[LifecycleStageId.EXTRACTION]["metrics"]["page_count"] == 1

        assert stages[LifecycleStageId.PII_REVIEW]["status"] == LifecycleStageStatus.PASSED
        assert stages[LifecycleStageId.PII_REVIEW]["metrics"]["accepted_findings"] == 1

        assert stages[LifecycleStageId.APPROVAL]["status"] == LifecycleStageStatus.PASSED
        assert stages[LifecycleStageId.INDEXING]["status"] == LifecycleStageStatus.PASSED
        assert stages[LifecycleStageId.INDEXING]["metrics"]["chunk_count"] == 1

        assert stages[LifecycleStageId.RETRIEVAL]["status"] == LifecycleStageStatus.PASSED
        assert stages[LifecycleStageId.RETRIEVAL]["metrics"]["retrieval_eligible"] is True

        assert stages[LifecycleStageId.CLEANUP]["status"] == LifecycleStageStatus.PENDING
        assert stages[LifecycleStageId.CLEANUP]["metrics"]["cleanup_pending"] is True
    finally:
        db.close()
