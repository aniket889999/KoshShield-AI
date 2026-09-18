import io

from fastapi.testclient import TestClient

from koshshield.database import get_db
from koshshield.main import app
from koshshield.models import (
    DocumentChunkRecord,
    DocumentPageRecord,
    DocumentRecord,
    DocumentState,
    DocumentVisualRegionRecord,
)


def test_evidence_catalog_unapproved_document_fails_closed():
    client = TestClient(app)
    headers = {
        "X-Tenant-ID": "tenant-zeta",
        "X-Actor-ID": "user-1",
        "X-Roles": "user",
    }

    # Upload document (starts in ENCRYPTED state)
    pdf_content = b"%PDF-1.4 sample for evidence gating test"
    resp = client.post(
        "/api/v1/documents",
        files={"file": ("raw.pdf", io.BytesIO(pdf_content), "application/pdf")},
        headers=headers,
    )
    doc_id = resp.json()["id"]

    # Attempting to fetch evidence catalog before approval must fail closed with 403
    evidence_resp = client.get(f"/api/v1/documents/{doc_id}/evidence", headers=headers)
    assert evidence_resp.status_code == 403
    assert "Evidence catalog restricted" in evidence_resp.json()["detail"]


def test_evidence_catalog_approved_indexed_document():
    client = TestClient(app)
    headers = {
        "X-Tenant-ID": "tenant-zeta",
        "X-Actor-ID": "user-1",
        "X-Roles": "user",
    }

    db = next(app.dependency_overrides.get(get_db, get_db)())
    try:
        doc = DocumentRecord(
            id="doc-evidence-approved",
            tenant_id="tenant-zeta",
            filename="procurement_spec.pdf",
            media_type="application/pdf",
            size_bytes=2048,
            sha256="sha256spec123",
            vault_path="encrypted/vault/spec",
            status=DocumentState.INDEXED,
            version=1,
            active_index_version=1,
        )
        db.add(doc)

        page = DocumentPageRecord(
            id="page-spec-1",
            document_id=doc.id,
            page_number=1,
            width=612.0,
            height=792.0,
            extraction_method="native_pdf",
            text_hash="pagehash1",
            encrypted_artifact_path="vault/spec1",
            masked_text="Section 1. Flow capacity is 80 m3/h at 16 bar. [REDACTED]",
        )
        db.add(page)

        region = DocumentVisualRegionRecord(
            id="reg-spec-1",
            tenant_id="tenant-zeta",
            document_id=doc.id,
            page_number=1,
            region_sequence=0,
            region_type="TABLE",
            source="native_layout",
            bbox_json={"bbox": [50.0, 100.0, 400.0, 300.0]},
            caption_text="Pump flow rate requirements table",
            caption_hash="captionhash123",
            masked_image_sha256="maskedimg123",
            redaction_version=1,
        )
        db.add(region)

        chunk = DocumentChunkRecord(
            id="chk-spec-0",
            document_id=doc.id,
            page_number=1,
            chunk_sequence=0,
            index_version=1,
            chunk_id="chunk-spec-p1-0",
            char_start=0,
            char_end=65,
            masked_content_hash="chkhash123",
        )
        db.add(chunk)
        db.commit()

        resp = client.get(f"/api/v1/documents/{doc.id}/evidence", headers=headers)
        assert resp.status_code == 200
        catalog = resp.json()

        assert catalog["document_id"] == doc.id
        assert catalog["tenant_id"] == "tenant-zeta"
        assert catalog["chunk_count"] == 1
        assert catalog["residual_pii_checked"] is True
        assert catalog["privacy_gate_verdict"] == "APPROVED_MASKED_ONLY"

        chunk_data = catalog["chunks"][0]
        assert chunk_data["chunk_id"] == "chunk-spec-p1-0"
        assert chunk_data["page_number"] == 1
        assert "80 m3/h" in chunk_data["masked_snippet"]
        assert chunk_data["citation_label"] == "[procurement_spec.pdf p.1 #0]"
        assert len(chunk_data["visual_regions"]) == 1
        assert chunk_data["visual_regions"][0]["region_type"] == "TABLE"
        assert chunk_data["visual_regions"][0]["image_available"] is True

        # Ensure no vault path or secret leaks
        assert "encrypted/vault/spec" not in resp.text
    finally:
        db.close()


def test_evidence_catalog_tenant_isolation():
    client = TestClient(app)
    headers_b = {"X-Tenant-ID": "tenant-b", "X-Actor-ID": "actor-b"}

    db = next(app.dependency_overrides.get(get_db, get_db)())
    try:
        doc = DocumentRecord(
            id="doc-evidence-tenant-a",
            tenant_id="tenant-a",
            filename="tenant_a_doc.pdf",
            media_type="application/pdf",
            size_bytes=100,
            sha256="hasha",
            vault_path="patha",
            status=DocumentState.INDEXED,
        )
        db.add(doc)
        db.commit()

        # Tenant B accessing Tenant A's document must return 404
        resp = client.get(f"/api/v1/documents/{doc.id}/evidence", headers=headers_b)
        assert resp.status_code == 404
    finally:
        db.close()


def test_evidence_catalog_residual_pii_fails_closed():
    client = TestClient(app)
    headers = {"X-Tenant-ID": "tenant-leak-test", "X-Actor-ID": "auditor-1"}

    db = next(app.dependency_overrides.get(get_db, get_db)())
    try:
        doc = DocumentRecord(
            id="doc-residual-pii",
            tenant_id="tenant-leak-test",
            filename="badly_masked.pdf",
            media_type="application/pdf",
            size_bytes=100,
            sha256="hashbad",
            vault_path="pathbad",
            status=DocumentState.INDEXED,
        )
        db.add(doc)

        # Simulate unmasked PAN remaining in text
        page = DocumentPageRecord(
            id="page-bad-1",
            document_id=doc.id,
            page_number=1,
            width=612.0,
            height=792.0,
            extraction_method="native_pdf",
            text_hash="hashpage",
            encrypted_artifact_path="vault/bad1",
            masked_text="Vendor contact email: analyst@secretcompany.in",
        )
        db.add(page)

        chunk = DocumentChunkRecord(
            id="chk-bad-0",
            document_id=doc.id,
            page_number=1,
            chunk_sequence=0,
            index_version=1,
            chunk_id="chk-bad",
            char_start=0,
            char_end=50,
            masked_content_hash="badchkhash",
        )
        db.add(chunk)
        db.commit()

        # Should fail closed with 500 Residual PII violation
        resp = client.get(f"/api/v1/documents/{doc.id}/evidence", headers=headers)
        assert resp.status_code == 500
        assert "residual PII pattern detected" in resp.json()["detail"]
    finally:
        db.close()
