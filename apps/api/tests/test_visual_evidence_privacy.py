import hashlib
import io
import os
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.orm import Session

from koshshield.api.routes.retrieval import get_embedding_provider, get_vector_store
from koshshield.database import engine
from koshshield.main import app
from koshshield.models import (
    DocumentChunkRecord,
    DocumentPageRecord,
    DocumentRecord,
    DocumentState,
    DocumentVisualRegionRecord,
    FindingStatus,
    RedactionFinding,
)
from koshshield.security.vault import EncryptedVault
from koshshield.services.extraction.interfaces import ExtractedPage
from koshshield.services.redaction import approve_redactions
from koshshield.services.retrieval.embeddings.deterministic_fake import (
    DeterministicEmbeddingProvider,
)
from koshshield.services.retrieval.image_redaction import (
    generate_masked_page_image_derivative,
    merge_bounding_boxes,
)
from koshshield.services.retrieval.vector_store import InMemoryVectorStore
from koshshield.services.retrieval.vector_store.interfaces import VectorStoreChunk
from koshshield.services.retrieval.visuals import (
    UUID_NAMESPACE_KOSHSHIELD_VISUAL,
    build_visual_region_drafts,
)


@pytest.fixture
def mock_dependencies() -> tuple[DeterministicEmbeddingProvider, InMemoryVectorStore]:
    fake_emb = DeterministicEmbeddingProvider()
    fake_store = InMemoryVectorStore()
    app.dependency_overrides[get_embedding_provider] = lambda: fake_emb
    app.dependency_overrides[get_vector_store] = lambda: fake_store
    yield fake_emb, fake_store
    app.dependency_overrides.pop(get_embedding_provider, None)
    app.dependency_overrides.pop(get_vector_store, None)


def _create_synthetic_png_bytes(color: tuple[int, int, int] = (255, 255, 255)) -> bytes:
    img = Image.new("RGB", (200, 200), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_merge_bounding_boxes() -> None:
    # Overlapping boxes
    boxes = [
        (10.0, 10.0, 50.0, 50.0),
        (30.0, 30.0, 70.0, 70.0),
        (100.0, 100.0, 150.0, 150.0),
    ]
    merged = merge_bounding_boxes(boxes)
    assert len(merged) == 2
    assert (10.0, 10.0, 70.0, 70.0) in merged
    assert (100.0, 100.0, 150.0, 150.0) in merged


def test_generate_masked_page_image_derivative_success() -> None:
    vault = EncryptedVault(
        Path(os.environ["KOSHSHIELD_VAULT_DIR"]),
        os.environ["KOSHSHIELD_MASTER_KEY_BASE64"],
    )
    doc_id = str(uuid.uuid4())
    orig_bytes = _create_synthetic_png_bytes((255, 255, 255))
    orig_hash = hashlib.sha256(orig_bytes).hexdigest()
    orig_path = vault.encrypt(
        document_id=f"{doc_id}_p1_image",
        evidence_hash=orig_hash,
        plaintext=orig_bytes,
    )

    page = DocumentPageRecord(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        page_number=1,
        width=200,
        height=200,
        encrypted_page_image_path=str(orig_path),
        page_image_sha256=orig_hash,
        page_image_media_type="image/png",
    )

    finding = RedactionFinding(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        page_number=1,
        finding_type="AADHAAR",
        salted_value_hash="dummy",
        detection_source="regex",
        masked_context="context",
        confidence=0.99,
        start_offset=0,
        end_offset=12,
        bbox_json={"bbox": [20.0, 20.0, 80.0, 80.0], "page_width": 200, "page_height": 200},
        status=FindingStatus.ACCEPTED,
    )

    enc_path, masked_sha256, media_type, status = generate_masked_page_image_derivative(
        vault=vault,
        document_id=doc_id,
        page=page,
        accepted_findings=[finding],
        target_version=2,
    )

    assert status == "APPROVED"
    assert enc_path is not None
    assert masked_sha256 is not None
    assert masked_sha256 != orig_hash
    assert media_type == "image/png"

    # Decrypt derivative and verify redacted pixels are black
    decrypted_bytes = vault.decrypt(
        document_id=f"{doc_id}_p1_v2_masked_image",
        evidence_hash=masked_sha256,
        path=enc_path,
    )
    with Image.open(io.BytesIO(decrypted_bytes)) as pil_img:
        # Check center of redacted area (50, 50) is black (0, 0, 0)
        pixel = pil_img.getpixel((50, 50))
        assert pixel == (0, 0, 0)
        # Check outside redacted area (180, 180) is white (255, 255, 255)
        pixel_outside = pil_img.getpixel((180, 180))
        assert pixel_outside == (255, 255, 255)


def test_fail_closed_unlocated_pii() -> None:
    vault = EncryptedVault(
        Path(os.environ["KOSHSHIELD_VAULT_DIR"]),
        os.environ["KOSHSHIELD_MASTER_KEY_BASE64"],
    )
    doc_id = str(uuid.uuid4())
    orig_bytes = _create_synthetic_png_bytes()
    orig_hash = hashlib.sha256(orig_bytes).hexdigest()
    orig_path = vault.encrypt(
        document_id=f"{doc_id}_p1_image",
        evidence_hash=orig_hash,
        plaintext=orig_bytes,
    )

    page = DocumentPageRecord(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        page_number=1,
        width=200,
        height=200,
        encrypted_page_image_path=str(orig_path),
        page_image_sha256=orig_hash,
        page_image_media_type="image/png",
    )

    # Finding with missing bbox_json
    finding = RedactionFinding(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        page_number=1,
        finding_type="PAN",
        salted_value_hash="dummy",
        detection_source="regex",
        masked_context="context",
        confidence=0.95,
        start_offset=0,
        end_offset=10,
        bbox_json=None,
        status=FindingStatus.ACCEPTED,
    )

    enc_path, masked_sha256, media_type, status = generate_masked_page_image_derivative(
        vault=vault,
        document_id=doc_id,
        page=page,
        accepted_findings=[finding],
        target_version=2,
    )

    assert status == "BLOCKED_UNLOCATED_PII"
    assert enc_path is None
    assert masked_sha256 is None


def test_fail_closed_residual_pii_detected_by_ocr() -> None:
    vault = EncryptedVault(
        Path(os.environ["KOSHSHIELD_VAULT_DIR"]),
        os.environ["KOSHSHIELD_MASTER_KEY_BASE64"],
    )
    doc_id = str(uuid.uuid4())
    orig_bytes = _create_synthetic_png_bytes()
    orig_hash = hashlib.sha256(orig_bytes).hexdigest()
    orig_path = vault.encrypt(
        document_id=f"{doc_id}_p1_image",
        evidence_hash=orig_hash,
        plaintext=orig_bytes,
    )

    page = DocumentPageRecord(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        page_number=1,
        width=200,
        height=200,
        encrypted_page_image_path=str(orig_path),
        page_image_sha256=orig_hash,
        page_image_media_type="image/png",
    )

    finding = RedactionFinding(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        page_number=1,
        finding_type="AADHAAR",
        salted_value_hash="dummy",
        detection_source="regex",
        masked_context="context",
        confidence=0.99,
        start_offset=0,
        end_offset=12,
        bbox_json={"bbox": [10.0, 10.0, 30.0, 30.0]},
        status=FindingStatus.ACCEPTED,
    )

    # Mock OCR adapter that discovers residual Aadhaar
    mock_ocr = MagicMock()
    mock_ocr.is_available.return_value = (True, "Ready")
    mock_ocr.extract_page_from_image.return_value = ExtractedPage(
        page_number=1,
        width=200.0,
        height=200.0,
        text="Applicant PAN: ABCPE1234F verified.",
    )

    enc_path, masked_sha256, media_type, status = generate_masked_page_image_derivative(
        vault=vault,
        document_id=doc_id,
        page=page,
        accepted_findings=[finding],
        target_version=2,
        ocr_adapter=mock_ocr,
    )

    assert status == "BLOCKED_RESIDUAL_PII"
    assert enc_path is None
    assert masked_sha256 is None


def test_build_visual_region_drafts_truthful_and_deterministic_uuidv5() -> None:
    tenant_id = "test-tenant"
    doc_id = str(uuid.uuid4())
    masked_text = "Table of allowances and schedule: INR 50,000"
    masked_hash = "m" * 64

    drafts = build_visual_region_drafts(
        tenant_id=tenant_id,
        document_id=doc_id,
        masked_text=masked_text,
        page_number=1,
        width=612.0,
        height=792.0,
        masked_image_sha256=masked_hash,
        redaction_version=2,
    )

    # Only truthful PAGE_IMAGE draft emitted; no fake heuristic tables/diagrams
    assert len(drafts) == 1
    assert drafts[0].region_type == "PAGE_IMAGE"
    assert drafts[0].source == "page_raster"
    assert drafts[0].masked_image_sha256 == masked_hash
    assert drafts[0].redaction_version == 2
    assert drafts[0].tenant_id == tenant_id

    # Verify deterministic UUIDv5 matching
    expected_id = str(uuid.uuid5(UUID_NAMESPACE_KOSHSHIELD_VISUAL, f"{tenant_id}:{doc_id}:1:0:2"))
    assert drafts[0].region_id == expected_id


def test_approve_redactions_generates_derivatives_and_updates_regions() -> None:
    vault = EncryptedVault(
        Path(os.environ["KOSHSHIELD_VAULT_DIR"]),
        os.environ["KOSHSHIELD_MASTER_KEY_BASE64"],
    )
    doc_id = str(uuid.uuid4())
    orig_bytes = _create_synthetic_png_bytes()
    orig_hash = hashlib.sha256(orig_bytes).hexdigest()
    orig_path = vault.encrypt(
        document_id=f"{doc_id}_p1_image",
        evidence_hash=orig_hash,
        plaintext=orig_bytes,
    )

    raw_text = "Officer Aadhaar: 9999 8888 7777 approved."
    raw_text_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    raw_text_path = vault.encrypt(
        document_id=f"{doc_id}_p1_raw",
        evidence_hash=raw_text_hash,
        plaintext=raw_text.encode("utf-8"),
    )

    with Session(bind=engine) as session:
        doc = DocumentRecord(
            id=doc_id,
            tenant_id="tenant-alpha",
            filename="confidential.pdf",
            media_type="application/pdf",
            size_bytes=2048,
            sha256="d" * 64,
            vault_path=f"vault/{doc_id}.ksh",
            status=DocumentState.REVIEW_REQUIRED,
            version=1,
        )
        page = DocumentPageRecord(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            page_number=1,
            width=200,
            height=200,
            extraction_method="native_pdf",
            text_hash=raw_text_hash,
            encrypted_artifact_path=str(raw_text_path),
            encrypted_page_image_path=str(orig_path),
            page_image_sha256=orig_hash,
            page_image_media_type="image/png",
        )
        finding = RedactionFinding(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            page_number=1,
            finding_type="AADHAAR",
            salted_value_hash="aadhaar-hash",
            detection_source="regex",
            masked_context="context",
            confidence=0.99,
            start_offset=17,
            end_offset=31,
            bbox_json={"bbox": [30.0, 30.0, 90.0, 50.0]},
            status=FindingStatus.ACCEPTED,
            version=1,
        )
        session.add_all([doc, page, finding])
        session.commit()

        updated_doc = approve_redactions(
            session=session,
            document_id=doc_id,
            actor_id="approver-1",
            vault=vault,
            tenant_id="tenant-alpha",
        )

        assert updated_doc.status == DocumentState.INDEX_READY
        assert updated_doc.version == 2

        # Verify page derivative metadata
        session.refresh(page)
        assert page.visual_privacy_status == "APPROVED"
        assert page.visual_redaction_version == 2
        assert page.encrypted_masked_page_image_path is not None
        assert page.masked_page_image_sha256 is not None
        assert page.masked_page_image_sha256 != orig_hash

        # Verify visual regions created with provenance
        from sqlalchemy import select

        regions = list(
            session.scalars(
                select(DocumentVisualRegionRecord).where(
                    DocumentVisualRegionRecord.document_id == doc_id
                )
            )
        )
        assert len(regions) == 1
        reg = regions[0]
        assert reg.tenant_id == "tenant-alpha"
        assert reg.redaction_version == 2
        assert reg.masked_image_sha256 == page.masked_page_image_sha256


def test_hardened_evidence_endpoint_blocks_cross_tenant_and_stale(
    client: TestClient,
    mock_dependencies: tuple[DeterministicEmbeddingProvider, InMemoryVectorStore],
) -> None:
    embedding_provider, vector_store = mock_dependencies
    vault = EncryptedVault(
        Path(os.environ["KOSHSHIELD_VAULT_DIR"]),
        os.environ["KOSHSHIELD_MASTER_KEY_BASE64"],
    )
    doc_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())
    evidence_hash = "h" * 64

    masked_img = _create_synthetic_png_bytes((0, 0, 0))
    masked_hash = hashlib.sha256(masked_img).hexdigest()
    masked_path = vault.encrypt(
        document_id=f"{doc_id}_p1_v2_masked_image",
        evidence_hash=masked_hash,
        plaintext=masked_img,
    )

    with Session(bind=engine) as session:
        doc = DocumentRecord(
            id=doc_id,
            tenant_id="tenant-secure",
            filename="secure-doc.pdf",
            media_type="application/pdf",
            size_bytes=1024,
            sha256=evidence_hash,
            vault_path=f"vault/{doc_id}.ksh",
            status=DocumentState.INDEXED,
            active_index_version=2,
        )
        page_id = str(uuid.uuid4())
        page = DocumentPageRecord(
            id=page_id,
            document_id=doc_id,
            page_number=1,
            width=200,
            height=200,
            extraction_method="native_pdf",
            text_hash="thash",
            encrypted_artifact_path="vault/raw.ksh",
            encrypted_masked_page_image_path=str(masked_path),
            masked_page_image_sha256=masked_hash,
            masked_page_image_media_type="image/png",
            visual_privacy_status="APPROVED",
            visual_redaction_version=2,
            masked_text="Clean content",
        )
        chunk = DocumentChunkRecord(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            page_number=1,
            chunk_sequence=0,
            index_version=2,
            chunk_id=chunk_id,
            char_start=0,
            char_end=13,
            masked_content_hash="c" * 64,
        )
        session.add_all([doc, page, chunk])
        session.commit()

    emb = embedding_provider.embed_query("Clean content")
    vector_store.upsert_chunks(
        [
            VectorStoreChunk(
                point_id=chunk_id,
                chunk_id=chunk_id,
                tenant_id="tenant-secure",
                document_id=doc_id,
                page_number=1,
                redaction_version=2,
                index_version=2,
                chunk_sequence=0,
                masked_text="Clean content",
                char_start=0,
                char_end=13,
                masked_content_hash="c" * 64,
                document_evidence_hash=evidence_hash,
                classification="RESTRICTED",
                document_filename="secure-doc.pdf",
                indexed_at="2026-09-06T12:00:00Z",
                dense_vector=emb.dense,
                sparse_indices=emb.sparse_indices,
                sparse_values=emb.sparse_values,
            )
        ]
    )

    # 1. Authorized request -> 200 with masked derivative and security headers
    res_ok = client.get(
        f"/api/v1/retrieval/evidence/{chunk_id}/page-image",
        headers={"X-Tenant-ID": "tenant-secure"},
    )
    assert res_ok.status_code == 200
    assert res_ok.content == masked_img
    assert res_ok.headers["cache-control"] == "no-store"
    assert res_ok.headers["x-content-type-options"] == "nosniff"
    assert res_ok.headers["x-koshshield-masked-image-hash"] == masked_hash

    # 2. Cross-tenant request -> 404
    res_cross = client.get(
        f"/api/v1/retrieval/evidence/{chunk_id}/page-image",
        headers={"X-Tenant-ID": "tenant-other"},
    )
    assert res_cross.status_code == 404

    # 3. Blocked visual privacy status -> 404
    with Session(bind=engine) as session:
        p = session.get(DocumentPageRecord, page_id)
        assert p is not None
        p.visual_privacy_status = "BLOCKED_UNLOCATED_PII"
        session.commit()

    res_blocked = client.get(
        f"/api/v1/retrieval/evidence/{chunk_id}/page-image",
        headers={"X-Tenant-ID": "tenant-secure"},
    )
    assert res_blocked.status_code == 404

    # 4. Stale visual redaction version -> 404
    with Session(bind=engine) as session:
        p = session.get(DocumentPageRecord, page_id)
        assert p is not None
        p.visual_privacy_status = "APPROVED"
        p.visual_redaction_version = 1  # Stale, active index version is 2
        session.commit()

    res_stale = client.get(
        f"/api/v1/retrieval/evidence/{chunk_id}/page-image",
        headers={"X-Tenant-ID": "tenant-secure"},
    )
    assert res_stale.status_code == 404
