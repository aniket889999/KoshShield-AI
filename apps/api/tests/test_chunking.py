import uuid

from koshshield.services.retrieval.chunking import DeterministicMaskedChunker


def test_chunking_preserves_page_boundaries() -> None:
    chunker = DeterministicMaskedChunker(target_tokens=50, overlap_tokens=10, chars_per_token=4)
    page1 = "Page 1 content. " * 30
    page2 = "Page 2 content. " * 30

    chunks1 = chunker.chunk_page(
        page_text=page1,
        document_id="doc-1",
        page_number=1,
        document_filename="test.pdf",
        document_evidence_hash="sha-1",
        redaction_version=1,
    )
    chunks2 = chunker.chunk_page(
        page_text=page2,
        document_id="doc-1",
        page_number=2,
        document_filename="test.pdf",
        document_evidence_hash="sha-1",
        redaction_version=1,
    )

    assert len(chunks1) >= 1
    assert len(chunks2) >= 1
    assert all(c.page_number == 1 for c in chunks1)
    assert all(c.page_number == 2 for c in chunks2)


def test_chunking_never_splits_redaction_placeholders() -> None:
    chunker = DeterministicMaskedChunker(target_tokens=15, overlap_tokens=5, chars_per_token=4)

    placeholder = "[REDACTED_AADHAAR]"
    text = (
        "Here is preliminary information. "
        f"The applicant identifier is {placeholder} and should remain protected. "
        "Further operational details follow in paragraph two."
    )

    chunks = chunker.chunk_page(
        page_text=text,
        document_id="doc-placeholder",
        page_number=1,
        document_filename="test.pdf",
        document_evidence_hash="sha-ph",
        redaction_version=1,
    )

    for chunk in chunks:
        if "REDACTED" in chunk.masked_text:
            assert placeholder in chunk.masked_text
            assert "[REDACTED" in chunk.masked_text
            assert "]" in chunk.masked_text


def test_chunking_deterministic_uuidv5_identities() -> None:
    chunker = DeterministicMaskedChunker(target_tokens=40, overlap_tokens=10, chars_per_token=4)
    text = "Section A contains financial disbursements. Section B contains contractor identities."

    chunks_run1 = chunker.chunk_page(
        page_text=text,
        document_id="doc-det-untruncated-full-identifier-1234567890",
        page_number=1,
        document_filename="det.pdf",
        document_evidence_hash="sha-det",
        redaction_version=1,
        tenant_id="tenant-alpha",
    )
    chunks_run2 = chunker.chunk_page(
        page_text=text,
        document_id="doc-det-untruncated-full-identifier-1234567890",
        page_number=1,
        document_filename="det.pdf",
        document_evidence_hash="sha-det",
        redaction_version=1,
        tenant_id="tenant-alpha",
    )

    assert len(chunks_run1) == len(chunks_run2)
    for c1, c2 in zip(chunks_run1, chunks_run2, strict=True):
        # 1. Standard UUIDv5 format verification
        parsed_uuid = uuid.UUID(c1.chunk_id)
        assert parsed_uuid.version == 5
        # 2. Exact stability across identical runs
        assert c1.chunk_id == c2.chunk_id
        assert c1.masked_content_hash == c2.masked_content_hash


def test_chunking_identity_changes_with_version() -> None:
    chunker = DeterministicMaskedChunker(target_tokens=40, overlap_tokens=10, chars_per_token=4)
    text = "Section A contains financial disbursements. Section B contains contractor identities."

    chunks_v1 = chunker.chunk_page(
        page_text=text,
        document_id="doc-versioned",
        page_number=1,
        document_filename="det.pdf",
        document_evidence_hash="sha-det",
        redaction_version=1,
        tenant_id="tenant-alpha",
    )
    chunks_v2 = chunker.chunk_page(
        page_text=text,
        document_id="doc-versioned",
        page_number=1,
        document_filename="det.pdf",
        document_evidence_hash="sha-det",
        redaction_version=2,
        tenant_id="tenant-alpha",
    )

    assert len(chunks_v1) == len(chunks_v2)
    for c1, c2 in zip(chunks_v1, chunks_v2, strict=True):
        # New redaction version must produce new chunk IDs
        assert c1.chunk_id != c2.chunk_id
        assert c1.index_version == 1
        assert c2.index_version == 2


def test_chunking_identity_changes_with_tenant() -> None:
    chunker = DeterministicMaskedChunker(target_tokens=40, overlap_tokens=10, chars_per_token=4)
    text = "Shared public circular content regarding administrative norms."

    chunks_t1 = chunker.chunk_page(
        page_text=text,
        document_id="doc-shared",
        page_number=1,
        document_filename="det.pdf",
        document_evidence_hash="sha-det",
        redaction_version=1,
        tenant_id="tenant-alpha",
    )
    chunks_t2 = chunker.chunk_page(
        page_text=text,
        document_id="doc-shared",
        page_number=1,
        document_filename="det.pdf",
        document_evidence_hash="sha-det",
        redaction_version=1,
        tenant_id="tenant-beta",
    )

    assert len(chunks_t1) == len(chunks_t2)
    for c1, c2 in zip(chunks_t1, chunks_t2, strict=True):
        # Different tenant must produce distinct chunk identities
        assert c1.chunk_id != c2.chunk_id


def test_chunking_collision_resistance() -> None:
    chunker = DeterministicMaskedChunker(target_tokens=30, overlap_tokens=5, chars_per_token=4)
    seen_ids: set[str] = set()
    total_generated = 0

    for page_idx in range(1, 21):
        text = f"Page {page_idx} operational clause with distinct paragraph index {page_idx}. " * 10
        chunks = chunker.chunk_page(
            page_text=text,
            document_id="doc-collision-test",
            page_number=page_idx,
            document_filename="collision.pdf",
            document_evidence_hash="sha-coll",
            redaction_version=1,
        )
        for c in chunks:
            assert c.chunk_id not in seen_ids, f"Collision detected for chunk ID: {c.chunk_id}"
            seen_ids.add(c.chunk_id)
            total_generated += 1

    assert total_generated > 30
    assert len(seen_ids) == total_generated
