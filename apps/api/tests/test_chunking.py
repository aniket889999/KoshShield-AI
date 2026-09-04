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
    # Small target to force splits around placeholders
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
        # Placeholder should never be cut into fragments like "[REDACTED_" or "AADHAAR]"
        if "REDACTED" in chunk.masked_text:
            assert placeholder in chunk.masked_text
            assert "[REDACTED" in chunk.masked_text
            assert "]" in chunk.masked_text


def test_chunking_deterministic_ids() -> None:
    chunker = DeterministicMaskedChunker(target_tokens=40, overlap_tokens=10, chars_per_token=4)
    text = "Section A contains financial disbursements. Section B contains contractor identities."

    chunks_run1 = chunker.chunk_page(
        page_text=text,
        document_id="doc-det",
        page_number=1,
        document_filename="det.pdf",
        document_evidence_hash="sha-det",
        redaction_version=1,
    )
    chunks_run2 = chunker.chunk_page(
        page_text=text,
        document_id="doc-det",
        page_number=1,
        document_filename="det.pdf",
        document_evidence_hash="sha-det",
        redaction_version=1,
    )

    assert len(chunks_run1) == len(chunks_run2)
    for c1, c2 in zip(chunks_run1, chunks_run2, strict=True):
        assert c1.chunk_id == c2.chunk_id
        assert c1.masked_content_hash == c2.masked_content_hash
        assert c1.masked_text == c2.masked_text
