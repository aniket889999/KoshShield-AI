import hashlib
from types import SimpleNamespace

import pytest

from koshshield.evaluation.evidence import check_citations, prepare_model_evidence
from koshshield.evaluation.fixtures import FixtureValidationError


def evidence_item(**changes):
    values = dict(
        chunk_id="chunk-1",
        document_id="doc-1",
        page_number=1,
        index_version=2,
        redaction_version=2,
        evidence_hash="a" * 64,
        masked_snippet="Approved masked evidence",
        masked_content_hash=hashlib.sha256(b"Approved masked evidence").hexdigest(),
    )
    return SimpleNamespace(**(values | changes))


def prepare(items, tenant="tenant-1"):
    return prepare_model_evidence(
        SimpleNamespace(tenant_id=tenant, items=items),
        tenant_id="tenant-1",
        document_id="doc-1",
        index_version=2,
        redaction_version=2,
        evidence_hash="a" * 64,
        max_page=2,
    )


def test_context_uses_real_retrieval_snippet_contract() -> None:
    result = prepare([evidence_item()])
    assert result == [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "page_number": 1,
            "masked_text": "Approved masked evidence",
        }
    ]


@pytest.mark.parametrize(
    "change",
    [
        {"document_id": "another"},
        {"index_version": 1},
        {"redaction_version": 1},
        {"page_number": 3},
        {"page_number": True},
        {"evidence_hash": "b" * 64},
        {"masked_snippet": "tampered"},
        {"masked_content_hash": "short"},
        {"chunk_id": ""},
    ],
)
def test_invalid_evidence_cannot_enter_model_context(change: dict) -> None:
    with pytest.raises(FixtureValidationError, match="EVIDENCE_INVALID"):
        prepare([evidence_item(**change)])


def test_foreign_tenant_and_duplicate_evidence_are_rejected() -> None:
    with pytest.raises(FixtureValidationError, match="EVIDENCE_TENANT_MISMATCH"):
        prepare([evidence_item()], tenant="another")
    with pytest.raises(FixtureValidationError, match="EVIDENCE_INVALID"):
        prepare([evidence_item(), evidence_item()])


@pytest.mark.parametrize("citations", [[], ["unknown"], ["chunk-1", "chunk-1"]])
def test_invalid_citations_do_not_pass(citations: list[str]) -> None:
    result = check_citations(citations, prepare([evidence_item()]), [1])
    assert not result.identities_valid
    assert not result.cited_pages_matched


def test_retrieved_page_is_not_credited_unless_cited() -> None:
    evidence = prepare([evidence_item(), evidence_item(chunk_id="chunk-2", page_number=2)])
    missing = check_citations(["chunk-1"], evidence, [1, 2])
    assert missing.identities_valid
    assert missing.retrieved_pages_matched
    assert not missing.cited_pages_matched
    complete = check_citations(["chunk-1", "chunk-2"], evidence, [1, 2])
    assert complete.cited_pages_matched
    assert complete.citation_count == 2
