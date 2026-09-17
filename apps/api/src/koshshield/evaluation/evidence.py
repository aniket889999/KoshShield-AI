"""Evidence identity checks for the smoke harness; not semantic entailment grading."""

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from koshshield.evaluation.fixtures import FixtureValidationError

if TYPE_CHECKING:
    from koshshield.services.retrieval.hybrid_search import RetrievalEvidencePack


def prepare_model_evidence(
    pack: "RetrievalEvidencePack",
    *,
    tenant_id: str,
    document_id: str,
    index_version: int,
    redaction_version: int,
    evidence_hash: str,
    max_page: int,
) -> list[dict]:
    if pack.tenant_id != tenant_id:
        raise FixtureValidationError("EVIDENCE_TENANT_MISMATCH")
    if len(pack.items) > 50 or type(index_version) is not int or index_version <= 0:
        raise FixtureValidationError("EVIDENCE_INVALID")
    seen = set()
    result = []
    for item in pack.items:
        if (
            not item.chunk_id
            or item.chunk_id in seen
            or item.document_id != document_id
            or type(item.index_version) is not int
            or item.index_version != index_version
            or type(item.redaction_version) is not int
            or item.redaction_version != redaction_version
            or type(item.page_number) is not int
            or not 1 <= item.page_number <= max_page
            or item.evidence_hash != evidence_hash
            or not isinstance(item.masked_snippet, str)
            or not item.masked_snippet
            or len(item.masked_snippet) > 64 * 1024
            or hashlib.sha256(item.masked_snippet.encode("utf-8")).hexdigest()
            != item.masked_content_hash
        ):
            raise FixtureValidationError("EVIDENCE_INVALID")
        seen.add(item.chunk_id)
        result.append(
            {
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "page_number": item.page_number,
                "masked_text": item.masked_snippet,
            }
        )
    return result


@dataclass(frozen=True)
class CitationCheck:
    identities_valid: bool
    cited_pages_matched: bool
    retrieved_pages_matched: bool
    citation_count: int


def check_citations(
    cited_ids: list[str], evidence: list[dict], required_pages: list[int]
) -> CitationCheck:
    """Grade only IDs actually returned by the model, not every retrieved page."""
    by_id = {item["chunk_id"]: item for item in evidence}
    valid = (
        bool(cited_ids)
        and len(cited_ids) <= 5
        and len(set(cited_ids)) == len(cited_ids)
        and len(by_id) == len(evidence)
        and all(identity in by_id for identity in cited_ids)
    )
    required = set(required_pages)
    cited_pages = {by_id[identity]["page_number"] for identity in cited_ids if identity in by_id}
    retrieved_pages = {item["page_number"] for item in evidence}
    return CitationCheck(
        identities_valid=valid,
        cited_pages_matched=valid and bool(required) and required <= cited_pages,
        retrieved_pages_matched=bool(required) and required <= retrieved_pages,
        citation_count=len(cited_ids),
    )
