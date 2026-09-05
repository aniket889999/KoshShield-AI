from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class VectorStoreChunk:
    point_id: str
    chunk_id: str
    tenant_id: str
    document_id: str
    page_number: int
    redaction_version: int
    chunk_sequence: int
    masked_text: str
    char_start: int
    char_end: int
    masked_content_hash: str
    document_evidence_hash: str
    classification: str
    document_filename: str
    indexed_at: str
    dense_vector: list[float]
    index_version: int = 1
    visual_regions: list[dict[str, Any]] = field(default_factory=list)
    sparse_indices: list[int] = field(default_factory=list)
    sparse_values: list[float] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "tenant_id": self.tenant_id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "redaction_version": self.redaction_version,
            "index_version": self.index_version,
            "chunk_sequence": self.chunk_sequence,
            "masked_text": self.masked_text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "masked_content_hash": self.masked_content_hash,
            "document_evidence_hash": self.document_evidence_hash,
            "classification": self.classification,
            "document_filename": self.document_filename,
            "indexed_at": self.indexed_at,
            "visual_regions": self.visual_regions,
        }


@dataclass
class VectorStoreSearchResult:
    point_id: str
    score: float
    payload: dict[str, Any]


class VectorStoreError(RuntimeError):
    """Raised when an operation on the vector store fails."""


class VectorStoreUnavailableError(VectorStoreError):
    """Raised when the vector store service is unreachable or unconfigured."""


@runtime_checkable
class VectorStore(Protocol):
    def is_available(self) -> tuple[bool, str]:
        """Truthfully report vector store readiness."""
        ...

    def ensure_collection(self, dense_dim: int) -> None:
        """Create or validate the collection schema with named dense and sparse vectors
        and payload indexes. Never silently recreate or delete an incompatible collection.
        """
        ...

    def upsert_chunks(self, chunks: list[VectorStoreChunk]) -> int:
        """Upsert a batch of chunks idempotently."""
        ...

    def verify_points(self, point_ids: list[str], tenant_id: str) -> bool:
        """Verify that all expected point IDs exist and strictly belong to tenant_id."""
        ...

    def delete_stale_chunks(self, document_id: str, tenant_id: str, active_version: int) -> int:
        """Remove points for a document with index_version < active_version,
        strictly scoped to tenant_id.
        """
        ...

    def delete_document_chunks(self, document_id: str, tenant_id: str) -> int:
        """Remove all chunks associated with a document, strictly scoped to tenant_id."""
        ...

    def retrieve_points(
        self, point_ids: list[str], tenant_id: str
    ) -> list[VectorStoreSearchResult]:
        """Retrieve points by ID, enforcing mandatory tenant isolation."""
        ...

    def search_dense(
        self,
        query_vector: list[float],
        tenant_id: str,
        permitted_document_ids: list[str] | None = None,
        classification: str | None = None,
        limit: int = 10,
    ) -> list[VectorStoreSearchResult]:
        """Search using dense vector with mandatory tenant isolation."""
        ...

    def search_sparse(
        self,
        indices: list[int],
        values: list[float],
        tenant_id: str,
        permitted_document_ids: list[str] | None = None,
        classification: str | None = None,
        limit: int = 10,
    ) -> list[VectorStoreSearchResult]:
        """Search using sparse lexical vector with mandatory tenant isolation."""
        ...

    def count_points(self, tenant_id: str | None = None) -> int:
        """Count total indexed points, optionally filtered by tenant."""
        ...
