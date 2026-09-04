from koshshield.services.retrieval.vector_store.interfaces import (
    VectorStore,
    VectorStoreChunk,
    VectorStoreSearchResult,
)


class InMemoryVectorStore(VectorStore):
    """In-memory mock VectorStore for deterministic unit testing and evaluation."""

    def __init__(self) -> None:
        self.chunks: list[VectorStoreChunk] = []

    def is_available(self) -> tuple[bool, str]:
        return True, "In-memory vector store ready"

    def ensure_collection(self, dense_dim: int) -> None:
        pass

    def upsert_chunks(self, chunks: list[VectorStoreChunk]) -> int:
        self.chunks.extend(chunks)
        return len(chunks)

    def delete_document_chunks(self, document_id: str) -> int:
        self.chunks = [c for c in self.chunks if c.document_id != document_id]
        return 1

    def search_dense(
        self,
        query_vector: list[float],
        tenant_id: str,
        permitted_document_ids: list[str] | None = None,
        classification: str | None = None,
        limit: int = 10,
    ) -> list[VectorStoreSearchResult]:
        filtered = [c for c in self.chunks if c.tenant_id == tenant_id]
        if permitted_document_ids is not None:
            filtered = [c for c in filtered if c.document_id in permitted_document_ids]
        if classification:
            filtered = [c for c in filtered if c.classification == classification]

        # Dot product
        scored: list[tuple[float, VectorStoreChunk]] = []
        for c in filtered:
            score = sum(x * y for x, y in zip(query_vector, c.dense_vector, strict=False))
            scored.append((score, c))
        scored.sort(key=lambda s: s[0], reverse=True)

        return [
            VectorStoreSearchResult(
                point_id=c.chunk_id,
                score=score,
                payload=c.to_payload(),
            )
            for score, c in scored[:limit]
        ]

    def search_sparse(
        self,
        indices: list[int],
        values: list[float],
        tenant_id: str,
        permitted_document_ids: list[str] | None = None,
        classification: str | None = None,
        limit: int = 10,
    ) -> list[VectorStoreSearchResult]:
        filtered = [c for c in self.chunks if c.tenant_id == tenant_id]
        if permitted_document_ids is not None:
            filtered = [c for c in filtered if c.document_id in permitted_document_ids]
        if classification:
            filtered = [c for c in filtered if c.classification == classification]

        q_map = dict(zip(indices, values, strict=False))
        scored: list[tuple[float, VectorStoreChunk]] = []
        for c in filtered:
            c_map = dict(zip(c.sparse_indices, c.sparse_values, strict=False))
            score = sum(q_map[idx] * c_map[idx] for idx in q_map if idx in c_map)
            scored.append((score, c))
        scored.sort(key=lambda s: s[0], reverse=True)

        return [
            VectorStoreSearchResult(
                point_id=c.chunk_id,
                score=score,
                payload=c.to_payload(),
            )
            for score, c in scored[:limit]
        ]

    def count_points(self, tenant_id: str | None = None) -> int:
        if tenant_id:
            return len([c for c in self.chunks if c.tenant_id == tenant_id])
        return len(self.chunks)
