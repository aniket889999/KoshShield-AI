from koshshield.services.retrieval.vector_store.interfaces import (
    VectorStore,
    VectorStoreChunk,
    VectorStoreError,
    VectorStoreSearchResult,
)


class InMemoryVectorStore(VectorStore):
    """In-memory mock VectorStore for deterministic unit testing and evaluation."""

    def __init__(self, expected_dim: int | None = None) -> None:
        self.chunks: list[VectorStoreChunk] = []
        self.expected_dim = expected_dim
        self.collection_created = False

    def is_available(self) -> tuple[bool, str]:
        return True, "In-memory vector store ready"

    def ensure_collection(self, dense_dim: int) -> None:
        if self.expected_dim is not None and self.expected_dim != dense_dim:
            raise VectorStoreError(
                f"Collection has text_dense dim {self.expected_dim}, expected {dense_dim}. "
                "Silently recreating or deleting collection is prohibited."
            )
        self.collection_created = True

    def upsert_chunks(self, chunks: list[VectorStoreChunk]) -> int:
        self.chunks.extend(chunks)
        return len(chunks)

    def verify_points(self, point_ids: list[str], tenant_id: str) -> bool:
        chunk_map = {c.point_id: c for c in self.chunks}
        for pid in point_ids:
            if pid not in chunk_map:
                return False
            if chunk_map[pid].tenant_id != tenant_id:
                return False
        return True

    def delete_stale_chunks(self, document_id: str, tenant_id: str, active_version: int) -> int:
        initial = len(self.chunks)
        self.chunks = [
            c
            for c in self.chunks
            if not (
                c.document_id == document_id
                and c.tenant_id == tenant_id
                and c.index_version < active_version
            )
        ]
        return initial - len(self.chunks)

    def delete_document_chunks(self, document_id: str, tenant_id: str) -> int:
        initial = len(self.chunks)
        self.chunks = [
            c
            for c in self.chunks
            if not (c.document_id == document_id and c.tenant_id == tenant_id)
        ]
        return initial - len(self.chunks)

    def retrieve_points(
        self, point_ids: list[str], tenant_id: str
    ) -> list[VectorStoreSearchResult]:
        chunk_map = {c.point_id: c for c in self.chunks if c.tenant_id == tenant_id}
        return [
            VectorStoreSearchResult(
                point_id=chunk_map[pid].chunk_id,
                score=1.0,
                payload=chunk_map[pid].to_payload(),
            )
            for pid in point_ids
            if pid in chunk_map
        ]

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
