import logging
import uuid
from typing import Any
from urllib.parse import urlparse

import httpx

from koshshield.services.retrieval.vector_store.interfaces import (
    VectorStore,
    VectorStoreChunk,
    VectorStoreError,
    VectorStoreSearchResult,
    VectorStoreUnavailableError,
)

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStore):
    """Local-first Qdrant vector store adapter over HTTP REST API.

    Implements named vectors:
    - text_dense: dense cosine vector
    - text_sparse: sparse lexical vector
    Guarantees strict tenant-isolation filters and idempotent operations.
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "koshshield_masked_docs",
        timeout_seconds: float = 10.0,
    ) -> None:
        self.qdrant_url = qdrant_url.rstrip("/")
        self.collection_name = collection_name
        self.timeout = timeout_seconds
        self._validate_local_url(self.qdrant_url)

    @staticmethod
    def _validate_local_url(url: str) -> None:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        is_loopback = host in {"localhost", "127.0.0.1", "::1"}
        is_container_name = bool(host) and "." not in host
        if parsed.scheme not in {"http", "https"} or not (is_loopback or is_container_name):
            raise ValueError(
                f"Qdrant URL must target localhost or a private container name, got: {url}"
            )

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.qdrant_url,
            timeout=self.timeout,
            trust_env=False,
        )

    def is_available(self) -> tuple[bool, str]:
        try:
            with self._client() as client:
                res = client.get("/healthz")
                if res.status_code == 200:
                    return True, "Qdrant vector store is ready"
                return False, f"Qdrant returned HTTP {res.status_code}"
        except Exception as err:
            return False, f"Qdrant is unreachable at {self.qdrant_url}: {err}"

    def ensure_collection(self, dense_dim: int = 1024) -> None:
        """Create or validate the Qdrant collection with named dense and sparse vectors."""
        with self._client() as client:
            try:
                res = client.get(f"/collections/{self.collection_name}")
            except Exception as err:
                raise VectorStoreUnavailableError(
                    f"Cannot reach Qdrant at {self.qdrant_url}: {err}"
                ) from err

            if res.status_code == 404:
                logger.info(
                    "Creating Qdrant collection '%s' with text_dense (%d dims) and text_sparse...",
                    self.collection_name,
                    dense_dim,
                )
                create_payload = {
                    "vectors": {
                        "text_dense": {
                            "size": dense_dim,
                            "distance": "Cosine",
                        }
                    },
                    "sparse_vectors": {
                        "text_sparse": {},
                    },
                }
                create_res = client.put(
                    f"/collections/{self.collection_name}",
                    json=create_payload,
                )
                if create_res.status_code not in (200, 201):
                    raise VectorStoreError(
                        f"Failed to create collection '{self.collection_name}': {create_res.text}"
                    )

                # Create payload indexes for authorization and filtering
                for field in [
                    "tenant_id",
                    "document_id",
                    "classification",
                    "page_number",
                    "redaction_version",
                ]:
                    schema_type = (
                        "integer" if field in ("page_number", "redaction_version") else "keyword"
                    )

                    client.put(
                        f"/collections/{self.collection_name}/index",
                        json={"field_name": field, "field_schema": schema_type},
                    )
            elif res.status_code == 200:
                data = res.json().get("result", {})
                config = data.get("config", {}).get("params", {})
                vectors_cfg = config.get("vectors", {})
                if isinstance(vectors_cfg, dict) and "text_dense" in vectors_cfg:
                    configured_dim = vectors_cfg["text_dense"].get("size")
                    if configured_dim != dense_dim:
                        raise VectorStoreError(
                            f"Collection '{self.collection_name}' has text_dense dim "
                            f"{configured_dim}, expected {dense_dim}"
                        )

            else:
                raise VectorStoreError(
                    f"Unexpected Qdrant response inspecting collection: {res.text}"
                )

    def upsert_chunks(self, chunks: list[VectorStoreChunk]) -> int:
        if not chunks:
            return 0

        points: list[dict[str, Any]] = []
        for chunk in chunks:
            # Deterministic UUID from chunk_id
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            point: dict[str, Any] = {
                "id": point_id,
                "vector": {
                    "text_dense": chunk.dense_vector,
                },
                "payload": chunk.to_payload(),
            }
            if chunk.sparse_indices and chunk.sparse_values:
                point["vector"]["text_sparse"] = {
                    "indices": chunk.sparse_indices,
                    "values": chunk.sparse_values,
                }
            points.append(point)

        with self._client() as client:
            res = client.put(
                f"/collections/{self.collection_name}/points",
                params={"wait": "true"},
                json={"points": points},
            )
            if res.status_code != 200:
                raise VectorStoreError(f"Failed to upsert points into Qdrant: {res.text}")
        return len(points)

    def delete_document_chunks(self, document_id: str) -> int:
        """Removes all chunks for a document to ensure idempotent re-indexing."""
        with self._client() as client:
            res = client.post(
                f"/collections/{self.collection_name}/points/delete",
                params={"wait": "true"},
                json={
                    "filter": {
                        "must": [
                            {"key": "document_id", "match": {"value": document_id}},
                        ]
                    }
                },
            )
            if res.status_code not in (200, 404):
                raise VectorStoreError(f"Failed to delete document chunks from Qdrant: {res.text}")
        return 1

    @staticmethod
    def _build_filter(
        tenant_id: str,
        permitted_document_ids: list[str] | None = None,
        classification: str | None = None,
    ) -> dict[str, Any]:
        must_clauses: list[dict[str, Any]] = [
            {"key": "tenant_id", "match": {"value": tenant_id}},
        ]
        if permitted_document_ids is not None:
            if not permitted_document_ids:
                # If permitted_document_ids is an empty list, no documents are accessible!
                must_clauses.append({"key": "document_id", "match": {"value": "__NO_ACCESS__"}})
            elif len(permitted_document_ids) == 1:
                must_clauses.append(
                    {"key": "document_id", "match": {"value": permitted_document_ids[0]}}
                )
            else:
                must_clauses.append(
                    {"key": "document_id", "match": {"any": permitted_document_ids}}
                )
        if classification:
            must_clauses.append({"key": "classification", "match": {"value": classification}})
        return {"must": must_clauses}

    def search_dense(
        self,
        query_vector: list[float],
        tenant_id: str,
        permitted_document_ids: list[str] | None = None,
        classification: str | None = None,
        limit: int = 10,
    ) -> list[VectorStoreSearchResult]:
        q_filter = self._build_filter(tenant_id, permitted_document_ids, classification)
        payload = {
            "vector": {
                "name": "text_dense",
                "vector": query_vector,
            },
            "filter": q_filter,
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
        }
        with self._client() as client:
            res = client.post(
                f"/collections/{self.collection_name}/points/search",
                json=payload,
            )
            if res.status_code != 200:
                raise VectorStoreError(f"Dense search failed on Qdrant: {res.text}")
            hits = res.json().get("result", [])
            return [
                VectorStoreSearchResult(
                    point_id=str(hit.get("id")),
                    score=float(hit.get("score", 0.0)),
                    payload=hit.get("payload", {}),
                )
                for hit in hits
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
        if not indices or not values:
            return []

        q_filter = self._build_filter(tenant_id, permitted_document_ids, classification)
        payload = {
            "vector": {
                "name": "text_sparse",
                "vector": {
                    "indices": indices,
                    "values": values,
                },
            },
            "filter": q_filter,
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
        }
        with self._client() as client:
            res = client.post(
                f"/collections/{self.collection_name}/points/search",
                json=payload,
            )
            if res.status_code != 200:
                raise VectorStoreError(f"Sparse search failed on Qdrant: {res.text}")
            hits = res.json().get("result", [])
            return [
                VectorStoreSearchResult(
                    point_id=str(hit.get("id")),
                    score=float(hit.get("score", 0.0)),
                    payload=hit.get("payload", {}),
                )
                for hit in hits
            ]

    def count_points(self, tenant_id: str | None = None) -> int:
        with self._client() as client:
            count_payload: dict[str, Any] = {"exact": True}
            if tenant_id:
                count_payload["filter"] = {
                    "must": [{"key": "tenant_id", "match": {"value": tenant_id}}]
                }
            res = client.post(
                f"/collections/{self.collection_name}/points/count",
                json=count_payload,
            )
            if res.status_code != 200:
                return 0
            return int(res.json().get("result", {}).get("count", 0))
