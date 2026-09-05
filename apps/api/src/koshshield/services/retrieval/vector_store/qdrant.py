import logging
from urllib.parse import urlparse

from qdrant_client import QdrantClient, models

from koshshield.services.retrieval.vector_store.interfaces import (
    VectorStore,
    VectorStoreChunk,
    VectorStoreError,
    VectorStoreSearchResult,
    VectorStoreUnavailableError,
)

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStore):
    """Local-first Qdrant vector store adapter using the official qdrant-client.

    Guarantees:
    - Strictly connects to local loopback or local container URLs.
    - Uses REST transport (prefer_grpc=False) for MVP monolith stability.
    - Enforces typed Qdrant models for all queries, points, filters, and schemas.
    - Enforces mandatory tenant isolation on all operations.
    - Never silently recreates or deletes incompatible existing collections.
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "koshshield_masked_docs",
        timeout_seconds: float = 10.0,
        client: QdrantClient | None = None,
    ) -> None:
        self.qdrant_url = qdrant_url.rstrip("/")
        self.collection_name = collection_name
        self.timeout = timeout_seconds
        self._validate_local_url(self.qdrant_url)

        if client is not None:
            self._client = client
        else:
            self._client = QdrantClient(
                url=self.qdrant_url,
                timeout=self.timeout,
                prefer_grpc=False,
            )

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

    def is_available(self) -> tuple[bool, str]:
        try:
            self._client.get_collections()
            return True, "Qdrant vector store is ready"
        except Exception as err:
            return False, f"Qdrant is unreachable at {self.qdrant_url}: {err}"

    def ensure_collection(self, dense_dim: int) -> None:
        """Create or validate the Qdrant collection with named dense and sparse vectors."""
        try:
            exists = self._client.collection_exists(self.collection_name)
        except Exception as err:
            raise VectorStoreUnavailableError(
                f"Cannot reach Qdrant at {self.qdrant_url}: {err}"
            ) from err

        if not exists:
            logger.info(
                "Creating Qdrant collection '%s' with text_dense (%d dims) and text_sparse...",
                self.collection_name,
                dense_dim,
            )
            try:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "text_dense": models.VectorParams(
                            size=dense_dim,
                            distance=models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        "text_sparse": models.SparseVectorParams(),
                    },
                )
            except Exception as err:
                raise VectorStoreError(
                    f"Failed to create collection '{self.collection_name}': {err}"
                ) from err

            # Create payload indexes for authorization and filtering
            for field, schema_type in [
                ("tenant_id", models.PayloadSchemaType.KEYWORD),
                ("document_id", models.PayloadSchemaType.KEYWORD),
                ("classification", models.PayloadSchemaType.KEYWORD),
                ("page_number", models.PayloadSchemaType.INTEGER),
                ("redaction_version", models.PayloadSchemaType.INTEGER),
                ("index_version", models.PayloadSchemaType.INTEGER),
            ]:
                try:
                    self._client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=schema_type,
                    )
                except Exception as err:
                    logger.warning("Failed to create index for field '%s': %s", field, err)
        else:
            try:
                info = self._client.get_collection(self.collection_name)
            except Exception as err:
                raise VectorStoreError(f"Failed to inspect collection info: {err}") from err

            params = info.config.params
            vectors = params.vectors

            # Validate named dense vector
            dense_cfg = None
            if isinstance(vectors, dict) and "text_dense" in vectors:
                dense_cfg = vectors["text_dense"]
            elif hasattr(vectors, "get") and vectors.get("text_dense"):
                dense_cfg = vectors.get("text_dense")

            if not dense_cfg:
                raise VectorStoreError(
                    f"Collection '{self.collection_name}' is missing required named vector "
                    "'text_dense'. Silently recreating or deleting collection is prohibited."
                )

            configured_dim = getattr(dense_cfg, "size", None) or (
                dense_cfg.get("size") if isinstance(dense_cfg, dict) else None
            )
            if configured_dim != dense_dim:
                raise VectorStoreError(
                    f"Collection '{self.collection_name}' has text_dense dim {configured_dim}, "
                    f"expected {dense_dim}. Silently recreating collection is prohibited."
                )

            distance = getattr(dense_cfg, "distance", None) or (
                dense_cfg.get("distance") if isinstance(dense_cfg, dict) else None
            )
            if distance not in (models.Distance.COSINE, "Cosine", "cosine"):
                raise VectorStoreError(
                    f"Collection '{self.collection_name}' text_dense distance is {distance}, "
                    "expected Cosine. Silently recreating collection is prohibited."
                )

            # Validate named sparse vector
            sparse_cfg = params.sparse_vectors
            if not sparse_cfg or "text_sparse" not in (
                sparse_cfg if isinstance(sparse_cfg, dict) else getattr(sparse_cfg, "__dict__", {})
            ):
                raise VectorStoreError(
                    f"Collection '{self.collection_name}' is missing sparse vector 'text_sparse'. "
                    "Silently recreating collection is prohibited."
                )

    def upsert_chunks(self, chunks: list[VectorStoreChunk]) -> int:
        if not chunks:
            return 0

        points: list[models.PointStruct] = []
        for chunk in chunks:
            vector_dict: dict[str, models.Vector | models.SparseVector] = {
                "text_dense": chunk.dense_vector,
            }
            if chunk.sparse_indices and chunk.sparse_values:
                vector_dict["text_sparse"] = models.SparseVector(
                    indices=chunk.sparse_indices,
                    values=chunk.sparse_values,
                )

            points.append(
                models.PointStruct(
                    id=chunk.point_id,
                    vector=vector_dict,
                    payload=chunk.to_payload(),
                )
            )

        try:
            self._client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            return len(points)
        except Exception as err:
            raise VectorStoreError(f"Failed to upsert points into Qdrant: {err}") from err

    def verify_points(self, point_ids: list[str], tenant_id: str) -> bool:
        """Verify that all expected points exist and strictly belong to tenant_id."""
        if not point_ids:
            return True
        try:
            records = self._client.retrieve(
                collection_name=self.collection_name,
                ids=point_ids,
                with_payload=True,
            )
            if len(records) != len(point_ids):
                return False
            for r in records:
                payload = r.payload or {}
                if payload.get("tenant_id") != tenant_id:
                    return False
            return True
        except Exception as err:
            logger.error("Failed to verify points in Qdrant: %s", err)
            return False

    def delete_stale_chunks(self, document_id: str, tenant_id: str, active_version: int) -> int:
        """Remove points for a document with index_version < active_version,
        strictly scoped to tenant_id.
        """
        delete_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                ),
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id),
                ),
                models.FieldCondition(
                    key="index_version",
                    range=models.Range(lt=active_version),
                ),
            ]
        )
        try:
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=delete_filter),
                wait=True,
            )
            return 1
        except Exception as err:
            raise VectorStoreError(f"Failed to delete stale chunks from Qdrant: {err}") from err

    def delete_document_chunks(self, document_id: str, tenant_id: str) -> int:
        """Removes all chunks for a document strictly scoped to tenant_id."""
        delete_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                ),
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id),
                ),
            ]
        )
        try:
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=delete_filter),
                wait=True,
            )
            return 1
        except Exception as err:
            raise VectorStoreError(f"Failed to delete document chunks from Qdrant: {err}") from err

    def retrieve_points(
        self, point_ids: list[str], tenant_id: str
    ) -> list[VectorStoreSearchResult]:
        """Retrieve points by ID, enforcing mandatory tenant isolation."""
        if not point_ids:
            return []
        try:
            records = self._client.retrieve(
                collection_name=self.collection_name,
                ids=point_ids,
                with_payload=True,
            )
            return [
                VectorStoreSearchResult(
                    point_id=str(r.id),
                    score=1.0,
                    payload=r.payload or {},
                )
                for r in records
                if (r.payload or {}).get("tenant_id") == tenant_id
            ]
        except Exception as err:
            raise VectorStoreError(f"Failed to retrieve points: {err}") from err

    @staticmethod
    def _build_filter(
        tenant_id: str,
        permitted_document_ids: list[str] | None = None,
        classification: str | None = None,
    ) -> models.Filter:
        must_clauses: list[models.Condition] = [
            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
        ]
        if permitted_document_ids is not None:
            if not permitted_document_ids:
                # If permitted_document_ids is an empty list, no documents are accessible
                must_clauses.append(
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value="__NO_ACCESS__")
                    )
                )
            elif len(permitted_document_ids) == 1:
                must_clauses.append(
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=permitted_document_ids[0])
                    )
                )
            else:
                must_clauses.append(
                    models.FieldCondition(
                        key="document_id", match=models.MatchAny(any=permitted_document_ids)
                    )
                )
        if classification:
            must_clauses.append(
                models.FieldCondition(
                    key="classification", match=models.MatchValue(value=classification)
                )
            )
        return models.Filter(must=must_clauses)

    def search_dense(
        self,
        query_vector: list[float],
        tenant_id: str,
        permitted_document_ids: list[str] | None = None,
        classification: str | None = None,
        limit: int = 10,
    ) -> list[VectorStoreSearchResult]:
        q_filter = self._build_filter(tenant_id, permitted_document_ids, classification)
        try:
            hits = self._client.search(
                collection_name=self.collection_name,
                query_vector=models.NamedVector(
                    name="text_dense",
                    vector=query_vector,
                ),
                query_filter=q_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            return [
                VectorStoreSearchResult(
                    point_id=str(hit.id),
                    score=float(hit.score),
                    payload=hit.payload or {},
                )
                for hit in hits
            ]
        except Exception as err:
            raise VectorStoreError(f"Dense search failed on Qdrant: {err}") from err

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
        try:
            hits = self._client.search(
                collection_name=self.collection_name,
                query_vector=models.NamedSparseVector(
                    name="text_sparse",
                    vector=models.SparseVector(
                        indices=indices,
                        values=values,
                    ),
                ),
                query_filter=q_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            return [
                VectorStoreSearchResult(
                    point_id=str(hit.id),
                    score=float(hit.score),
                    payload=hit.payload or {},
                )
                for hit in hits
            ]
        except Exception as err:
            raise VectorStoreError(f"Sparse search failed on Qdrant: {err}") from err

    def count_points(self, tenant_id: str | None = None) -> int:
        count_filter = None
        if tenant_id:
            count_filter = models.Filter(
                must=[
                    models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))
                ]
            )
        try:
            res = self._client.count(
                collection_name=self.collection_name,
                count_filter=count_filter,
                exact=True,
            )
            return int(res.count)
        except Exception:
            return 0
