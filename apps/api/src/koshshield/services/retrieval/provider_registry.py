import threading
from typing import Any

from koshshield.config import Settings, get_settings
from koshshield.services.retrieval.embeddings.bge_m3 import BgeM3EmbeddingProvider
from koshshield.services.retrieval.embeddings.interfaces import (
    EmbeddingProvider,
    EmbeddingResult,
)
from koshshield.services.retrieval.vector_store.interfaces import VectorStore
from koshshield.services.retrieval.vector_store.qdrant import QdrantVectorStore


class ConcurrencyBoundedEmbeddingProvider(BgeM3EmbeddingProvider):
    """Wraps an EmbeddingProvider with a semaphore to constrain concurrent model inference."""

    def __init__(self, target: EmbeddingProvider, max_concurrency: int = 2) -> None:
        self._target = target
        self._semaphore = threading.BoundedSemaphore(value=max(1, max_concurrency))
        if isinstance(target, BgeM3EmbeddingProvider):
            self.device = target.device
            self.batch_size = target.batch_size
            self.model_dir = target.model_dir
            self._init_lock = target._init_lock

    @property
    def _model(self) -> Any:
        return getattr(self._target, "_model", None)

    @_model.setter
    def _model(self, val: Any) -> None:
        if hasattr(self._target, "_model"):
            self._target._model = val

    @property
    def dense_dim(self) -> int:
        return self._target.dense_dim

    def is_available(self) -> tuple[bool, str]:
        return self._target.is_available()

    def _ensure_model(self) -> Any:
        if hasattr(self._target, "_ensure_model"):
            return self._target._ensure_model()
        return None

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        with self._semaphore:
            return self._target.embed_texts(texts)

    def embed_query(self, query: str) -> EmbeddingResult:
        with self._semaphore:
            return self._target.embed_query(query)

    def unwrap(self) -> EmbeddingProvider:
        return self._target


_provider_lock = threading.Lock()
_store_lock = threading.Lock()

_embedding_provider_instance: EmbeddingProvider | None = None
_vector_store_instance: VectorStore | None = None

_embedding_override: EmbeddingProvider | None = None
_vector_store_override: VectorStore | None = None


def get_singleton_embedding_provider(
    settings: Settings | None = None,
    concurrency_limit: int = 2,
) -> EmbeddingProvider:
    """Returns the application-scoped singleton EmbeddingProvider with thread-safe lazy init
    and bounded embedding concurrency.
    """
    global _embedding_provider_instance
    if _embedding_override is not None:
        return _embedding_override

    if _embedding_provider_instance is not None:
        return _embedding_provider_instance

    with _provider_lock:
        if _embedding_override is not None:
            return _embedding_override
        if _embedding_provider_instance is None:
            cfg = settings or get_settings()
            base_provider = BgeM3EmbeddingProvider(
                model_dir=cfg.embedding_model_dir,
                device=cfg.embedding_device,
                batch_size=cfg.embedding_batch_size,
            )
            _embedding_provider_instance = ConcurrencyBoundedEmbeddingProvider(
                target=base_provider,
                max_concurrency=concurrency_limit,
            )
        return _embedding_provider_instance


def get_singleton_vector_store(settings: Settings | None = None) -> VectorStore:
    """Returns the application-scoped singleton VectorStore with thread-safe lazy init."""
    global _vector_store_instance
    if _vector_store_override is not None:
        return _vector_store_override

    if _vector_store_instance is not None:
        return _vector_store_instance

    with _store_lock:
        if _vector_store_override is not None:
            return _vector_store_override
        if _vector_store_instance is None:
            cfg = settings or get_settings()
            _vector_store_instance = QdrantVectorStore(
                qdrant_url=cfg.qdrant_url,
                collection_name=cfg.qdrant_collection,
            )
        return _vector_store_instance


def set_embedding_provider_override(provider: EmbeddingProvider | None) -> None:
    """Explicitly inject an embedding provider for tests."""
    global _embedding_override
    _embedding_override = provider


def set_vector_store_override(store: VectorStore | None) -> None:
    """Explicitly inject a vector store for tests."""
    global _vector_store_override
    _vector_store_override = store


def reset_provider_registry() -> None:
    """Resets all singleton instances and test overrides."""
    global _embedding_provider_instance, _vector_store_instance
    global _embedding_override, _vector_store_override
    with _provider_lock:
        _embedding_provider_instance = None
        _embedding_override = None
    with _store_lock:
        _vector_store_instance = None
        _vector_store_override = None
