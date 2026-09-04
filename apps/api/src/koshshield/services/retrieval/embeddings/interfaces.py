from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class EmbeddingResult:
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


class ModelUnavailableError(RuntimeError):
    """Raised when the requested embedding model is unconfigured or unavailable."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def dense_dim(self) -> int:
        """Dimensionality of the dense vector."""
        ...

    def is_available(self) -> tuple[bool, str]:
        """Truthfully report whether the local model is loaded/loadable without network access."""
        ...

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate dense and sparse embeddings for a batch of text chunks."""
        ...

    def embed_query(self, query: str) -> EmbeddingResult:
        """Generate dense and sparse embeddings for a single search query."""
        ...
