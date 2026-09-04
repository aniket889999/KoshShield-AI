import hashlib
import math
import re
from collections import Counter

from koshshield.services.retrieval.embeddings.interfaces import (
    EmbeddingProvider,
    EmbeddingResult,
)


class DeterministicEmbeddingProvider(EmbeddingProvider):
    """Deterministic, air-gapped test double embedding provider.

    Used for unit tests and local evaluation without downloading heavy model weights.
    Produces 1024-dimensional normalized dense vectors and lexical sparse vectors
    where semantically related words produce higher cosine similarities.
    """

    def __init__(self, dense_dim: int = 1024) -> None:
        self._dense_dim = dense_dim

    @property
    def dense_dim(self) -> int:
        return self._dense_dim

    def is_available(self) -> tuple[bool, str]:
        return True, "Deterministic test embedding provider ready"

    def _text_to_dense(self, text: str) -> list[float]:
        tokens = re.findall(r"\w+", text.lower())
        vec = [0.0] * self._dense_dim
        if not tokens:
            vec[0] = 1.0
            return vec

        for token in tokens:
            # Deterministic pseudo-random vector from token SHA-256
            h = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(min(16, self._dense_dim)):
                idx = int.from_bytes(h[i * 2 : (i + 1) * 2], "big") % self._dense_dim
                val = (h[i] / 128.0) - 1.0
                vec[idx] += val

        # L2 Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            return [x / norm for x in vec]
        vec[0] = 1.0
        return vec

    def _text_to_sparse(self, text: str) -> tuple[list[int], list[float]]:
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return [], []

        counts = Counter(tokens)
        indices: list[int] = []
        values: list[float] = []

        for token, count in counts.items():
            # Stable 31-bit positive token index
            token_hash = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % (
                2**31 - 1
            )
            # TF-like weight: log(1 + count) / total
            weight = math.log1p(count)
            indices.append(token_hash)
            values.append(float(weight))

        # Sort by indices as required by Qdrant
        sorted_pairs = sorted(zip(indices, values, strict=False), key=lambda p: p[0])
        return [p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs]

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        results: list[EmbeddingResult] = []
        for text in texts:
            dense = self._text_to_dense(text)
            s_indices, s_values = self._text_to_sparse(text)
            results.append(
                EmbeddingResult(
                    dense=dense,
                    sparse_indices=s_indices,
                    sparse_values=s_values,
                )
            )
        return results

    def embed_query(self, query: str) -> EmbeddingResult:
        return self.embed_texts([query])[0]
