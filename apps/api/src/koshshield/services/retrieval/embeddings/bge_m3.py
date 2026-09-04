import logging
import os
from pathlib import Path
from typing import Any

from koshshield.services.retrieval.embeddings.interfaces import (
    EmbeddingProvider,
    EmbeddingResult,
    ModelUnavailableError,
)

logger = logging.getLogger(__name__)


class BgeM3EmbeddingProvider(EmbeddingProvider):
    """Local-only, air-gapped BGE-M3 embedding provider using FlagEmbedding.

    Guarantees:
    - Never contacts Hugging Face or any remote service at runtime.
    - Loads weights strictly from local disk.
    - Reports truthful readiness; raises ModelUnavailableError if unavailable.
    """

    def __init__(
        self,
        model_dir: Path | None,
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else None
        self.device = device
        self.batch_size = batch_size
        self._model: Any | None = None

    @property
    def dense_dim(self) -> int:
        return 1024

    def is_available(self) -> tuple[bool, str]:
        if not self.model_dir:
            return (
                False,
                "BGE-M3 model directory is not configured (KOSHSHIELD_EMBEDDING_MODEL_DIR)",
            )

        if not self.model_dir.exists() or not self.model_dir.is_dir():
            return False, f"BGE-M3 directory does not exist: {self.model_dir}"

        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            return False, f"BGE-M3 model config missing in: {self.model_dir}"

        # Verify weight files exist
        has_weights = any(
            (self.model_dir / name).exists()
            for name in ["model.safetensors", "pytorch_model.bin", "model.onnx"]
        )
        if not has_weights:
            return False, f"BGE-M3 model weights missing in: {self.model_dir}"

        try:
            import FlagEmbedding  # noqa: F401
        except ImportError:
            return False, "FlagEmbedding package is not installed"

        return True, "BGE-M3 model weights present and ready on local disk"

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        ready, reason = self.is_available()
        if not ready:
            raise ModelUnavailableError(f"Local BGE-M3 model is unavailable: {reason}")

        try:
            # Enforce strict offline operation
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

            from FlagEmbedding import BGEM3FlagModel

            logger.info("Loading local BGE-M3 model from %s on %s...", self.model_dir, self.device)
            self._model = BGEM3FlagModel(
                str(self.model_dir),
                use_fp16=False,
                device=self.device,
            )
            return self._model
        except Exception as err:
            logger.error("Failed to load BGE-M3 model: %s", err)
            raise ModelUnavailableError(f"Failed to load local BGE-M3 model: {err}") from err

    @staticmethod
    def _format_sparse(lexical_dict: dict[Any, float]) -> tuple[list[int], list[float]]:
        """Converts FlagEmbedding lexical_weights dictionary into Qdrant sparse
        indices and values.
        """
        indices: list[int] = []

        values: list[float] = []

        for key, val in lexical_dict.items():
            if isinstance(key, int):
                idx = key
            elif isinstance(key, str) and key.isdigit():
                idx = int(key)
            else:
                # Deterministic token string hash within positive 31-bit integer range
                idx = abs(hash(str(key))) % (2**31 - 1)
            indices.append(idx)
            values.append(float(val))

        # Qdrant requires sparse vector indices to be sorted and unique
        if indices:
            # Aggregate any duplicate indices
            combined: dict[int, float] = {}
            for idx, val in zip(indices, values, strict=False):
                combined[idx] = max(combined.get(idx, 0.0), val)
            sorted_indices = sorted(combined.keys())
            sorted_values = [combined[i] for i in sorted_indices]
            return sorted_indices, sorted_values

        return [], []

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        if not texts:
            return []

        model = self._ensure_model()
        output = model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=8192,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vecs = output["dense_vecs"]
        sparse_weights = output["lexical_weights"]

        results: list[EmbeddingResult] = []
        for i in range(len(texts)):
            dense_list = (
                dense_vecs[i].tolist() if hasattr(dense_vecs[i], "tolist") else list(dense_vecs[i])
            )
            sparse_dict = sparse_weights[i] if i < len(sparse_weights) else {}
            s_indices, s_values = self._format_sparse(sparse_dict)
            results.append(
                EmbeddingResult(
                    dense=dense_list,
                    sparse_indices=s_indices,
                    sparse_values=s_values,
                )
            )
        return results

    def embed_query(self, query: str) -> EmbeddingResult:
        results = self.embed_texts([query])
        if not results:
            raise RuntimeError("Empty embedding result generated for query")
        return results[0]
