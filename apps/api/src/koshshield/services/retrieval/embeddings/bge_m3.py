import json
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
    - Never contacts Hugging Face or remote services at runtime (HF_HUB_OFFLINE=1).
    - Strictly rejects URLs or remote Hugging Face repository IDs.
    - Dynamically obtains and verifies dimensions from local config and real model output.
    - Reports truthful readiness; raises ModelUnavailableError if unavailable.
    """

    def __init__(
        self,
        model_dir: Path | str | None,
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        # Enforce offline flags immediately upon instantiation
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

        self.device = device
        self.batch_size = batch_size
        self._model: Any | None = None
        self._configured_dim: int | None = None
        self._validated_dim: int | None = None

        if model_dir:
            model_str = str(model_dir).strip()
            # Explicitly reject URLs or remote HF repo IDs
            is_url = model_str.startswith(("http://", "https://", "ftp://"))
            is_remote_repo = (
                not os.path.exists(model_str)
                and "/" in model_str
                and not model_str.startswith(("/", "."))
            )
            if is_url or is_remote_repo:
                raise ValueError(
                    f"Remote model identifiers and URLs are prohibited in air-gapped mode: "
                    f"'{model_str}'. Must provide a valid local directory path."
                )
            self.model_dir: Path | None = Path(model_str)
            self._read_config_dim()
        else:
            self.model_dir = None

    def _read_config_dim(self) -> None:
        if not self.model_dir or not self.model_dir.is_dir():
            return
        config_path = self.model_dir / "config.json"
        if config_path.is_file():
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
                dim = data.get("hidden_size") or data.get("dim") or data.get("d_model")
                if isinstance(dim, int):
                    self._configured_dim = dim
            except Exception as err:
                logger.warning("Could not parse BGE-M3 config.json: %s", err)

    @property
    def dense_dim(self) -> int:
        return self._validated_dim or self._configured_dim or 1024

    def is_available(self) -> tuple[bool, str]:
        if not self.model_dir:
            return (
                False,
                "BGE-M3 model directory is not configured (KOSHSHIELD_EMBEDDING_MODEL_DIR)",
            )

        if not self.model_dir.exists() or not self.model_dir.is_dir():
            return False, f"BGE-M3 directory does not exist on local disk: {self.model_dir}"

        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            return False, f"BGE-M3 model config missing in: {self.model_dir}"

        # Verify local weight files exist
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
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

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
        indices: list[int] = []
        values: list[float] = []

        for key, val in lexical_dict.items():
            if isinstance(key, int):
                idx = key
            elif isinstance(key, str) and key.isdigit():
                idx = int(key)
            else:
                idx = abs(hash(str(key))) % (2**31 - 1)
            indices.append(idx)
            values.append(float(val))

        if indices:
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

            # Tri-point verification: validate actual output vector dimension against configured dim
            actual_dim = len(dense_list)
            if self._configured_dim is not None and actual_dim != self._configured_dim:
                raise ModelUnavailableError(
                    f"Embedding dimension mismatch: local config declared {self._configured_dim}, "
                    f"but model produced {actual_dim} dimensions."
                )
            self._validated_dim = actual_dim

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
