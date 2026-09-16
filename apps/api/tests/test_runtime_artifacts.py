import json
import os
import struct
from pathlib import Path

import pytest

from koshshield.config import Settings
from koshshield.runtime_artifacts import (
    ArtifactCheckError,
    embedding_dimension,
    inspect_bge_bundle,
    inspect_gguf,
    read_artifact_json,
)
from koshshield.runtime_preflight import PrerequisiteStatus, check_qwen_gguf
from koshshield.services.retrieval.embeddings.bge_m3 import BgeM3EmbeddingProvider
from koshshield.services.retrieval.embeddings.interfaces import ModelUnavailableError


def gguf_header(version: int = 3, tensors: int = 1, metadata: int = 1) -> bytes:
    # Deliberately not a loadable model: tests cover header inspection only.
    return struct.pack("<4sIQQ", b"GGUF", version, tensors, metadata) + bytes(128)


@pytest.mark.parametrize("version", [2, 3])
def test_gguf_header_inspection_does_not_claim_integrity(tmp_path: Path, version: int) -> None:
    model, projector = tmp_path / "model.gguf", tmp_path / "projector.gguf"
    model.write_bytes(gguf_header(version))
    projector.write_bytes(gguf_header(version))
    result = check_qwen_gguf(
        Settings(_env_file=None, llama_cpp_model_path=model, llama_cpp_mmproj_path=projector)
    )
    assert result.status == PrerequisiteStatus.READY
    assert result.metadata["integrity_verified"] is False
    assert result.metadata["validation_scope"] == "header_only"
    assert result.metadata["model_version"] == version
    assert str(tmp_path) not in str(result)


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (b"", "ARTIFACT_EMPTY"),
        (b"GGUF", "GGUF_HEADER_TRUNCATED"),
        (bytes(128), "GGUF_MAGIC_INVALID"),
        (gguf_header(99), "GGUF_VERSION_UNSUPPORTED"),
        (gguf_header(tensors=0), "GGUF_COUNTS_INVALID"),
        (gguf_header(metadata=2**63), "GGUF_COUNTS_INVALID"),
        (gguf_header()[:24], "GGUF_HEADER_TRUNCATED"),
    ],
)
def test_gguf_rejects_invalid_headers(tmp_path: Path, payload: bytes, code: str) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(payload)
    with pytest.raises(ArtifactCheckError, match=code):
        inspect_gguf(path)


def test_gguf_rejects_missing_symlink_and_fifo_without_reading(tmp_path: Path) -> None:
    path = tmp_path / "missing"
    with pytest.raises(ArtifactCheckError, match="ARTIFACT_MISSING"):
        inspect_gguf(path)
    target = tmp_path / "real"
    target.write_bytes(gguf_header())
    path.symlink_to(target)
    with pytest.raises(ArtifactCheckError, match="ARTIFACT_SYMLINK"):
        inspect_gguf(path)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ArtifactCheckError, match="ARTIFACT_NOT_REGULAR"):
        inspect_gguf(fifo)


def test_gguf_rejects_model_projector_hardlink(tmp_path: Path) -> None:
    model, projector = tmp_path / "model", tmp_path / "projector"
    model.write_bytes(gguf_header())
    os.link(model, projector)
    result = check_qwen_gguf(
        Settings(_env_file=None, llama_cpp_model_path=model, llama_cpp_mmproj_path=projector)
    )
    assert result.status == PrerequisiteStatus.CONTRACT_MISMATCH
    assert result.metadata["failure_code"] == "GGUF_MODEL_PROJECTOR_IDENTICAL"


def bge_bundle(root: Path) -> None:
    (root / "config.json").write_text('{"hidden_size": 1024}')
    (root / "tokenizer_config.json").write_text("{}")
    (root / "special_tokens_map.json").write_text("{}")
    (root / "tokenizer.json").write_text('{"model": {"vocab": [["test", 1.0]]}}')
    for name in (
        "pytorch_model.bin",
        "sentencepiece.bpe.model",
        "sparse_linear.pt",
        "colbert_linear.pt",
    ):
        (root / name).write_bytes(b"synthetic structure fixture; not real weights")


def test_bge_bundle_checks_structure_without_deserializing_weights(tmp_path: Path) -> None:
    bge_bundle(tmp_path)
    result = inspect_bge_bundle(tmp_path)
    assert result["dense_dimension"] == 1024
    assert result["integrity_verified"] is False
    assert result["model_loaded"] is False


@pytest.mark.parametrize("name", ["sparse_linear.pt", "colbert_linear.pt", "tokenizer.json"])
def test_bge_missing_required_component_blocks_runtime(tmp_path: Path, name: str) -> None:
    bge_bundle(tmp_path)
    (tmp_path / name).unlink()
    provider = BgeM3EmbeddingProvider(tmp_path)
    assert provider.is_available()[0] is False
    with pytest.raises(ModelUnavailableError, match="ARTIFACT_MISSING"):
        provider._ensure_model()


@pytest.mark.parametrize("dimension", [True, 0, -1, 1.5, "1024", None, 65537])
def test_bge_dimension_is_a_bounded_positive_integer(tmp_path: Path, dimension: object) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": dimension}))
    with pytest.raises(ArtifactCheckError, match="EMBEDDING_DIMENSION_INVALID"):
        embedding_dimension(tmp_path)
    with pytest.raises(ModelUnavailableError):
        _ = BgeM3EmbeddingProvider(tmp_path).dense_dim


@pytest.mark.parametrize("payload", [b"[]", b"invalid", b"\xff"])
def test_artifact_config_rejects_non_object_json(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "config.json"
    path.write_bytes(payload)
    with pytest.raises(ArtifactCheckError, match="ARTIFACT_JSON_INVALID"):
        read_artifact_json(path)


def test_artifact_json_read_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_bytes(b" " * 1025)
    with pytest.raises(ArtifactCheckError, match="ARTIFACT_JSON_TOO_LARGE"):
        read_artifact_json(path, max_bytes=1024)


def test_bge_onnx_only_is_not_a_flagembedding_bundle(tmp_path: Path) -> None:
    bge_bundle(tmp_path)
    (tmp_path / "pytorch_model.bin").rename(tmp_path / "model.onnx")
    with pytest.raises(ArtifactCheckError, match="ARTIFACT_MISSING"):
        inspect_bge_bundle(tmp_path)
