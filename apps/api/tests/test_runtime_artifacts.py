import os
import struct
from pathlib import Path

import pytest

from koshshield.config import Settings
from koshshield.runtime_artifacts import ArtifactCheckError, inspect_gguf
from koshshield.runtime_preflight import PrerequisiteStatus, check_qwen_gguf


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
