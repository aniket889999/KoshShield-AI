"""Bounded, read-only artifact inspection. Never deserialize model weights."""

from __future__ import annotations

import errno
import importlib.metadata
import importlib.util
import json
import os
import stat
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


class ArtifactCheckError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@contextmanager
def open_artifact(path: Path) -> Iterator[BinaryIO]:
    """Do not follow final symlinks or block on special files such as FIFOs."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ArtifactCheckError("ARTIFACT_NOT_REGULAR")
            if info.st_size == 0:
                raise ArtifactCheckError("ARTIFACT_EMPTY")
            yield stream
    except FileNotFoundError:
        raise ArtifactCheckError("ARTIFACT_MISSING") from None
    except OSError as exc:
        code = "ARTIFACT_SYMLINK" if exc.errno == errno.ELOOP else "ARTIFACT_UNREADABLE"
        raise ArtifactCheckError(code) from None


def inspect_gguf(path: Path) -> dict[str, int]:
    """Check only the fixed little-endian v2/v3 header, not tensor contents or identity."""
    with open_artifact(path) as stream:
        info = os.fstat(stream.fileno())
        header = stream.read(24)
    if len(header) != 24 or info.st_size <= 24:
        raise ArtifactCheckError("GGUF_HEADER_TRUNCATED")
    magic, version, tensors, metadata = struct.unpack("<4sIQQ", header)
    if magic != b"GGUF":
        raise ArtifactCheckError("GGUF_MAGIC_INVALID")
    if version not in (2, 3):
        raise ArtifactCheckError("GGUF_VERSION_UNSUPPORTED")
    # Even empty names/values require these minimum descriptor sizes.
    if tensors == 0 or metadata == 0 or 24 + tensors * 24 + metadata * 13 > info.st_size:
        raise ArtifactCheckError("GGUF_COUNTS_INVALID")
    return {
        "version": version,
        "tensor_count": tensors,
        "metadata_count": metadata,
        "size_bytes": info.st_size,
        "device": info.st_dev,
        "inode": info.st_ino,
    }


def read_artifact_json(path: Path, max_bytes: int = 1024 * 1024) -> dict:
    with open_artifact(path) as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ArtifactCheckError("ARTIFACT_JSON_TOO_LARGE")
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError, RecursionError):
        raise ArtifactCheckError("ARTIFACT_JSON_INVALID") from None
    if not isinstance(value, dict):
        raise ArtifactCheckError("ARTIFACT_JSON_INVALID")
    return value


def embedding_dimension(model_dir: Path) -> int:
    config = read_artifact_json(model_dir / "config.json")
    dimension = config.get("hidden_size")
    if type(dimension) is not int or not 0 < dimension <= 65536:
        raise ArtifactCheckError("EMBEDDING_DIMENSION_INVALID")
    return dimension


def inspect_bge_bundle(model_dir: Path) -> dict[str, int | str | bool]:
    """Inspect the supported unsharded FlagEmbedding layout without loading pickle files."""
    dimension = embedding_dimension(model_dir)
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        read_artifact_json(model_dir / name)
    tokenizer = read_artifact_json(model_dir / "tokenizer.json", max_bytes=32 * 1024 * 1024)
    if not isinstance(tokenizer.get("model"), dict) or not tokenizer["model"].get("vocab"):
        raise ArtifactCheckError("TOKENIZER_STRUCTURE_INVALID")

    weight_name = "model.safetensors"
    if not (model_dir / weight_name).exists():
        weight_name = "pytorch_model.bin"
    # FlagEmbedding initializes BOTH heads randomly when either trained head is absent.
    required = (weight_name, "sentencepiece.bpe.model", "sparse_linear.pt", "colbert_linear.pt")
    for name in required:
        with open_artifact(model_dir / name):
            pass
    return {
        "dense_dimension": dimension,
        "weight_format": "safetensors" if weight_name.endswith("safetensors") else "pytorch",
        "validation_scope": "bundle_structure_only",
        "integrity_verified": False,
        "model_loaded": False,
    }


def inspect_ocr_bundle(model_dir: Path) -> str:
    """Check matching PaddleOCR 2.x inference program/parameter files."""
    for prefix in ("model", "inference"):
        program = model_dir / f"{prefix}.pdmodel"
        parameters = model_dir / f"{prefix}.pdiparams"
        if program.exists() and parameters.exists():
            with open_artifact(program), open_artifact(parameters):
                return prefix
    raise ArtifactCheckError("OCR_INFERENCE_FILES_MISSING")


def inspect_ocr_runtime() -> str:
    """Inspect installed metadata without importing or initializing PaddleOCR."""
    if any(importlib.util.find_spec(name) is None for name in ("paddleocr", "paddle")):
        raise ArtifactCheckError("OCR_DEPENDENCY_MISSING")
    try:
        version = importlib.metadata.version("paddleocr")
    except importlib.metadata.PackageNotFoundError:
        raise ArtifactCheckError("OCR_DEPENDENCY_MISSING") from None
    if version.split(".", 1)[0] != "2":
        raise ArtifactCheckError("OCR_API_VERSION_UNSUPPORTED")
    return version
