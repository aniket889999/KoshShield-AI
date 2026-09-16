"""Bounded, read-only artifact inspection. Never deserialize model weights."""

from __future__ import annotations

import errno
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
