"""Offline inspection of operator-supplied Python wheels. Never extracts or imports them."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

MAX_METADATA_BYTES = 1024 * 1024


class DependencyPreparationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class WheelInfo:
    filename: str
    name: str
    version: str
    size_bytes: int
    sha256: str
    requires_python: str
    requirements: tuple[str, ...]
    tags: frozenset[str]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_wheel(path: Path, wheelhouse: Path) -> WheelInfo:
    root = wheelhouse.resolve()
    if (
        path.is_symlink()
        or not path.resolve().is_relative_to(root)
        or not path.is_file()
    ):
        raise DependencyPreparationError("WHEEL_PATH_UNSAFE")
    try:
        name, version, _, tags = parse_wheel_filename(path.name)
        with ZipFile(path) as archive:
            entries = archive.infolist()
            metadata = [
                entry
                for entry in entries
                if entry.filename.count("/") == 1
                and entry.filename.endswith(".dist-info/METADATA")
            ]
            if len(metadata) != 1 or metadata[0].file_size > MAX_METADATA_BYTES:
                raise DependencyPreparationError("WHEEL_METADATA_INVALID")
            message = BytesParser(policy=policy.default).parsebytes(
                archive.read(metadata[0])
            )
            if (
                len(message.get_all("Name", [])) != 1
                or len(message.get_all("Version", [])) != 1
            ):
                raise DependencyPreparationError("WHEEL_METADATA_INVALID")
            if (
                canonicalize_name(str(message["Name"])) != name
                or Version(str(message["Version"])) != version
            ):
                raise DependencyPreparationError("WHEEL_IDENTITY_MISMATCH")
            return WheelInfo(
                filename=path.name,
                name=name,
                version=str(version),
                size_bytes=path.stat().st_size,
                sha256=file_sha256(path),
                requires_python=str(message.get("Requires-Python", "")),
                requirements=tuple(
                    str(item) for item in message.get_all("Requires-Dist", [])
                ),
                tags=frozenset(str(tag) for tag in tags),
            )
    except DependencyPreparationError:
        raise
    except (ValueError, OSError, BadZipFile, KeyError, RuntimeError) as exc:
        raise DependencyPreparationError("WHEEL_METADATA_INVALID") from exc


def inventory_wheels(wheelhouse: Path) -> list[WheelInfo]:
    if not wheelhouse.is_dir():
        raise DependencyPreparationError("WHEELHOUSE_MISSING")
    paths = sorted(wheelhouse.glob("*.whl"))
    if not paths:
        raise DependencyPreparationError("WHEELHOUSE_EMPTY")
    return [inspect_wheel(path, wheelhouse) for path in paths]
