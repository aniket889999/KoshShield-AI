"""Read and snapshot manifest-listed synthetic fixtures without executing their contents."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError, field_validator

from koshshield.runtime_artifacts import ArtifactCheckError, open_artifact, read_artifact_json

MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_BUNDLE_BYTES = 64 * 1024 * 1024


class FixtureValidationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class FixtureFile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str = Field(min_length=1, max_length=256)
    role: Literal["input", "evaluation_only"]
    bytes: int = Field(gt=0, le=MAX_FILE_BYTES)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("path")
    @classmethod
    def relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or "\\" in value
            or any(p in {"", ".", ".."} for p in value.split("/"))
        ):
            raise ValueError("Unsafe fixture path")
        return value


class FixtureManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    case_id: str = Field(pattern=r"^[A-Z0-9-]{1,64}$")
    synthetic: StrictBool
    files: list[FixtureFile] = Field(min_length=1, max_length=64)


@dataclass(frozen=True)
class FixtureBundle:
    case_id: str
    files: Mapping[str, bytes] = field(repr=False)
    roles: Mapping[str, str]

    def input_pdf(self, name: str) -> bytes:
        if PurePosixPath(name).name != name or not name.endswith(".pdf"):
            raise FixtureValidationError("FIXTURE_INPUT_FORBIDDEN")
        path = f"inputs/{name}"
        if self.roles.get(path) != "input":
            raise FixtureValidationError("FIXTURE_INPUT_FORBIDDEN")
        content = self.files[path]
        if not content.startswith(b"%PDF-"):
            raise FixtureValidationError("FIXTURE_PDF_INVALID")
        return content

    def evaluation_json(self, path: str) -> dict:
        if self.roles.get(path) != "evaluation_only":
            raise FixtureValidationError("FIXTURE_EVALUATION_FILE_MISSING")
        try:
            value = json.loads(self.files[path])
        except (ValueError, UnicodeError, RecursionError):
            raise FixtureValidationError("FIXTURE_JSON_INVALID") from None
        if not isinstance(value, dict):
            raise FixtureValidationError("FIXTURE_JSON_INVALID")
        return value


def load_fixture_bundle(root: Path) -> FixtureBundle:
    """Hashes establish consistency with the local manifest, not publisher authenticity."""
    try:
        manifest = FixtureManifest.model_validate(
            read_artifact_json(root / "manifest.json", max_bytes=256 * 1024)
        )
        if not manifest.synthetic:
            raise FixtureValidationError("FIXTURE_NOT_SYNTHETIC")
        names = [item.path for item in manifest.files]
        if len(set(names)) != len(names):
            raise FixtureValidationError("FIXTURE_DUPLICATE_PATH")
        if sum(item.bytes for item in manifest.files) > MAX_BUNDLE_BYTES:
            raise FixtureValidationError("FIXTURE_BUNDLE_TOO_LARGE")
        files = {}
        roles = {}
        for item in manifest.files:
            path = root / item.path
            if not path.resolve().is_relative_to(root.resolve()):
                raise FixtureValidationError("FIXTURE_PATH_UNSAFE")
            with open_artifact(path) as stream:
                content = stream.read(item.bytes + 1)
            if len(content) != item.bytes or hashlib.sha256(content).hexdigest() != item.sha256:
                raise FixtureValidationError("FIXTURE_INTEGRITY_MISMATCH")
            files[item.path] = content
            roles[item.path] = item.role
        return FixtureBundle(manifest.case_id, MappingProxyType(files), MappingProxyType(roles))
    except (ArtifactCheckError, OSError):
        raise FixtureValidationError("FIXTURE_UNREADABLE") from None
    except ValidationError:
        raise FixtureValidationError("FIXTURE_MANIFEST_INVALID") from None
