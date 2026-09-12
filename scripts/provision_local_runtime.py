"""Reproducible local runtime provisioning guardrails for KoshShield AI.

Reads docs/runtime_artifacts_manifest.json to inventory, validate, and check
capacity for local-first runtime artifacts without downloading or mutating state.

In strict compliance with KoshShield AI local-first policies (AGENTS.md):
- Application startup and inference NEVER trigger downloads.
- Provisioning is an explicit, operator-initiated setup procedure.
- Prerequisite checks fail closed: missing or unverified integrity metadata,
  insufficient disk space, or incomplete dependency locks block provisioning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "docs" / "runtime_artifacts_manifest.json"
DEFAULT_MODEL_DIR = REPO_ROOT / "data" / "models"

GIB_BYTES: int = 1024**3
MIB_BYTES: int = 1024**2
RESERVED_HEADROOM_BYTES: int = 10 * GIB_BYTES

# Estimated additional space requirements for uninstalled components (integer bytes)
ESTIMATED_OVERHEAD_BYTES: dict[str, int] = {
    "dependencies_installed": int(2.5 * GIB_BYTES),
    "temp_extraction_space": int(2.0 * GIB_BYTES),
    "docker_storage_growth": int(1.5 * GIB_BYTES),
    "runtime_caches": int(1.0 * GIB_BYTES),
}

SHA256_HEX_REGEX = re.compile(r"^[0-9a-fA-F]{64}$")
IMAGE_DIGEST_REGEX = re.compile(r"^sha256:[0-9a-fA-F]{64}$")

REQUIRED_RUNTIME_CATEGORIES = {
    "embedding",
    "multimodal_llm",
    "ocr",
    "inference_server",
    "vector_store",
}

REQUIRED_RUNTIME_PACKAGES = {
    "fastapi",
    "pydantic",
    "sqlalchemy",
    "cryptography",
    "httpx",
    "pymupdf",
    "pillow",
    "qdrant-client",
    "flagembedding",
    "torch",
    "paddlepaddle",
    "paddleocr",
}


class BlockerCode(StrEnum):
    CAPACITY_INSUFFICIENT = "CAPACITY_INSUFFICIENT"
    CAPACITY_CHECK_FAILED = "CAPACITY_CHECK_FAILED"
    ARTIFACT_INTEGRITY_UNVERIFIED = "ARTIFACT_INTEGRITY_UNVERIFIED"
    ARTIFACT_INTEGRITY_MISSING = "ARTIFACT_INTEGRITY_MISSING"
    ARTIFACT_INTEGRITY_INVALID = "ARTIFACT_INTEGRITY_INVALID"
    ARTIFACT_CHECKSUM_MISMATCH = "ARTIFACT_CHECKSUM_MISMATCH"
    MANIFEST_INVALID = "MANIFEST_INVALID"
    DEPENDENCY_LOCK_INCOMPLETE = "DEPENDENCY_LOCK_INCOMPLETE"
    QDRANT_VERSION_MISMATCH = "QDRANT_VERSION_MISMATCH"
    APPLY_NOT_IMPLEMENTED = "APPLY_NOT_IMPLEMENTED"


@dataclass
class Blocker:
    code: BlockerCode
    message: str
    details: dict[str, Any] | None = None


@dataclass
class FilesystemCapacity:
    target_path: str
    nearest_existing_parent: str
    device_id: int
    total_bytes: int
    available_bytes: int
    required_bytes: int
    headroom_bytes: int
    deficit_bytes: int
    is_sufficient: bool

    @property
    def available_gib(self) -> float:
        return self.available_bytes / GIB_BYTES

    @property
    def required_gib(self) -> float:
        return self.required_bytes / GIB_BYTES

    @property
    def deficit_gib(self) -> float:
        return self.deficit_bytes / GIB_BYTES


@dataclass
class CapacityReport:
    is_sufficient: bool
    blockers: list[Blocker] = field(default_factory=list)
    filesystems: list[FilesystemCapacity] = field(default_factory=list)
    net_artifact_bytes: int = 0
    estimated_overhead_bytes: dict[str, int] = field(default_factory=dict)
    total_peak_bytes: int = 0
    headroom_bytes: int = RESERVED_HEADROOM_BYTES


@dataclass
class ManifestValidationResult:
    is_valid: bool
    blockers: list[Blocker] = field(default_factory=list)
    total_manifest_bytes: int = 0
    existing_verified_bytes: int = 0
    net_required_bytes: int = 0
    model_artifact_bytes: int = 0
    existing_verified_model_bytes: int = 0
    net_model_artifact_bytes: int = 0
    non_model_artifact_bytes: int = 0
    existing_verified_non_model_bytes: int = 0
    net_non_model_artifact_bytes: int = 0
    artifacts_inventory: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ProvisioningReport:
    mode: str
    status: str
    blockers: list[Blocker]
    capacity: CapacityReport
    manifest_validation: ManifestValidationResult
    dependency_lock_status: str
    timestamp: str


def check_path_safety(path_str: str, allow_empty: bool = False) -> None:
    """Ensures paths are relative and do not perform directory traversal."""
    if not path_str:
        if allow_empty:
            return
        raise ValueError("Path string cannot be empty")
    p = Path(path_str)
    if p.is_absolute():
        raise ValueError(f"Path '{path_str}' must be relative, not absolute")
    if ".." in p.parts:
        raise ValueError(f"Path '{path_str}' contains forbidden '..' path traversal")


def check_identifier_safety(ident: str) -> None:
    """Ensures an artifact identifier is a safe slug without path traversal."""
    if not ident or not isinstance(ident, str) or not ident.strip():
        raise ValueError("Artifact identifier cannot be empty")
    p = Path(ident)
    if p.is_absolute():
        raise ValueError(f"Artifact identifier '{ident}' must not be absolute")
    if ".." in p.parts or "/" in ident or "\\" in ident:
        raise ValueError(
            f"Artifact identifier '{ident}' contains path separators or '..' traversal"
        )
    if not re.match(r"^[a-zA-Z0-9_\-\.]+$", ident):
        raise ValueError(f"Artifact identifier '{ident}' contains invalid characters")


class ManifestFileModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    filename: StrictStr
    size_bytes: StrictInt
    sha256: StrictStr | None = None
    integrity_status: StrictStr = "UNVERIFIED"
    provenance: StrictStr | None = None

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        check_path_safety(v)
        return v

    @field_validator("size_bytes")
    @classmethod
    def validate_size_bytes(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"size_bytes must be a positive integer, got {v}")
        return v

    @field_validator("integrity_status")
    @classmethod
    def validate_integrity_status(cls, v: str) -> str:
        if v not in {"VERIFIED", "UNVERIFIED"}:
            raise ValueError(
                f"integrity_status must be 'VERIFIED' or 'UNVERIFIED', got '{v}'"
            )
        return v

    @model_validator(mode="after")
    def validate_sha256_and_provenance(self) -> ManifestFileModel:
        if self.sha256 is not None and not SHA256_HEX_REGEX.match(self.sha256):
            raise ValueError(
                f"sha256 digest '{self.sha256}' for '{self.filename}' is not a valid 64-hexadecimal character string"
            )
        if self.integrity_status == "VERIFIED":
            if not self.sha256:
                raise ValueError(
                    f"integrity_status is 'VERIFIED' but sha256 digest is missing for '{self.filename}'"
                )
            if not self.provenance or not self.provenance.strip():
                raise ValueError(
                    f"integrity_status is 'VERIFIED' but recorded upstream checksum provenance is missing for '{self.filename}'"
                )
        return self


class ManifestArchiveModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source_archive: StrictStr
    archive_filename: StrictStr
    target_dir: StrictStr
    size_bytes: StrictInt
    sha256: StrictStr | None = None
    integrity_status: StrictStr = "UNVERIFIED"
    provenance: StrictStr | None = None

    @field_validator("archive_filename", "target_dir")
    @classmethod
    def validate_archive_paths(cls, v: str) -> str:
        check_path_safety(v)
        return v

    @field_validator("size_bytes")
    @classmethod
    def validate_size_bytes(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"size_bytes must be a positive integer, got {v}")
        return v

    @field_validator("integrity_status")
    @classmethod
    def validate_integrity_status(cls, v: str) -> str:
        if v not in {"VERIFIED", "UNVERIFIED"}:
            raise ValueError(
                f"integrity_status must be 'VERIFIED' or 'UNVERIFIED', got '{v}'"
            )
        return v

    @model_validator(mode="after")
    def validate_sha256_and_provenance(self) -> ManifestArchiveModel:
        if self.sha256 is not None and not SHA256_HEX_REGEX.match(self.sha256):
            raise ValueError(
                f"sha256 digest '{self.sha256}' for archive '{self.archive_filename}' is invalid"
            )
        if self.integrity_status == "VERIFIED":
            if not self.sha256:
                raise ValueError(
                    f"integrity_status is 'VERIFIED' but sha256 digest is missing for archive '{self.archive_filename}'"
                )
            if not self.provenance or not self.provenance.strip():
                raise ValueError(
                    f"integrity_status is 'VERIFIED' but recorded upstream checksum provenance is missing for archive '{self.archive_filename}'"
                )
        return self


class LlamaServerArtifactModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    id: StrictStr
    category: StrictStr
    official_source: StrictStr = ""
    license: StrictStr = ""
    pinned_release: StrictStr = ""
    pinned_build: StrictStr = ""
    pinned_commit: StrictStr = ""
    target_binary: StrictStr
    target_architecture: StrictStr = "darwin-arm64"
    estimated_size_bytes: StrictInt
    estimated_size_human: StrictStr | None = None
    sha256: StrictStr | None = None
    integrity_status: StrictStr = "UNVERIFIED"
    provenance: StrictStr | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        check_identifier_safety(v)
        return v

    @field_validator("target_binary")
    @classmethod
    def validate_binary(cls, v: str) -> str:
        check_path_safety(v)
        return v

    @field_validator("estimated_size_bytes")
    @classmethod
    def validate_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(
                f"estimated_size_bytes must be a positive integer, got {v}"
            )
        return v

    @field_validator("integrity_status")
    @classmethod
    def validate_integrity_status(cls, v: str) -> str:
        if v not in {"VERIFIED", "UNVERIFIED"}:
            raise ValueError(
                f"integrity_status must be 'VERIFIED' or 'UNVERIFIED', got '{v}'"
            )
        return v

    @model_validator(mode="after")
    def validate_sha256_and_provenance(self) -> LlamaServerArtifactModel:
        if self.sha256 is not None and not SHA256_HEX_REGEX.match(self.sha256):
            raise ValueError(
                f"sha256 digest '{self.sha256}' for binary '{self.target_binary}' is invalid"
            )
        if self.integrity_status == "VERIFIED":
            if not self.sha256:
                raise ValueError(
                    f"integrity_status is 'VERIFIED' but sha256 digest is missing for binary '{self.target_binary}'"
                )
            if not self.provenance or not self.provenance.strip():
                raise ValueError(
                    f"integrity_status is 'VERIFIED' but recorded upstream checksum provenance is missing for binary '{self.target_binary}'"
                )
        return self


class QdrantArtifactModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    id: StrictStr
    category: StrictStr
    official_source: StrictStr = ""
    license: StrictStr = ""
    pinned_version: StrictStr
    image_reference: StrictStr
    image_digest: StrictStr | None = None
    integrity_status: StrictStr = "UNVERIFIED"
    estimated_size_bytes: StrictInt
    estimated_size_human: StrictStr | None = None
    default_url: StrictStr | None = None
    client_compatibility: StrictStr | None = None
    provenance: StrictStr | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        check_identifier_safety(v)
        return v

    @field_validator("estimated_size_bytes")
    @classmethod
    def validate_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(
                f"estimated_size_bytes must be a positive integer, got {v}"
            )
        return v

    @field_validator("pinned_version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        if v != "v1.15.4":
            raise ValueError(f"Qdrant pinned_version must be 'v1.15.4', got '{v}'")
        return v

    @field_validator("integrity_status")
    @classmethod
    def validate_integrity_status(cls, v: str) -> str:
        if v not in {"VERIFIED", "UNVERIFIED"}:
            raise ValueError(
                f"integrity_status must be 'VERIFIED' or 'UNVERIFIED', got '{v}'"
            )
        return v

    @model_validator(mode="after")
    def validate_image_digest_and_provenance(self) -> QdrantArtifactModel:
        if self.image_digest is not None and not IMAGE_DIGEST_REGEX.match(
            self.image_digest
        ):
            raise ValueError(
                f"image_digest '{self.image_digest}' for Qdrant is not in format 'sha256:<64 hex chars>'"
            )
        if self.integrity_status == "VERIFIED":
            if not self.image_digest:
                raise ValueError(
                    "integrity_status is 'VERIFIED' but image_digest is missing for Qdrant image"
                )
            if not self.provenance or not self.provenance.strip():
                raise ValueError(
                    "integrity_status is 'VERIFIED' but recorded upstream provenance is missing for Qdrant image"
                )
        return self


class MultiFileArtifactModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    id: StrictStr
    category: StrictStr
    official_source: StrictStr = ""
    license: StrictStr = ""
    target_dir: StrictStr = ""
    files: list[ManifestFileModel]
    revision: StrictStr | None = None
    pinned_runtime_contract: dict[str, Any] | None = None
    estimated_size_bytes: StrictInt | None = None
    estimated_size_human: StrictStr | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        check_identifier_safety(v)
        return v

    @field_validator("target_dir")
    @classmethod
    def validate_target_dir(cls, v: str) -> str:
        check_path_safety(v, allow_empty=True)
        return v

    @field_validator("files")
    @classmethod
    def validate_files_non_empty(
        cls, v: list[ManifestFileModel]
    ) -> list[ManifestFileModel]:
        if not v:
            raise ValueError("files list must not be empty")
        return v


class PaddleOCRArtifactModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    id: StrictStr
    category: StrictStr
    official_source: StrictStr = ""
    license: StrictStr = ""
    detection: ManifestArchiveModel
    recognition: ManifestArchiveModel
    estimated_size_bytes: StrictInt | None = None
    estimated_size_human: StrictStr | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        check_identifier_safety(v)
        return v


def find_nearest_existing_parent(path: Path) -> Path:
    """Finds the nearest existing parent without modifying the filesystem."""
    current = path.resolve()
    while not current.exists():
        parent = current.parent
        if parent == current:
            break
        current = parent
    return current


def compute_file_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Streams and computes sha256 digest of a local file."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def load_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    """Loads JSON manifest from disk fail-closed."""
    target = manifest_path or DEFAULT_MANIFEST_PATH
    if not target.is_file():
        raise FileNotFoundError(f"Manifest not found at {target}")
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception as err:
        raise ValueError(f"Manifest JSON syntax invalid: {err}") from err


def parse_dependency_lock_requirements(
    content: str,
) -> tuple[bool, set[str], list[str]]:
    """Parses requirements lock content supporting continuations, extras, markers, and sha256 hashes.

    Returns:
        (is_valid_syntax, set_of_canonical_package_names, list_of_error_strings)
    """
    from packaging.requirements import InvalidRequirement, Requirement

    # 1. Join line continuations ending in backslash
    raw_lines = content.splitlines()
    logical_lines: list[str] = []
    current_line = ""
    for raw in raw_lines:
        stripped = raw.strip()
        if stripped.endswith("\\"):
            current_line += " " + stripped[:-1].strip()
        else:
            if current_line:
                current_line += " " + stripped
                logical_lines.append(current_line.strip())
                current_line = ""
            else:
                if stripped:
                    logical_lines.append(stripped)
    if current_line:
        logical_lines.append(current_line.strip())

    found_packages: set[str] = set()
    errors: list[str] = []
    hash_re = re.compile(r"--hash=([^\s]+)")

    for line in logical_lines:
        if line.startswith("#"):
            continue

        all_hashes = hash_re.findall(line)
        req_part = hash_re.sub("", line).strip()
        if " #" in req_part:
            req_part = req_part.split(" #", 1)[0].strip()

        if not req_part:
            continue

        if not all_hashes:
            errors.append(
                f"Requirement '{req_part}' lacks sha256 hash (--hash=sha256:<64 hex chars>)"
            )
            continue

        for h in all_hashes:
            if not h.startswith("sha256:") or not SHA256_HEX_REGEX.match(h[7:]):
                errors.append(
                    f"Malformed hash '{h}' for requirement '{req_part}': must be sha256:<64 hex chars>"
                )

        try:
            req = Requirement(req_part)
        except InvalidRequirement as e:
            errors.append(f"Invalid requirement syntax '{req_part}': {e}")
            continue

        clauses = list(req.specifier)
        if len(clauses) != 1 or clauses[0].operator != "==":
            errors.append(
                f"Package '{req.name}' must be pinned with exact '==' version, got '{req.specifier}'"
            )
            continue

        pkg_canonical = req.name.lower().replace("_", "-")
        found_packages.add(pkg_canonical)

    is_valid = len(errors) == 0 and len(found_packages) > 0
    return is_valid, found_packages, errors


def check_dependency_lock(repo_root: Path) -> tuple[bool, list[Blocker]]:
    """Verifies that an authoritative, hash-pinned runtime dependency lock exists.

    Validates:
    - Lock file exists and is not comment-only or empty.
    - Each dependency entry is an exact pin with valid sha256 hash syntax.
    - Supports standard continuations, extras, and markers.
    - Rejects unpinned packages, malformed hashes, and comment-based fake resolution.
    - Requires complete coverage of core runtime dependencies.
    - Without genuine resolution evidence, keeps DEPENDENCY_LOCK_INCOMPLETE.
    """
    candidates = [
        repo_root / "docs" / "requirements-runtime.lock",
        repo_root / "requirements-lock.txt",
    ]
    lock_file: Path | None = None
    for candidate in candidates:
        if candidate.is_file():
            lock_file = candidate
            break

    if lock_file is None:
        return False, [
            Blocker(
                code=BlockerCode.DEPENDENCY_LOCK_INCOMPLETE,
                message=(
                    "Runtime dependency lock file (requirements-lock.txt) is not present. "
                    "Untracked wheel caches or unpinned pyproject ranges do not constitute a verified dependency lock."
                ),
            )
        ]

    content = lock_file.read_text(encoding="utf-8")
    _is_valid_syntax, found_packages, errors = parse_dependency_lock_requirements(
        content
    )

    if errors:
        return False, [
            Blocker(
                code=BlockerCode.DEPENDENCY_LOCK_INCOMPLETE,
                message=f"Dependency lock contains invalid entries: {'; '.join(errors)}",
            )
        ]

    if not found_packages:
        return False, [
            Blocker(
                code=BlockerCode.DEPENDENCY_LOCK_INCOMPLETE,
                message="Dependency lock file contains no valid requirement entries (comment-only or empty).",
            )
        ]

    missing_required = REQUIRED_RUNTIME_PACKAGES - found_packages
    if missing_required:
        return False, [
            Blocker(
                code=BlockerCode.DEPENDENCY_LOCK_INCOMPLETE,
                message=(
                    f"Dependency lock does not provide complete runtime coverage; "
                    f"missing required packages: {sorted(missing_required)}"
                ),
            )
        ]

    # Without genuine dependency resolution evidence from an authoritative package resolver,
    # keep DEPENDENCY_LOCK_INCOMPLETE. A comment or unverified pin list does not establish resolver provenance.
    return False, [
        Blocker(
            code=BlockerCode.DEPENDENCY_LOCK_INCOMPLETE,
            message=(
                "Dependency lock syntax is valid, but genuine dependency resolution has not been verified "
                "for the target environment (darwin-arm64 cp312). Annotations or fabricated pins do not establish resolver provenance."
            ),
        )
    ]


def validate_manifest(
    manifest: dict[str, Any],
    repo_root: Path,
    model_dir: Path,
    require_all_categories: bool = True,
) -> ManifestValidationResult:
    """Performs fail-closed validation of manifest metadata, integrity, and local files."""
    blockers: list[Blocker] = []
    if not isinstance(manifest, dict):
        blockers.append(
            Blocker(
                code=BlockerCode.MANIFEST_INVALID,
                message="Manifest root must be a JSON object.",
            )
        )
        return ManifestValidationResult(is_valid=False, blockers=blockers)

    if require_all_categories:
        m_version = manifest.get("manifest_version")
        if m_version is None or m_version != 1:
            blockers.append(
                Blocker(
                    code=BlockerCode.MANIFEST_INVALID,
                    message=f"Manifest missing or unsupported 'manifest_version' (expected 1, got {m_version!r}).",
                )
            )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        blockers.append(
            Blocker(
                code=BlockerCode.MANIFEST_INVALID,
                message="Manifest does not define a non-empty 'artifacts' list.",
            )
        )
        return ManifestValidationResult(is_valid=False, blockers=blockers)

    seen_ids: set[str] = set()
    found_categories: set[str] = set()

    total_manifest_bytes = 0
    existing_verified_bytes = 0
    model_artifact_bytes = 0
    existing_verified_model_bytes = 0
    non_model_artifact_bytes = 0
    existing_verified_non_model_bytes = 0
    inventory: list[dict[str, Any]] = []

    for art_idx, art in enumerate(artifacts):
        if not isinstance(art, dict) or not art:
            blockers.append(
                Blocker(
                    code=BlockerCode.MANIFEST_INVALID,
                    message=f"Artifact at index {art_idx} is empty or not an object.",
                )
            )
            continue

        art_id = art.get("id")
        category = art.get("category")

        if not isinstance(art_id, str) or not art_id.strip():
            blockers.append(
                Blocker(
                    code=BlockerCode.MANIFEST_INVALID,
                    message=f"Artifact at index {art_idx} missing valid 'id'.",
                )
            )
            continue

        try:
            check_identifier_safety(art_id)
        except ValueError as err:
            blockers.append(
                Blocker(
                    code=BlockerCode.MANIFEST_INVALID,
                    message=f"Artifact at index {art_idx} has invalid identifier '{art_id}': {err}",
                )
            )
            continue

        if art_id in seen_ids:
            blockers.append(
                Blocker(
                    code=BlockerCode.MANIFEST_INVALID,
                    message=f"Duplicate artifact identifier detected: '{art_id}'.",
                )
            )
        seen_ids.add(art_id)

        if isinstance(category, str):
            found_categories.add(category)

        is_model_category = category in {"embedding", "multimodal_llm", "ocr"}

        inv_entry: dict[str, Any] = {
            "id": art_id,
            "category": category,
            "source": art.get("official_source"),
            "license": art.get("license"),
            "files": [],
        }

        # Multi-file model artifacts (e.g. bge-m3, qwen3-vl)
        if category in {"embedding", "multimodal_llm"}:
            try:
                parsed_model = MultiFileArtifactModel.model_validate(art)
            except ValidationError as val_err:
                for err in val_err.errors():
                    field_name = ".".join(str(loc) for loc in err["loc"])
                    msg = err["msg"]
                    clean_details = {
                        "artifact_id": art_id,
                        "field": field_name,
                        "error_type": str(err.get("type", "validation_error")),
                        "message": str(msg),
                    }
                    if "provenance" in msg.lower() or "unverified" in msg.lower():
                        code = BlockerCode.ARTIFACT_INTEGRITY_UNVERIFIED
                    elif (
                        "sha256" in field_name
                        or "digest" in field_name
                        or "sha256" in msg.lower()
                        or "digest" in msg.lower()
                    ):
                        code = BlockerCode.ARTIFACT_INTEGRITY_INVALID
                    elif "pinned_version" in field_name or "pinned_version" in msg:
                        code = BlockerCode.QDRANT_VERSION_MISMATCH
                    else:
                        code = BlockerCode.MANIFEST_INVALID
                    blockers.append(
                        Blocker(
                            code=code,
                            message=f"Artifact '{art_id}' field '{field_name}': {msg}",
                            details=clean_details,
                        )
                    )
                continue

            target_sub = parsed_model.target_dir
            if target_sub.startswith("data/models/"):
                base_dest = model_dir / Path(target_sub).relative_to("data/models")
                intended_root = model_dir
            elif target_sub:
                base_dest = repo_root / target_sub
                intended_root = repo_root
            else:
                base_dest = model_dir / art_id
                intended_root = model_dir

            # Verify base_dest containment within intended_root
            try:
                base_resolved = base_dest.resolve()
                root_resolved = intended_root.resolve()
                if not base_resolved.is_relative_to(root_resolved):
                    blockers.append(
                        Blocker(
                            code=BlockerCode.MANIFEST_INVALID,
                            message=f"Artifact '{art_id}' destination '{base_dest}' escapes storage root '{intended_root}'.",
                        )
                    )
                    continue
            except (ValueError, OSError) as e:
                blockers.append(
                    Blocker(
                        code=BlockerCode.MANIFEST_INVALID,
                        message=f"Artifact '{art_id}' path error: {e}",
                    )
                )
                continue

            for f_spec in parsed_model.files:
                fname = f_spec.filename
                fsize = f_spec.size_bytes
                expected_sha = f_spec.sha256
                integrity_status = f_spec.integrity_status
                provenance = f_spec.provenance

                has_provenance = bool(provenance and str(provenance).strip())
                is_verified = (
                    (integrity_status == "VERIFIED")
                    and bool(expected_sha)
                    and has_provenance
                )

                total_manifest_bytes += fsize
                if is_model_category:
                    model_artifact_bytes += fsize
                else:
                    non_model_artifact_bytes += fsize

                inv_entry["files"].append(
                    {
                        "filename": fname,
                        "size_bytes": fsize,
                        "integrity_status": integrity_status,
                        "sha256": expected_sha,
                        "provenance": provenance,
                    }
                )

                if not is_verified:
                    if integrity_status == "VERIFIED" and not has_provenance:
                        msg = f"Artifact '{art_id}' file '{fname}' is marked VERIFIED but lacks recorded upstream checksum provenance."
                    elif not expected_sha:
                        msg = f"Artifact {art_id} file '{fname}' lacks verified SHA-256 digest."
                    else:
                        msg = f"Artifact {art_id} file '{fname}' is UNVERIFIED."
                    blockers.append(
                        Blocker(
                            code=BlockerCode.ARTIFACT_INTEGRITY_UNVERIFIED,
                            message=msg,
                            details={"artifact_id": art_id, "filename": fname},
                        )
                    )

                local_path = base_dest / fname
                try:
                    check_path_safety(fname)
                    local_resolved = local_path.resolve()
                    if not local_resolved.is_relative_to(intended_root.resolve()):
                        blockers.append(
                            Blocker(
                                code=BlockerCode.MANIFEST_INVALID,
                                message=f"Artifact '{art_id}' file '{fname}' escapes storage root '{intended_root}'.",
                            )
                        )
                        continue
                except ValueError as pe:
                    blockers.append(
                        Blocker(
                            code=BlockerCode.MANIFEST_INVALID,
                            message=f"Artifact '{art_id}' file '{fname}' unsafe path: {pe}",
                        )
                    )
                    continue

                if local_path.is_file():
                    local_stat = local_path.stat()
                    if local_stat.st_size == fsize and is_verified and expected_sha:
                        local_sha = compute_file_sha256(local_path)
                        if local_sha == expected_sha.lower():
                            existing_verified_bytes += fsize
                            if is_model_category:
                                existing_verified_model_bytes += fsize
                            else:
                                existing_verified_non_model_bytes += fsize
                        else:
                            blockers.append(
                                Blocker(
                                    code=BlockerCode.ARTIFACT_CHECKSUM_MISMATCH,
                                    message=f"Local file {local_path} checksum mismatch (expected {expected_sha}, got {local_sha}).",
                                    details={"path": str(local_path)},
                                )
                            )

        # Archive-based artifacts (e.g. paddleocr)
        elif category == "ocr":
            try:
                parsed_ocr = PaddleOCRArtifactModel.model_validate(art)
            except ValidationError as val_err:
                for err in val_err.errors():
                    field_name = ".".join(str(loc) for loc in err["loc"])
                    msg = err["msg"]
                    clean_details = {
                        "artifact_id": art_id,
                        "field": field_name,
                        "error_type": str(err.get("type", "validation_error")),
                        "message": str(msg),
                    }
                    if "provenance" in msg.lower() or "unverified" in msg.lower():
                        code = BlockerCode.ARTIFACT_INTEGRITY_UNVERIFIED
                    elif (
                        "sha256" in field_name
                        or "digest" in field_name
                        or "sha256" in msg.lower()
                        or "digest" in msg.lower()
                    ):
                        code = BlockerCode.ARTIFACT_INTEGRITY_INVALID
                    elif "pinned_version" in field_name or "pinned_version" in msg:
                        code = BlockerCode.QDRANT_VERSION_MISMATCH
                    else:
                        code = BlockerCode.MANIFEST_INVALID
                    blockers.append(
                        Blocker(
                            code=code,
                            message=f"OCR artifact '{art_id}' field '{field_name}': {msg}",
                            details=clean_details,
                        )
                    )
                continue

            for arch_obj in [parsed_ocr.detection, parsed_ocr.recognition]:
                asize = arch_obj.size_bytes
                total_manifest_bytes += asize
                model_artifact_bytes += asize
                has_arch_prov = bool(
                    arch_obj.provenance and str(arch_obj.provenance).strip()
                )
                is_arch_verified = (
                    (arch_obj.integrity_status == "VERIFIED")
                    and bool(arch_obj.sha256)
                    and has_arch_prov
                )

                inv_entry["files"].append(
                    {
                        "archive": arch_obj.archive_filename,
                        "size_bytes": asize,
                        "integrity_status": arch_obj.integrity_status,
                        "sha256": arch_obj.sha256,
                        "provenance": arch_obj.provenance,
                    }
                )

                if not is_arch_verified:
                    if arch_obj.integrity_status == "VERIFIED" and not has_arch_prov:
                        msg = f"OCR archive '{arch_obj.archive_filename}' is marked VERIFIED but lacks recorded upstream checksum provenance."
                    elif not arch_obj.sha256:
                        msg = f"OCR archive '{arch_obj.archive_filename}' lacks verified SHA-256 digest."
                    else:
                        msg = (
                            f"OCR archive '{arch_obj.archive_filename}' is UNVERIFIED."
                        )
                    blockers.append(
                        Blocker(
                            code=BlockerCode.ARTIFACT_INTEGRITY_UNVERIFIED,
                            message=msg,
                            details={
                                "artifact_id": art_id,
                                "archive": arch_obj.archive_filename,
                            },
                        )
                    )

        # Binary artifacts (e.g. llama-server)
        elif category == "inference_server":
            try:
                parsed_bin = LlamaServerArtifactModel.model_validate(art)
            except ValidationError as val_err:
                for err in val_err.errors():
                    field_name = ".".join(str(loc) for loc in err["loc"])
                    msg = err["msg"]
                    clean_details = {
                        "artifact_id": art_id,
                        "field": field_name,
                        "error_type": str(err.get("type", "validation_error")),
                        "message": str(msg),
                    }
                    if "provenance" in msg.lower() or "unverified" in msg.lower():
                        code = BlockerCode.ARTIFACT_INTEGRITY_UNVERIFIED
                    elif (
                        "sha256" in field_name
                        or "digest" in field_name
                        or "sha256" in msg.lower()
                        or "digest" in msg.lower()
                    ):
                        code = BlockerCode.ARTIFACT_INTEGRITY_INVALID
                    elif "pinned_version" in field_name or "pinned_version" in msg:
                        code = BlockerCode.QDRANT_VERSION_MISMATCH
                    else:
                        code = BlockerCode.MANIFEST_INVALID
                    blockers.append(
                        Blocker(
                            code=code,
                            message=f"Inference server artifact '{art_id}' field '{field_name}': {msg}",
                            details=clean_details,
                        )
                    )
                continue

            bsize = parsed_bin.estimated_size_bytes
            total_manifest_bytes += bsize
            non_model_artifact_bytes += bsize
            has_bin_prov = bool(
                parsed_bin.provenance and str(parsed_bin.provenance).strip()
            )
            is_bin_verified = (
                (parsed_bin.integrity_status == "VERIFIED")
                and bool(parsed_bin.sha256)
                and has_bin_prov
            )

            inv_entry["files"].append(
                {
                    "binary": parsed_bin.target_binary,
                    "size_bytes": bsize,
                    "integrity_status": parsed_bin.integrity_status,
                    "sha256": parsed_bin.sha256,
                    "provenance": parsed_bin.provenance,
                }
            )

            if not is_bin_verified:
                if parsed_bin.integrity_status == "VERIFIED" and not has_bin_prov:
                    msg = f"Inference binary '{parsed_bin.target_binary}' is marked VERIFIED but lacks recorded upstream checksum provenance."
                elif not parsed_bin.sha256:
                    msg = f"Inference binary '{parsed_bin.target_binary}' lacks verified SHA-256 digest."
                else:
                    msg = (
                        f"Inference binary '{parsed_bin.target_binary}' is UNVERIFIED."
                    )
                blockers.append(
                    Blocker(
                        code=BlockerCode.ARTIFACT_INTEGRITY_UNVERIFIED,
                        message=msg,
                        details={"artifact_id": art_id},
                    )
                )

        # Container image artifacts (e.g. qdrant)
        elif category == "vector_store":
            try:
                parsed_qdrant = QdrantArtifactModel.model_validate(art)
            except ValidationError as val_err:
                for err in val_err.errors():
                    field_name = ".".join(str(loc) for loc in err["loc"])
                    msg = err["msg"]
                    clean_details = {
                        "artifact_id": art_id,
                        "field": field_name,
                        "error_type": str(err.get("type", "validation_error")),
                        "message": str(msg),
                    }
                    if "provenance" in msg.lower() or "unverified" in msg.lower():
                        code = BlockerCode.ARTIFACT_INTEGRITY_UNVERIFIED
                    elif (
                        "sha256" in field_name
                        or "digest" in field_name
                        or "sha256" in msg.lower()
                        or "digest" in msg.lower()
                    ):
                        code = BlockerCode.ARTIFACT_INTEGRITY_INVALID
                    elif "pinned_version" in field_name or "pinned_version" in msg:
                        code = BlockerCode.QDRANT_VERSION_MISMATCH
                    else:
                        code = BlockerCode.MANIFEST_INVALID
                    blockers.append(
                        Blocker(
                            code=code,
                            message=f"Vector store artifact '{art_id}' field '{field_name}': {msg}",
                            details=clean_details,
                        )
                    )
                continue

            csize = parsed_qdrant.estimated_size_bytes
            total_manifest_bytes += csize
            non_model_artifact_bytes += csize
            has_qdrant_prov = bool(
                parsed_qdrant.provenance and str(parsed_qdrant.provenance).strip()
            )
            is_qdrant_verified = (
                (parsed_qdrant.integrity_status == "VERIFIED")
                and bool(parsed_qdrant.image_digest)
                and has_qdrant_prov
            )

            inv_entry["container"] = {
                "image": parsed_qdrant.image_reference,
                "pinned_version": parsed_qdrant.pinned_version,
                "image_digest": parsed_qdrant.image_digest,
                "integrity_status": parsed_qdrant.integrity_status,
                "provenance": parsed_qdrant.provenance,
            }

            if not is_qdrant_verified:
                if parsed_qdrant.integrity_status == "VERIFIED" and not has_qdrant_prov:
                    msg = f"Qdrant container image '{parsed_qdrant.image_reference}' is marked VERIFIED but lacks recorded upstream provenance."
                elif not parsed_qdrant.image_digest:
                    msg = f"Qdrant container image '{parsed_qdrant.image_reference}' lacks verified digest."
                else:
                    msg = f"Qdrant container image '{parsed_qdrant.image_reference}' is UNVERIFIED."
                blockers.append(
                    Blocker(
                        code=BlockerCode.ARTIFACT_INTEGRITY_UNVERIFIED,
                        message=msg,
                        details={"artifact_id": art_id},
                    )
                )
        else:
            blockers.append(
                Blocker(
                    code=BlockerCode.MANIFEST_INVALID,
                    message=f"Artifact '{art_id}' has unrecognized category '{category}'.",
                )
            )
            continue

        inventory.append(inv_entry)

    if require_all_categories:
        missing_categories = REQUIRED_RUNTIME_CATEGORIES - found_categories
        if missing_categories:
            blockers.append(
                Blocker(
                    code=BlockerCode.MANIFEST_INVALID,
                    message=f"Manifest missing required runtime categories: {sorted(missing_categories)}",
                )
            )

    net_model_bytes = max(0, model_artifact_bytes - existing_verified_model_bytes)
    net_non_model_bytes = max(
        0, non_model_artifact_bytes - existing_verified_non_model_bytes
    )
    net_required_bytes = net_model_bytes + net_non_model_bytes

    return ManifestValidationResult(
        is_valid=len(blockers) == 0,
        blockers=blockers,
        total_manifest_bytes=total_manifest_bytes,
        existing_verified_bytes=existing_verified_bytes,
        net_required_bytes=net_required_bytes,
        model_artifact_bytes=model_artifact_bytes,
        existing_verified_model_bytes=existing_verified_model_bytes,
        net_model_artifact_bytes=net_model_bytes,
        non_model_artifact_bytes=non_model_artifact_bytes,
        existing_verified_non_model_bytes=existing_verified_non_model_bytes,
        net_non_model_artifact_bytes=net_non_model_bytes,
        artifacts_inventory=inventory,
    )


def check_capacity(
    repo_root: Path,
    model_dir: Path,
    net_artifact_bytes: int,
    net_model_artifact_bytes: int | None = None,
    net_non_model_artifact_bytes: int | None = None,
    model_artifact_bytes: int | None = None,
    non_model_artifact_bytes: int | None = None,
) -> CapacityReport:
    """Calculates disk capacity requirements per filesystem in integer bytes with 10 GiB headroom."""
    blockers: list[Blocker] = []
    filesystems: list[FilesystemCapacity] = []

    model_parent = find_nearest_existing_parent(model_dir)
    repo_parent = find_nearest_existing_parent(repo_root)

    total_overhead = sum(ESTIMATED_OVERHEAD_BYTES.values())
    total_peak = net_artifact_bytes + total_overhead

    model_dev = model_parent.stat().st_dev
    repo_dev = repo_parent.stat().st_dev

    is_same_fs = model_dev == repo_dev

    effective_net_model = (
        net_model_artifact_bytes
        if net_model_artifact_bytes is not None
        else (
            model_artifact_bytes
            if model_artifact_bytes is not None
            else net_artifact_bytes
        )
    )
    effective_net_non_model = (
        net_non_model_artifact_bytes
        if net_non_model_artifact_bytes is not None
        else (non_model_artifact_bytes if non_model_artifact_bytes is not None else 0)
    )

    if is_same_fs:
        usage = shutil.disk_usage(model_parent)
        required_total = total_peak + RESERVED_HEADROOM_BYTES
        deficit = max(0, required_total - usage.free)
        sufficient = deficit == 0

        fs_cap = FilesystemCapacity(
            target_path=str(model_dir),
            nearest_existing_parent=str(model_parent),
            device_id=model_dev,
            total_bytes=usage.total,
            available_bytes=usage.free,
            required_bytes=required_total,
            headroom_bytes=RESERVED_HEADROOM_BYTES,
            deficit_bytes=deficit,
            is_sufficient=sufficient,
        )
        filesystems.append(fs_cap)

        if not sufficient:
            blockers.append(
                Blocker(
                    code=BlockerCode.CAPACITY_INSUFFICIENT,
                    message=(
                        f"Target filesystem ({model_parent}) has {usage.free / GIB_BYTES:.2f} GiB available; "
                        f"requires {required_total / GIB_BYTES:.2f} GiB "
                        f"(deficit: {deficit / GIB_BYTES:.2f} GiB, retaining {RESERVED_HEADROOM_BYTES / GIB_BYTES:.2f} GiB headroom)."
                    ),
                    details={
                        "target_parent": str(model_parent),
                        "available_bytes": usage.free,
                        "required_bytes": required_total,
                        "deficit_bytes": deficit,
                    },
                )
            )
    else:
        # Separate filesystems: external model storage vs project storage
        # 1. Model storage filesystem (accurately uses net model bytes after deduplication)
        m_usage = shutil.disk_usage(model_parent)
        m_required = effective_net_model + RESERVED_HEADROOM_BYTES
        m_deficit = max(0, m_required - m_usage.free)
        m_sufficient = m_deficit == 0

        filesystems.append(
            FilesystemCapacity(
                target_path=str(model_dir),
                nearest_existing_parent=str(model_parent),
                device_id=model_dev,
                total_bytes=m_usage.total,
                available_bytes=m_usage.free,
                required_bytes=m_required,
                headroom_bytes=RESERVED_HEADROOM_BYTES,
                deficit_bytes=m_deficit,
                is_sufficient=m_sufficient,
            )
        )

        if not m_sufficient:
            blockers.append(
                Blocker(
                    code=BlockerCode.CAPACITY_INSUFFICIENT,
                    message=(
                        f"External model filesystem ({model_parent}) has {m_usage.free / GIB_BYTES:.2f} GiB available; "
                        f"requires {m_required / GIB_BYTES:.2f} GiB (deficit: {m_deficit / GIB_BYTES:.2f} GiB)."
                    ),
                    details={"fs": "model_storage", "deficit_bytes": m_deficit},
                )
            )

        # 2. Repo / System filesystem
        r_usage = shutil.disk_usage(repo_parent)
        r_required = effective_net_non_model + total_overhead + RESERVED_HEADROOM_BYTES
        r_deficit = max(0, r_required - r_usage.free)
        r_sufficient = r_deficit == 0

        filesystems.append(
            FilesystemCapacity(
                target_path=str(repo_root),
                nearest_existing_parent=str(repo_parent),
                device_id=repo_dev,
                total_bytes=r_usage.total,
                available_bytes=r_usage.free,
                required_bytes=r_required,
                headroom_bytes=RESERVED_HEADROOM_BYTES,
                deficit_bytes=r_deficit,
                is_sufficient=r_sufficient,
            )
        )

        if not r_sufficient:
            blockers.append(
                Blocker(
                    code=BlockerCode.CAPACITY_INSUFFICIENT,
                    message=(
                        f"Project repository filesystem ({repo_parent}) has {r_usage.free / GIB_BYTES:.2f} GiB available; "
                        f"requires {r_required / GIB_BYTES:.2f} GiB for dependencies, Docker, caches and headroom (deficit: {r_deficit / GIB_BYTES:.2f} GiB)."
                    ),
                    details={"fs": "repo_storage", "deficit_bytes": r_deficit},
                )
            )

    all_sufficient = len(blockers) == 0
    return CapacityReport(
        is_sufficient=all_sufficient,
        blockers=blockers,
        filesystems=filesystems,
        net_artifact_bytes=net_artifact_bytes,
        estimated_overhead_bytes=ESTIMATED_OVERHEAD_BYTES,
        total_peak_bytes=total_peak,
        headroom_bytes=RESERVED_HEADROOM_BYTES,
    )


def assess_readiness(
    manifest_path: Path | None = None,
    model_dir: Path | None = None,
    repo_root: Path | None = None,
    mode: str = "dry-run",
) -> ProvisioningReport:
    """Executes read-only validation of manifest, dependency lock, and capacity."""
    r_root = repo_root or REPO_ROOT
    m_dir = model_dir or DEFAULT_MODEL_DIR
    m_path = manifest_path or DEFAULT_MANIFEST_PATH

    from datetime import UTC, datetime

    manifest_data = load_manifest(m_path)
    manifest_res = validate_manifest(
        manifest_data, repo_root=r_root, model_dir=m_dir, require_all_categories=True
    )

    lock_ok, lock_blockers = check_dependency_lock(repo_root=r_root)

    capacity_res = check_capacity(
        repo_root=r_root,
        model_dir=m_dir,
        net_artifact_bytes=manifest_res.net_required_bytes,
        net_model_artifact_bytes=manifest_res.net_model_artifact_bytes,
        net_non_model_artifact_bytes=manifest_res.net_non_model_artifact_bytes,
    )

    all_blockers: list[Blocker] = []
    all_blockers.extend(manifest_res.blockers)
    all_blockers.extend(lock_blockers)
    all_blockers.extend(capacity_res.blockers)

    overall_status = "READY" if not all_blockers else "BLOCKED"

    return ProvisioningReport(
        mode=mode,
        status=overall_status,
        blockers=all_blockers,
        capacity=capacity_res,
        manifest_validation=manifest_res,
        dependency_lock_status="READY" if lock_ok else "BLOCKED",
        timestamp=datetime.now(UTC).isoformat(),
    )


def format_report(report: ProvisioningReport) -> str:
    """Generates structured, human-readable terminal report."""
    lines: list[str] = [
        "=" * 72,
        "KoshShield AI Local Runtime Provisioning Guardrails Report",
        "=" * 72,
        f"Mode:                   {report.mode.upper()}",
        f"Readiness Status:       {report.status}",
        f"Dependency Lock Status: {report.dependency_lock_status}",
        f"Timestamp:              {report.timestamp}",
        "",
        "--- CAPACITY ASSESSMENT (1024-based GiB with 10 GiB Headroom) ---",
        f"Net Artifact Requirement: {report.capacity.net_artifact_bytes / GIB_BYTES:.2f} GiB ({report.capacity.net_artifact_bytes} bytes)",
        f"Estimated Overhead:       {report.capacity.total_peak_bytes - report.capacity.net_artifact_bytes} bytes ({sum(report.capacity.estimated_overhead_bytes.values()) / GIB_BYTES:.2f} GiB)",
    ]

    for k, v in report.capacity.estimated_overhead_bytes.items():
        lines.append(f"  - {k}: {v / GIB_BYTES:.2f} GiB ({v} bytes) [ESTIMATED]")

    lines.append(
        f"Total Peak Footprint:     {report.capacity.total_peak_bytes / GIB_BYTES:.2f} GiB"
    )
    lines.append(
        f"Reserved Headroom:        {report.capacity.headroom_bytes / GIB_BYTES:.2f} GiB (mandatory policy)"
    )
    lines.append("")

    for idx, fs in enumerate(report.capacity.filesystems, 1):
        lines.append(f"Filesystem {idx} [{fs.nearest_existing_parent}]:")
        lines.append(
            f"  Available: {fs.available_gib:.2f} GiB ({fs.available_bytes} bytes)"
        )
        lines.append(
            f"  Required:  {fs.required_gib:.2f} GiB ({fs.required_bytes} bytes)"
        )
        if not fs.is_sufficient:
            lines.append(
                f"  Deficit:   {fs.deficit_gib:.2f} GiB ({fs.deficit_bytes} bytes) [INSUFFICIENT]"
            )
        else:
            lines.append("  Status:    SUFFICIENT")

    lines.append("")
    lines.append(f"--- ACTIVE BLOCKERS ({len(report.blockers)}) ---")
    if not report.blockers:
        lines.append("No active blockers detected.")
    else:
        for b in report.blockers:
            lines.append(f"  [{b.code.value}] {b.message}")

    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check prerequisites and guardrails for KoshShield local runtime provisioning."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Perform read-only inventory, integrity, and capacity guardrail checks (default)",
    )
    group.add_argument(
        "--apply",
        action="store_true",
        default=None,
        help="Execute local provisioning after fail-closed validation of all prerequisites",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help=f"Target directory for model weights (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(DEFAULT_MANIFEST_PATH),
        help=f"Path to artifact manifest (default: {DEFAULT_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report in JSON format",
    )

    args = parser.parse_args(argv)

    is_apply = bool(args.apply)
    mode = "apply" if is_apply else "dry-run"
    model_dir = Path(args.model_dir)
    manifest_path = Path(args.manifest)

    try:
        report = assess_readiness(
            manifest_path=manifest_path,
            model_dir=model_dir,
            repo_root=REPO_ROOT,
            mode=mode,
        )
    except (ValueError, OSError, RuntimeError, json.JSONDecodeError) as err:
        print(f"FATAL: Provisioning assessment failed: {err}", file=sys.stderr)
        return 1

    if args.json:
        # Convert dataclasses and non-primitives safely to dict
        def _convert_report(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _convert_report(v) for k, v in asdict(obj).items()}
            if isinstance(obj, (list, tuple, set)):
                return [_convert_report(i) for i in obj]
            if isinstance(obj, dict):
                return {str(k): _convert_report(v) for k, v in obj.items()}
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, Exception):
                return str(obj)
            if isinstance(obj, (int, float, bool, str)) or obj is None:
                return obj
            return str(obj)

        print(json.dumps(_convert_report(report), indent=2, default=str))
    else:
        print(format_report(report))

    if is_apply:
        if report.status != "READY":
            print(
                "\nERROR: Apply cannot proceed because prerequisite checks failed (BLOCKED).",
                file=sys.stderr,
            )
            return 1

        # Downloader/installer execution is not implemented in this checkpoint
        print(
            "\nERROR: APPLY_NOT_IMPLEMENTED: Execution pipeline (model downloading / package installation) "
            "is not implemented in this checkpoint. Stage 0 remains incomplete.",
            file=sys.stderr,
        )
        return 2

    # In dry-run mode: the inspection executed cleanly without error.
    # The output report truthfully communicates whether status is READY or BLOCKED.
    return 0


if __name__ == "__main__":
    sys.exit(main())
