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
import shutil
import sys
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

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
    non_model_artifact_bytes: int = 0
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


def check_dependency_lock(repo_root: Path) -> tuple[bool, list[Blocker]]:
    """Verifies that an authoritative, hash-pinned runtime dependency lock exists."""
    candidates = [
        repo_root / "docs" / "requirements-runtime.lock",
        repo_root / "requirements-lock.txt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            content = candidate.read_text(encoding="utf-8")
            if "--hash=sha256:" in content:
                return True, []

    blocker = Blocker(
        code=BlockerCode.DEPENDENCY_LOCK_INCOMPLETE,
        message=(
            "Complete verified runtime dependency lock with hashes is not present. "
            "Untracked wheel caches or unpinned pyproject ranges do not constitute a verified dependency lock."
        ),
    )
    return False, [blocker]


def validate_manifest(
    manifest: dict[str, Any], repo_root: Path, model_dir: Path
) -> ManifestValidationResult:
    """Performs fail-closed validation of manifest metadata, integrity, and local files."""
    blockers: list[Blocker] = []
    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        blockers.append(
            Blocker(
                code=BlockerCode.MANIFEST_INVALID,
                message="Manifest does not define a valid 'artifacts' list.",
            )
        )
        return ManifestValidationResult(is_valid=False, blockers=blockers)

    total_manifest_bytes = 0
    existing_verified_bytes = 0
    model_artifact_bytes = 0
    non_model_artifact_bytes = 0
    inventory: list[dict[str, Any]] = []

    for art in artifacts:
        art_id = art.get("id", "unknown")
        category = art.get("category", "unknown")
        is_model_category = category in {"embedding", "multimodal_llm", "ocr"}

        inv_entry: dict[str, Any] = {
            "id": art_id,
            "category": category,
            "source": art.get("official_source"),
            "license": art.get("license"),
            "estimated_size_bytes": art.get("estimated_size_bytes", 0),
            "files": [],
        }

        # Multi-file model artifacts (e.g. bge-m3, qwen3-vl)
        if "files" in art and isinstance(art["files"], list):
            target_sub = art.get("target_dir", "")
            base_dest = (
                (model_dir / art_id)
                if not target_sub
                else (model_dir / Path(target_sub).relative_to("data/models"))
                if target_sub.startswith("data/models/")
                else repo_root / target_sub
            )

            for f_spec in art["files"]:
                fname = f_spec.get("filename", "")
                fsize = int(f_spec.get("size_bytes", 0))
                expected_sha = f_spec.get("sha256")
                integrity_status = f_spec.get("integrity_status", "UNVERIFIED")
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
                    }
                )

                if not expected_sha or integrity_status != "VERIFIED":
                    blockers.append(
                        Blocker(
                            code=BlockerCode.ARTIFACT_INTEGRITY_UNVERIFIED,
                            message=f"Artifact {art_id} file '{fname}' lacks verified SHA-256 digest.",
                            details={"artifact_id": art_id, "filename": fname},
                        )
                    )
                elif len(expected_sha) != 64 or not all(
                    c in "0123456789abcdef" for c in expected_sha.lower()
                ):
                    blockers.append(
                        Blocker(
                            code=BlockerCode.ARTIFACT_INTEGRITY_INVALID,
                            message=f"Artifact {art_id} file '{fname}' has invalid SHA-256 digest format.",
                            details={
                                "artifact_id": art_id,
                                "filename": fname,
                                "sha256": expected_sha,
                            },
                        )
                    )

                local_path = base_dest / fname
                if (
                    local_path.is_file()
                    and expected_sha
                    and integrity_status == "VERIFIED"
                ):
                    local_sha = compute_file_sha256(local_path)
                    if local_sha == expected_sha.lower():
                        existing_verified_bytes += fsize
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
            for arch_key in ["detection", "recognition"]:
                sub = art.get(arch_key, {})
                if isinstance(sub, dict):
                    arch_name = sub.get(
                        "archive_filename", sub.get("source_archive", "")
                    )
                    asize = int(sub.get("size_bytes", 0))
                    total_manifest_bytes += asize
                    model_artifact_bytes += asize
                    expected_sha = sub.get("sha256")
                    integrity_status = sub.get("integrity_status", "UNVERIFIED")

                    inv_entry["files"].append(
                        {
                            "archive": arch_name,
                            "size_bytes": asize,
                            "integrity_status": integrity_status,
                            "sha256": expected_sha,
                        }
                    )

                    if not expected_sha or integrity_status != "VERIFIED":
                        blockers.append(
                            Blocker(
                                code=BlockerCode.ARTIFACT_INTEGRITY_UNVERIFIED,
                                message=f"OCR archive '{arch_name}' lacks verified SHA-256 digest.",
                                details={"artifact_id": art_id, "archive": arch_name},
                            )
                        )

        # Binary artifacts (e.g. llama-server)
        elif category == "inference_server":
            bsize = int(art.get("estimated_size_bytes", 0))
            total_manifest_bytes += bsize
            non_model_artifact_bytes += bsize
            expected_sha = art.get("sha256")
            integrity_status = art.get("integrity_status", "UNVERIFIED")

            inv_entry["files"].append(
                {
                    "binary": art.get("target_binary", "llama-server"),
                    "size_bytes": bsize,
                    "integrity_status": integrity_status,
                    "sha256": expected_sha,
                }
            )

            if not expected_sha or integrity_status != "VERIFIED":
                blockers.append(
                    Blocker(
                        code=BlockerCode.ARTIFACT_INTEGRITY_UNVERIFIED,
                        message=f"Inference binary '{art.get('target_binary')}' lacks verified SHA-256 digest.",
                        details={"artifact_id": art_id},
                    )
                )

        # Container image artifacts (e.g. qdrant)
        elif category == "vector_store":
            csize = int(art.get("estimated_size_bytes", 0))
            total_manifest_bytes += csize
            non_model_artifact_bytes += csize
            pinned_version = art.get("pinned_version", "")
            image_digest = art.get("image_digest")
            integrity_status = art.get("integrity_status", "UNVERIFIED")

            inv_entry["container"] = {
                "image": art.get("image_reference", "qdrant/qdrant:v1.15.4"),
                "pinned_version": pinned_version,
                "image_digest": image_digest,
                "integrity_status": integrity_status,
            }

            if pinned_version != "v1.15.4":
                blockers.append(
                    Blocker(
                        code=BlockerCode.QDRANT_VERSION_MISMATCH,
                        message=f"Qdrant pinned version '{pinned_version}' does not match repository configuration 'v1.15.4'.",
                        details={
                            "pinned_version": pinned_version,
                            "expected": "v1.15.4",
                        },
                    )
                )

            if not image_digest or integrity_status != "VERIFIED":
                blockers.append(
                    Blocker(
                        code=BlockerCode.ARTIFACT_INTEGRITY_UNVERIFIED,
                        message=f"Qdrant container image '{art.get('image_reference')}' lacks verified digest.",
                        details={"artifact_id": art_id},
                    )
                )

        inventory.append(inv_entry)

    net_required_bytes = max(0, total_manifest_bytes - existing_verified_bytes)
    return ManifestValidationResult(
        is_valid=len(blockers) == 0,
        blockers=blockers,
        total_manifest_bytes=total_manifest_bytes,
        existing_verified_bytes=existing_verified_bytes,
        net_required_bytes=net_required_bytes,
        model_artifact_bytes=model_artifact_bytes,
        non_model_artifact_bytes=non_model_artifact_bytes,
        artifacts_inventory=inventory,
    )


def check_capacity(
    repo_root: Path,
    model_dir: Path,
    net_artifact_bytes: int,
    model_artifact_bytes: int,
    non_model_artifact_bytes: int,
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
        # 1. Model storage filesystem
        m_usage = shutil.disk_usage(model_parent)
        m_required = model_artifact_bytes + RESERVED_HEADROOM_BYTES
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
        r_required = non_model_artifact_bytes + total_overhead + RESERVED_HEADROOM_BYTES
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
    manifest_res = validate_manifest(manifest_data, repo_root=r_root, model_dir=m_dir)

    lock_ok, lock_blockers = check_dependency_lock(repo_root=r_root)

    capacity_res = check_capacity(
        repo_root=r_root,
        model_dir=m_dir,
        net_artifact_bytes=manifest_res.net_required_bytes,
        model_artifact_bytes=manifest_res.model_artifact_bytes,
        non_model_artifact_bytes=manifest_res.non_model_artifact_bytes,
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
        # Convert dataclasses to dict
        def _convert_report(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _convert_report(v) for k, v in asdict(obj).items()}
            if isinstance(obj, list):
                return [_convert_report(i) for i in obj]
            if isinstance(obj, dict):
                return {k: _convert_report(v) for k, v in obj.items()}
            return obj

        print(json.dumps(_convert_report(report), indent=2))
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
