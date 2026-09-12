"""Regression tests for local runtime provisioning guardrails.

Verifies:
1. Mutually exclusive --dry-run and --apply flags (default to dry-run).
2. Read-only prerequisite checks, honest READY/BLOCKED status, and stable blocker codes.
3. Capacity arithmetic using exact integer bytes, 1024-based GiB, and 10 GiB headroom.
4. Nonexistent destination inspection via nearest existing parent without mkdir.
5. Fail-closed manifest validation (missing, malformed, or unverified integrity metadata).
6. Existing verified artifact deduplication without double counting.
7. Dependency lock verification.
8. Unimplemented apply behavior failing closed with nonzero exit code.
9. Interception of forbidden network, installer, and service-launch entry points.
10. Strict immutability of configuration and artifact directories.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.provision_local_runtime import (  # noqa: E402
    ESTIMATED_OVERHEAD_BYTES,
    GIB_BYTES,
    RESERVED_HEADROOM_BYTES,
    BlockerCode,
    assess_readiness,
    check_capacity,
    check_dependency_lock,
    compute_file_sha256,
    find_nearest_existing_parent,
    format_report,
    main,
    validate_manifest,
)


@pytest.fixture(autouse=True)
def guard_forbidden_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail immediately if any network connection, subprocess, or installer is called."""

    def forbidden_network(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "FORBIDDEN_NETWORK_CALL: Provisioning guardrails must be strictly offline."
        )

    def forbidden_subprocess(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "FORBIDDEN_SUBPROCESS_CALL: Subprocesses are forbidden during validation."
        )

    monkeypatch.setattr(urllib.request, "urlopen", forbidden_network)
    monkeypatch.setattr(httpx.Client, "send", forbidden_network)
    monkeypatch.setattr(socket.socket, "connect", forbidden_network)
    monkeypatch.setattr(subprocess, "run", forbidden_subprocess)
    monkeypatch.setattr(subprocess, "Popen", forbidden_subprocess)


def test_mutually_exclusive_flags() -> None:
    """Test that passing both --dry-run and --apply raises ArgumentError (SystemExit 2)."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--dry-run", "--apply"])
    assert exc_info.value.code == 2


def test_default_mode_is_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that omitting execution flags defaults to dry-run and reports BLOCKED honestly."""
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Mode:                   DRY-RUN" in captured.out
    assert "Readiness Status:       BLOCKED" in captured.out
    assert "ACTIVE BLOCKERS" in captured.out


def test_nonexistent_destination_nearest_parent_no_mkdir(tmp_path: Path) -> None:
    """For nonexistent destinations, inspect nearest existing parent without mkdir."""
    nonexistent = tmp_path / "level1" / "level2" / "models"
    assert not (tmp_path / "level1").exists()

    parent = find_nearest_existing_parent(nonexistent)
    assert parent == tmp_path.resolve()

    # Crucial guarantee: directory must NOT have been created
    assert not (tmp_path / "level1").exists()
    assert not nonexistent.exists()


def test_capacity_arithmetic_exact_integer_bytes(tmp_path: Path) -> None:
    """Verify capacity math uses integer bytes and exact 10 GiB headroom."""
    fake_free = 14 * GIB_BYTES  # 14 GiB available
    fake_total = 100 * GIB_BYTES

    fake_usage = shutil._ntuple_diskusage(
        total=fake_total, used=fake_total - fake_free, free=fake_free
    )

    net_artifacts = int(5.49 * GIB_BYTES)
    overhead = sum(ESTIMATED_OVERHEAD_BYTES.values())
    total_required = net_artifacts + overhead + RESERVED_HEADROOM_BYTES
    expected_deficit = total_required - fake_free

    with patch("shutil.disk_usage", return_value=fake_usage):
        report = check_capacity(
            repo_root=tmp_path,
            model_dir=tmp_path / "models",
            net_artifact_bytes=net_artifacts,
            model_artifact_bytes=net_artifacts,
            non_model_artifact_bytes=0,
        )

    assert report.is_sufficient is False
    assert len(report.blockers) == 1
    assert report.blockers[0].code == BlockerCode.CAPACITY_INSUFFICIENT

    fs_cap = report.filesystems[0]
    assert fs_cap.available_bytes == fake_free
    assert fs_cap.required_bytes == total_required
    assert fs_cap.deficit_bytes == expected_deficit
    assert fs_cap.headroom_bytes == RESERVED_HEADROOM_BYTES
    assert fs_cap.is_sufficient is False


def test_multi_filesystem_capacity_isolation(tmp_path: Path) -> None:
    """External model path does not relocate project or Docker storage."""
    model_mount = tmp_path / "ext_models"
    repo_mount = tmp_path / "repo"
    model_mount.mkdir()
    repo_mount.mkdir()

    # Mock different device IDs
    stat_model = MagicMock(st_dev=100)
    stat_repo = MagicMock(st_dev=200)

    def fake_stat(self: Path, *args: Any, **kwargs: Any) -> Any:
        if "ext_models" in str(self):
            return stat_model
        return stat_repo

    usage_model = shutil._ntuple_diskusage(
        total=100 * GIB_BYTES, used=50 * GIB_BYTES, free=50 * GIB_BYTES
    )
    # Repo filesystem has only 8 GiB free (deficit for overhead + headroom)
    usage_repo = shutil._ntuple_diskusage(
        total=50 * GIB_BYTES, used=42 * GIB_BYTES, free=8 * GIB_BYTES
    )

    def fake_usage(path: Any) -> Any:
        if "ext_models" in str(path):
            return usage_model
        return usage_repo

    with (
        patch.object(Path, "stat", fake_stat),
        patch("shutil.disk_usage", fake_usage),
    ):
        report = check_capacity(
            repo_root=repo_mount,
            model_dir=model_mount,
            net_artifact_bytes=5 * GIB_BYTES,
            model_artifact_bytes=5 * GIB_BYTES,
            non_model_artifact_bytes=0,
        )

    assert report.is_sufficient is False
    assert len(report.filesystems) == 2
    # Model storage has 50 GiB free, which exceeds 5 GiB + 10 GiB headroom
    assert report.filesystems[0].is_sufficient is True
    # Repo storage has 8 GiB free; requires 7 GiB overhead + 10 GiB headroom (17 GiB required)
    assert report.filesystems[1].is_sufficient is False
    assert report.filesystems[1].deficit_bytes > 0


def test_existing_verified_artifacts_deduplication(tmp_path: Path) -> None:
    """Existing verified artifact files with matching SHA-256 are not double-counted."""
    model_dir = tmp_path / "models"
    target_dir = model_dir / "test-model"
    target_dir.mkdir(parents=True)

    dummy_file = target_dir / "weights.safetensors"
    dummy_content = b"koshshield-verified-weights-test-content"
    dummy_file.write_bytes(dummy_content)
    expected_sha = compute_file_sha256(dummy_file)
    file_size = len(dummy_content)

    manifest: dict[str, Any] = {
        "artifacts": [
            {
                "id": "test-model",
                "category": "embedding",
                "files": [
                    {
                        "filename": "weights.safetensors",
                        "size_bytes": file_size,
                        "sha256": expected_sha,
                        "integrity_status": "VERIFIED",
                        "provenance": "https://example.com/weights.sha256",
                    }
                ],
            }
        ]
    }

    res = validate_manifest(
        manifest, repo_root=tmp_path, model_dir=model_dir, require_all_categories=False
    )
    assert res.is_valid is True
    assert len(res.blockers) == 0
    assert res.total_manifest_bytes == file_size
    assert res.existing_verified_bytes == file_size
    assert res.net_required_bytes == 0


def test_manifest_fails_on_unverified_or_invalid_integrity(tmp_path: Path) -> None:
    """Missing, empty, or unverified integrity metadata must fail validation."""
    manifest_unverified: dict[str, Any] = {
        "artifacts": [
            {
                "id": "unverified-model",
                "category": "embedding",
                "files": [
                    {
                        "filename": "weights.safetensors",
                        "size_bytes": 1000,
                        "sha256": None,
                        "integrity_status": "UNVERIFIED",
                    }
                ],
            },
            {
                "id": "bad-hash-model",
                "category": "embedding",
                "files": [
                    {
                        "filename": "config.json",
                        "size_bytes": 100,
                        "sha256": "not-a-valid-hex-digest",
                        "integrity_status": "VERIFIED",
                        "provenance": "test",
                    }
                ],
            },
            {
                "id": "qdrant",
                "category": "vector_store",
                "pinned_version": "v1.12.1",  # Mismatch with v1.15.4
                "image_reference": "qdrant/qdrant:v1.12.1",
                "image_digest": None,
                "integrity_status": "UNVERIFIED",
            },
        ]
    }

    res = validate_manifest(
        manifest_unverified,
        repo_root=tmp_path,
        model_dir=tmp_path / "models",
        require_all_categories=False,
    )
    assert res.is_valid is False
    codes = [b.code for b in res.blockers]
    assert BlockerCode.ARTIFACT_INTEGRITY_UNVERIFIED in codes
    assert BlockerCode.ARTIFACT_INTEGRITY_INVALID in codes
    assert BlockerCode.QDRANT_VERSION_MISMATCH in codes


def test_dependency_lock_incomplete_check(tmp_path: Path) -> None:
    """Detects missing or incomplete dependency lock file."""
    ok, blockers = check_dependency_lock(tmp_path)
    assert ok is False
    assert len(blockers) == 1
    assert blockers[0].code == BlockerCode.DEPENDENCY_LOCK_INCOMPLETE


def test_manifest_rejects_empty_artifact_object(tmp_path: Path) -> None:
    """Reject empty artifact objects."""
    manifest: dict[str, Any] = {"artifacts": [{}]}
    res = validate_manifest(
        manifest, repo_root=tmp_path, model_dir=tmp_path / "models", require_all_categories=False
    )
    assert res.is_valid is False
    assert any(b.code == BlockerCode.MANIFEST_INVALID for b in res.blockers)


def test_manifest_rejects_embedding_with_empty_files(tmp_path: Path) -> None:
    """Reject embedding artifact with empty files list."""
    manifest: dict[str, Any] = {
        "artifacts": [
            {
                "id": "bge-m3",
                "category": "embedding",
                "official_source": "https://huggingface.co/BAAI/bge-m3",
                "files": [],
            }
        ]
    }
    res = validate_manifest(
        manifest, repo_root=tmp_path, model_dir=tmp_path / "models", require_all_categories=False
    )
    assert res.is_valid is False
    assert any(b.code == BlockerCode.MANIFEST_INVALID for b in res.blockers)


def test_manifest_rejects_inference_binary_with_invalid_sha256(tmp_path: Path) -> None:
    """Reject inference binary with invalid SHA-256 digest format."""
    manifest: dict[str, Any] = {
        "artifacts": [
            {
                "id": "llama-server",
                "category": "inference_server",
                "official_source": "https://github.com/ggerganov/llama.cpp",
                "target_binary": "llama-server",
                "pinned_release": "v0.4.0",
                "pinned_build": "b10809",
                "pinned_commit": "5266f24",
                "estimated_size_bytes": 55000000,
                "sha256": "x",
                "integrity_status": "VERIFIED",
                "provenance": "upstream",
            }
        ]
    }
    res = validate_manifest(
        manifest, repo_root=tmp_path, model_dir=tmp_path / "models", require_all_categories=False
    )
    assert res.is_valid is False
    assert any(b.code == BlockerCode.ARTIFACT_INTEGRITY_INVALID for b in res.blockers)


def test_manifest_rejects_qdrant_with_invalid_image_digest(tmp_path: Path) -> None:
    """Reject Qdrant container image with invalid image_digest format."""
    manifest: dict[str, Any] = {
        "artifacts": [
            {
                "id": "qdrant",
                "category": "vector_store",
                "official_source": "https://github.com/qdrant/qdrant",
                "image_reference": "qdrant/qdrant:v1.15.4",
                "pinned_version": "v1.15.4",
                "estimated_size_bytes": 100000000,
                "image_digest": "x",
                "integrity_status": "VERIFIED",
                "provenance": "upstream",
            }
        ]
    }
    res = validate_manifest(
        manifest, repo_root=tmp_path, model_dir=tmp_path / "models", require_all_categories=False
    )
    assert res.is_valid is False
    assert any(b.code == BlockerCode.ARTIFACT_INTEGRITY_INVALID for b in res.blockers)


def test_manifest_rejects_negative_or_non_integer_size_bytes(tmp_path: Path) -> None:
    """Reject negative size_bytes, booleans, and numeric strings."""
    manifest_neg: dict[str, Any] = {
        "artifacts": [
            {
                "id": "bge-m3",
                "category": "embedding",
                "files": [
                    {
                        "filename": "model.safetensors",
                        "size_bytes": -100,
                        "sha256": None,
                        "integrity_status": "UNVERIFIED",
                    }
                ],
            }
        ]
    }
    res_neg = validate_manifest(
        manifest_neg,
        repo_root=tmp_path,
        model_dir=tmp_path / "models",
        require_all_categories=False,
    )
    assert res_neg.is_valid is False
    assert any(b.code == BlockerCode.MANIFEST_INVALID for b in res_neg.blockers)

    manifest_bool: dict[str, Any] = {
        "artifacts": [
            {
                "id": "bge-m3",
                "category": "embedding",
                "files": [
                    {
                        "filename": "model.safetensors",
                        "size_bytes": True,
                        "sha256": None,
                        "integrity_status": "UNVERIFIED",
                    }
                ],
            }
        ]
    }
    res_bool = validate_manifest(
        manifest_bool,
        repo_root=tmp_path,
        model_dir=tmp_path / "models",
        require_all_categories=False,
    )
    assert res_bool.is_valid is False
    assert any(b.code == BlockerCode.MANIFEST_INVALID for b in res_bool.blockers)


def test_manifest_rejects_path_traversal_and_absolute_paths(tmp_path: Path) -> None:
    """Reject filenames or target directories containing path traversal or absolute paths."""
    manifest: dict[str, Any] = {
        "artifacts": [
            {
                "id": "bge-m3",
                "category": "embedding",
                "target_dir": "/tmp/absolute_escape",
                "files": [
                    {
                        "filename": "../secret.txt",
                        "size_bytes": 100,
                        "sha256": None,
                        "integrity_status": "UNVERIFIED",
                    }
                ],
            }
        ]
    }
    res = validate_manifest(
        manifest, repo_root=tmp_path, model_dir=tmp_path / "models", require_all_categories=False
    )
    assert res.is_valid is False
    assert any(b.code == BlockerCode.MANIFEST_INVALID for b in res.blockers)


def test_dependency_lock_rejects_comment_only_file(tmp_path: Path) -> None:
    """A file containing only comments and a hash string is not a valid lock."""
    lock_file = tmp_path / "requirements-lock.txt"
    lock_file.write_text("# --hash=sha256:not-a-real-hash\n", encoding="utf-8")
    ok, blockers = check_dependency_lock(tmp_path)
    assert ok is False
    assert any(b.code == BlockerCode.DEPENDENCY_LOCK_INCOMPLETE for b in blockers)


def test_dependency_lock_rejects_unpinned_or_malformed_hashes(tmp_path: Path) -> None:
    """Reject unpinned dependencies and malformed hash strings in lock."""
    lock_file = tmp_path / "requirements-lock.txt"
    lock_file.write_text("fastapi>=0.115.0 --hash=sha256:short\n", encoding="utf-8")
    ok, blockers = check_dependency_lock(tmp_path)
    assert ok is False
    assert any(b.code == BlockerCode.DEPENDENCY_LOCK_INCOMPLETE for b in blockers)


def test_dependency_lock_rejects_incomplete_package_coverage(tmp_path: Path) -> None:
    """Reject lock file missing core runtime dependencies."""
    lock_file = tmp_path / "requirements-lock.txt"
    fake_hash = "a" * 64
    lock_file.write_text(f"fastapi==0.115.0 --hash=sha256:{fake_hash}\n", encoding="utf-8")
    ok, blockers = check_dependency_lock(tmp_path)
    assert ok is False
    assert any("missing required packages" in b.message for b in blockers)


def test_fabricated_lock_with_comment_cannot_become_ready(tmp_path: Path) -> None:
    """A lock file with fabricated pins and comment cannot become READY."""
    from scripts.provision_local_runtime import REQUIRED_RUNTIME_PACKAGES

    lock_file = tmp_path / "requirements-lock.txt"
    fake_hash = "a" * 64
    lines = ["# Verified-Resolution: darwin-arm64-cp312"]
    for pkg in REQUIRED_RUNTIME_PACKAGES:
        lines.append(f"{pkg}==0.0.0 --hash=sha256:{fake_hash}")
    lock_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, blockers = check_dependency_lock(tmp_path)
    assert ok is False
    assert any(b.code == BlockerCode.DEPENDENCY_LOCK_INCOMPLETE for b in blockers)


def test_dependency_lock_parser_supports_continuations_extras_and_markers(tmp_path: Path) -> None:
    """Dependency lock parser supports standard line continuations, extras, and markers."""
    from scripts.provision_local_runtime import parse_dependency_lock_requirements

    content = (
        "# Comment header\n"
        "fastapi[all]==0.115.0 \\\n"
        "    --hash=sha256:" + ("a" * 64) + " \\\n"
        "    --hash=sha256:" + ("b" * 64) + "\n"
        "torch==2.3.0; sys_platform == 'darwin' \\\n"
        "    --hash=sha256:" + ("c" * 64) + "\n"
    )
    is_valid, found_pkgs, errors = parse_dependency_lock_requirements(content)
    assert is_valid is True
    assert "fastapi" in found_pkgs
    assert "torch" in found_pkgs
    assert len(errors) == 0


def test_manifest_rejects_verified_status_when_provenance_is_missing(tmp_path: Path) -> None:
    """Require recorded upstream checksum provenance before accepting VERIFIED status."""
    valid_hash = "a" * 64
    manifest: dict[str, Any] = {
        "artifacts": [
            {
                "id": "bge-m3",
                "category": "embedding",
                "official_source": "https://huggingface.co/BAAI/bge-m3",
                "files": [
                    {
                        "filename": "model.safetensors",
                        "size_bytes": 1000,
                        "sha256": valid_hash,
                        "integrity_status": "VERIFIED",
                        # provenance is omitted / None
                    }
                ],
            }
        ]
    }
    res = validate_manifest(
        manifest,
        repo_root=tmp_path,
        model_dir=tmp_path / "models",
        require_all_categories=False,
    )
    assert res.is_valid is False
    assert any(
        b.code == BlockerCode.ARTIFACT_INTEGRITY_UNVERIFIED and "provenance" in b.message.lower()
        for b in res.blockers
    )


def test_assess_readiness_requires_manifest_version_unconditionally(
    tmp_path: Path,
) -> None:
    """Production readiness path unconditionally requires manifest_version and all categories."""
    manifest_file = tmp_path / "partial_manifest.json"
    manifest_content = {
        # manifest_version intentionally omitted
        "artifacts": [
            {
                "id": "bge-m3",
                "category": "embedding",
                "official_source": "https://huggingface.co/BAAI/bge-m3",
                "files": [
                    {
                        "filename": "model.safetensors",
                        "size_bytes": 1000,
                        "sha256": None,
                        "integrity_status": "UNVERIFIED",
                    }
                ],
            }
        ]
    }
    import json

    manifest_file.write_text(json.dumps(manifest_content), encoding="utf-8")

    report = assess_readiness(
        manifest_path=manifest_file,
        repo_root=tmp_path,
        model_dir=tmp_path / "models",
    )
    assert report.status == "BLOCKED"
    messages = " ".join(b.message for b in report.blockers)
    assert "manifest_version" in messages.lower()
    assert "missing required runtime categories" in messages.lower()


def test_main_json_output_with_malformed_manifest_does_not_crash(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """main(['--json']) with malformed synthetic manifests returns valid JSON without traceback."""
    import json

    bad_manifest = tmp_path / "bad_manifest.json"
    bad_manifest.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "artifacts": [
                    {
                        "id": "bge-m3",
                        "category": "embedding",
                        "files": [
                            {
                                "filename": "model.safetensors",
                                "size_bytes": -100,  # triggers ValueError in Pydantic
                                "sha256": None,
                                "integrity_status": "UNVERIFIED",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["--dry-run", "--json", "--manifest", str(bad_manifest)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    parsed = json.loads(captured.out)
    assert parsed["status"] == "BLOCKED"
    assert len(parsed["blockers"]) > 0


def test_artifact_id_path_traversal_is_blocked_and_out_of_root_files_never_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validate identifiers and destinations: out-of-root paths must be rejected and never read."""
    model_dir = tmp_path / "data" / "models"
    model_dir.mkdir(parents=True)
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir(parents=True)

    secret_file = outside_dir / "weights.safetensors"
    secret_file.write_bytes(b"forbidden-content-never-read")

    manifest: dict[str, Any] = {
        "artifacts": [
            {
                "id": "../outside_dir",
                "category": "embedding",
                "files": [
                    {
                        "filename": "weights.safetensors",
                        "size_bytes": len(b"forbidden-content-never-read"),
                        "sha256": "a" * 64,
                        "integrity_status": "VERIFIED",
                        "provenance": "upstream",
                    }
                ],
            }
        ]
    }

    # Spy on compute_file_sha256: it must NEVER be called on secret_file
    read_paths: list[Path] = []
    real_compute = compute_file_sha256

    def spy_compute(p: Path, *args: Any, **kwargs: Any) -> str:
        read_paths.append(p.resolve())
        return real_compute(p, *args, **kwargs)

    monkeypatch.setattr("scripts.provision_local_runtime.compute_file_sha256", spy_compute)

    res = validate_manifest(
        manifest,
        repo_root=tmp_path,
        model_dir=model_dir,
        require_all_categories=False,
    )
    assert res.is_valid is False
    assert any(b.code == BlockerCode.MANIFEST_INVALID for b in res.blockers)
    # Crucial guarantee: secret_file was NEVER read or hashed
    assert secret_file.resolve() not in read_paths


def test_external_filesystem_deducts_verified_model_artifacts(tmp_path: Path) -> None:
    """Verify local verified model files on an external mount are deducted from required space."""
    model_mount = tmp_path / "ext_models"
    repo_mount = tmp_path / "repo"
    model_mount.mkdir()
    repo_mount.mkdir()

    target_model_dir = model_mount / "bge-m3"
    target_model_dir.mkdir(parents=True)
    f = target_model_dir / "model.safetensors"
    content = b"sample-model-weights"
    f.write_bytes(content)
    file_sha = compute_file_sha256(f)
    file_size = len(content)

    manifest: dict[str, Any] = {
        "artifacts": [
            {
                "id": "bge-m3",
                "category": "embedding",
                "target_dir": "data/models/bge-m3",
                "files": [
                    {
                        "filename": "model.safetensors",
                        "size_bytes": file_size,
                        "sha256": file_sha,
                        "integrity_status": "VERIFIED",
                        "provenance": "https://example.com/bge-m3.sha256",
                    }
                ],
            }
        ]
    }
    res = validate_manifest(
        manifest,
        repo_root=repo_mount,
        model_dir=model_mount,
        require_all_categories=False,
    )
    assert res.existing_verified_bytes == file_size
    assert res.net_model_artifact_bytes == 0

    stat_model = MagicMock(st_dev=100)
    stat_repo = MagicMock(st_dev=200)

    def fake_stat(self: Path, *args: Any, **kwargs: Any) -> Any:
        if "ext_models" in str(self):
            return stat_model
        return stat_repo

    usage_model = shutil._ntuple_diskusage(
        total=100 * GIB_BYTES, used=50 * GIB_BYTES, free=15 * GIB_BYTES
    )
    usage_repo = shutil._ntuple_diskusage(
        total=50 * GIB_BYTES, used=10 * GIB_BYTES, free=40 * GIB_BYTES
    )

    def fake_usage(path: Any) -> Any:
        if "ext_models" in str(path):
            return usage_model
        return usage_repo

    with (
        patch.object(Path, "stat", fake_stat),
        patch("shutil.disk_usage", fake_usage),
    ):
        report = check_capacity(
            repo_root=repo_mount,
            model_dir=model_mount,
            net_artifact_bytes=res.net_required_bytes,
            net_model_artifact_bytes=res.net_model_artifact_bytes,
            net_non_model_artifact_bytes=res.net_non_model_artifact_bytes,
        )

    # Net model artifact bytes is 0; required on model mount is strictly 10 GiB headroom
    assert report.filesystems[0].required_bytes == RESERVED_HEADROOM_BYTES
    assert report.filesystems[0].is_sufficient is True


def test_apply_fails_closed_when_prerequisites_blocked(capsys: pytest.CaptureFixture[str]) -> None:
    """Apply must abort immediately with code 1 if any prerequisite is blocked."""
    exit_code = main(["--apply"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Apply cannot proceed because prerequisite checks failed (BLOCKED)" in captured.err


def test_apply_unimplemented_fails_closed_with_code_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When prerequisites are mocked READY, apply must exit 2 (APPLY_NOT_IMPLEMENTED), never 0."""
    from scripts.provision_local_runtime import (
        CapacityReport,
        ManifestValidationResult,
        ProvisioningReport,
    )

    mock_report = ProvisioningReport(
        mode="apply",
        status="READY",
        blockers=[],
        capacity=CapacityReport(is_sufficient=True),
        manifest_validation=ManifestValidationResult(is_valid=True),
        dependency_lock_status="READY",
        timestamp="2026-09-12T00:00:00Z",
    )

    monkeypatch.setattr(
        "scripts.provision_local_runtime.assess_readiness", lambda **kw: mock_report
    )

    exit_code = main(["--apply"])
    assert exit_code == 2  # Nonzero code representing APPLY_NOT_IMPLEMENTED


def test_directory_immutability_during_dry_run(tmp_path: Path) -> None:
    """Verify that running dry-run leaves the repo and target directories completely untouched."""
    model_dir = tmp_path / "target_models"
    assert not model_dir.exists()

    report = assess_readiness(
        manifest_path=REPO_ROOT / "docs" / "runtime_artifacts_manifest.json",
        model_dir=model_dir,
        repo_root=REPO_ROOT,
        mode="dry-run",
    )
    assert report.status == "BLOCKED"
    # Target directory was not created or modified
    assert not model_dir.exists()


def test_format_report_structure() -> None:
    """Verify report formatting output contains stable section headers and metrics."""
    report = assess_readiness()
    text = format_report(report)
    assert "KoshShield AI Local Runtime Provisioning Guardrails Report" in text
    assert "--- CAPACITY ASSESSMENT" in text
    assert "Reserved Headroom:" in text
    assert "--- ACTIVE BLOCKERS" in text
    assert "ARTIFACT_INTEGRITY_UNVERIFIED" in text


def test_manifest_contains_verified_upstream_metadata_and_correct_filenames() -> None:
    """Verify that authoritative manifest contains reconciled upstream identities and digests."""
    from scripts.provision_local_runtime import DEFAULT_MANIFEST_PATH, load_manifest

    manifest = load_manifest(DEFAULT_MANIFEST_PATH)
    artifacts = {a["id"]: a for a in manifest["artifacts"]}

    # 1. BGE-M3
    bge = artifacts["bge-m3"]
    bge_files = {f["filename"]: f for f in bge["files"]}
    assert "pytorch_model.bin" in bge_files
    assert "model.safetensors" not in bge_files  # Corrected from guessed name
    assert bge_files["pytorch_model.bin"]["size_bytes"] == 2271145830
    assert (
        bge_files["pytorch_model.bin"]["sha256"]
        == "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38"
    )
    assert bge_files["pytorch_model.bin"]["integrity_status"] == "VERIFIED"
    assert bge_files["config.json"]["integrity_status"] == "UNVERIFIED"

    # 2. Qwen3-VL
    qwen = artifacts["qwen3-vl-4b-gguf"]
    qwen_files = {f["filename"]: f for f in qwen["files"]}
    assert "Qwen3VL-4B-Instruct-Q4_K_M.gguf" in qwen_files
    assert "mmproj-Qwen3VL-4B-Instruct-F16.gguf" in qwen_files
    assert qwen_files["Qwen3VL-4B-Instruct-Q4_K_M.gguf"]["size_bytes"] == 2497281664
    assert (
        qwen_files["Qwen3VL-4B-Instruct-Q4_K_M.gguf"]["sha256"]
        == "66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a"
    )
    assert qwen_files["mmproj-Qwen3VL-4B-Instruct-F16.gguf"]["size_bytes"] == 836180256
    assert (
        qwen_files["mmproj-Qwen3VL-4B-Instruct-F16.gguf"]["sha256"]
        == "256f3a43bd4205ffef48d6b92715e1e70b5b0e9aef06522584967513a9985331"
    )

    # 3. PaddleOCR
    ocr = artifacts["paddleocr-v4-en"]
    assert ocr["detection"]["integrity_status"] == "UNVERIFIED"
    assert ocr["detection"]["sha256"] is None
    assert ocr["recognition"]["integrity_status"] == "UNVERIFIED"
    assert ocr["recognition"]["sha256"] is None

    # 4. llama-server
    llama = artifacts["llama-server"]
    assert llama["target_binary"] == "llama-server"
    assert llama["estimated_size_bytes"] == 11123196
    assert llama["integrity_status"] == "UNVERIFIED"
    assert llama["sha256"] is None

    # 5. Qdrant
    qdrant = artifacts["qdrant"]
    assert (
        qdrant["image_digest"]
        == "sha256:cc374f58d68768be3b83a78c2d57782eb1661e5c9c63e7c495d40c76c539f469"
    )
    assert qdrant["integrity_status"] == "VERIFIED"
    assert qdrant["estimated_size_bytes"] == 65266145


def test_authoritative_manifest_retains_fail_closed_blocking() -> None:
    """The authoritative manifest must remain fail-closed BLOCKED due to unverified items."""
    from scripts.provision_local_runtime import DEFAULT_MANIFEST_PATH, load_manifest

    manifest = load_manifest(DEFAULT_MANIFEST_PATH)
    res = validate_manifest(manifest, repo_root=REPO_ROOT, model_dir=REPO_ROOT / "data" / "models")
    assert res.is_valid is False
    assert len(res.blockers) > 0
    assert any(b.code == BlockerCode.ARTIFACT_INTEGRITY_UNVERIFIED for b in res.blockers)

    report = assess_readiness()
    assert report.status == "BLOCKED"
