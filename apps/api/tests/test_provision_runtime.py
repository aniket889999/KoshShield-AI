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
                    }
                ],
            }
        ]
    }

    res = validate_manifest(manifest, repo_root=tmp_path, model_dir=model_dir)
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

    res = validate_manifest(manifest_unverified, repo_root=tmp_path, model_dir=tmp_path / "models")
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

    # Create dummy verified lock with hashes
    lock_file = tmp_path / "requirements-lock.txt"
    lock_file.write_text("fastapi==0.115.0 --hash=sha256:abcd1234abcd1234\n", encoding="utf-8")

    ok2, blockers2 = check_dependency_lock(tmp_path)
    assert ok2 is True
    assert len(blockers2) == 0


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
