"""Synthetic-wheel checks for offline dependency preparation; no installed package mutations."""

from __future__ import annotations

import copy
import hashlib
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.runtime_dependencies import (  # noqa: E402
    DependencyPreparationError,
    build_resolution_evidence,
    compatible_wheels,
    inspect_wheel,
    inventory_wheels,
    local_requirement,
    missing_root_requirements,
    resolve_offline,
    runtime_requirements,
)


def make_wheel(
    root: Path,
    name: str = "demo",
    version: str = "1.0",
    requirements: tuple[str, ...] = (),
    requires_python: str = ">=3.12",
    tag: str = "py3-none-any",
    metadata_name: str | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}-{version}-{tag}.whl"
    metadata = (
        f"Metadata-Version: 2.1\nName: {metadata_name or name}\nVersion: {version}\n"
        f"Requires-Python: {requires_python}\n"
        + "".join(f"Requires-Dist: {item}\n" for item in requirements)
        + "\n"
    )
    with ZipFile(path, "w") as archive:
        prefix = f"{name}-{version}.dist-info"
        archive.writestr(f"{prefix}/METADATA", metadata)
        archive.writestr(
            f"{prefix}/WHEEL", f"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: {tag}\n"
        )
        archive.writestr(f"{prefix}/RECORD", "")
    return path


def test_inventory_records_identity_metadata_and_exact_hash(tmp_path: Path) -> None:
    path = make_wheel(tmp_path, requirements=("child>=1",))
    before = set(tmp_path.iterdir())
    wheel = inventory_wheels(tmp_path)[0]
    assert wheel.name == "demo"
    assert wheel.requirements == ("child>=1",)
    assert wheel.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert wheel.size_bytes == path.stat().st_size
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("case", ["missing", "empty", "invalid_zip", "identity"])
def test_invalid_wheelhouse_is_rejected(tmp_path: Path, case: str) -> None:
    root = tmp_path / "wheels"
    if case != "missing":
        root.mkdir()
    if case == "invalid_zip":
        (root / "demo-1.0-py3-none-any.whl").write_bytes(b"not a zip")
    elif case == "identity":
        make_wheel(root, metadata_name="different")
    with pytest.raises(DependencyPreparationError):
        inventory_wheels(root)


def test_symlink_wheel_is_not_read(tmp_path: Path) -> None:
    outside = make_wheel(tmp_path / "outside")
    root = tmp_path / "wheels"
    root.mkdir()
    link = root / outside.name
    link.symlink_to(outside)
    with pytest.raises(DependencyPreparationError, match="WHEEL_PATH_UNSAFE"):
        inspect_wheel(link, root)


def test_compatibility_filters_python_and_platform(tmp_path: Path) -> None:
    make_wheel(tmp_path, name="usable")
    make_wheel(tmp_path, name="future", requires_python=">=99")
    make_wheel(tmp_path, name="foreign", tag="cp27-none-win32")
    wheels = compatible_wheels(inventory_wheels(tmp_path), {"py3-none-any"}, "3.12.0")
    assert [wheel.name for wheel in wheels] == ["usable"]
    assert missing_root_requirements(wheels, ("usable>=1", "missing", "usable>=2")) == [
        "missing",
        "usable>=2",
    ]


@pytest.mark.parametrize("reference", ["https://example.invalid/pkg.whl", "file:///tmp/pkg.whl"])
def test_direct_references_are_rejected_in_roots_and_wheel_metadata(
    tmp_path: Path,
    reference: str,
) -> None:
    requirement = f"child @ {reference}"
    with pytest.raises(DependencyPreparationError, match="DIRECT_REFERENCE_FORBIDDEN"):
        local_requirement(requirement)
    make_wheel(tmp_path, requirements=(requirement,))
    with pytest.raises(DependencyPreparationError, match="DIRECT_REFERENCE_FORBIDDEN"):
        compatible_wheels(inventory_wheels(tmp_path))


def test_project_constraints_are_preserved_and_models_added(tmp_path: Path) -> None:
    project = tmp_path / "apps/api/pyproject.toml"
    project.parent.mkdir(parents=True)
    project.write_text(
        '[project]\ndependencies = ["fastapi>=0.115,<1", "uvicorn[standard]>=0.34"]\n'
    )
    requirements = runtime_requirements(tmp_path)
    assert "fastapi<1,>=0.115" in requirements
    assert "uvicorn[standard]>=0.34" in requirements
    assert set(("FlagEmbedding", "torch", "paddleocr", "transformers")) <= set(requirements)


def test_real_pip_resolves_synthetic_local_wheels_without_installing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib.metadata import distributions

    make_wheel(tmp_path, name="fixture_parent", requirements=("fixture-child>=1",))
    make_wheel(tmp_path, name="fixture_child")
    before = sorted((dist.metadata["Name"], dist.version) for dist in distributions())
    real_run = subprocess.run
    calls = []

    def capture_run(command, **kwargs):
        calls.append((command, kwargs))
        return real_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", capture_run)
    report = resolve_offline(tmp_path, ("fixture-parent",))
    assert {item["metadata"]["name"] for item in report["install"]} == {
        "fixture_parent",
        "fixture_child",
    }
    assert sorted((dist.metadata["Name"], dist.version) for dist in distributions()) == before
    command, options = calls[0]
    assert {
        "--dry-run",
        "--ignore-installed",
        "--no-index",
        "--only-binary=:all:",
        "--no-cache-dir",
    } <= set(command)
    assert options["env"]["PIP_CONFIG_FILE"] == "/dev/null"
    assert "PIP_INDEX_URL" not in options["env"]


def test_real_pip_reports_transitive_constraint_conflict(tmp_path: Path) -> None:
    make_wheel(tmp_path, name="fixture_parent", requirements=("fixture-child<2",))
    make_wheel(tmp_path, name="fixture_child", version="2.0")
    with pytest.raises(DependencyPreparationError, match="OFFLINE_RESOLUTION_FAILED"):
        resolve_offline(tmp_path, ("fixture-parent", "fixture-child>=2"))


def test_missing_root_wheels_do_not_launch_pip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_wheel(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("pip must not run when root wheels are missing")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(DependencyPreparationError, match="ROOT_WHEELS_MISSING"):
        resolve_offline(tmp_path, ("missing",))


@pytest.fixture
def resolved_fixture(tmp_path: Path):
    make_wheel(tmp_path, name="fixture_parent", requirements=("fixture-child>=1",))
    make_wheel(tmp_path, name="fixture_child")
    roots = ("fixture-parent",)
    return tmp_path, roots, resolve_offline(tmp_path, roots)


def test_resolution_evidence_binds_wheels_roots_and_lock(resolved_fixture) -> None:
    wheelhouse, roots, report = resolved_fixture
    lock, evidence = build_resolution_evidence(wheelhouse, roots, report)
    assert evidence["lock_sha256"] == hashlib.sha256(lock.encode()).hexdigest()
    assert evidence["source"] == "pip-dry-run-no-index"
    assert len(evidence["wheels"]) == 2
    assert "fixture-parent==1.0 --hash=sha256:" in lock
    assert lock == build_resolution_evidence(wheelhouse, roots, report)[0]


@pytest.mark.parametrize("tamper", ["remote", "hash", "environment", "missing_child"])
def test_resolution_evidence_rejects_inconsistent_reports(resolved_fixture, tamper: str) -> None:
    wheelhouse, roots, original = resolved_fixture
    report = copy.deepcopy(original)
    if tamper == "remote":
        report["install"][0]["download_info"]["url"] = "https://example.invalid/package.whl"
    elif tamper == "hash":
        report["install"][0]["download_info"]["archive_info"]["hashes"]["sha256"] = "0" * 64
    elif tamper == "environment":
        report["environment"]["python_version"] = "0.0"
    else:
        report["install"] = [
            item for item in report["install"] if item["metadata"]["name"] == "fixture_parent"
        ]
    with pytest.raises(DependencyPreparationError):
        build_resolution_evidence(wheelhouse, roots, report)
