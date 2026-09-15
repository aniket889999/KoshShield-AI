"""Synthetic-wheel checks for offline dependency preparation; no installed package mutations."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.runtime_dependencies import (  # noqa: E402
    DependencyPreparationError,
    inspect_wheel,
    inventory_wheels,
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
