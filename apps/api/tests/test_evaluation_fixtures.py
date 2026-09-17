import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from koshshield.evaluation.fixtures import FixtureValidationError, load_fixture_bundle
from koshshield.smoke import _find_fixture_dir, run_smoke


def bundle(root: Path, path: str = "inputs/demo.pdf", content: bytes = b"%PDF-demo") -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "case_id": "DEMO",
                "synthetic": True,
                "files": [
                    {
                        "path": path,
                        "role": "input",
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                ],
            }
        )
    )


def test_real_fixture_manifest_and_snapshot() -> None:
    result = load_fixture_bundle(_find_fixture_dir())
    assert result.case_id == "PROC-001"
    assert len(result.files) == 8
    assert result.input_pdf("proc001-native.pdf").startswith(b"%PDF-")
    assert result.evaluation_json("expected/proc001-ground-truth.json")["case_id"] == result.case_id
    with pytest.raises(TypeError):
        result.files["new"] = b"data"


def test_fixture_inputs_use_verified_bytes_not_later_disk_content(tmp_path: Path) -> None:
    bundle(tmp_path)
    result = load_fixture_bundle(tmp_path)
    (tmp_path / "inputs/demo.pdf").write_bytes(b"tampered later")
    assert result.input_pdf("demo.pdf") == b"%PDF-demo"


@pytest.mark.parametrize(
    "change", ["hash", "size", "duplicate", "traversal", "absolute", "bool_size"]
)
def test_invalid_manifest_or_file_is_rejected(tmp_path: Path, change: str) -> None:
    bundle(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    entry = manifest["files"][0]
    if change == "hash":
        entry["sha256"] = "a" * 64
    elif change == "size":
        entry["bytes"] += 1
    elif change == "duplicate":
        manifest["files"].append(entry.copy())
    elif change == "bool_size":
        entry["bytes"] = True
    else:
        entry["path"] = "../outside.pdf" if change == "traversal" else "/private/secret.pdf"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(FixtureValidationError):
        load_fixture_bundle(tmp_path)


def test_parent_symlink_cannot_escape_fixture_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    bundle(root)
    (root / "inputs").rename(tmp_path / "outside")
    (root / "inputs").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(FixtureValidationError, match="FIXTURE_PATH_UNSAFE"):
        load_fixture_bundle(root)


@pytest.mark.parametrize("name", ["../source_case.json", "source_case.json", "unlisted.pdf"])
def test_evaluation_files_and_unlisted_inputs_cannot_be_ingested(name: str) -> None:
    with pytest.raises(FixtureValidationError, match="FIXTURE_INPUT_FORBIDDEN"):
        load_fixture_bundle(_find_fixture_dir()).input_pdf(name)


def test_integrity_failure_stops_smoke_before_provider_execution(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    shutil.copytree(_find_fixture_dir(), root)
    (root / "inputs/proc001-native.pdf").write_bytes(b"changed")
    with (
        patch("koshshield.smoke._find_fixture_dir", return_value=root),
        patch("koshshield.smoke.run_single_variant") as run,
    ):
        result = run_smoke()
    run.assert_not_called()
    assert result.status == "FAILED"
    assert result.failure_code == "FIXTURE_INTEGRITY_MISMATCH"
    assert not any(result.real_providers_executed.values())
    assert str(root) not in str(result)
