import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from koshshield.evaluation.fixtures import default_fixture_dir
from koshshield.evaluation.reference import run_reference_evaluation


def test_reference_reports_calculations_separately_from_model_quality() -> None:
    result = run_reference_evaluation()
    assert result["status"] == "PASSED"
    assert result["fixture_integrity_verified"] is True
    assert result["comparison_checks"] == 12
    assert result["measurement_checks"] == 3
    assert result["missing_proposal_values"] == 1
    assert result["comparison_mismatches"] == result["measurement_mismatches"] == 0
    assert result["model_evaluation_executed"] is False
    for pii in ("TESTP0000A", "9000000000", "proc001@example.invalid"):
        assert pii not in json.dumps(result)


def test_consistent_hashes_do_not_hide_wrong_expected_labels(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    shutil.copytree(default_fixture_dir(), root)
    relative = "expected/proc001-ground-truth.json"
    expected = json.loads((root / relative).read_text())
    expected["comparison_expected"]["Supplier A"]["R1"] = "CONTRADICTED"
    content = json.dumps(expected).encode()
    (root / relative).write_bytes(content)
    manifest = json.loads((root / "manifest.json").read_text())
    for entry in manifest["files"]:
        if entry["path"] == relative:
            entry.update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    (root / "manifest.json").write_text(json.dumps(manifest))
    result = run_reference_evaluation(root)
    assert result["fixture_integrity_verified"] is True
    assert result["status"] == "FAILED"
    assert result["failure_code"] == "REFERENCE_RESULT_MISMATCH"
    assert result["comparison_mismatches"] == 1


def test_reference_cli_missing_fixture_is_json_not_traceback(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "koshshield.evaluation.reference",
            "--fixture-dir",
            str(tmp_path / "private-secret"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["failure_code"] == "FIXTURE_UNREADABLE"
    assert "private-secret" not in result.stdout + result.stderr
    assert "Traceback" not in result.stderr


def test_reference_cli_has_no_network_writes_or_model_imports(tmp_path: Path) -> None:
    script = r"""
import os, sys
violations = []
def audit(event, args):
    forbidden = event.startswith("socket.") or event in {
        "subprocess.Popen", "os.system", "os.mkdir", "os.remove", "os.rmdir",
        "os.rename", "os.link", "os.symlink", "os.truncate"
    }
    if event == "open":
        _, mode, flags = args
        forbidden = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
    if forbidden:
        violations.append(event)
        raise RuntimeError("Unexpected side effect")
sys.addaudithook(audit)
from koshshield.evaluation.reference import main
code = main([])
assert not violations, violations
assert not ({"torch", "FlagEmbedding", "paddle", "paddleocr", "qdrant_client"} & set(sys.modules))
sys.exit(code)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "PASSED"
    assert list(tmp_path.iterdir()) == []
