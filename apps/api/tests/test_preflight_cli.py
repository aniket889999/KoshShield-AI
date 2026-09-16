import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from koshshield import runtime_preflight as preflight
from koshshield.config import Settings


def test_default_inspection_never_invokes_services_or_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = []
    for name in ("check_llama_cpp", "check_qdrant", "check_isolated_storage"):
        mock = MagicMock(side_effect=AssertionError("Must not be invoked"))
        monkeypatch.setattr(preflight, name, mock)
        blocked.append(mock)
    result = preflight.run_runtime_preflight(Settings(_env_file=None))
    for mock in blocked:
        mock.assert_not_called()
    assert result.runtime_verified is False
    assert result.generation_executed is False
    assert result.service_probes_enabled is False
    assert result.storage_probe_enabled is False
    assert result.status == "NOT_READY"


@pytest.mark.parametrize(("services", "storage"), [(True, False), (False, True), (True, True)])
def test_probe_flags_are_independent(
    monkeypatch: pytest.MonkeyPatch, services: bool, storage: bool
) -> None:
    ready = preflight.CheckResult(preflight.PrerequisiteStatus.READY, "synthetic test")
    mocks = {}
    for name in ("llama_cpp", "qdrant", "isolated_storage"):
        mocks[name] = MagicMock(return_value=ready)
        monkeypatch.setattr(preflight, f"check_{name}", mocks[name])
    result = preflight.run_runtime_preflight(
        Settings(_env_file=None), probe_services=services, probe_storage=storage
    )
    assert mocks["llama_cpp"].call_count == int(services)
    assert mocks["qdrant"].call_count == int(services)
    assert mocks["isolated_storage"].call_count == int(storage)
    assert result.service_probes_enabled is services
    assert result.storage_probe_enabled is storage
    assert result.runtime_verified is False


def test_default_cli_under_no_network_no_write_audit_guard(tmp_path: Path) -> None:
    script = r"""
import os
import sys
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
        raise RuntimeError("forbidden preflight side effect")
sys.addaudithook(audit)
from koshshield.runtime_preflight import main
code = main([])
assert not violations, violations
assert not ({"torch", "FlagEmbedding", "paddle", "paddleocr"} & set(sys.modules))
sys.exit(code)
"""
    env = {key: value for key, value in os.environ.items() if not key.startswith("KOSHSHIELD_")}
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    assert "Traceback" not in result.stderr
    report = json.loads(result.stdout)
    assert report["prerequisites"]["dependencies"]["status"] == "READY"
    for name in ("llama_cpp", "qdrant", "isolated_storage"):
        assert report["prerequisites"][name]["status"] == "NOT_EXECUTED"
    assert list(tmp_path.iterdir()) == []


def test_cli_invalid_configuration_has_json_without_secret(tmp_path: Path) -> None:
    env = {key: value for key, value in os.environ.items() if not key.startswith("KOSHSHIELD_")}
    env["KOSHSHIELD_MAX_UPLOAD_BYTES"] = "SECRET_INVALID_ENV_VALUE"
    result = subprocess.run(
        [sys.executable, "-B", "-m", "koshshield.runtime_preflight"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert (
        report["prerequisites"]["configuration"]["metadata"]["failure_code"]
        == "CONFIGURATION_INVALID"
    )
    assert "SECRET_INVALID_ENV_VALUE" not in result.stdout + result.stderr
    assert "Traceback" not in result.stderr


def test_cli_passes_flags_and_restores_logging(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import logging

    report = preflight.run_runtime_preflight(Settings(_env_file=None))
    run = MagicMock(return_value=report)
    monkeypatch.setattr(preflight, "run_runtime_preflight", run)
    previous = logging.root.manager.disable
    assert preflight.main(["--probe-services", "--probe-storage"]) == 1
    run.assert_called_once_with(probe_services=True, probe_storage=True)
    assert logging.root.manager.disable == previous
    assert json.loads(capsys.readouterr().out)["runtime_verified"] is False
