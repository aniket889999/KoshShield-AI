import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from koshshield import runtime_preflight as preflight
from koshshield.config import Settings
from koshshield.services.retrieval.llama_cpp_client import (
    LlamaCppIncapableError,
    LlamaCppSecurityError,
    LlamaCppUnavailableError,
)

CHECKS = (
    "bge_m3",
    "qwen_gguf",
    "ocr_models",
    "dependencies",
    "llama_cpp",
    "qdrant",
    "isolated_storage",
)
SECRET = "/private/operator-secret?token=credential"


@pytest.mark.parametrize("broken", CHECKS)
def test_one_failed_check_does_not_abort_report(
    monkeypatch: pytest.MonkeyPatch, broken: str
) -> None:
    completed = []

    def succeed(*_):
        completed.append(True)
        return preflight.CheckResult(preflight.PrerequisiteStatus.READY, "synthetic test check")

    def fail(*_):
        raise OSError(SECRET)

    for name in CHECKS:
        monkeypatch.setattr(preflight, f"check_{name}", fail if name == broken else succeed)
    result = preflight.run_runtime_preflight(
        Settings(_env_file=None), probe_services=True, probe_storage=True
    )
    assert len(completed) == len(CHECKS) - 1
    assert result.status == "NOT_READY"
    assert result.missing_categories == [broken]
    assert result.prerequisites[broken]["metadata"]["failure_code"] == "CHECK_FAILED"
    assert SECRET not in json.dumps(asdict(result))


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (LlamaCppIncapableError, "LLAMA_CONTRACT_MISMATCH"),
        (LlamaCppSecurityError, "LLAMA_SECURITY_REJECTED"),
        (LlamaCppUnavailableError, "LLAMA_UNAVAILABLE"),
        (RuntimeError, "LLAMA_CHECK_FAILED"),
    ],
)
def test_llama_failure_details_are_sanitized(error: type[Exception], code: str) -> None:
    with patch(
        "koshshield.services.retrieval.llama_cpp_client.LlamaCppMultimodalClient",
        side_effect=error(SECRET),
    ):
        result = preflight.check_llama_cpp(Settings(_env_file=None))
    assert result.metadata["failure_code"] == code
    assert SECRET not in str(result)


def test_invalid_settings_produce_structured_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def invalid():
        raise ValueError(SECRET)

    monkeypatch.setattr(preflight, "get_settings", invalid)
    result = preflight.run_runtime_preflight()
    assert result.missing_categories == ["configuration"]
    assert (
        result.prerequisites["configuration"]["metadata"]["failure_code"] == "CONFIGURATION_INVALID"
    )
    assert SECRET not in json.dumps(asdict(result))


def test_scratch_creation_failure_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        preflight.tempfile, "TemporaryDirectory", MagicMock(side_effect=OSError(SECRET))
    )
    result = preflight.check_isolated_storage()
    assert result.metadata["failure_code"] == "SCRATCH_STORAGE_FAILED"
    assert SECRET not in str(result)


def test_scratch_failure_disposes_engine_before_removing_directory(tmp_path: Path) -> None:
    scratch = tmp_path / "owned-scratch"
    scratch.mkdir()
    engine = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = str(scratch)
    context.__exit__.side_effect = lambda *_: engine.dispose.assert_called_once()
    with (
        patch.object(preflight.tempfile, "TemporaryDirectory", return_value=context),
        patch("sqlalchemy.create_engine", return_value=engine),
        patch("koshshield.database.Base.metadata.create_all", side_effect=RuntimeError(SECRET)),
    ):
        result = preflight.check_isolated_storage()
    assert result.status == preflight.PrerequisiteStatus.ERROR
    context.__exit__.assert_called_once()
    assert SECRET not in str(result)
