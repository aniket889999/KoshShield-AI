"""Unit and contract tests for runtime preflight and smoke harness."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from koshshield.runtime_preflight import (
    CheckResult,
    PrerequisiteStatus,
    check_dependencies,
    check_isolated_storage,
    run_runtime_preflight,
)
from koshshield.smoke import (
    FORBIDDEN_SYNTHETIC_PII,
    run_smoke,
)


def test_runtime_preflight_reports_honest_structure_and_false_generation() -> None:
    report = run_runtime_preflight()

    assert report.report_type == "runtime_preflight"
    assert report.generation_executed is False
    assert report.status in {"READY", "NOT_READY"}

    expected_prereqs = {
        "bge_m3",
        "qwen_gguf",
        "ocr_models",
        "dependencies",
        "llama_cpp",
        "qdrant",
        "isolated_storage",
    }
    assert set(report.prerequisites.keys()) == expected_prereqs

    # Dependencies and isolated storage should always succeed in dev env
    assert report.prerequisites["dependencies"]["status"] == "READY"
    assert report.prerequisites["isolated_storage"]["status"] == "READY"

    # Verify no raw env or secret values leaked in details
    for _name, p in report.prerequisites.items():
        assert "status" in p
        assert "details" in p
        for forbidden in FORBIDDEN_SYNTHETIC_PII:
            assert forbidden not in p["details"]


def test_runtime_preflight_dependencies_and_isolated_storage_pass() -> None:
    dep_res = check_dependencies()
    assert dep_res.status == PrerequisiteStatus.READY

    storage_res = check_isolated_storage()
    assert storage_res.status == PrerequisiteStatus.READY


def test_runtime_preflight_ready_when_all_prerequisites_met() -> None:
    ready = CheckResult(status=PrerequisiteStatus.READY, details="verified ready")
    with (
        patch("koshshield.runtime_preflight.check_bge_m3", return_value=ready),
        patch("koshshield.runtime_preflight.check_qwen_gguf", return_value=ready),
        patch("koshshield.runtime_preflight.check_ocr_models", return_value=ready),
        patch("koshshield.runtime_preflight.check_llama_cpp", return_value=ready),
        patch("koshshield.runtime_preflight.check_qdrant", return_value=ready),
    ):
        report = run_runtime_preflight()
        assert report.status == "READY"
        assert len(report.missing_categories) == 0
        assert report.generation_executed is False


def test_smoke_harness_honest_execution_and_clean_pii() -> None:
    report = run_smoke()

    assert report.report_type == "smoke_local"
    assert report.synthetic_fixture == "PROC-001"
    assert "disclaimer" in report.__dict__
    assert report.real_providers_configured["bge_m3"] is True
    assert report.real_providers_configured["qdrant"] is True
    assert report.real_providers_configured["ocr"] is True
    assert report.real_providers_configured["llama_cpp"] is True

    # When services are missing, executed flags must all be False
    assert report.real_providers_executed["bge_m3"] is False
    assert report.real_providers_executed["qdrant"] is False
    assert report.real_providers_executed["ocr"] is False
    assert report.real_providers_executed["llama_cpp"] is False

    # Both variants must have run independently
    assert "proc001-native.pdf" in report.variants
    assert "proc001-scanned.pdf" in report.variants

    native_rep = report.variants["proc001-native.pdf"]
    scanned_rep = report.variants["proc001-scanned.pdf"]

    # Native PDF must have executed vault ingestion, extraction, and masking
    assert native_rep["stages"]["vault_ingestion"]["status"] == "PASSED"
    assert native_rep["stages"]["extraction"]["status"] == "PASSED"
    assert native_rep["stages"]["pii_review"]["status"] == "PASSED"
    assert native_rep["stages"]["masked_text_verification"]["status"] == "PASSED"
    assert native_rep["synthetic_pii_clean"] is True

    # Missing OCR must NOT produce a passing visual privacy verification
    assert native_rep["stages"]["visual_derivative_verification"]["status"] == "NOT_EXECUTED"
    assert (
        native_rep["stages"]["visual_derivative_verification"]["failure_code"] == "OCR_UNAVAILABLE"
    )

    # Scanned PDF without real OCR should record NOT_EXECUTED for extraction
    assert scanned_rep["stages"]["vault_ingestion"]["status"] == "PASSED"
    assert scanned_rep["stages"]["extraction"]["status"] == "NOT_EXECUTED"
    assert scanned_rep["stages"]["extraction"]["failure_code"] == "OCR_UNAVAILABLE"
    # When extraction did not execute, synthetic_pii_clean must be None (not checked)
    assert scanned_rep["synthetic_pii_clean"] is None


def test_smoke_harness_distinguishes_configured_from_executed() -> None:
    report = run_smoke()
    # Configured providers indicate real production classes were used
    assert all(report.real_providers_configured.values())
    # But unexecuted providers must never be marked executed
    assert not any(report.real_providers_executed.values())


def test_smoke_harness_missing_ocr_fails_closed_on_visual_privacy() -> None:
    report = run_smoke()
    native_stages = report.variants["proc001-native.pdf"]["stages"]
    # Even though text redaction passed, visual derivative must NOT pass without real OCR
    assert native_stages["masked_text_verification"]["status"] == "PASSED"
    assert native_stages["visual_derivative_verification"]["status"] == "NOT_EXECUTED"
    assert native_stages["visual_derivative_verification"]["failure_code"] == "OCR_UNAVAILABLE"


def test_strict_integration_fails_truthfully_when_server_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_multimodal_answering import (
        test_real_llama_cpp_integration_readiness_and_generation,
    )

    # When KOSHSHIELD_STRICT_INTEGRATION=1, offline server must raise failure, not skip
    monkeypatch.setenv("KOSHSHIELD_STRICT_INTEGRATION", "1")
    with pytest.raises(pytest.fail.Exception, match="Strict integration requires local llama.cpp"):
        test_real_llama_cpp_integration_readiness_and_generation()
