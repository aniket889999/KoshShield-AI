from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from koshshield.config import Settings
from koshshield.services.readiness import (
    ComponentStatus,
    DemoReadinessReport,
    check_component_bge_m3,
    check_component_llama_cpp,
    check_component_qwen_gguf,
    evaluate_demo_readiness,
    sanitize_text,
)


def test_sanitize_text_removes_absolute_paths():
    sample = "Model at /Users/aniket/models/bge-m3 failed with /var/log/syslog"
    sanitized = sanitize_text(sample)
    assert "/Users" not in sanitized
    assert "/var" not in sanitized
    assert "[LOCAL_PATH]" in sanitized


def test_unconfigured_readiness_report():
    settings = Settings(
        embedding_model_dir=None,
        llama_cpp_model_path=None,
        llama_cpp_mmproj_path=None,
        ocr_det_model_dir=None,
        ocr_rec_model_dir=None,
        master_key_base64="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
    )
    engine = create_engine("sqlite:///:memory:")
    with Session(engine) as session:
        report = evaluate_demo_readiness(settings, session, probe_services=False)

    assert isinstance(report, DemoReadinessReport)
    assert report.report_type == "demo_readiness"
    assert report.overall_status == "DEMO_RESTRICTED"
    assert report.can_run_offline_demo is True
    assert report.can_run_live_inference is False
    assert report.total_components_count == 7
    assert report.executable_components_count == 2  # DB + vault

    # Check component failure codes
    components_by_id = {c.component_id: c for c in report.components}
    assert components_by_id["bge_m3"].status == ComponentStatus.NOT_CONFIGURED
    assert components_by_id["bge_m3"].failure_code == "BGE_PATH_UNCONFIGURED"
    assert components_by_id["bge_m3"].executable is False

    assert components_by_id["qwen_gguf"].status == ComponentStatus.NOT_CONFIGURED
    assert components_by_id["qwen_gguf"].failure_code == "GGUF_PATHS_UNCONFIGURED"
    assert components_by_id["qwen_gguf"].executable is False

    assert components_by_id["paddle_ocr"].status == ComponentStatus.NOT_CONFIGURED
    assert components_by_id["paddle_ocr"].failure_code == "OCR_PATHS_UNCONFIGURED"

    assert components_by_id["llama_cpp"].status == ComponentStatus.NOT_EXECUTED
    assert components_by_id["llama_cpp"].failure_code == "PROBE_NOT_REQUESTED"

    assert components_by_id["qdrant"].status == ComponentStatus.NOT_EXECUTED
    assert components_by_id["qdrant"].failure_code == "PROBE_NOT_REQUESTED"

    assert components_by_id["relational_db"].status == ComponentStatus.READY
    assert components_by_id["relational_db"].executable is True

    assert components_by_id["encrypted_vault"].status == ComponentStatus.READY
    assert components_by_id["encrypted_vault"].executable is True

    # Disclaimer checks
    assert "DPDP-aligned" in report.disclaimer
    assert "not an air-gapped" in report.disclaimer


def test_vault_unconfigured_sets_not_ready():
    settings = Settings(master_key_base64="")
    report = evaluate_demo_readiness(settings, session=None, probe_services=False)
    assert report.can_run_offline_demo is False
    assert report.overall_status == "NOT_READY"


def test_sanitization_across_all_components(tmp_path: Path):
    leak_path = tmp_path / "secret_models" / "weights.bin"
    settings = Settings(
        embedding_model_dir=str(leak_path),
        llama_cpp_model_path=str(leak_path),
        llama_cpp_mmproj_path=str(leak_path),
    )
    bge_check = check_component_bge_m3(settings)
    assert str(leak_path) not in bge_check.details

    qwen_check = check_component_qwen_gguf(settings)
    assert str(leak_path) not in qwen_check.details


def test_probe_services_llama_cpp_offline():
    settings = Settings(
        llama_base_url="http://127.0.0.1:54321",
        llama_cpp_timeout_seconds=1.0,
    )
    llama_check = check_component_llama_cpp(settings, probe_services=True)
    assert llama_check.status == ComponentStatus.SERVICE_UNAVAILABLE
    assert llama_check.failure_code == "LLAMA_CPP_OFFLINE"
    assert llama_check.executable is False
    assert llama_check.live_status == "FAILED"
