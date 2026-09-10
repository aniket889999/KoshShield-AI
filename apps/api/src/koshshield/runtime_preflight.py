"""Local runtime preflight checker for KoshShield AI.

Verifies configured runtime prerequisites independently:
1. Local model artifacts: BGE-M3, Qwen GGUF, projector, OCR resources.
2. Dependencies and llama.cpp build/alias/vision capability contract.
3. Qdrant readiness on local allowlisted URL with collection schema checks.
4. Isolated database and encrypted scratch vault usability.

Outputs structured JSON with prerequisite statuses and exits nonzero if any fail.
Never prints raw keys, env dumps, unrestricted paths, or unhandled exceptions.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from koshshield.config import Settings, get_settings


class PrerequisiteStatus(StrEnum):
    READY = "READY"
    MISSING_ARTIFACT = "MISSING_ARTIFACT"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    CONTRACT_MISMATCH = "CONTRACT_MISMATCH"
    ERROR = "ERROR"


@dataclass
class CheckResult:
    status: PrerequisiteStatus
    details: str
    metadata: dict[str, Any] | None = None


@dataclass
class PreflightReport:
    report_type: str
    status: str
    generation_executed: bool
    timestamp: str
    prerequisites: dict[str, dict[str, Any]]
    missing_categories: list[str]


def check_bge_m3(settings: Settings) -> CheckResult:
    """Verifies local BGE-M3 embedding artifact configuration and weights."""
    if not settings.embedding_model_dir:
        return CheckResult(
            status=PrerequisiteStatus.MISSING_ARTIFACT,
            details="BGE-M3 model directory not configured (KOSHSHIELD_EMBEDDING_MODEL_DIR)",
        )

    model_dir = Path(settings.embedding_model_dir)
    if not model_dir.exists() or not model_dir.is_dir():
        return CheckResult(
            status=PrerequisiteStatus.MISSING_ARTIFACT,
            details="BGE-M3 model directory does not exist on local disk",
        )

    config_file = model_dir / "config.json"
    if not config_file.is_file():
        return CheckResult(
            status=PrerequisiteStatus.MISSING_ARTIFACT,
            details="BGE-M3 config.json missing in model directory",
        )

    has_weights = any(
        (model_dir / name).is_file()
        for name in ["model.safetensors", "pytorch_model.bin", "model.onnx"]
    )
    if not has_weights:
        return CheckResult(
            status=PrerequisiteStatus.MISSING_ARTIFACT,
            details="BGE-M3 model weights missing in model directory",
        )

    if importlib.util.find_spec("FlagEmbedding") is None:
        return CheckResult(
            status=PrerequisiteStatus.ERROR,
            details="FlagEmbedding package is not installed",
        )

    return CheckResult(
        status=PrerequisiteStatus.READY,
        details="BGE-M3 model weights and configuration verified on local disk",
    )


def check_qwen_gguf(settings: Settings) -> CheckResult:
    """Verifies local Qwen GGUF model and mmproj multimodal projector paths."""
    if not settings.llama_cpp_model_path or not settings.llama_cpp_mmproj_path:
        return CheckResult(
            status=PrerequisiteStatus.MISSING_ARTIFACT,
            details=(
                "Qwen GGUF model or projector path not configured "
                "(KOSHSHIELD_LLAMA_CPP_MODEL_PATH / KOSHSHIELD_LLAMA_CPP_MMPROJ_PATH)"
            ),
        )

    model_path = Path(settings.llama_cpp_model_path)
    mmproj_path = Path(settings.llama_cpp_mmproj_path)

    if not model_path.is_file():
        return CheckResult(
            status=PrerequisiteStatus.MISSING_ARTIFACT,
            details="Qwen GGUF model file not found on local disk",
        )

    if not mmproj_path.is_file():
        return CheckResult(
            status=PrerequisiteStatus.MISSING_ARTIFACT,
            details="Qwen mmproj projector file not found on local disk",
        )

    return CheckResult(
        status=PrerequisiteStatus.READY,
        details="Qwen GGUF model and projector files verified on local disk",
        metadata={
            "model_file_name": model_path.name,
            "mmproj_file_name": mmproj_path.name,
        },
    )


def check_ocr_models(settings: Settings) -> CheckResult:
    """Verifies local PaddleOCR resources and directory configurations."""
    if importlib.util.find_spec("paddleocr") is None:
        return CheckResult(
            status=PrerequisiteStatus.ERROR,
            details="paddleocr package is not installed",
        )

    if not settings.ocr_det_model_dir or not settings.ocr_rec_model_dir:
        return CheckResult(
            status=PrerequisiteStatus.MISSING_ARTIFACT,
            details=(
                "OCR model paths not configured "
                "(KOSHSHIELD_OCR_DET_MODEL_DIR / KOSHSHIELD_OCR_REC_MODEL_DIR)"
            ),
        )

    det_path = Path(settings.ocr_det_model_dir)
    rec_path = Path(settings.ocr_rec_model_dir)

    if not det_path.is_dir():
        return CheckResult(
            status=PrerequisiteStatus.MISSING_ARTIFACT,
            details="OCR detection model directory not found on local disk",
        )

    if not rec_path.is_dir():
        return CheckResult(
            status=PrerequisiteStatus.MISSING_ARTIFACT,
            details="OCR recognition model directory not found on local disk",
        )

    return CheckResult(
        status=PrerequisiteStatus.READY,
        details="OCR detection and recognition model directories verified on local disk",
    )


def check_dependencies() -> CheckResult:
    """Verifies core application dependencies are importable."""
    required_packages = [
        "fastapi",
        "pydantic",
        "sqlalchemy",
        "cryptography",
        "httpx",
        "fitz",  # PyMuPDF
        "PIL",  # Pillow
        "qdrant_client",
    ]
    missing = [pkg for pkg in required_packages if importlib.util.find_spec(pkg) is None]
    if missing:
        return CheckResult(
            status=PrerequisiteStatus.ERROR,
            details=f"Missing required dependencies: {', '.join(missing)}",
        )

    return CheckResult(
        status=PrerequisiteStatus.READY,
        details="All core runtime dependencies verified importable",
    )


def check_llama_cpp(settings: Settings) -> CheckResult:
    """Verifies llama.cpp reachability and adherence to selected build/alias/vision contract."""
    from koshshield.services.retrieval.llama_cpp_client import (
        LlamaCppIncapableError,
        LlamaCppMultimodalClient,
        LlamaCppSecurityError,
        LlamaCppUnavailableError,
    )

    try:
        client = LlamaCppMultimodalClient(
            base_url=settings.llama_base_url,
            model_id=settings.llama_cpp_model_id,
            service_name=settings.llama_cpp_service_name,
            allowlisted_release=settings.llama_cpp_release,
            allowlisted_build=settings.llama_cpp_build,
            allowlisted_commit=settings.llama_cpp_commit,
            timeout_seconds=min(settings.llama_cpp_timeout_seconds, 5.0),
        )
        model_info = client.check_health_and_capability()
        return CheckResult(
            status=PrerequisiteStatus.READY,
            details="llama.cpp server verified online and compliant with target contract",
            metadata={
                "model_id": model_info.model_id,
                "build": model_info.build,
                "commit": model_info.commit,
                "vision": model_info.supports_vision,
            },
        )
    except LlamaCppUnavailableError:
        return CheckResult(
            status=PrerequisiteStatus.SERVICE_UNAVAILABLE,
            details="Local llama.cpp service is unreachable on configured loopback endpoint",
        )
    except LlamaCppIncapableError as err:
        return CheckResult(
            status=PrerequisiteStatus.CONTRACT_MISMATCH,
            details=f"Local llama.cpp service contract mismatch: {err}",
        )
    except LlamaCppSecurityError as err:
        return CheckResult(
            status=PrerequisiteStatus.ERROR,
            details=f"llama.cpp security validation error: {err}",
        )
    except Exception as err:
        return CheckResult(
            status=PrerequisiteStatus.ERROR,
            details=f"llama.cpp check encountered unexpected error: {err.__class__.__name__}",
        )


def check_qdrant(settings: Settings) -> CheckResult:
    """Verifies Qdrant vector store reachability and collection schema preservation."""
    from koshshield.services.retrieval.vector_store.interfaces import VectorStoreError
    from koshshield.services.retrieval.vector_store.qdrant import QdrantVectorStore

    try:
        store = QdrantVectorStore(
            qdrant_url=settings.qdrant_url,
            collection_name=settings.qdrant_collection,
            timeout_seconds=3.0,
        )
        available, _ = store.is_available()
        if not available:
            return CheckResult(
                status=PrerequisiteStatus.SERVICE_UNAVAILABLE,
                details="Local Qdrant service is unreachable on configured endpoint",
            )

        # If collection exists, verify schema conformance without destructive changes
        try:
            exists = store._client.collection_exists(settings.qdrant_collection)
            if exists:
                store.ensure_collection(dense_dim=1024)
        except VectorStoreError as err:
            return CheckResult(
                status=PrerequisiteStatus.CONTRACT_MISMATCH,
                details=f"Existing Qdrant collection schema mismatch: {err}",
            )

        return CheckResult(
            status=PrerequisiteStatus.READY,
            details="Local Qdrant service verified online with valid schema compatibility",
        )
    except Exception as err:
        return CheckResult(
            status=PrerequisiteStatus.SERVICE_UNAVAILABLE,
            details=f"Local Qdrant check failed: {err.__class__.__name__}",
        )


def check_isolated_storage() -> CheckResult:
    """Verifies ability to create an isolated database and encrypted scratch vault."""
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from koshshield.database import Base
    from koshshield.models import DocumentRecord
    from koshshield.security.vault import EncryptedVault

    scratch_dir = tempfile.mkdtemp(prefix="koshshield_preflight_")
    try:
        scratch_path = Path(scratch_dir)
        db_path = scratch_path / "scratch.db"
        vault_path = scratch_path / "scratch_vault"

        # 1. Verify isolated database creation
        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(bind=engine)
        with Session(engine) as session:
            records = list(session.scalars(select(DocumentRecord)))
            assert len(records) == 0

        # 2. Verify encrypted scratch vault roundtrip
        ephemeral_key = base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")
        vault = EncryptedVault(vault_path, ephemeral_key)
        test_content = b"preflight-encryption-probe"
        test_hash = "probe-hash-1234"
        doc_id = "preflight-test-doc"

        enc_path = vault.encrypt(doc_id, test_hash, test_content)
        decrypted = vault.decrypt(doc_id, test_hash, enc_path)
        if decrypted != test_content:
            return CheckResult(
                status=PrerequisiteStatus.ERROR,
                details="Scratch vault encrypt/decrypt roundtrip mismatch",
            )

        return CheckResult(
            status=PrerequisiteStatus.READY,
            details="Isolated database and encrypted scratch vault verified successfully",
        )
    except Exception as err:
        return CheckResult(
            status=PrerequisiteStatus.ERROR,
            details=f"Isolated storage verification failed: {err.__class__.__name__}",
        )
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def run_runtime_preflight(settings: Settings | None = None) -> PreflightReport:
    """Executes all preflight checks independently and compiles structured report."""
    cfg = settings or get_settings()

    checks: dict[str, CheckResult] = {
        "bge_m3": check_bge_m3(cfg),
        "qwen_gguf": check_qwen_gguf(cfg),
        "ocr_models": check_ocr_models(cfg),
        "dependencies": check_dependencies(),
        "llama_cpp": check_llama_cpp(cfg),
        "qdrant": check_qdrant(cfg),
        "isolated_storage": check_isolated_storage(),
    }

    missing_categories: list[str] = [
        name for name, res in checks.items() if res.status != PrerequisiteStatus.READY
    ]

    overall_status = "READY" if not missing_categories else "NOT_READY"

    prereqs_dict = {
        name: {
            "status": res.status.value,
            "details": res.details,
            **({"metadata": res.metadata} if res.metadata else {}),
        }
        for name, res in checks.items()
    }

    return PreflightReport(
        report_type="runtime_preflight",
        status=overall_status,
        generation_executed=False,
        timestamp=datetime.now(UTC).isoformat(),
        prerequisites=prereqs_dict,
        missing_categories=missing_categories,
    )


def main() -> int:
    import logging

    # Suppress verbose loggers so stdout receives only structured JSON
    logging.disable(logging.CRITICAL)
    report = run_runtime_preflight()
    print(json.dumps(asdict(report), indent=2))
    return 0 if report.status == "READY" else 1


if __name__ == "__main__":
    sys.exit(main())
