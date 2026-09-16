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
import contextlib
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
from koshshield.runtime_artifacts import (
    ArtifactCheckError,
    embedding_dimension,
    inspect_bge_bundle,
    inspect_gguf,
    inspect_ocr_bundle,
    inspect_ocr_runtime,
)


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

    try:
        metadata = inspect_bge_bundle(Path(settings.embedding_model_dir))
    except ArtifactCheckError as exc:
        return CheckResult(
            status=(
                PrerequisiteStatus.MISSING_ARTIFACT
                if exc.code == "ARTIFACT_MISSING"
                else PrerequisiteStatus.CONTRACT_MISMATCH
            ),
            details="BGE-M3 local bundle inspection failed",
            metadata={"failure_code": exc.code},
        )

    if importlib.util.find_spec("FlagEmbedding") is None:
        return CheckResult(
            status=PrerequisiteStatus.ERROR,
            details="FlagEmbedding package is not installed",
        )

    return CheckResult(
        status=PrerequisiteStatus.READY,
        details="BGE-M3 bundle structure present; model loading and integrity unverified",
        metadata=metadata,
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

    try:
        model = inspect_gguf(model_path)
        projector = inspect_gguf(mmproj_path)
        if (model["device"], model["inode"]) == (projector["device"], projector["inode"]):
            raise ArtifactCheckError("GGUF_MODEL_PROJECTOR_IDENTICAL")
    except ArtifactCheckError as exc:
        return CheckResult(
            status=(
                PrerequisiteStatus.MISSING_ARTIFACT
                if exc.code == "ARTIFACT_MISSING"
                else PrerequisiteStatus.CONTRACT_MISMATCH
            ),
            details="GGUF model/projector inspection failed",
            metadata={"failure_code": exc.code},
        )

    return CheckResult(
        status=PrerequisiteStatus.READY,
        details="Distinct GGUF files passed header checks; compatibility and integrity unverified",
        metadata={
            "validation_scope": "header_only",
            "integrity_verified": False,
            "model_version": model["version"],
            "projector_version": projector["version"],
            "model_size_bytes": model["size_bytes"],
            "projector_size_bytes": projector["size_bytes"],
        },
    )


def check_ocr_models(settings: Settings) -> CheckResult:
    """Verifies local PaddleOCR resources and directory configurations."""
    if not settings.ocr_det_model_dir or not settings.ocr_rec_model_dir:
        return CheckResult(
            status=PrerequisiteStatus.MISSING_ARTIFACT,
            details=(
                "OCR model paths not configured "
                "(KOSHSHIELD_OCR_DET_MODEL_DIR / KOSHSHIELD_OCR_REC_MODEL_DIR)"
            ),
        )

    try:
        version = inspect_ocr_runtime()
        paths = [settings.ocr_det_model_dir, settings.ocr_rec_model_dir]
        if settings.ocr_cls_model_dir:
            paths.append(settings.ocr_cls_model_dir)
        for path in paths:
            inspect_ocr_bundle(Path(path))
    except ArtifactCheckError as exc:
        return CheckResult(
            status=PrerequisiteStatus.CONTRACT_MISMATCH,
            details="OCR dependency or local inference bundle inspection failed",
            metadata={"failure_code": exc.code},
        )

    return CheckResult(
        status=PrerequisiteStatus.READY,
        details="OCR 2.x bundle structure present; loading and recognition unverified",
        metadata={
            "paddleocr_version": version,
            "validation_scope": "bundle_structure_only",
            "integrity_verified": False,
            "model_loaded": False,
            "angle_classifier_configured": bool(settings.ocr_cls_model_dir),
        },
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
    from koshshield.services.retrieval.vector_store.interfaces import (
        VectorStoreError,
        VectorStoreUnavailableError,
    )
    from koshshield.services.retrieval.vector_store.qdrant import QdrantVectorStore

    store = None
    try:
        if not settings.embedding_model_dir:
            raise ArtifactCheckError("EMBEDDING_DIMENSION_UNAVAILABLE")
        dimension = embedding_dimension(Path(settings.embedding_model_dir))
        store = QdrantVectorStore(
            qdrant_url=settings.qdrant_url,
            collection_name=settings.qdrant_collection,
            timeout_seconds=3.0,
        )
        store.validate_collection(dense_dim=dimension)

        return CheckResult(
            status=PrerequisiteStatus.READY,
            details="Existing Qdrant schema matches configured embedding dimension and indexes",
            metadata={"dense_dimension": dimension, "schema_modified": False},
        )
    except ArtifactCheckError:
        return CheckResult(
            status=PrerequisiteStatus.CONTRACT_MISMATCH,
            details="Qdrant schema inspection requires valid local embedding configuration",
            metadata={"failure_code": "EMBEDDING_DIMENSION_UNAVAILABLE"},
        )
    except VectorStoreUnavailableError:
        return CheckResult(
            status=PrerequisiteStatus.SERVICE_UNAVAILABLE,
            details="Local Qdrant service is unavailable",
            metadata={"failure_code": "QDRANT_UNAVAILABLE"},
        )
    except VectorStoreError:
        return CheckResult(
            status=PrerequisiteStatus.CONTRACT_MISMATCH,
            details="Qdrant collection or required schema is missing or incompatible",
            metadata={"failure_code": "QDRANT_SCHEMA_MISMATCH"},
        )
    except Exception:
        return CheckResult(
            status=PrerequisiteStatus.ERROR,
            details="Local Qdrant inspection failed",
            metadata={"failure_code": "QDRANT_CHECK_FAILED"},
        )
    finally:
        if store is not None:
            with contextlib.suppress(Exception):
                store.close()


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
