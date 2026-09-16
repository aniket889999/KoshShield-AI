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

import argparse
import base64
import contextlib
import importlib.util
import json
import os
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from functools import partial
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
    NOT_EXECUTED = "NOT_EXECUTED"


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
    service_probes_enabled: bool = False
    storage_probe_enabled: bool = False
    runtime_verified: bool = False


def check_bge_m3(settings: Settings) -> CheckResult:
    """Verifies local BGE-M3 embedding artifact configuration and weights."""
    if not settings.embedding_model_dir:
        return CheckResult(
            status=PrerequisiteStatus.MISSING_ARTIFACT,
            details="BGE-M3 model directory not configured (KOSHSHIELD_EMBEDDING_MODEL_DIR)",
            metadata={"failure_code": "BGE_PATH_UNCONFIGURED"},
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
            metadata={"failure_code": "EMBEDDING_DEPENDENCY_MISSING"},
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
            metadata={"failure_code": "GGUF_PATHS_UNCONFIGURED"},
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
            metadata={"failure_code": "OCR_PATHS_UNCONFIGURED"},
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
    """Discover core module specs; this does not import or test their native libraries."""
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
            metadata={"failure_code": "CORE_DEPENDENCY_MISSING"},
        )

    return CheckResult(
        status=PrerequisiteStatus.READY,
        details="Core module specs discovered; imports and native-library compatibility untested",
        metadata={"validation_scope": "module_discovery_only"},
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
            metadata={"failure_code": "LLAMA_UNAVAILABLE"},
        )
    except LlamaCppIncapableError:
        return CheckResult(
            status=PrerequisiteStatus.CONTRACT_MISMATCH,
            details="Local llama.cpp service does not match the required runtime contract",
            metadata={"failure_code": "LLAMA_CONTRACT_MISMATCH"},
        )
    except LlamaCppSecurityError:
        return CheckResult(
            status=PrerequisiteStatus.ERROR,
            details="llama.cpp endpoint or response security validation failed",
            metadata={"failure_code": "LLAMA_SECURITY_REJECTED"},
        )
    except Exception:
        return CheckResult(
            status=PrerequisiteStatus.ERROR,
            details="llama.cpp inspection failed",
            metadata={"failure_code": "LLAMA_CHECK_FAILED"},
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

    try:
        with tempfile.TemporaryDirectory(prefix="koshshield_preflight_") as scratch_dir:
            scratch_path = Path(scratch_dir)
            engine = create_engine(f"sqlite:///{scratch_path / 'scratch.db'}")
            try:
                Base.metadata.create_all(bind=engine)
                with Session(engine) as session:
                    if list(session.scalars(select(DocumentRecord))):
                        raise RuntimeError("Scratch database is not empty")

                ephemeral_key = base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")
                vault = EncryptedVault(scratch_path / "scratch_vault", ephemeral_key)
                test_content = b"preflight-encryption-probe"
                test_hash = "probe-hash-1234"
                doc_id = "preflight-test-doc"
                enc_path = vault.encrypt(doc_id, test_hash, test_content)
                if vault.decrypt(doc_id, test_hash, enc_path) != test_content:
                    raise RuntimeError("Scratch vault roundtrip mismatch")
            finally:
                engine.dispose()

        return CheckResult(
            status=PrerequisiteStatus.READY,
            details=(
                "Temporary SQLite and encrypted vault roundtrip passed; deployment storage untested"
            ),
            metadata={"validation_scope": "temporary_storage_only"},
        )
    except Exception:
        return CheckResult(
            status=PrerequisiteStatus.ERROR,
            details="Isolated storage verification failed",
            metadata={"failure_code": "SCRATCH_STORAGE_FAILED"},
        )


def _run_check(check: Callable[[], CheckResult]) -> CheckResult:
    try:
        return check()
    except Exception:
        return CheckResult(
            status=PrerequisiteStatus.ERROR,
            details="Prerequisite inspection failed; other checks continue independently",
            metadata={"failure_code": "CHECK_FAILED"},
        )


def run_runtime_preflight(
    settings: Settings | None = None,
    *,
    probe_services: bool = False,
    probe_storage: bool = False,
) -> PreflightReport:
    """Inspect locally by default. Network probes and scratch writes require explicit flags."""
    try:
        cfg = settings or get_settings()
    except Exception:
        return _report(
            {
                "configuration": CheckResult(
                    status=PrerequisiteStatus.ERROR,
                    details="Runtime configuration is invalid or could not be read",
                    metadata={"failure_code": "CONFIGURATION_INVALID"},
                )
            },
            probe_services=probe_services,
            probe_storage=probe_storage,
        )

    checks: dict[str, Callable[[], CheckResult]] = {
        "bge_m3": partial(check_bge_m3, cfg),
        "qwen_gguf": partial(check_qwen_gguf, cfg),
        "ocr_models": partial(check_ocr_models, cfg),
        "dependencies": check_dependencies,
        "llama_cpp": partial(check_llama_cpp, cfg),
        "qdrant": partial(check_qdrant, cfg),
        "isolated_storage": check_isolated_storage,
    }
    results = {}
    for name, check in checks.items():
        disabled = (name in {"llama_cpp", "qdrant"} and not probe_services) or (
            name == "isolated_storage" and not probe_storage
        )
        results[name] = (
            CheckResult(
                status=PrerequisiteStatus.NOT_EXECUTED,
                details="Explicit probe flag required; this check was not executed",
                metadata={"failure_code": "PROBE_NOT_REQUESTED"},
            )
            if disabled
            else _run_check(check)
        )
    return _report(results, probe_services=probe_services, probe_storage=probe_storage)


def _report(
    checks: dict[str, CheckResult], *, probe_services: bool, probe_storage: bool
) -> PreflightReport:
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
        service_probes_enabled=probe_services,
        storage_probe_enabled=probe_storage,
    )


def main(argv: list[str] | None = None) -> int:
    import logging

    parser = argparse.ArgumentParser(description="Inspect local runtime prerequisites")
    parser.add_argument(
        "--probe-services", action="store_true", help="Probe configured local llama.cpp and Qdrant"
    )
    parser.add_argument(
        "--probe-storage", action="store_true", help="Create and clean up owned temporary DB/vault"
    )
    args = parser.parse_args(argv)

    # Suppress verbose loggers so stdout receives only structured JSON
    previous_disable = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        report = run_runtime_preflight(
            probe_services=args.probe_services, probe_storage=args.probe_storage
        )
        print(json.dumps(asdict(report), indent=2))
    finally:
        logging.disable(previous_disable)
    return 0 if report.status == "READY" else 1


if __name__ == "__main__":
    sys.exit(main())
