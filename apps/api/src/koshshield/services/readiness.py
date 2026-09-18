"""Typed demo-readiness domain model and local component inspection service.

Evaluates local runtime readiness across all seven core components:
1. BGE-M3 embedding artifact & dependencies
2. Qwen GGUF multimodal LLM weights & projector
3. PaddleOCR detection/recognition bundles & runtime
4. Local llama.cpp inference server (loopback only)
5. Local Qdrant vector database (loopback only)
6. Relational metadata store (PostgreSQL or development SQLite)
7. Encrypted local vault with AES-256-GCM master key

Enforces strict sanitization: never exposes absolute filesystem paths,
environment keys, raw exceptions, raw prompts, or document contents.
Does not make remote network requests or download any model weights.
"""

from __future__ import annotations

import contextlib
import importlib.util
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from koshshield.config import Settings
from koshshield.runtime_artifacts import (
    ArtifactCheckError,
    inspect_bge_bundle,
    inspect_gguf,
    inspect_ocr_bundle,
    inspect_ocr_runtime,
)

_PATH_PATTERN = re.compile(r"(?:/[a-zA-Z0-9_\.\-]+)+|[A-Za-z]:\\[a-zA-Z0-9_\.\-\\]+")


def sanitize_text(value: str) -> str:
    """Replaces filesystem paths and sensitive substrings with safe tokens."""
    if not value:
        return ""
    sanitized = _PATH_PATTERN.sub("[LOCAL_PATH]", value)
    return sanitized


class ComponentStatus(StrEnum):
    READY = "READY"
    MISSING_ARTIFACT = "MISSING_ARTIFACT"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    CONTRACT_MISMATCH = "CONTRACT_MISMATCH"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    NOT_EXECUTED = "NOT_EXECUTED"
    ERROR = "ERROR"


class ComponentCategory(StrEnum):
    EMBEDDING = "EMBEDDING"
    INFERENCE_SERVER = "INFERENCE_SERVER"
    LLM_WEIGHTS = "LLM_WEIGHTS"
    OCR = "OCR"
    VECTOR_DATABASE = "VECTOR_DATABASE"
    RELATIONAL_DATABASE = "RELATIONAL_DATABASE"
    ENCRYPTED_VAULT = "ENCRYPTED_VAULT"


class ComponentReadiness(BaseModel):
    component_id: str
    display_name: str
    category: ComponentCategory
    status: ComponentStatus
    failure_code: str | None = None
    details: str
    executable: bool
    local_only: bool = True
    configuration_status: str = Field(description="'CONFIGURED', 'PARTIAL', or 'UNCONFIGURED'")
    live_status: str = Field(description="'PASSED', 'FAILED', or 'NOT_EXECUTED'")


class DemoReadinessReport(BaseModel):
    report_type: str = "demo_readiness"
    overall_status: str = Field(
        description="'READY', 'DEGRADED', 'DEMO_RESTRICTED', or 'NOT_READY'"
    )
    can_run_offline_demo: bool
    can_run_live_inference: bool
    executable_components_count: int
    total_components_count: int
    components: list[ComponentReadiness]
    missing_components: list[str]
    summary: str
    timestamp: str
    disclaimer: str


def check_component_bge_m3(settings: Settings) -> ComponentReadiness:
    """Inspects local BGE-M3 embedding artifact configuration without loading weights."""
    if not settings.embedding_model_dir:
        return ComponentReadiness(
            component_id="bge_m3",
            display_name="BGE-M3 Dense Embeddings",
            category=ComponentCategory.EMBEDDING,
            status=ComponentStatus.NOT_CONFIGURED,
            failure_code="BGE_PATH_UNCONFIGURED",
            details="BGE-M3 embedding directory not configured in environment.",
            executable=False,
            configuration_status="UNCONFIGURED",
            live_status="NOT_EXECUTED",
        )

    model_dir = Path(settings.embedding_model_dir)
    try:
        inspect_bge_bundle(model_dir)
    except ArtifactCheckError as exc:
        is_missing = exc.code == "ARTIFACT_MISSING"
        return ComponentReadiness(
            component_id="bge_m3",
            display_name="BGE-M3 Dense Embeddings",
            category=ComponentCategory.EMBEDDING,
            status=(
                ComponentStatus.MISSING_ARTIFACT
                if is_missing
                else ComponentStatus.CONTRACT_MISMATCH
            ),
            failure_code=exc.code,
            details=(
                "BGE-M3 local model weights or bundle files missing on disk."
                if is_missing
                else "BGE-M3 local artifact structure does not match expected layout."
            ),
            executable=False,
            configuration_status="CONFIGURED",
            live_status="FAILED",
        )

    if importlib.util.find_spec("FlagEmbedding") is None:
        return ComponentReadiness(
            component_id="bge_m3",
            display_name="BGE-M3 Dense Embeddings",
            category=ComponentCategory.EMBEDDING,
            status=ComponentStatus.ERROR,
            failure_code="EMBEDDING_DEPENDENCY_MISSING",
            details="FlagEmbedding package is not installed in local environment.",
            executable=False,
            configuration_status="CONFIGURED",
            live_status="FAILED",
        )

    return ComponentReadiness(
        component_id="bge_m3",
        display_name="BGE-M3 Dense Embeddings",
        category=ComponentCategory.EMBEDDING,
        status=ComponentStatus.READY,
        failure_code=None,
        details="BGE-M3 model structure and tokenizer verified locally.",
        executable=True,
        configuration_status="CONFIGURED",
        live_status="PASSED",
    )


def check_component_qwen_gguf(settings: Settings) -> ComponentReadiness:
    """Inspects local Qwen GGUF model and mmproj multimodal projector paths."""
    if not settings.llama_cpp_model_path or not settings.llama_cpp_mmproj_path:
        return ComponentReadiness(
            component_id="qwen_gguf",
            display_name="Qwen Vision LLM Weights",
            category=ComponentCategory.LLM_WEIGHTS,
            status=ComponentStatus.NOT_CONFIGURED,
            failure_code="GGUF_PATHS_UNCONFIGURED",
            details="Qwen GGUF model or projector path not configured.",
            executable=False,
            configuration_status="UNCONFIGURED",
            live_status="NOT_EXECUTED",
        )

    model_path = Path(settings.llama_cpp_model_path)
    mmproj_path = Path(settings.llama_cpp_mmproj_path)

    try:
        model = inspect_gguf(model_path)
        projector = inspect_gguf(mmproj_path)
        if (model["device"], model["inode"]) == (projector["device"], projector["inode"]):
            raise ArtifactCheckError("GGUF_MODEL_PROJECTOR_IDENTICAL")
    except ArtifactCheckError as exc:
        is_missing = exc.code == "ARTIFACT_MISSING"
        return ComponentReadiness(
            component_id="qwen_gguf",
            display_name="Qwen Vision LLM Weights",
            category=ComponentCategory.LLM_WEIGHTS,
            status=(
                ComponentStatus.MISSING_ARTIFACT
                if is_missing
                else ComponentStatus.CONTRACT_MISMATCH
            ),
            failure_code=exc.code,
            details=(
                "Qwen GGUF model or projector file missing on local disk."
                if is_missing
                else "Qwen GGUF header inspection failed or invalid structure."
            ),
            executable=False,
            configuration_status="CONFIGURED",
            live_status="FAILED",
        )

    return ComponentReadiness(
        component_id="qwen_gguf",
        display_name="Qwen Vision LLM Weights",
        category=ComponentCategory.LLM_WEIGHTS,
        status=ComponentStatus.READY,
        failure_code=None,
        details="Distinct GGUF model and mmproj files verified via header inspection.",
        executable=True,
        configuration_status="CONFIGURED",
        live_status="PASSED",
    )


def check_component_ocr(settings: Settings) -> ComponentReadiness:
    """Inspects local PaddleOCR resources and directory configurations."""
    if not settings.ocr_det_model_dir or not settings.ocr_rec_model_dir:
        return ComponentReadiness(
            component_id="paddle_ocr",
            display_name="PaddleOCR Document Text Extraction",
            category=ComponentCategory.OCR,
            status=ComponentStatus.NOT_CONFIGURED,
            failure_code="OCR_PATHS_UNCONFIGURED",
            details="OCR detection and recognition model paths not configured.",
            executable=False,
            configuration_status="UNCONFIGURED",
            live_status="NOT_EXECUTED",
        )

    try:
        inspect_ocr_runtime()
        paths = [settings.ocr_det_model_dir, settings.ocr_rec_model_dir]
        if settings.ocr_cls_model_dir:
            paths.append(settings.ocr_cls_model_dir)
        for path in paths:
            inspect_ocr_bundle(Path(path))
    except ArtifactCheckError as exc:
        return ComponentReadiness(
            component_id="paddle_ocr",
            display_name="PaddleOCR Document Text Extraction",
            category=ComponentCategory.OCR,
            status=ComponentStatus.CONTRACT_MISMATCH,
            failure_code=exc.code,
            details="PaddleOCR local model bundle or runtime inspection failed.",
            executable=False,
            configuration_status="CONFIGURED",
            live_status="FAILED",
        )

    return ComponentReadiness(
        component_id="paddle_ocr",
        display_name="PaddleOCR Document Text Extraction",
        category=ComponentCategory.OCR,
        status=ComponentStatus.READY,
        failure_code=None,
        details="Local OCR inference bundles and runtime verified.",
        executable=True,
        configuration_status="CONFIGURED",
        live_status="PASSED",
    )


def check_component_llama_cpp(
    settings: Settings, *, probe_services: bool = False
) -> ComponentReadiness:
    """Inspects local llama.cpp server configuration and optional loopback health."""
    if not probe_services:
        return ComponentReadiness(
            component_id="llama_cpp",
            display_name="llama.cpp Multimodal Server",
            category=ComponentCategory.INFERENCE_SERVER,
            status=ComponentStatus.NOT_EXECUTED,
            failure_code="PROBE_NOT_REQUESTED",
            details="Local llama.cpp inference service probe was not requested.",
            executable=False,
            configuration_status="CONFIGURED" if settings.llama_base_url else "UNCONFIGURED",
            live_status="NOT_EXECUTED",
        )

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
            timeout_seconds=min(settings.llama_cpp_timeout_seconds, 2.0),
        )
        client.check_health_and_capability()
        return ComponentReadiness(
            component_id="llama_cpp",
            display_name="llama.cpp Multimodal Server",
            category=ComponentCategory.INFERENCE_SERVER,
            status=ComponentStatus.READY,
            failure_code=None,
            details="Local llama.cpp server verified online and matches target contract.",
            executable=True,
            configuration_status="CONFIGURED",
            live_status="PASSED",
        )
    except LlamaCppUnavailableError:
        return ComponentReadiness(
            component_id="llama_cpp",
            display_name="llama.cpp Multimodal Server",
            category=ComponentCategory.INFERENCE_SERVER,
            status=ComponentStatus.SERVICE_UNAVAILABLE,
            failure_code="LLAMA_CPP_OFFLINE",
            details="Local llama.cpp server is offline on configured loopback endpoint.",
            executable=False,
            configuration_status="CONFIGURED",
            live_status="FAILED",
        )
    except (LlamaCppIncapableError, LlamaCppSecurityError, Exception) as exc:
        return ComponentReadiness(
            component_id="llama_cpp",
            display_name="llama.cpp Multimodal Server",
            category=ComponentCategory.INFERENCE_SERVER,
            status=ComponentStatus.CONTRACT_MISMATCH,
            failure_code=getattr(exc, "code", "LLAMA_CONTRACT_MISMATCH"),
            details="Local llama.cpp contract mismatch or capability rejection.",
            executable=False,
            configuration_status="CONFIGURED",
            live_status="FAILED",
        )


def check_component_qdrant(
    settings: Settings, *, probe_services: bool = False
) -> ComponentReadiness:
    """Inspects local Qdrant vector database configuration and optional loopback health."""
    if not probe_services:
        return ComponentReadiness(
            component_id="qdrant",
            display_name="Qdrant Vector Database",
            category=ComponentCategory.VECTOR_DATABASE,
            status=ComponentStatus.NOT_EXECUTED,
            failure_code="PROBE_NOT_REQUESTED",
            details="Local Qdrant vector store probe was not requested.",
            executable=False,
            configuration_status="CONFIGURED" if settings.qdrant_url else "UNCONFIGURED",
            live_status="NOT_EXECUTED",
        )

    from koshshield.services.retrieval.vector_store.interfaces import (
        VectorStoreError,
        VectorStoreUnavailableError,
    )
    from koshshield.services.retrieval.vector_store.qdrant import QdrantVectorStore

    store = None
    try:
        dim = 1024
        if settings.embedding_model_dir:
            from koshshield.runtime_artifacts import embedding_dimension

            try:
                dim = embedding_dimension(Path(settings.embedding_model_dir))
            except Exception:
                dim = 1024

        store = QdrantVectorStore(
            qdrant_url=settings.qdrant_url,
            collection_name=settings.qdrant_collection,
            timeout_seconds=2.0,
        )
        store.validate_collection(dense_dim=dim)
        return ComponentReadiness(
            component_id="qdrant",
            display_name="Qdrant Vector Database",
            category=ComponentCategory.VECTOR_DATABASE,
            status=ComponentStatus.READY,
            failure_code=None,
            details="Local Qdrant collection validated and matches dense dimension.",
            executable=True,
            configuration_status="CONFIGURED",
            live_status="PASSED",
        )
    except VectorStoreUnavailableError:
        return ComponentReadiness(
            component_id="qdrant",
            display_name="Qdrant Vector Database",
            category=ComponentCategory.VECTOR_DATABASE,
            status=ComponentStatus.SERVICE_UNAVAILABLE,
            failure_code="QDRANT_OFFLINE",
            details="Local Qdrant vector database is offline or unreachable.",
            executable=False,
            configuration_status="CONFIGURED",
            live_status="FAILED",
        )
    except (VectorStoreError, Exception) as exc:
        return ComponentReadiness(
            component_id="qdrant",
            display_name="Qdrant Vector Database",
            category=ComponentCategory.VECTOR_DATABASE,
            status=ComponentStatus.CONTRACT_MISMATCH,
            failure_code=getattr(exc, "code", "QDRANT_SCHEMA_MISMATCH"),
            details="Local Qdrant collection is missing or schema is incompatible.",
            executable=False,
            configuration_status="CONFIGURED",
            live_status="FAILED",
        )
    finally:
        if store is not None:
            with contextlib.suppress(Exception):
                store.close()


def check_component_database(session: Session | None = None) -> ComponentReadiness:
    """Inspects relational database connectivity and schema readiness."""
    if session is None:
        return ComponentReadiness(
            component_id="relational_db",
            display_name="Relational Metadata Store",
            category=ComponentCategory.RELATIONAL_DATABASE,
            status=ComponentStatus.NOT_EXECUTED,
            failure_code="SESSION_UNAVAILABLE",
            details="Database session not provided for inspection.",
            executable=False,
            configuration_status="CONFIGURED",
            live_status="NOT_EXECUTED",
        )

    try:
        session.execute(text("SELECT 1"))
        return ComponentReadiness(
            component_id="relational_db",
            display_name="Relational Metadata Store",
            category=ComponentCategory.RELATIONAL_DATABASE,
            status=ComponentStatus.READY,
            failure_code=None,
            details="Relational database is connected and responsive.",
            executable=True,
            configuration_status="CONFIGURED",
            live_status="PASSED",
        )
    except Exception:
        return ComponentReadiness(
            component_id="relational_db",
            display_name="Relational Metadata Store",
            category=ComponentCategory.RELATIONAL_DATABASE,
            status=ComponentStatus.ERROR,
            failure_code="DATABASE_UNAVAILABLE",
            details="Relational database query failed or connection is unavailable.",
            executable=False,
            configuration_status="CONFIGURED",
            live_status="FAILED",
        )


def check_component_vault(settings: Settings) -> ComponentReadiness:
    """Inspects encrypted vault configuration without leaking secret keys."""
    if not settings.vault_configured:
        return ComponentReadiness(
            component_id="encrypted_vault",
            display_name="Encrypted Document Vault",
            category=ComponentCategory.ENCRYPTED_VAULT,
            status=ComponentStatus.NOT_CONFIGURED,
            failure_code="VAULT_KEY_UNCONFIGURED",
            details="Encrypted vault master key is unconfigured or invalid.",
            executable=False,
            configuration_status="UNCONFIGURED",
            live_status="FAILED",
        )

    return ComponentReadiness(
        component_id="encrypted_vault",
        display_name="Encrypted Document Vault",
        category=ComponentCategory.ENCRYPTED_VAULT,
        status=ComponentStatus.READY,
        failure_code=None,
        details="AES-256-GCM vault master key is configured and active.",
        executable=True,
        configuration_status="CONFIGURED",
        live_status="PASSED",
    )


def evaluate_demo_readiness(
    settings: Settings,
    session: Session | None = None,
    *,
    probe_services: bool = False,
) -> DemoReadinessReport:
    """Computes a complete, typed, sanitized demo-readiness report.

    Distinguishes clearly between offline demonstration readiness (which can show
    encrypted intake, deterministic redactions, reviewer queue, approval workflows,
    and audit trails) versus live AI inference (which requires real weights/services).
    """
    components = [
        check_component_database(session),
        check_component_vault(settings),
        check_component_bge_m3(settings),
        check_component_qwen_gguf(settings),
        check_component_ocr(settings),
        check_component_llama_cpp(settings, probe_services=probe_services),
        check_component_qdrant(settings, probe_services=probe_services),
    ]

    executable_count = sum(1 for c in components if c.executable)
    total_count = len(components)

    missing_components = [c.component_id for c in components if not c.executable]

    db_ready = any(c.component_id == "relational_db" and c.executable for c in components)
    vault_ready = any(c.component_id == "encrypted_vault" and c.executable for c in components)
    can_run_offline_demo = db_ready and vault_ready

    llm_ready = any(c.component_id == "qwen_gguf" and c.executable for c in components)
    llama_ready = any(c.component_id == "llama_cpp" and c.executable for c in components)
    bge_ready = any(c.component_id == "bge_m3" and c.executable for c in components)
    qdrant_ready = any(c.component_id == "qdrant" and c.executable for c in components)
    can_run_live_inference = llm_ready and llama_ready and bge_ready and qdrant_ready

    if executable_count == total_count:
        overall_status = "READY"
        summary = "All local components, AI models, and services are fully operational."
    elif can_run_offline_demo:
        overall_status = "DEMO_RESTRICTED"
        summary = (
            "Offline demonstration mode active. Encrypted document vault and metadata store "
            "are operational. Local AI models and vector services are unconfigured or offline, "
            "failing closed as designed."
        )
    else:
        overall_status = "NOT_READY"
        summary = "Core storage or database prerequisites are offline. System cannot operate."

    timestamp = datetime.now(UTC).isoformat()
    disclaimer = (
        "Demonstration operations readiness report. Missing local model weights or offline "
        "services fail closed and will not return mock responses as genuine AI intelligence. "
        "All controls are DPDP-aligned; this is not an air-gapped or statutory compliance claim."
    )

    return DemoReadinessReport(
        report_type="demo_readiness",
        overall_status=overall_status,
        can_run_offline_demo=can_run_offline_demo,
        can_run_live_inference=can_run_live_inference,
        executable_components_count=executable_count,
        total_components_count=total_count,
        components=components,
        missing_components=missing_components,
        summary=summary,
        timestamp=timestamp,
        disclaimer=disclaimer,
    )
