import asyncio
from typing import Annotated, Literal

import httpx
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from koshshield.config import Settings, get_settings
from koshshield.database import get_db
from koshshield.services.extraction.paddle_ocr import PaddleOcrAdapter

router = APIRouter()
SessionDependency = Annotated[Session, Depends(get_db)]
SettingsDependency = Annotated[Settings, Depends(get_settings)]


class DependencyState(BaseModel):
    status: Literal["ready", "unavailable", "not_configured"]
    endpoint: str | None = None


class SystemStatus(BaseModel):
    application: str
    environment: str
    processing_boundary: str
    external_ai_enabled: bool
    metadata_backend: str
    vault: DependencyState
    metadata_store: DependencyState
    vector_store: DependencyState
    local_model: DependencyState
    ocr: DependencyState
    embedding: DependencyState


async def probe(endpoint: str) -> DependencyState:
    try:
        async with httpx.AsyncClient(timeout=0.35, trust_env=False) as client:
            response = await client.get(endpoint)
            response.raise_for_status()
        return DependencyState(status="ready", endpoint=endpoint)
    except (httpx.HTTPError, OSError):
        return DependencyState(status="unavailable", endpoint=endpoint)


@router.get("/health/live")
def liveness() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/health/ready")
def readiness(session: SessionDependency) -> dict[str, str]:
    session.execute(text("SELECT 1"))
    return {"status": "ready"}


@router.get("/system/status", response_model=SystemStatus)
async def system_status(
    session: SessionDependency,
    settings: SettingsDependency,
) -> SystemStatus:
    session.execute(text("SELECT 1"))
    qdrant, model = await asyncio.gather(
        probe(settings.qdrant_url),
        probe(f"{settings.llama_base_url}/models"),
    )
    vault_status: Literal["ready", "not_configured"] = (
        "ready" if settings.vault_configured else "not_configured"
    )
    metadata_backend = (
        "PostgreSQL"
        if settings.database_url.startswith(("postgresql://", "postgresql+"))
        else "SQLite development"
    )
    ocr_adapter = PaddleOcrAdapter(
        det_model_dir=settings.ocr_det_model_dir,
        rec_model_dir=settings.ocr_rec_model_dir,
        cls_model_dir=settings.ocr_cls_model_dir,
    )
    ocr_ready, ocr_reason = ocr_adapter.is_available()
    ocr_status: Literal["ready", "unavailable", "not_configured"] = (
        "ready"
        if ocr_ready
        else "not_configured"
        if not settings.ocr_det_model_dir
        else "unavailable"
    )

    from koshshield.services.retrieval.embeddings.bge_m3 import BgeM3EmbeddingProvider

    bge_provider = BgeM3EmbeddingProvider(
        model_dir=settings.embedding_model_dir,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    emb_ready, emb_reason = bge_provider.is_available()
    emb_status: Literal["ready", "unavailable", "not_configured"] = (
        "ready"
        if emb_ready
        else "not_configured"
        if not settings.embedding_model_dir
        else "unavailable"
    )

    return SystemStatus(
        application=settings.app_name,
        environment=settings.environment,
        processing_boundary="local-only",
        external_ai_enabled=False,
        metadata_backend=metadata_backend,
        vault=DependencyState(status=vault_status),
        metadata_store=DependencyState(status="ready"),
        vector_store=qdrant,
        local_model=model,
        ocr=DependencyState(status=ocr_status, endpoint=ocr_reason),
        embedding=DependencyState(status=emb_status, endpoint=emb_reason),
    )
