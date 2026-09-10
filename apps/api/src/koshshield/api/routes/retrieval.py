import base64
import hashlib
import io
import logging
import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Response, status
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from koshshield.config import Settings, get_settings
from koshshield.database import get_db
from koshshield.models import DocumentChunkRecord, DocumentPageRecord, DocumentRecord, DocumentState
from koshshield.schemas import (
    AnswerCitation,
    CleanupPendingRequest,
    CleanupPendingResponse,
    IndexingStatusResponse,
    RetrievalAnswerRequest,
    RetrievalAnswerResponse,
    RetrievalEvidenceItem,
    RetrievalResponse,
    RetrievalSearchRequest,
    RetrievalStatusResponse,
    RetrievalVisualRegion,
)
from koshshield.security.context import (
    AdminContextDependency,
    ExecutorContextDependency,
    RequestContextDependency,
)
from koshshield.security.pii.indian_pii import REDACTION_PLACEHOLDERS, IndianPiiDetector
from koshshield.security.vault import EncryptedVault, VaultConfigurationError
from koshshield.services.audit import record_audit_event
from koshshield.services.retrieval.chunking import DeterministicMaskedChunker
from koshshield.services.retrieval.embeddings.interfaces import (
    EmbeddingProvider,
    ModelUnavailableError,
)
from koshshield.services.retrieval.hybrid_search import HybridRetrievalService
from koshshield.services.retrieval.indexing_service import DocumentIndexingService
from koshshield.services.retrieval.llama_cpp_client import (
    LlamaCppClientError,
    LlamaCppIncapableError,
    LlamaCppMultimodalClient,
    LlamaCppResponseInvalidError,
    LlamaCppSecurityError,
    LlamaCppUnavailableError,
)
from koshshield.services.retrieval.privacy_gate import (
    DocumentNotApprovedError,
    PrivacyGateError,
    ResidualPiiDetectedError,
    RetrievalPrivacyGate,
    UnresolvedFindingsError,
)
from koshshield.services.retrieval.provider_registry import (
    get_singleton_embedding_provider,
    get_singleton_vector_store,
)
from koshshield.services.retrieval.vector_store.interfaces import (
    VectorStore,
    VectorStoreError,
    VectorStoreUnavailableError,
)

logger = logging.getLogger(__name__)

router = APIRouter()
SessionDependency = Annotated[Session, Depends(get_db)]
SettingsDependency = Annotated[Settings, Depends(get_settings)]


def get_embedding_provider(settings: SettingsDependency) -> EmbeddingProvider:
    return get_singleton_embedding_provider(settings)


def get_vector_store(settings: SettingsDependency) -> VectorStore:
    return get_singleton_vector_store(settings)


def get_privacy_gate(settings: SettingsDependency) -> RetrievalPrivacyGate:
    return RetrievalPrivacyGate(pii_salt=settings.pii_salt)


def get_llama_client(settings: SettingsDependency) -> LlamaCppMultimodalClient:
    return LlamaCppMultimodalClient(
        base_url=settings.llama_base_url,
        model_id=settings.llama_cpp_model_id,
        service_name=settings.llama_cpp_service_name,
        allowlisted_release=settings.llama_cpp_release,
        allowlisted_build=settings.llama_cpp_build,
        allowlisted_commit=settings.llama_cpp_commit,
        timeout_seconds=settings.llama_cpp_timeout_seconds,
        max_tokens=settings.llama_cpp_max_tokens,
    )


def get_indexing_service(
    embedding_provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    privacy_gate: Annotated[RetrievalPrivacyGate, Depends(get_privacy_gate)],
) -> DocumentIndexingService:
    return DocumentIndexingService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        privacy_gate=privacy_gate,
        chunker=DeterministicMaskedChunker(),
    )


def get_retrieval_service(
    embedding_provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    settings: SettingsDependency,
) -> HybridRetrievalService:
    return HybridRetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        rrf_k=settings.retrieval_rrf_k,
    )


@router.post(
    "/documents/{document_id}/index",
    response_model=IndexingStatusResponse,
    status_code=status.HTTP_200_OK,
)
def index_document(
    document_id: str,
    session: SessionDependency,
    indexing_service: Annotated[DocumentIndexingService, Depends(get_indexing_service)],
    settings: SettingsDependency,
    context: ExecutorContextDependency,
) -> IndexingStatusResponse:
    try:
        result = indexing_service.index_document(
            session=session,
            document_id=document_id,
            actor_id=context.actor_id,
            tenant_id=context.tenant_id,
        )
        return IndexingStatusResponse(
            document_id=result.document_id,
            status=result.status,
            chunk_count=result.chunk_count,
            redaction_version=result.redaction_version,
            active_index_version=result.active_index_version,
            tenant_id=result.tenant_id,
            collection_name=settings.qdrant_collection,
            completed_at=result.completed_at,
        )
    except (DocumentNotApprovedError, UnresolvedFindingsError, ResidualPiiDetectedError) as err:
        logger.warning("Privacy Gate rejected indexing: %s", err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Privacy Gate rejected indexing: document not approved or findings unresolved.",
        ) from err
    except PrivacyGateError as err:
        logger.warning("Privacy Gate error: %s", err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Privacy Gate validation failed.",
        ) from err
    except ModelUnavailableError as err:
        logger.error("Local embedding model unavailable: %s", err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Local embedding model unavailable.",
        ) from err
    except VectorStoreUnavailableError as err:
        logger.error("Vector store unavailable: %s", err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store unavailable.",
        ) from err
    except VectorStoreError as err:
        logger.error("Vector store error: %s", err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Vector store error.",
        ) from err
    except ValueError as err:
        logger.warning("Invalid argument during indexing: %s", err)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or invalid request.",
        ) from err
    except Exception as err:
        logger.error("Unexpected error during document indexing: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document indexing failed.",
        ) from err


@router.get(
    "/documents/{document_id}/indexing",
    response_model=IndexingStatusResponse,
)
def get_document_indexing(
    document_id: str,
    session: SessionDependency,
    settings: SettingsDependency,
    context: RequestContextDependency,
) -> IndexingStatusResponse:
    doc = session.scalar(
        select(DocumentRecord).where(
            DocumentRecord.id == document_id,
            DocumentRecord.tenant_id == context.tenant_id,
        )
    )
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    chunk_count = (
        session.scalar(
            select(func.count(DocumentChunkRecord.id)).where(
                DocumentChunkRecord.document_id == document_id
            )
        )
        or 0
    )

    return IndexingStatusResponse(
        document_id=doc.id,
        status=doc.status,
        chunk_count=int(chunk_count),
        redaction_version=doc.version,
        active_index_version=doc.active_index_version,
        tenant_id=doc.tenant_id,
        collection_name=settings.qdrant_collection,
        completed_at=doc.updated_at.isoformat() if doc.status == DocumentState.INDEXED else None,
    )


@router.get(
    "/retrieval/status",
    response_model=RetrievalStatusResponse,
)
def retrieval_status(
    session: SessionDependency,
    embedding_provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    settings: SettingsDependency,
    context: RequestContextDependency,
) -> RetrievalStatusResponse:
    vs_ready, vs_reason = vector_store.is_available()
    emb_ready, emb_reason = embedding_provider.is_available()

    total_chunks = 0
    if vs_ready:
        try:
            total_chunks = vector_store.count_points(tenant_id=context.tenant_id)
        except (VectorStoreError, VectorStoreUnavailableError) as err:
            logger.error("Vector store error counting points: %s", err)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector store error counting points.",
            ) from err

    indexed_docs = (
        session.scalar(
            select(func.count(DocumentRecord.id)).where(
                DocumentRecord.status == DocumentState.INDEXED,
                DocumentRecord.tenant_id == context.tenant_id,
            )
        )
        or 0
    )

    public_emb_reason = (
        "BGE-M3 model ready"
        if emb_ready
        else (
            "BGE-M3 model not configured"
            if not settings.embedding_model_dir
            else "BGE-M3 model unavailable"
        )
    )

    return RetrievalStatusResponse(
        collection_name=settings.qdrant_collection,
        vector_store_status="ready" if vs_ready else "unavailable",
        embedding_model_status="ready" if emb_ready else "unavailable",
        embedding_model_reason=public_emb_reason,
        total_chunks=total_chunks,
        indexed_documents_count=int(indexed_docs),
    )


@router.post(
    "/retrieval/cleanup-pending",
    response_model=CleanupPendingResponse,
    status_code=status.HTTP_200_OK,
)
def cleanup_pending(
    session: SessionDependency,
    indexing_service: Annotated[DocumentIndexingService, Depends(get_indexing_service)],
    context: AdminContextDependency,
    request: Annotated[CleanupPendingRequest | None, Body()] = None,
) -> CleanupPendingResponse:
    """Operational endpoint for tenant-scoped reconciliation of pending stale index cleanups.
    Requires administrative authorization and enforces strict tenant boundaries.
    """
    try:
        limit = request.limit if request is not None else 50
        result = indexing_service.reconcile_pending_cleanups(
            session=session,
            tenant_id=context.tenant_id,
            limit=limit,
            actor_id=context.actor_id,
        )
        return CleanupPendingResponse(
            tenant_id=result.tenant_id,
            processed_count=result.processed_count,
            succeeded_count=result.succeeded_count,
            failed_count=result.failed_count,
            failure_codes=result.failure_codes,
        )
    except HTTPException:
        raise
    except Exception as err:
        logger.error("Failed to reconcile pending index cleanups: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Pending index cleanup reconciliation failed.",
        ) from None


@router.post(
    "/retrieval/search",
    response_model=RetrievalResponse,
)
def search_retrieval(
    request: RetrievalSearchRequest,
    session: SessionDependency,
    retrieval_service: Annotated[HybridRetrievalService, Depends(get_retrieval_service)],
    context: RequestContextDependency,
) -> RetrievalResponse:
    """Executes local hybrid search across authorized masked document chunks."""
    try:
        evidence_pack = retrieval_service.search(
            query=request.query,
            tenant_id=context.tenant_id,
            permitted_document_ids=request.permitted_document_ids,
            classification=request.classification,
            top_k=request.top_k,
            session=session,
            actor_id=context.actor_id,
        )

        return RetrievalResponse(
            query_length=evidence_pack.query_length,
            duration_ms=evidence_pack.duration_ms,
            tenant_id=evidence_pack.tenant_id,
            top_k=evidence_pack.top_k,
            total_found=evidence_pack.total_found,
            results=[
                RetrievalEvidenceItem(
                    rank=item.rank,
                    fused_score=item.fused_score,
                    sources=item.sources,
                    masked_snippet=item.masked_snippet,
                    document_id=item.document_id,
                    document_filename=item.document_filename,
                    page_number=item.page_number,
                    chunk_id=item.chunk_id,
                    evidence_hash=item.evidence_hash,
                    masked_content_hash=item.masked_content_hash,
                    redaction_version=item.redaction_version,
                    index_version=item.index_version,
                    citation_label=item.citation_label,
                    visual_regions=[
                        RetrievalVisualRegion(
                            region_id=region.region_id,
                            region_type=region.region_type,
                            page_number=region.page_number,
                            bbox=region.bbox,
                            page_width=region.page_width,
                            page_height=region.page_height,
                            caption=region.caption,
                            caption_hash=region.caption_hash,
                            image_sha256=region.image_sha256,
                            image_available=region.image_available,
                            source=region.source,
                        )
                        for region in item.visual_regions
                    ],
                )
                for item in evidence_pack.items
            ],
        )
    except ModelUnavailableError as err:
        logger.error("Local embedding model unavailable: %s", err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Local embedding model unavailable.",
        ) from err
    except VectorStoreUnavailableError as err:
        logger.error("Vector store unavailable: %s", err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store unavailable.",
        ) from err
    except VectorStoreError as err:
        logger.error("Vector store error: %s", err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Vector store error.",
        ) from err
    except ValueError as err:
        logger.warning("Invalid retrieval request: %s", err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid retrieval query or parameters.",
        ) from err


@router.get("/retrieval/evidence/{chunk_id}/page-image")
def get_authorized_evidence_page_image(
    chunk_id: str,
    session: SessionDependency,
    settings: SettingsDependency,
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    context: RequestContextDependency,
) -> Response:
    """Return a privacy-masked page image derivative only after
    tenant-scoped evidence authorization.
    """
    try:
        hits = vector_store.retrieve_points(point_ids=[chunk_id], tenant_id=context.tenant_id)
    except VectorStoreUnavailableError as err:
        logger.error("Vector store unavailable retrieving evidence: %s", err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store unavailable.",
        ) from err
    except VectorStoreError as err:
        logger.error("Vector store error retrieving evidence: %s", err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Vector store error.",
        ) from err

    if not hits:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence not found")

    payload = hits[0].payload
    document_id = str(payload.get("document_id") or "")
    try:
        page_number = int(payload.get("page_number") or 0)
        index_version = int(payload.get("index_version") or 0)
    except (TypeError, ValueError) as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evidence payload is incomplete",
        ) from err

    document = session.scalar(
        select(DocumentRecord).where(
            DocumentRecord.id == document_id,
            DocumentRecord.tenant_id == context.tenant_id,
        )
    )
    if not document or document.status != DocumentState.INDEXED:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence not found")
    if document.sha256 != str(payload.get("document_evidence_hash") or ""):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence not found")
    if document.active_index_version is None or document.active_index_version != index_version:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence not found")

    # Authoritative chunk validation against DB
    db_chunk = session.scalar(
        select(DocumentChunkRecord).where(
            DocumentChunkRecord.chunk_id == chunk_id,
            DocumentChunkRecord.document_id == document.id,
            DocumentChunkRecord.index_version == index_version,
        )
    )
    if not db_chunk or db_chunk.page_number != page_number:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence not found")

    payload_content_hash = str(payload.get("masked_content_hash") or "")
    if payload_content_hash and payload_content_hash != db_chunk.masked_content_hash:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence not found")

    page = session.scalar(
        select(DocumentPageRecord).where(
            DocumentPageRecord.document_id == document_id,
            DocumentPageRecord.page_number == page_number,
        )
    )
    if not page:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence not found")

    # Privacy and version checks for visual evidence derivative
    if page.visual_privacy_status != "APPROVED":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Visual evidence is unavailable or blocked for privacy safety",
        )
    if page.visual_redaction_version is None or page.visual_redaction_version != index_version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Visual evidence version is stale or mismatched",
        )
    if not page.encrypted_masked_page_image_path or not page.masked_page_image_sha256:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Visual evidence derivative is missing",
        )
    if (page.masked_page_image_media_type or "image/png") != "image/png":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid visual derivative media type",
        )

    try:
        vault = EncryptedVault(settings.vault_dir, settings.master_key_base64)
        masked_artifact_id = (
            f"{document_id}_p{page_number}_v{page.visual_redaction_version}_masked_image"
        )
        image_bytes = vault.decrypt(
            document_id=masked_artifact_id,
            evidence_hash=page.masked_page_image_sha256,
            path=Path(page.encrypted_masked_page_image_path),
        )
    except VaultConfigurationError as err:
        logger.error("Vault configuration error (VAULT_UNAVAILABLE)")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vault storage is unavailable or misconfigured.",
        ) from err
    except Exception as err:
        logger.error("Failed to decrypt evidence image derivative (DECRYPTION_FAILED)")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evidence not found",
        ) from err

    # Validate stored hash
    if hashlib.sha256(image_bytes).hexdigest() != page.masked_page_image_sha256:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence not found")

    # Validate PNG magic signature
    if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence not found")

    # Validate byte limit before decoding
    if len(image_bytes) > min(2_500_000, settings.max_upload_bytes):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence not found")

    # Configure decompression bomb limit and validate dimension limits
    Image.MAX_IMAGE_PIXELS = settings.max_image_dimension * settings.max_image_dimension
    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            w, h = pil_img.size
            if (
                w > settings.max_image_dimension
                or h > settings.max_image_dimension
                or (w * h) > Image.MAX_IMAGE_PIXELS
            ):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="Evidence not found"
                )
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Exception) as err:
        logger.error("Failed to decode evidence image (DECODE_FAILED)")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Evidence not found"
        ) from err

    return Response(
        content=image_bytes,
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "X-KoshShield-Evidence-Hash": document.sha256,
            "X-KoshShield-Masked-Image-Hash": page.masked_page_image_sha256,
        },
    )


@router.post(
    "/retrieval/answer",
    response_model=RetrievalAnswerResponse,
)
def answer_retrieval(
    request: RetrievalAnswerRequest,
    session: SessionDependency,
    settings: SettingsDependency,
    retrieval_service: Annotated[HybridRetrievalService, Depends(get_retrieval_service)],
    llama_client: Annotated[LlamaCppMultimodalClient, Depends(get_llama_client)],
    context: RequestContextDependency,
) -> RetrievalAnswerResponse:
    """Executes grounded local multimodal answering using local Qwen3-VL and llama.cpp."""
    # 1. Feature flag boundary check
    if not settings.enable_multimodal_answering:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Multimodal answering feature is disabled.",
        )

    start_time = time.perf_counter()

    # 2. Hybrid search first (bound top_k to at most 5)
    top_k = min(request.top_k, 5)
    evidence_pack = retrieval_service.search(
        query=request.query,
        tenant_id=context.tenant_id,
        permitted_document_ids=request.permitted_document_ids,
        classification=request.classification,
        top_k=top_k,
        session=session,
        actor_id=context.actor_id,
    )

    selected_chunks = evidence_pack.items[:5]
    if not selected_chunks:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        record_audit_event(
            session=session,
            tenant_id=context.tenant_id,
            actor_id=context.actor_id,
            event_type="MULTIMODAL_ANSWER_GENERATED",
            resource_type="retrieval",
            resource_id=None,
            details={
                "actor_id": context.actor_id,
                "tenant_id": context.tenant_id,
                "query_length": len(request.query),
                "selected_chunk_ids": [],
                "masked_image_hashes": [],
                "model_id": settings.llama_cpp_model_id,
                "duration_ms": round(duration_ms, 2),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "policy_result": "INSUFFICIENT_EVIDENCE",
                "failure_code": "INSUFFICIENT_EVIDENCE",
            },
        )
        session.commit()
        return RetrievalAnswerResponse(
            answer="Insufficient evidence to answer query.",
            cited_chunk_ids=[],
            citations=[],
            insufficient_evidence=True,
            model_id=settings.llama_cpp_model_id,
            duration_ms=round(duration_ms, 2),
            tenant_id=context.tenant_id,
        )

    # 3. Gather at most 2 masked derivative images matching active versions
    vault = EncryptedVault(settings.vault_dir, settings.master_key_base64)
    masked_images_data_urls: list[str] = []
    masked_image_hashes: list[str] = []
    seen_pages: set[tuple[str, int]] = set()

    for chunk in selected_chunks:
        if len(masked_images_data_urls) >= 2:
            break
        page_key = (chunk.document_id, chunk.page_number)
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)

        page_rec = session.scalar(
            select(DocumentPageRecord).where(
                DocumentPageRecord.document_id == chunk.document_id,
                DocumentPageRecord.page_number == chunk.page_number,
            )
        )
        if (
            page_rec
            and page_rec.visual_privacy_status == "APPROVED"
            and page_rec.visual_redaction_version == chunk.index_version
            and page_rec.encrypted_masked_page_image_path
            and page_rec.masked_page_image_sha256
            and (page_rec.masked_page_image_media_type or "image/png") == "image/png"
        ):
            try:
                ver = page_rec.visual_redaction_version
                masked_artifact_id = f"{chunk.document_id}_p{chunk.page_number}_v{ver}_masked_image"
                img_bytes = vault.decrypt(
                    document_id=masked_artifact_id,
                    evidence_hash=page_rec.masked_page_image_sha256,
                    path=Path(page_rec.encrypted_masked_page_image_path),
                )
                # Stored hash check
                if hashlib.sha256(img_bytes).hexdigest() != page_rec.masked_page_image_sha256:
                    continue
                # PNG magic signature check
                if not img_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                    continue
                # Enforce bound on image bytes (max 2.5MB and upload limit)
                if len(img_bytes) > min(2_500_000, settings.max_upload_bytes):
                    continue
                # Dimension checks with Pillow and decompression bomb protection
                Image.MAX_IMAGE_PIXELS = settings.max_image_dimension * settings.max_image_dimension
                with Image.open(io.BytesIO(img_bytes)) as pil_img:
                    w, h = pil_img.size
                    if (
                        w > settings.max_image_dimension
                        or h > settings.max_image_dimension
                        or (w * h) > Image.MAX_IMAGE_PIXELS
                    ):
                        continue

                b64_str = base64.b64encode(img_bytes).decode("ascii")
                masked_images_data_urls.append(f"data:image/png;base64,{b64_str}")
                masked_image_hashes.append(page_rec.masked_page_image_sha256)
            except Exception:
                logger.warning("Failed to decrypt/validate masked derivative (DERIVATIVE_INVALID)")

    # 4. Format chunks into bounded untrusted context
    evidence_dicts = [
        {
            "chunk_id": c.chunk_id,
            "document_filename": c.document_filename,
            "page_number": c.page_number,
            "masked_snippet": c.masked_snippet[:1500],
        }
        for c in selected_chunks
    ]

    # 5. Model generation with structured output (performs exactly 1 capability check)
    try:
        model_result = llama_client.generate_grounded_answer(
            query=request.query,
            evidence_chunks=evidence_dicts,
            masked_images_data_urls=masked_images_data_urls,
        )
    except LlamaCppResponseInvalidError as err:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.warning("Local multimodal model response invalid (MODEL_RESPONSE_INVALID)")
        record_audit_event(
            session=session,
            tenant_id=context.tenant_id,
            actor_id=context.actor_id,
            event_type="MULTIMODAL_ANSWER_GENERATED",
            resource_type="retrieval",
            resource_id=None,
            details={
                "actor_id": context.actor_id,
                "tenant_id": context.tenant_id,
                "query_length": len(request.query),
                "selected_chunk_ids": [c.chunk_id for c in selected_chunks],
                "masked_image_hashes": masked_image_hashes,
                "model_id": settings.llama_cpp_model_id,
                "duration_ms": round(duration_ms, 2),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "policy_result": "OUTPUT_BLOCKED",
                "failure_code": "MODEL_RESPONSE_INVALID",
            },
        )
        session.commit()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Invalid response from local model service.",
        ) from err
    except LlamaCppIncapableError as err:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.warning("Local multimodal model lacks vision capability (MODEL_INCAPABLE)")
        record_audit_event(
            session=session,
            tenant_id=context.tenant_id,
            actor_id=context.actor_id,
            event_type="MULTIMODAL_ANSWER_GENERATED",
            resource_type="retrieval",
            resource_id=None,
            details={
                "actor_id": context.actor_id,
                "tenant_id": context.tenant_id,
                "query_length": len(request.query),
                "selected_chunk_ids": [c.chunk_id for c in selected_chunks],
                "masked_image_hashes": masked_image_hashes,
                "model_id": settings.llama_cpp_model_id,
                "duration_ms": round(duration_ms, 2),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "policy_result": "OUTPUT_BLOCKED",
                "failure_code": "MODEL_INCAPABLE",
            },
        )
        session.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Local multimodal model lacks vision capability.",
        ) from err
    except LlamaCppUnavailableError as err:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.warning("Local multimodal answering service is unavailable (MODEL_UNAVAILABLE)")
        record_audit_event(
            session=session,
            tenant_id=context.tenant_id,
            actor_id=context.actor_id,
            event_type="MULTIMODAL_ANSWER_GENERATED",
            resource_type="retrieval",
            resource_id=None,
            details={
                "actor_id": context.actor_id,
                "tenant_id": context.tenant_id,
                "query_length": len(request.query),
                "selected_chunk_ids": [c.chunk_id for c in selected_chunks],
                "masked_image_hashes": masked_image_hashes,
                "model_id": settings.llama_cpp_model_id,
                "duration_ms": round(duration_ms, 2),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "policy_result": "OUTPUT_BLOCKED",
                "failure_code": "MODEL_UNAVAILABLE",
            },
        )
        session.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Local multimodal answering service is unavailable.",
        ) from err
    except LlamaCppSecurityError as err:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.warning("Local multimodal service security error (SSRF_BLOCKED)")
        record_audit_event(
            session=session,
            tenant_id=context.tenant_id,
            actor_id=context.actor_id,
            event_type="MULTIMODAL_ANSWER_GENERATED",
            resource_type="retrieval",
            resource_id=None,
            details={
                "actor_id": context.actor_id,
                "tenant_id": context.tenant_id,
                "query_length": len(request.query),
                "selected_chunk_ids": [c.chunk_id for c in selected_chunks],
                "masked_image_hashes": masked_image_hashes,
                "model_id": settings.llama_cpp_model_id,
                "duration_ms": round(duration_ms, 2),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "policy_result": "OUTPUT_BLOCKED",
                "failure_code": "SSRF_BLOCKED",
            },
        )
        session.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid local model service configuration.",
        ) from err
    except LlamaCppClientError as err:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.warning("Local multimodal answer generation failed (MODEL_ERROR)")
        record_audit_event(
            session=session,
            tenant_id=context.tenant_id,
            actor_id=context.actor_id,
            event_type="MULTIMODAL_ANSWER_GENERATED",
            resource_type="retrieval",
            resource_id=None,
            details={
                "actor_id": context.actor_id,
                "tenant_id": context.tenant_id,
                "query_length": len(request.query),
                "selected_chunk_ids": [c.chunk_id for c in selected_chunks],
                "masked_image_hashes": masked_image_hashes,
                "model_id": settings.llama_cpp_model_id,
                "duration_ms": round(duration_ms, 2),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "policy_result": "OUTPUT_BLOCKED",
                "failure_code": "MODEL_ERROR",
            },
        )
        session.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Local multimodal answering service is unavailable.",
        ) from err

    # 6. Grounding validation & authoritative citation reconstruction
    allowed_chunk_map = {c.chunk_id: c for c in selected_chunks}
    candidate_cids = model_result.get("cited_chunk_ids", [])
    valid_cids: list[str] = []
    seen_cids: set[str] = set()
    for cid in candidate_cids:
        if cid in allowed_chunk_map and cid not in seen_cids:
            valid_cids.append(cid)
            seen_cids.add(cid)

    insufficient = bool(model_result.get("insufficient_evidence", False))
    raw_answer = str(model_result.get("answer", "")).strip()

    # Fixed insufficient-evidence response if insufficient_evidence is True or no valid citations
    if insufficient or not valid_cids:
        insufficient = True
        final_answer = "Insufficient verified evidence to answer the query."
        citations = []
        valid_cids = []
    else:
        final_answer = raw_answer if raw_answer else "Answer based on verified evidence."
        citations = [
            AnswerCitation(
                chunk_id=cid,
                citation_label=allowed_chunk_map[cid].citation_label,
                document_id=allowed_chunk_map[cid].document_id,
                document_filename=allowed_chunk_map[cid].document_filename,
                page_number=allowed_chunk_map[cid].page_number,
                evidence_hash=allowed_chunk_map[cid].evidence_hash,
                masked_content_hash=allowed_chunk_map[cid].masked_content_hash,
                image_available=any(
                    r.image_available for r in allowed_chunk_map[cid].visual_regions
                ),
            )
            for cid in valid_cids
        ]

    # 7. Output PII detection & masking
    detector = IndianPiiDetector()
    pii_findings = detector.detect(final_answer)
    redacted = False
    blocked = False
    if pii_findings:
        for f in sorted(pii_findings, key=lambda x: x.start, reverse=True):
            placeholder = REDACTION_PLACEHOLDERS.get(f.finding_type, "[REDACTED]")
            final_answer = final_answer[: f.start] + placeholder + final_answer[f.end :]
        redacted = True
        if detector.detect(final_answer):
            final_answer = "Response blocked due to residual sensitive information."
            insufficient = True
            citations = []
            valid_cids = []
            blocked = True

    # 8. Determine truthful policy result
    if blocked:
        policy_result = "OUTPUT_BLOCKED"
        failure_code = "RESIDUAL_PII_BLOCKED"
    elif insufficient:
        policy_result = "INSUFFICIENT_EVIDENCE"
        failure_code = "INSUFFICIENT_EVIDENCE"
    elif redacted:
        policy_result = "OUTPUT_REDACTED"
        failure_code = None
    else:
        policy_result = "ALLOWED"
        failure_code = None

    duration_ms = (time.perf_counter() - start_time) * 1000.0

    # 9. Privacy-safe auditing: strictly metadata only
    record_audit_event(
        session=session,
        tenant_id=context.tenant_id,
        actor_id=context.actor_id,
        event_type="MULTIMODAL_ANSWER_GENERATED",
        resource_type="retrieval",
        resource_id=None,
        details={
            "actor_id": context.actor_id,
            "tenant_id": context.tenant_id,
            "query_length": len(request.query),
            "selected_chunk_ids": [c.chunk_id for c in selected_chunks],
            "masked_image_hashes": masked_image_hashes,
            "model_id": settings.llama_cpp_model_id,
            "duration_ms": round(duration_ms, 2),
            "prompt_tokens": model_result.get("prompt_tokens", 0),
            "completion_tokens": model_result.get("completion_tokens", 0),
            "total_tokens": model_result.get("total_tokens", 0),
            "policy_result": policy_result,
            "failure_code": failure_code,
        },
    )
    session.commit()

    return RetrievalAnswerResponse(
        answer=final_answer,
        cited_chunk_ids=valid_cids,
        citations=citations,
        insufficient_evidence=insufficient,
        model_id=settings.llama_cpp_model_id,
        duration_ms=round(duration_ms, 2),
        tenant_id=context.tenant_id,
    )
