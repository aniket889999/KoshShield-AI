import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from koshshield.config import Settings, get_settings
from koshshield.database import get_db
from koshshield.models import DocumentChunkRecord, DocumentPageRecord, DocumentRecord, DocumentState
from koshshield.schemas import (
    IndexingStatusResponse,
    RetrievalEvidenceItem,
    RetrievalResponse,
    RetrievalSearchRequest,
    RetrievalStatusResponse,
    RetrievalVisualRegion,
)
from koshshield.security.context import (
    ExecutorContextDependency,
    RequestContextDependency,
)
from koshshield.security.vault import EncryptedVault, VaultConfigurationError
from koshshield.services.retrieval.chunking import DeterministicMaskedChunker
from koshshield.services.retrieval.embeddings.interfaces import (
    EmbeddingProvider,
    ModelUnavailableError,
)
from koshshield.services.retrieval.hybrid_search import HybridRetrievalService
from koshshield.services.retrieval.indexing_service import DocumentIndexingService
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
    """Return a cited page image only after tenant-scoped evidence authorization."""
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
    if document.active_index_version is not None and document.active_index_version != index_version:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence not found")

    page = session.scalar(
        select(DocumentPageRecord).where(
            DocumentPageRecord.document_id == document_id,
            DocumentPageRecord.page_number == page_number,
        )
    )
    if (
        not page
        or not page.encrypted_page_image_path
        or not page.page_image_sha256
        or not page.page_image_media_type
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Page visual evidence is unavailable",
        )

    try:
        vault = EncryptedVault(settings.vault_dir, settings.master_key_base64)
        image_bytes = vault.decrypt(
            document_id=f"{document_id}_p{page_number}_image",
            evidence_hash=page.page_image_sha256,
            path=Path(page.encrypted_page_image_path),
        )
    except VaultConfigurationError as err:
        logger.error("Vault configuration error: %s", err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vault storage is unavailable or misconfigured.",
        ) from err
    except Exception as err:
        logger.error("Failed to decrypt evidence image: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve visual evidence image.",
        ) from err

    return Response(
        content=image_bytes,
        media_type=page.page_image_media_type,
        headers={
            "X-KoshShield-Evidence-Hash": document.sha256,
            "X-KoshShield-Page-Image-Hash": page.page_image_sha256,
        },
    )
