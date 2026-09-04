from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from koshshield.config import Settings, get_settings
from koshshield.database import get_db
from koshshield.models import DocumentChunkRecord, DocumentRecord, DocumentState
from koshshield.schemas import (
    IndexingStatusResponse,
    RetrievalEvidenceItem,
    RetrievalResponse,
    RetrievalSearchRequest,
    RetrievalStatusResponse,
)
from koshshield.services.retrieval.chunking import DeterministicMaskedChunker
from koshshield.services.retrieval.embeddings.bge_m3 import BgeM3EmbeddingProvider
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
from koshshield.services.retrieval.vector_store.interfaces import (
    VectorStore,
    VectorStoreError,
    VectorStoreUnavailableError,
)
from koshshield.services.retrieval.vector_store.qdrant import QdrantVectorStore

router = APIRouter()
SessionDependency = Annotated[Session, Depends(get_db)]
SettingsDependency = Annotated[Settings, Depends(get_settings)]


def get_embedding_provider(settings: SettingsDependency) -> EmbeddingProvider:
    return BgeM3EmbeddingProvider(
        model_dir=settings.embedding_model_dir,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )


def get_vector_store(settings: SettingsDependency) -> VectorStore:
    return QdrantVectorStore(
        qdrant_url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
    )


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
    x_tenant_id: str = Header(default="default"),
    x_actor_id: str = Header(default="demo-admin"),
) -> IndexingStatusResponse:
    try:
        result = indexing_service.index_document(
            session=session,
            document_id=document_id,
            actor_id=x_actor_id,
            tenant_id=x_tenant_id,
        )
        return IndexingStatusResponse(
            document_id=result.document_id,
            status=result.status,
            chunk_count=result.chunk_count,
            redaction_version=result.redaction_version,
            tenant_id=result.tenant_id,
            collection_name=settings.qdrant_collection,
            completed_at=result.completed_at,
        )
    except (DocumentNotApprovedError, UnresolvedFindingsError, ResidualPiiDetectedError) as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Privacy Gate rejected indexing: {err}",
        ) from err
    except PrivacyGateError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)) from err
    except ModelUnavailableError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Local embedding model unavailable: {err}",
        ) from err
    except VectorStoreUnavailableError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector store unavailable: {err}",
        ) from err
    except VectorStoreError as err:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Vector store error: {err}",
        ) from err
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err)) from err


@router.get(
    "/documents/{document_id}/indexing",
    response_model=IndexingStatusResponse,
)
def get_document_indexing(
    document_id: str,
    session: SessionDependency,
    settings: SettingsDependency,
    x_tenant_id: str = Header(default="default"),
) -> IndexingStatusResponse:
    doc = session.scalar(select(DocumentRecord).where(DocumentRecord.id == document_id))
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
        tenant_id=x_tenant_id,
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
) -> RetrievalStatusResponse:
    vs_ready, vs_reason = vector_store.is_available()
    emb_ready, emb_reason = embedding_provider.is_available()

    total_chunks = 0
    if vs_ready:
        try:
            total_chunks = vector_store.count_points()
        except Exception:
            total_chunks = 0

    indexed_docs = (
        session.scalar(
            select(func.count(DocumentRecord.id)).where(
                DocumentRecord.status == DocumentState.INDEXED
            )
        )
        or 0
    )

    return RetrievalStatusResponse(
        collection_name=settings.qdrant_collection,
        vector_store_status="ready" if vs_ready else "unavailable",
        embedding_model_status="ready" if emb_ready else "unavailable",
        embedding_model_reason=emb_reason,
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
    x_tenant_id: str = Header(default="default"),
    x_actor_id: str = Header(default="demo-user"),
) -> RetrievalResponse:
    try:
        evidence_pack = retrieval_service.search(
            query=request.query,
            tenant_id=x_tenant_id,
            permitted_document_ids=request.permitted_document_ids,
            classification=request.classification,
            top_k=request.top_k,
            session=session,
            actor_id=x_actor_id,
        )

        return RetrievalResponse(
            query_hash=evidence_pack.query_hash,
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
                    redaction_version=item.redaction_version,
                    citation_label=item.citation_label,
                )
                for item in evidence_pack.items
            ],
        )
    except ModelUnavailableError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Local embedding model unavailable: {err}",
        ) from err
    except VectorStoreUnavailableError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector store unavailable: {err}",
        ) from err
    except VectorStoreError as err:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Vector store error: {err}",
        ) from err
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)) from err
