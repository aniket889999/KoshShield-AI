import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from koshshield.models import (
    DocumentChunkRecord,
    DocumentRecord,
    DocumentState,
    validate_transition,
)
from koshshield.services.audit import record_audit_event
from koshshield.services.retrieval.chunking import DeterministicMaskedChunker, MaskedChunk
from koshshield.services.retrieval.embeddings.interfaces import EmbeddingProvider
from koshshield.services.retrieval.privacy_gate import RetrievalPrivacyGate
from koshshield.services.retrieval.vector_store.interfaces import (
    VectorStore,
    VectorStoreChunk,
)

logger = logging.getLogger(__name__)


@dataclass
class IndexingResult:
    document_id: str
    status: str
    chunk_count: int
    redaction_version: int
    tenant_id: str
    completed_at: str


class DocumentIndexingService:
    """Orchestrates secure local indexing of privacy-approved documents into Qdrant."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        privacy_gate: RetrievalPrivacyGate,
        chunker: DeterministicMaskedChunker | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.privacy_gate = privacy_gate
        self.chunker = chunker or DeterministicMaskedChunker()

    def index_document(
        self,
        session: Session,
        document_id: str,
        actor_id: str = "demo-admin",
        tenant_id: str = "default",
        classification: str = "CONFIDENTIAL",
    ) -> IndexingResult:
        doc = session.scalar(select(DocumentRecord).where(DocumentRecord.id == document_id))
        if not doc:
            raise ValueError(f"Document '{document_id}' not found")

        # 1. Run Privacy Gate
        pages = self.privacy_gate.validate_document_for_indexing(
            session=session,
            document=doc,
            actor_id=actor_id,
        )

        # 2. State transition to INDEXING
        validate_transition(doc.status, DocumentState.INDEXING)
        doc.status = DocumentState.INDEXING
        session.commit()
        session.refresh(doc)

        record_audit_event(
            session=session,
            actor_id=actor_id,
            event_type="DOCUMENT_INDEXING_STARTED",
            resource_type="document",
            resource_id=doc.id,
            details={"tenant_id": tenant_id, "version": doc.version},
        )

        try:
            # 3. Deterministically chunk approved masked text
            all_chunks: list[MaskedChunk] = []
            for page in pages:
                if not page.masked_text:
                    continue
                page_chunks = self.chunker.chunk_page(
                    page_text=page.masked_text,
                    document_id=doc.id,
                    page_number=page.page_number,
                    document_filename=doc.filename,
                    document_evidence_hash=doc.sha256,
                    redaction_version=doc.version,
                    tenant_id=tenant_id,
                    classification=classification,
                )
                all_chunks.extend(page_chunks)

            if not all_chunks:
                raise ValueError("Document yielded no text chunks for indexing")

            # 4. Generate local BGE-M3 embeddings
            texts = [c.masked_text for c in all_chunks]
            embedding_results = self.embedding_provider.embed_texts(texts)
            if len(embedding_results) != len(all_chunks):
                raise RuntimeError(
                    f"Embedding count mismatch: expected {len(all_chunks)}, "
                    f"got {len(embedding_results)}"
                )

            # 5. Ensure Qdrant collection exists with proper dimensions
            self.vector_store.ensure_collection(dense_dim=self.embedding_provider.dense_dim)

            # 6. Idempotent cleanup: delete old chunks for this document
            self.vector_store.delete_document_chunks(doc.id)
            session.execute(
                delete(DocumentChunkRecord).where(DocumentChunkRecord.document_id == doc.id)
            )

            # 7. Convert to VectorStoreChunk and upsert into Qdrant
            vs_chunks: list[VectorStoreChunk] = []
            db_chunk_records: list[DocumentChunkRecord] = []

            for chunk, emb in zip(all_chunks, embedding_results, strict=True):
                vs_chunk = VectorStoreChunk(
                    point_id=chunk.chunk_id,
                    chunk_id=chunk.chunk_id,
                    tenant_id=chunk.tenant_id,
                    document_id=chunk.document_id,
                    page_number=chunk.page_number,
                    redaction_version=chunk.redaction_version,
                    chunk_sequence=chunk.chunk_sequence,
                    masked_text=chunk.masked_text,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    masked_content_hash=chunk.masked_content_hash,
                    document_evidence_hash=chunk.document_evidence_hash,
                    classification=chunk.classification,
                    document_filename=chunk.document_filename,
                    indexed_at=chunk.indexed_at,
                    dense_vector=emb.dense,
                    sparse_indices=emb.sparse_indices,
                    sparse_values=emb.sparse_values,
                )
                vs_chunks.append(vs_chunk)

                db_chunk_records.append(
                    DocumentChunkRecord(
                        id=str(uuid.uuid4()),
                        document_id=doc.id,
                        page_number=chunk.page_number,
                        chunk_sequence=chunk.chunk_sequence,
                        chunk_id=chunk.chunk_id,
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                        masked_content_hash=chunk.masked_content_hash,
                    )
                )

            self.vector_store.upsert_chunks(vs_chunks)
            session.add_all(db_chunk_records)

            # 8. Mark document as INDEXED
            validate_transition(doc.status, DocumentState.INDEXED)
            doc.status = DocumentState.INDEXED
            completed_at = datetime.now(UTC).isoformat()

            record_audit_event(
                session=session,
                actor_id=actor_id,
                event_type="DOCUMENT_INDEXED",
                resource_type="document",
                resource_id=doc.id,
                details={
                    "tenant_id": tenant_id,
                    "chunk_count": len(all_chunks),
                    "redaction_version": doc.version,
                    "evidence_hash": doc.sha256[:16],
                },
            )

            session.commit()
            session.refresh(doc)

            return IndexingResult(
                document_id=doc.id,
                status=doc.status,
                chunk_count=len(all_chunks),
                redaction_version=doc.version,
                tenant_id=tenant_id,
                completed_at=completed_at,
            )

        except Exception as err:
            logger.error("Failed to index document '%s': %s", doc.id, err)
            session.rollback()
            # Set to INDEX_FAILED in a clean transaction
            doc_fail = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc.id))
            if doc_fail:
                doc_fail.status = DocumentState.INDEX_FAILED
                record_audit_event(
                    session=session,
                    actor_id=actor_id,
                    event_type="DOCUMENT_INDEXING_FAILED",
                    resource_type="document",
                    resource_id=doc_fail.id,
                    details={"error": str(err)[:200]},
                )
                session.commit()
            raise
