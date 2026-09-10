import logging
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from koshshield.models import (
    DocumentChunkRecord,
    DocumentRecord,
    DocumentState,
    DocumentVisualRegionRecord,
    validate_transition,
)
from koshshield.services.audit import record_audit_event
from koshshield.services.retrieval.chunking import DeterministicMaskedChunker, MaskedChunk
from koshshield.services.retrieval.embeddings.interfaces import EmbeddingProvider
from koshshield.services.retrieval.privacy_gate import RetrievalPrivacyGate
from koshshield.services.retrieval.vector_store.interfaces import (
    VectorStore,
    VectorStoreChunk,
    VectorStoreError,
)

logger = logging.getLogger(__name__)

_HEX_64_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass
class IndexingResult:
    document_id: str
    status: str
    chunk_count: int
    redaction_version: int
    active_index_version: int
    tenant_id: str
    completed_at: str


class DocumentIndexingService:
    """Orchestrates secure, failure-safe local indexing of privacy-approved documents into Qdrant.

    Enforces strict failure-safe sequencing:
    1. Validate document state and privacy gate.
    2. Generate and validate all chunks with current version.
    3. Generate all embeddings and verify dimensions.
    4. Upsert the new index version into vector store.
    5. Verify expected point IDs and point count in vector store.
    6. Mark the new version active atomically in metadata database.
    7. Delete stale points from older versions (failure preserves valid active index).
    8. Mark the document INDEXED.
    """

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
        doc = session.scalar(
            select(DocumentRecord).where(
                DocumentRecord.id == document_id,
                DocumentRecord.tenant_id == tenant_id,
            )
        )
        if not doc:
            raise ValueError(f"Document '{document_id}' not found")

        doc_tenant_id = doc.tenant_id

        target_version = doc.version
        previous_active_version = doc.active_index_version

        # Verified same-version indexing no-op
        # Allowed only after verifying all expected active points against authoritative
        # DB chunk records, tenant, document, index version, and content hashes.
        if (
            previous_active_version is not None
            and target_version == previous_active_version
            and doc.status == DocumentState.INDEXED
        ):
            db_chunks = list(
                session.scalars(
                    select(DocumentChunkRecord).where(
                        DocumentChunkRecord.document_id == doc.id,
                        DocumentChunkRecord.index_version == target_version,
                    )
                )
            )
            if not db_chunks:
                raise VectorStoreError(
                    f"Same-version verification failed for doc '{doc.id}': "
                    f"no DB chunk records found for active version {target_version}"
                )

            expected_pids = [c.chunk_id for c in db_chunks]
            if not self.vector_store.verify_points(expected_pids, tenant_id=doc_tenant_id):
                raise VectorStoreError(
                    f"Same-version verification failed for doc '{doc.id}': "
                    "points missing in vector store"
                )

            retrieved = self.vector_store.retrieve_points(expected_pids, tenant_id=doc_tenant_id)
            if len(retrieved) != len(expected_pids):
                raise VectorStoreError(
                    f"Same-version verification failed for doc '{doc.id}': "
                    f"expected {len(expected_pids)} points, retrieved {len(retrieved)}"
                )

            retrieved_map = {r.point_id: r.payload for r in retrieved}
            for db_c in db_chunks:
                pl = retrieved_map.get(db_c.chunk_id)
                if not pl:
                    raise VectorStoreError(
                        f"Same-version verification failed for doc '{doc.id}': "
                        f"point '{db_c.chunk_id}' missing in payload"
                    )
                if pl.get("tenant_id") != doc_tenant_id:
                    raise VectorStoreError(
                        f"Same-version verification failed for doc '{doc.id}': tenant mismatch"
                    )
                if pl.get("document_id") != doc.id:
                    raise VectorStoreError(
                        f"Same-version verification failed for doc '{doc.id}': document mismatch"
                    )
                if int(pl.get("index_version", 0)) != target_version:
                    raise VectorStoreError(
                        f"Same-version verification failed for doc '{doc.id}': "
                        "index version mismatch"
                    )
                if pl.get("masked_content_hash") != db_c.masked_content_hash:
                    raise VectorStoreError(
                        f"Same-version verification failed for doc '{doc.id}': "
                        "content hash mismatch"
                    )

            # Active points verified; return idempotent no-op without mutating state
            return IndexingResult(
                document_id=doc.id,
                status=doc.status,
                chunk_count=len(db_chunks),
                redaction_version=target_version,
                active_index_version=target_version,
                tenant_id=doc_tenant_id,
                completed_at=doc.updated_at.isoformat(),
            )

        # 0. Validate evidence hash strictly
        if not doc.sha256 or not _HEX_64_PATTERN.match(doc.sha256):
            raise ValueError(
                f"Invalid document evidence hash for document '{doc.id}': "
                "must be a lowercase 64-character hexadecimal SHA-256 string"
            )

        current_phase = "INIT"
        try:
            # 1. Validate Privacy Gate
            current_phase = "PRIVACY_GATE"
            pages = self.privacy_gate.validate_document_for_indexing(
                session=session,
                document=doc,
                actor_id=actor_id,
            )

            # Transition to INDEXING
            validate_transition(doc.status, DocumentState.INDEXING)
            doc.status = DocumentState.INDEXING
            session.commit()
            session.refresh(doc)

            record_audit_event(
                session=session,
                tenant_id=doc_tenant_id,
                actor_id=actor_id,
                event_type="DOCUMENT_INDEXING_STARTED",
                resource_type="document",
                resource_id=doc.id,
                details={
                    "tenant_id": doc_tenant_id,
                    "target_version": target_version,
                    "previous_active_version": previous_active_version,
                },
            )

            # 2. Deterministically chunk approved masked text
            current_phase = "CHUNKING"
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
                    redaction_version=target_version,
                    tenant_id=doc_tenant_id,
                    classification=classification,
                )
                all_chunks.extend(page_chunks)

            if not all_chunks:
                raise ValueError("Document yielded no text chunks for indexing")

            visual_regions_by_page = self._load_visual_regions_by_page(
                session=session,
                document_id=doc.id,
            )

            # 3. Generate local BGE-M3 embeddings
            current_phase = "EMBEDDING"
            texts = [
                self._build_embedding_text(
                    c.masked_text, visual_regions_by_page.get(c.page_number, [])
                )
                for c in all_chunks
            ]
            embedding_results = self.embedding_provider.embed_texts(texts)
            if len(embedding_results) != len(all_chunks):
                raise RuntimeError(
                    f"Embedding count mismatch: expected {len(all_chunks)}, "
                    f"got {len(embedding_results)}"
                )

            # Tri-point dimension validation: model provider vs Qdrant collection schema
            expected_dim = self.embedding_provider.dense_dim
            if embedding_results and len(embedding_results[0].dense) != expected_dim:
                raise RuntimeError(
                    f"Provider dense_dim ({expected_dim}) differs from actual vector dimension "
                    f"({len(embedding_results[0].dense)})"
                )

            # 4. Ensure collection exists and schema matches
            current_phase = "SCHEMA"
            self.vector_store.ensure_collection(dense_dim=expected_dim)

            # 5. Convert to VectorStoreChunk and upsert new index version
            current_phase = "UPSERT"
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
                    index_version=target_version,
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
                    visual_regions=visual_regions_by_page.get(chunk.page_number, []),
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
                        index_version=target_version,
                        chunk_id=chunk.chunk_id,
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                        masked_content_hash=chunk.masked_content_hash,
                    )
                )

            self.vector_store.upsert_chunks(vs_chunks)

            # 6. Verify expected point IDs and point count in vector store
            current_phase = "VERIFICATION"
            point_ids = [c.point_id for c in vs_chunks]
            points_verified = self.vector_store.verify_points(
                point_ids=point_ids,
                tenant_id=doc_tenant_id,
            )
            if not points_verified:
                raise VectorStoreError(
                    f"Index verification failed: upserted points for document '{doc.id}' "
                    f"version {target_version} could not be verified in vector store"
                )

            # 7. Atomically mark new version active in metadata database
            current_phase = "METADATA_COMMIT"
            session.execute(
                delete(DocumentChunkRecord).where(DocumentChunkRecord.document_id == doc.id)
            )
            session.add_all(db_chunk_records)
            doc.active_index_version = target_version
            doc.index_cleanup_pending = False
            validate_transition(doc.status, DocumentState.INDEXED)
            doc.status = DocumentState.INDEXED
            session.commit()
            session.refresh(doc)

            # 8. Delete stale points from older versions
            current_phase = "STALE_CLEANUP"
            try:
                self.vector_store.delete_stale_chunks(
                    document_id=doc.id,
                    tenant_id=doc_tenant_id,
                    active_version=target_version,
                )
            except Exception as stale_err:
                logger.warning(
                    "Stale chunk deletion failed for doc '%s': %s. Marking cleanup pending.",
                    doc.id,
                    stale_err,
                )
                doc.index_cleanup_pending = True
                session.commit()

            completed_at = datetime.now(UTC).isoformat()
            record_audit_event(
                session=session,
                tenant_id=doc_tenant_id,
                actor_id=actor_id,
                event_type="DOCUMENT_INDEXED",
                resource_type="document",
                resource_id=doc.id,
                details={
                    "tenant_id": doc_tenant_id,
                    "chunk_count": len(all_chunks),
                    "active_index_version": target_version,
                    "evidence_hash": doc.sha256,
                    "evidence_sha256": doc.sha256,
                },
            )
            session.commit()

            return IndexingResult(
                document_id=doc.id,
                status=doc.status,
                chunk_count=len(all_chunks),
                redaction_version=target_version,
                active_index_version=target_version,
                tenant_id=doc_tenant_id,
                completed_at=completed_at,
            )

        except Exception as err:
            logger.error("Failed to index document '%s': %s", doc.id, err)
            # Orphan version cleanup executes only when target version is strictly newer
            # than previous active version. Enforces invariant preventing active deletion.
            if previous_active_version is not None and target_version == previous_active_version:
                logger.warning(
                    "Skipping orphan cleanup for doc '%s': "
                    "target version %s matches active generation",
                    doc.id,
                    target_version,
                )
            elif previous_active_version is None or target_version > previous_active_version:
                assert (
                    previous_active_version is None or target_version != previous_active_version
                ), "Cannot delete active generation"
                try:
                    self.vector_store.delete_version_chunks(
                        document_id=doc.id,
                        tenant_id=doc_tenant_id,
                        index_version=target_version,
                    )
                except Exception as cleanup_err:
                    logger.warning(
                        "Failed to clean orphan target points for doc '%s' v%s: %s",
                        doc.id,
                        target_version,
                        cleanup_err,
                    )

            session.rollback()
            doc_fail = session.scalar(select(DocumentRecord).where(DocumentRecord.id == doc.id))
            if doc_fail:
                error_code = self._classify_error_code(err, phase=current_phase)
                if previous_active_version is None:
                    # First-index failure
                    doc_fail.status = DocumentState.INDEX_FAILED
                    event_type = "DOCUMENT_INDEXING_FAILED"
                    audit_details = {
                        "error_code": error_code,
                        "failure_code": error_code,
                        "target_version": target_version,
                        "failure_type": "FIRST_INDEX",
                    }
                else:
                    # Failed reindex preserves previous active generation as searchable
                    doc_fail.status = DocumentState.INDEXED
                    doc_fail.active_index_version = previous_active_version
                    event_type = "DOCUMENT_REINDEXING_FAILED"
                    audit_details = {
                        "error_code": error_code,
                        "failure_code": error_code,
                        "target_version": target_version,
                        "active_index_version": previous_active_version,
                        "failure_type": "REINDEX",
                    }

                record_audit_event(
                    session=session,
                    tenant_id=doc_tenant_id,
                    actor_id=actor_id,
                    event_type=event_type,
                    resource_type="document",
                    resource_id=doc_fail.id,
                    details=audit_details,
                )
                session.commit()
            raise

    @staticmethod
    def _classify_error_code(err: Exception, phase: str | None = None) -> str:
        name = type(err).__name__
        if (
            "Privacy" in name
            or "Unresolved" in name
            or "Pii" in name
            or "Approved" in name
            or "Finding" in name
            or phase == "PRIVACY_GATE"
        ):
            return "PRIVACY_GATE_REJECTED"
        if "ModelUnavailable" in name:
            return "MODEL_UNAVAILABLE"
        if phase == "EMBEDDING":
            return "EMBEDDING_FAILURE"
        if "VectorStoreUnavailable" in name:
            return "VECTOR_STORE_UNAVAILABLE"
        if "VectorStore" in name or phase in ("SCHEMA", "UPSERT", "VERIFICATION"):
            return "VECTOR_STORE_ERROR"
        if isinstance(err, ValueError):
            return "INVALID_ARGUMENT"
        return "INTERNAL_INDEXING_ERROR"

    @staticmethod
    def _load_visual_regions_by_page(
        *,
        session: Session,
        document_id: str,
    ) -> dict[int, list[dict[str, object]]]:
        records = list(
            session.scalars(
                select(DocumentVisualRegionRecord)
                .where(DocumentVisualRegionRecord.document_id == document_id)
                .order_by(
                    DocumentVisualRegionRecord.page_number.asc(),
                    DocumentVisualRegionRecord.region_sequence.asc(),
                )
            )
        )
        regions_by_page: dict[int, list[dict[str, object]]] = {}
        for record in records:
            bbox_payload = record.bbox_json if isinstance(record.bbox_json, dict) else {}
            bbox = bbox_payload.get("bbox") if isinstance(bbox_payload.get("bbox"), list) else None
            regions_by_page.setdefault(record.page_number, []).append(
                {
                    "region_id": record.id,
                    "region_type": record.region_type,
                    "page_number": record.page_number,
                    "bbox": bbox,
                    "page_width": bbox_payload.get("page_width"),
                    "page_height": bbox_payload.get("page_height"),
                    "caption": record.caption_text,
                    "caption_hash": record.caption_hash,
                    "image_sha256": record.image_sha256,
                    "image_available": bool(record.image_sha256),
                    "source": record.source,
                }
            )
        return regions_by_page

    @staticmethod
    def _build_embedding_text(masked_text: str, visual_regions: list[dict[str, object]]) -> str:
        captions = [
            str(region.get("caption", "")).strip()
            for region in visual_regions
            if str(region.get("caption", "")).strip()
        ]
        if not captions:
            return masked_text
        return f"{masked_text}\n\nVisual captions:\n" + "\n".join(captions)
