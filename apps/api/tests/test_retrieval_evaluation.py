import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from koshshield.database import engine
from koshshield.models import DocumentPageRecord, DocumentRecord, DocumentState
from koshshield.services.retrieval.chunking import DeterministicMaskedChunker
from koshshield.services.retrieval.embeddings.deterministic_fake import (
    DeterministicEmbeddingProvider,
)
from koshshield.services.retrieval.hybrid_search import HybridRetrievalService
from koshshield.services.retrieval.indexing_service import DocumentIndexingService
from koshshield.services.retrieval.privacy_gate import RetrievalPrivacyGate
from koshshield.services.retrieval.vector_store import InMemoryVectorStore


@dataclass
class GoldenQuery:
    query: str
    target_doc_id: str
    target_page: int
    tenant_id: str


def test_retrieval_evaluation_benchmark() -> None:
    """Evaluates Recall@5, MRR, citation completeness, and cross-tenant leakage."""
    embedding_provider = DeterministicEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    privacy_gate = RetrievalPrivacyGate()
    indexing_service = DocumentIndexingService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        privacy_gate=privacy_gate,
        chunker=DeterministicMaskedChunker(),
    )
    retrieval_service = HybridRetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        rrf_k=60,
    )

    # 1. Build synthetic corpus (5 documents across 2 tenants)
    corpus_specs = [
        {
            "doc_id": "doc-eval-1",
            "filename": "DefenseProcurement-2026.pdf",
            "tenant": "tenant-defense",
            "pages": [
                (1, "Ministry of Defense encrypted communications equipment procurement."),
                (2, "Specifications for ruggedized satellite transceivers and telemetry."),
            ],
        },
        {
            "doc_id": "doc-eval-2",
            "filename": "RailwayExpansion-2026.pdf",
            "tenant": "tenant-railways",
            "pages": [
                (1, "High-speed rail corridor automated train control tenders."),
                (2, "Track electrification and power substation maintenance schedules."),
            ],
        },
        {
            "doc_id": "doc-eval-3",
            "filename": "HealthcareSubsidies-2026.pdf",
            "tenant": "tenant-health",
            "pages": [
                (1, "Public health clinic equipment allocation and rural immunization logistics."),
                (2, "Medical supply chain cold storage compliance requirements."),
            ],
        },
        {
            "doc_id": "doc-eval-4",
            "filename": "PowerGridGridModernization-2026.pdf",
            "tenant": "tenant-power",
            "pages": [
                (1, "National smart grid SCADA cybersecurity directives."),
                (2, "Renewable energy integration into regional high-voltage transmission."),
            ],
        },
        {
            "doc_id": "doc-eval-5",
            "filename": "WaterSanitationSurvey-2026.pdf",
            "tenant": "tenant-sanitation",
            "pages": [
                (1, "Urban water quality monitoring sensor deployment and reservoir audits."),
                (2, "Effluent treatment plant zero-liquid-discharge guidelines."),
            ],
        },
    ]

    with Session(bind=engine) as session:
        for spec in corpus_specs:
            doc = DocumentRecord(
                id=spec["doc_id"],
                filename=spec["filename"],
                media_type="application/pdf",
                size_bytes=2048,
                sha256=f"hash-{spec['doc_id']}-abcdef1234567890",
                vault_path=f"vault/{spec['doc_id']}.ksh",
                status=DocumentState.INDEX_READY,
            )
            pages = [
                DocumentPageRecord(
                    id=str(uuid.uuid4()),
                    document_id=doc.id,
                    page_number=p_num,
                    extraction_method="native_pdf",
                    text_hash=f"thash-{p_num}",
                    encrypted_artifact_path=f"vault/{doc.id}_p{p_num}.ksh",
                    masked_text=p_text,
                )
                for p_num, p_text in spec["pages"]
            ]
            session.add(doc)
            session.add_all(pages)
            session.commit()

            indexing_service.index_document(
                session=session,
                document_id=doc.id,
                tenant_id=spec["tenant"],
            )

    # 2. Golden query set
    golden_queries = [
        GoldenQuery("satellite transceivers telemetry", "doc-eval-1", 2, "tenant-defense"),
        GoldenQuery("automated train control signaling", "doc-eval-2", 1, "tenant-railways"),
        GoldenQuery("rural immunization medical supply", "doc-eval-3", 1, "tenant-health"),
        GoldenQuery("smart grid cybersecurity scada", "doc-eval-4", 1, "tenant-power"),
        GoldenQuery("water quality monitoring reservoir", "doc-eval-5", 1, "tenant-sanitation"),
    ]

    hits_at_5 = 0
    reciprocal_ranks: list[float] = []
    citations_complete = 0
    cross_tenant_leaks = 0
    total_retrieved = 0

    for gq in golden_queries:
        pack = retrieval_service.search(
            query=gq.query,
            tenant_id=gq.tenant_id,
            top_k=5,
        )

        # Check for cross-tenant leaks
        for item in pack.items:
            # Look up document owner
            owner_tenant = next(
                s["tenant"] for s in corpus_specs if s["doc_id"] == item.document_id
            )
            if owner_tenant != gq.tenant_id:
                cross_tenant_leaks += 1

        # Check ranking of target document
        matched_rank = None
        for item in pack.items:
            if item.document_id == gq.target_doc_id and item.page_number == gq.target_page:
                matched_rank = item.rank
                break

        if matched_rank is not None:
            hits_at_5 += 1
            reciprocal_ranks.append(1.0 / matched_rank)
        else:
            reciprocal_ranks.append(0.0)

        # Check citation completeness
        for item in pack.items:
            total_retrieved += 1
            if (
                item.citation_label.startswith("[Document:")
                and "Page:" in item.citation_label
                and "Evidence:" in item.citation_label
            ):
                citations_complete += 1

    total_queries = len(golden_queries)
    recall_at_5 = hits_at_5 / total_queries
    mrr = sum(reciprocal_ranks) / total_queries
    citation_rate = citations_complete / max(1, total_retrieved)

    # Assertions for evaluation benchmarks
    assert recall_at_5 == 1.0, f"Expected Recall@5 == 1.0, got {recall_at_5}"
    assert mrr >= 0.8, f"Expected MRR >= 0.8, got {mrr}"
    assert cross_tenant_leaks == 0, f"Cross-tenant leaks detected: {cross_tenant_leaks}"
    assert citation_rate == 1.0, f"Citation rate must be 1.0, got {citation_rate}"
