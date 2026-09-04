from pathlib import Path

import pytest

from koshshield.services.retrieval.embeddings.bge_m3 import BgeM3EmbeddingProvider
from koshshield.services.retrieval.embeddings.deterministic_fake import (
    DeterministicEmbeddingProvider,
)
from koshshield.services.retrieval.embeddings.interfaces import ModelUnavailableError


def test_bge_m3_unconfigured_availability() -> None:
    provider = BgeM3EmbeddingProvider(model_dir=None)
    ready, reason = provider.is_available()
    assert ready is False
    assert "not configured" in reason

    with pytest.raises(ModelUnavailableError):
        provider.embed_query("test query")


def test_bge_m3_missing_directory_availability(tmp_path: Path) -> None:
    non_existent = tmp_path / "missing_model"
    provider = BgeM3EmbeddingProvider(model_dir=non_existent)
    ready, reason = provider.is_available()
    assert ready is False
    assert "does not exist" in reason


def test_deterministic_embedding_provider() -> None:
    provider = DeterministicEmbeddingProvider(dense_dim=1024)
    assert provider.dense_dim == 1024

    ready, reason = provider.is_available()
    assert ready is True

    res = provider.embed_query("Tender security clearance for Ministry")
    assert len(res.dense) == 1024
    # Check L2 normalization (sum of squares is ~1.0)
    norm_sq = sum(x * x for x in res.dense)
    assert abs(norm_sq - 1.0) < 1e-4

    assert len(res.sparse_indices) > 0
    assert len(res.sparse_indices) == len(res.sparse_values)
    # Check sparse indices are strictly ascending
    assert res.sparse_indices == sorted(res.sparse_indices)


def test_deterministic_embedding_semantic_correlation() -> None:
    provider = DeterministicEmbeddingProvider(dense_dim=1024)

    text_a = "confidential procurement contract tender guidelines"
    text_b = "tender contract guidelines for procurement bids"
    text_c = "weather report and rainy seasonal forecasts"

    emb_a = provider.embed_query(text_a)
    emb_b = provider.embed_query(text_b)
    emb_c = provider.embed_query(text_c)

    # Dot product of normalized vectors = Cosine similarity
    sim_ab = sum(x * y for x, y in zip(emb_a.dense, emb_b.dense, strict=False))
    sim_ac = sum(x * y for x, y in zip(emb_a.dense, emb_c.dense, strict=False))

    # Text A and B share almost all words, so sim_ab must be significantly higher than sim_ac
    assert sim_ab > sim_ac
