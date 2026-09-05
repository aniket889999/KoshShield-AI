import os
import socket
from pathlib import Path

import pytest

from koshshield.api.routes.retrieval import get_embedding_provider
from koshshield.config import Settings
from koshshield.services.retrieval.embeddings.bge_m3 import BgeM3EmbeddingProvider
from koshshield.services.retrieval.embeddings.deterministic_fake import (
    DeterministicEmbeddingProvider,
)
from koshshield.services.retrieval.embeddings.interfaces import ModelUnavailableError


def test_bgem3_offline_flags_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    # Clear env
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_TELEMETRY", raising=False)

    BgeM3EmbeddingProvider(model_dir=None)

    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert os.environ.get("HF_HUB_DISABLE_TELEMETRY") == "1"


def test_bgem3_rejects_urls_and_remote_repo_ids() -> None:
    # Prohibit remote URLs
    with pytest.raises(ValueError) as exc1:
        BgeM3EmbeddingProvider(model_dir="https://huggingface.co/BAAI/bge-m3")
    assert "Remote model identifiers and URLs are prohibited" in str(exc1.value)

    with pytest.raises(ValueError) as exc2:
        BgeM3EmbeddingProvider(model_dir="http://models.internal.local/bge-m3")
    assert "Remote model identifiers and URLs are prohibited" in str(exc2.value)

    # Prohibit Hugging Face repo IDs when not on disk
    with pytest.raises(ValueError) as exc3:
        BgeM3EmbeddingProvider(model_dir="BAAI/bge-m3")
    assert "Remote model identifiers and URLs are prohibited" in str(exc3.value)


def test_bgem3_no_outbound_network_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Monkeypatches socket.socket to guarantee no outbound connections occur during check."""
    network_attempted = False

    def guarded_connect(*args, **kwargs):
        nonlocal network_attempted
        network_attempted = True
        raise OSError("Air-gap boundary violation: outbound network attempt intercepted!")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)

    model_dir = tmp_path / "bge-m3"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"hidden_size": 1024}')
    (model_dir / "model.safetensors").write_text("fake-weights")

    provider = BgeM3EmbeddingProvider(model_dir=model_dir)
    ready, reason = provider.is_available()

    assert not network_attempted, "Network call was attempted during provider check!"


def test_production_configuration_cannot_instantiate_fake_provider() -> None:
    """Production dependency injection must never instantiate or return the fake provider."""
    prod_settings = Settings(
        embedding_model_dir=None,
        app_env="production",
    )
    provider = get_embedding_provider(prod_settings)

    # Must be BgeM3EmbeddingProvider
    assert isinstance(provider, BgeM3EmbeddingProvider)
    assert not isinstance(provider, DeterministicEmbeddingProvider)

    # When model weights missing in production, fail closed
    ready, reason = provider.is_available()
    assert ready is False
    assert "not configured" in reason

    with pytest.raises(ModelUnavailableError):
        provider._ensure_model()
