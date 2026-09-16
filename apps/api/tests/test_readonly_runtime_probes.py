from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client import models

from koshshield.config import Settings
from koshshield.runtime_preflight import PrerequisiteStatus, check_qdrant
from koshshield.services.retrieval.vector_store.interfaces import VectorStoreError
from koshshield.services.retrieval.vector_store.qdrant import QdrantVectorStore


def collection_client(dimension: int = 768) -> MagicMock:
    client = MagicMock()
    client.collection_exists.return_value = True
    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors={"text_dense": models.VectorParams(size=dimension, distance="Cosine")},
                sparse_vectors={"text_sparse": models.SparseVectorParams()},
            )
        ),
        payload_schema=dict(QdrantVectorStore.REQUIRED_PAYLOAD_INDEXES),
    )
    return client


def assert_no_mutations(client: MagicMock) -> None:
    assert {call[0] for call in client.mock_calls} <= {
        "collection_exists",
        "get_collection",
        "close",
    }


def test_qdrant_readonly_success_uses_supplied_dimension() -> None:
    client = collection_client()
    store = QdrantVectorStore(client=client)
    store.validate_collection(768)
    assert_no_mutations(client)


@pytest.mark.parametrize("fault", ["absent", "dimension", "distance", "sparse", "index", "type"])
def test_qdrant_readonly_rejects_schema_faults_without_repair(fault: str) -> None:
    client = collection_client()
    info = client.get_collection.return_value
    if fault == "absent":
        client.collection_exists.return_value = False
    elif fault == "dimension":
        info.config.params.vectors["text_dense"].size = 64
    elif fault == "distance":
        info.config.params.vectors["text_dense"].distance = models.Distance.EUCLID
    elif fault == "sparse":
        info.config.params.sparse_vectors = {}
    elif fault == "index":
        del info.payload_schema["tenant_id"]
    else:
        info.payload_schema["tenant_id"] = models.PayloadSchemaType.INTEGER
    with pytest.raises(VectorStoreError):
        QdrantVectorStore(client=client).validate_collection(768)
    assert_no_mutations(client)


def test_preflight_uses_dynamic_dimension_and_closes_owned_client(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"hidden_size": 768}')
    client = collection_client()
    with patch(
        "koshshield.services.retrieval.vector_store.qdrant.QdrantClient", return_value=client
    ):
        result = check_qdrant(Settings(_env_file=None, embedding_model_dir=tmp_path))
    assert result.status == PrerequisiteStatus.READY
    assert result.metadata == {"dense_dimension": 768, "schema_modified": False}
    client.close.assert_called_once()
    assert_no_mutations(client)


def test_preflight_closes_client_on_failure_and_hides_exception(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"hidden_size": 768}')
    client = collection_client()
    client.collection_exists.side_effect = RuntimeError("secret tenant path /private/secret")
    with patch(
        "koshshield.services.retrieval.vector_store.qdrant.QdrantClient", return_value=client
    ):
        result = check_qdrant(Settings(_env_file=None, embedding_model_dir=tmp_path))
    assert result.status == PrerequisiteStatus.SERVICE_UNAVAILABLE
    assert "secret" not in str(result)
    client.close.assert_called_once()
    assert_no_mutations(client)


def test_preflight_does_not_guess_dimension_or_contact_qdrant(tmp_path: Path) -> None:
    with patch("koshshield.services.retrieval.vector_store.qdrant.QdrantClient") as client:
        result = check_qdrant(Settings(_env_file=None, embedding_model_dir=tmp_path))
    assert result.metadata["failure_code"] == "EMBEDDING_DIMENSION_UNAVAILABLE"
    client.assert_not_called()
