import base64
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ["KOSHSHIELD_DATABASE_URL"] = "sqlite:///:memory:"
os.environ["KOSHSHIELD_MASTER_KEY_BASE64"] = base64.urlsafe_b64encode(b"k" * 32).decode()
os.environ["KOSHSHIELD_VAULT_DIR"] = "/tmp/koshshield-test-vault"
os.environ["KOSHSHIELD_AUTO_CREATE_SCHEMA"] = "true"

from koshshield.database import Base, engine  # noqa: E402
from koshshield.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def reset_state() -> None:
    vault = Path(os.environ["KOSHSHIELD_VAULT_DIR"])
    if vault.exists():
        for path in vault.iterdir():
            path.unlink()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client
