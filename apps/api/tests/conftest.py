import base64
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

os.environ["KOSHSHIELD_DATABASE_URL"] = "sqlite:///:memory:"
os.environ["KOSHSHIELD_MASTER_KEY_BASE64"] = base64.urlsafe_b64encode(b"k" * 32).decode()
os.environ["KOSHSHIELD_VAULT_DIR"] = "/tmp/koshshield-test-vault"

from alembic.script import ScriptDirectory  # noqa: E402

from koshshield.database import Base, engine  # noqa: E402
from koshshield.database.migration import get_alembic_config  # noqa: E402
from koshshield.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def reset_state() -> None:
    vault = Path(os.environ["KOSHSHIELD_VAULT_DIR"])
    if vault.exists():
        for path in vault.iterdir():
            path.unlink()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    cfg = get_alembic_config()
    script = ScriptDirectory.from_config(cfg)
    head_rev = script.get_current_head()

    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS alembic_version "
                "(version_num VARCHAR(32) NOT NULL, PRIMARY KEY (version_num))"
            )
        )
        conn.execute(
            text(f"INSERT OR REPLACE INTO alembic_version (version_num) VALUES ('{head_rev}')")
        )


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client
