import os
import tempfile
from datetime import UTC, datetime

from alembic.config import Config
from sqlalchemy import create_engine, text

from alembic import command


def test_migration_fresh_database() -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        os.environ["KOSHSHIELD_DATABASE_URL"] = f"sqlite:///{db_path}"
        cfg = Config("apps/api/alembic.ini")
        cfg.set_main_option("script_location", "apps/api/alembic")
        cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")

        command.upgrade(cfg, "head")

        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            # Check documents columns
            doc_cols = conn.execute(text("PRAGMA table_info(documents)")).fetchall()
            doc_col_names = {c[1]: c for c in doc_cols}
            assert "tenant_id" in doc_col_names
            assert doc_col_names["tenant_id"][2].upper().startswith("VARCHAR")
            # notnull is at index 3 in table_info
            assert doc_col_names["tenant_id"][3] == 1

            # Check audit_events columns
            audit_cols = conn.execute(text("PRAGMA table_info(audit_events)")).fetchall()
            audit_col_names = {c[1]: c for c in audit_cols}
            assert "tenant_id" in audit_col_names
            assert audit_col_names["tenant_id"][3] == 1

            # Check index existence
            indexes = conn.execute(text("PRAGMA index_list(documents)")).fetchall()
            index_names = [idx[1] for idx in indexes]
            assert "ix_documents_tenant_id" in index_names

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_migration_from_old_schema_preserves_data() -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        cfg = Config("apps/api/alembic.ini")
        cfg.set_main_option("script_location", "apps/api/alembic")
        cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")

        # 1. Upgrade to old schema (0001)
        command.upgrade(cfg, "0001_initial_schema")

        engine = create_engine(f"sqlite:///{db_path}")
        now_str = datetime.now(UTC).isoformat()
        doc_id = "doc-pre-existing-123"
        audit_id = "audit-pre-existing-456"

        with engine.connect() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO documents (
                        id, filename, media_type, size_bytes, sha256,
                        vault_path, status, version, created_at, updated_at
                    ) VALUES (
                        :id, :filename, :media_type, :size_bytes, :sha256,
                        :vault_path, :status, :version, :created_at, :updated_at
                    )
                    """
                ),
                {
                    "id": doc_id,
                    "filename": "legacy_tender.pdf",
                    "media_type": "application/pdf",
                    "size_bytes": 2048,
                    "sha256": "a" * 64,
                    "vault_path": "/vault/legacy.ksh",
                    "status": "ENCRYPTED",
                    "version": 1,
                    "created_at": now_str,
                    "updated_at": now_str,
                },
            )
            conn.execute(
                text(
                    """
                    INSERT INTO audit_events (
                        id, actor_id, event_type, resource_type,
                        resource_id, details, event_hash, created_at
                    ) VALUES (
                        :id, :actor_id, :event_type, :resource_type,
                        :resource_id, :details, :event_hash, :created_at
                    )
                    """
                ),
                {
                    "id": audit_id,
                    "actor_id": "legacy-admin",
                    "event_type": "document.accepted",
                    "resource_type": "document",
                    "resource_id": doc_id,
                    "details": '{"storage": "encrypted"}',
                    "event_hash": "e" * 64,
                    "created_at": now_str,
                },
            )
            conn.commit()

        # 2. Upgrade to head (0002_add_tenant_ownership)
        command.upgrade(cfg, "head")

        # 3. Verify data preservation and default tenant backfill
        with engine.connect() as conn:
            doc_row = conn.execute(
                text("SELECT id, filename, tenant_id, status FROM documents WHERE id = :id"),
                {"id": doc_id},
            ).fetchone()
            assert doc_row is not None
            assert doc_row[0] == doc_id
            assert doc_row[1] == "legacy_tender.pdf"
            assert doc_row[2] == "default"
            assert doc_row[3] == "ENCRYPTED"

            audit_row = conn.execute(
                text("SELECT id, actor_id, tenant_id, event_type FROM audit_events WHERE id = :id"),
                {"id": audit_id},
            ).fetchone()
            assert audit_row is not None
            assert audit_row[0] == audit_id
            assert audit_row[1] == "legacy-admin"
            assert audit_row[2] == "default"
            assert audit_row[3] == "document.accepted"

        # 4. Verify downgrade works cleanly
        command.downgrade(cfg, "0001_initial_schema")
        with engine.connect() as conn:
            doc_cols = conn.execute(text("PRAGMA table_info(documents)")).fetchall()
            doc_col_names = [c[1] for c in doc_cols]
            assert "tenant_id" not in doc_col_names

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)
