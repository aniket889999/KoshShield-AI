import os
import sys
from pathlib import Path
from typing import Any

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Connection, Engine, inspect, text

from alembic import command
from koshshield.config import get_settings


class IncompatibleSchemaError(RuntimeError):
    """Raised when an unrecognized or corrupted schema is encountered during migration."""


RECOGNIZED_LEGACY_TABLES = {
    "documents",
    "extraction_jobs",
    "document_pages",
    "document_visual_regions",
    "redaction_findings",
    "audit_events",
    "document_chunks",
    "agent_runs",
    "agent_approvals",
}


def find_alembic_ini() -> Path:
    candidates = [
        Path("alembic.ini"),
        Path("apps/api/alembic.ini"),
        Path(__file__).resolve().parent.parent.parent.parent / "alembic.ini",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    # Fallback to apps/api/alembic.ini relative to this file
    return (
        Path(__file__).resolve().parent.parent.parent.parent / "apps" / "api" / "alembic.ini"
    ).resolve()


def get_alembic_config(
    database_url: str | None = None,
    config_path: str | Path | None = None,
) -> Config:
    ini_path = Path(config_path).resolve() if config_path else find_alembic_ini()
    if not ini_path.is_file():
        raise FileNotFoundError(f"Alembic configuration file not found at {ini_path}")
    cfg = Config(str(ini_path))
    db_url = database_url or get_settings().database_url
    cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


def inspect_schema_state(conn: Connection) -> tuple[str, Any]:
    """Inspects the database schema and returns:
    - ("managed", current_rev) if alembic_version exists
    - ("empty", None) if no user tables exist
    - ("legacy_pre_alembic", "0001") if it exactly matches the recognized pre-Alembic schema
    - ("incompatible", reason) if unrecognized tables or incompatible column layouts exist
    """
    inspector = inspect(conn)
    tables = set(inspector.get_table_names())

    if "alembic_version" in tables:
        result = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
        return "managed", result

    # Exclude SQLite internal tables
    user_tables = {t for t in tables if not t.startswith("sqlite_")}
    if not user_tables:
        return "empty", None

    # Check for unrecognized tables
    unrecognized = user_tables - RECOGNIZED_LEGACY_TABLES
    if unrecognized:
        return "incompatible", f"Unrecognized tables found in database: {sorted(unrecognized)}"

    # Check that core tables exist
    core_tables = {"documents", "audit_events"}
    missing_core = core_tables - user_tables
    if missing_core:
        return (
            "incompatible",
            f"Partially initialized schema: missing core tables {sorted(missing_core)}",
        )

    # Check documents table columns
    doc_cols = {c["name"] for c in inspector.get_columns("documents")}
    required_doc_cols = {"id", "sha256", "vault_path", "status"}
    if not required_doc_cols.issubset(doc_cols):
        return (
            "incompatible",
            f"documents table missing required legacy columns: {required_doc_cols - doc_cols}",
        )

    # Check audit_events table columns
    audit_cols = {c["name"] for c in inspector.get_columns("audit_events")}
    required_audit_cols = {"id", "actor_id", "event_type", "event_hash"}
    if not required_audit_cols.issubset(audit_cols):
        missing_cols = required_audit_cols - audit_cols
        return (
            "incompatible",
            f"audit_events table missing required legacy columns: {missing_cols}",
        )

    # In a genuine pre-Alembic database, tenant_id is NOT yet in documents
    if "tenant_id" in doc_cols and "tenant_id" in audit_cols:
        # Tables have tenant_id but no alembic_version -> irregular state
        return "incompatible", "Database has tenant columns but no alembic_version stamp"

    return "legacy_pre_alembic", "0001_initial_schema"


def bootstrap_and_upgrade(
    database_url: str | None = None,
    config_path: str | Path | None = None,
) -> str:
    """Safely bootstraps genuine pre-Alembic databases or empty databases and upgrades to head."""
    cfg = get_alembic_config(database_url=database_url, config_path=config_path)
    from sqlalchemy import create_engine

    db_url = cfg.get_main_option("sqlalchemy.url")
    engine = create_engine(db_url)

    with engine.connect() as conn:
        state, detail = inspect_schema_state(conn)

    if state == "incompatible":
        raise IncompatibleSchemaError(
            f"Cannot migrate database safely: {detail}. "
            "Manual inspection required; aborting without modifying schema."
        )

    if state == "legacy_pre_alembic":
        # Stamp revision 0001_initial_schema representing baseline legacy schema
        command.stamp(cfg, "0001_initial_schema")

    # Upgrade to head
    command.upgrade(cfg, "head")

    script = ScriptDirectory.from_config(cfg)
    return script.get_current_head()


def check_schema_at_head(
    engine: Engine,
    config_path: str | Path | None = None,
) -> None:
    """Validates that the database has been migrated to Alembic head.
    Raises RuntimeError if uninitialized or behind head.
    """
    cfg = get_alembic_config(config_path=config_path)
    script = ScriptDirectory.from_config(cfg)
    head_rev = script.get_current_head()

    with engine.connect() as conn:
        state, current_rev = inspect_schema_state(conn)

    if state == "empty":
        raise RuntimeError(
            "Database schema is uninitialized (no tables found). "
            "Run 'make migrate' or 'alembic upgrade head' before starting the application."
        )
    if state == "legacy_pre_alembic":
        raise RuntimeError(
            "Database contains an unmigrated pre-Alembic schema. "
            "Run 'make migrate' before starting the application."
        )
    if state == "incompatible":
        raise RuntimeError(
            f"Database schema is incompatible or corrupt ({current_rev}). "
            "Run 'make migrate' or inspect database."
        )

    if state == "managed" and current_rev != head_rev:
        raise RuntimeError(
            f"Database schema revision '{current_rev}' is behind head '{head_rev}'. "
            "Run 'make migrate' before starting the application."
        )


def main() -> None:
    db_url = os.environ.get("KOSHSHIELD_DATABASE_URL")
    print("Running KoshShield database migration...")
    try:
        head = bootstrap_and_upgrade(database_url=db_url)
        print(f"Database migrated successfully to revision: {head}")
    except IncompatibleSchemaError as err:
        print(f"Migration aborted: {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"Migration failed: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
