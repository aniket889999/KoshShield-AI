import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Connection, Engine, inspect, text

from alembic import command
from koshshield.config import get_settings


class IncompatibleSchemaError(RuntimeError):
    """Raised when an unrecognized, drifted, or corrupted schema is encountered during migration."""


@dataclass(frozen=True)
class ColumnContract:
    name: str
    primary_key: bool = False
    nullable: bool | None = None


@dataclass(frozen=True)
class ForeignKeyContract:
    constrained_columns: tuple[str, ...]
    referred_table: str
    referred_columns: tuple[str, ...]


@dataclass(frozen=True)
class IndexContract:
    columns: tuple[str, ...]
    unique: bool = False


@dataclass(frozen=True)
class TableContract:
    name: str
    columns: dict[str, ColumnContract]
    forbidden_columns: tuple[str, ...] = ()
    foreign_keys: tuple[ForeignKeyContract, ...] = ()
    indexes: tuple[IndexContract, ...] = ()
    unique_constraints: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class SchemaContract:
    revision_id: str
    tables: dict[str, TableContract]


def _build_0001_contract() -> SchemaContract:
    tables = {
        "documents": TableContract(
            name="documents",
            columns={
                "id": ColumnContract("id", primary_key=True, nullable=False),
                "filename": ColumnContract("filename", nullable=False),
                "media_type": ColumnContract("media_type", nullable=False),
                "size_bytes": ColumnContract("size_bytes", nullable=False),
                "sha256": ColumnContract("sha256", nullable=False),
                "vault_path": ColumnContract("vault_path", nullable=False),
                "status": ColumnContract("status", nullable=False),
                "version": ColumnContract("version", nullable=False),
                "active_index_version": ColumnContract("active_index_version", nullable=True),
                "index_cleanup_pending": ColumnContract("index_cleanup_pending", nullable=None),
                "created_at": ColumnContract("created_at", nullable=False),
                "updated_at": ColumnContract("updated_at", nullable=False),
            },
            forbidden_columns=("tenant_id",),
            indexes=(IndexContract(columns=("sha256",)),),
        ),
        "extraction_jobs": TableContract(
            name="extraction_jobs",
            columns={
                "id": ColumnContract("id", primary_key=True, nullable=False),
                "document_id": ColumnContract("document_id", nullable=False),
                "status": ColumnContract("status", nullable=False),
                "pages_processed": ColumnContract("pages_processed", nullable=False),
                "total_pages": ColumnContract("total_pages", nullable=False),
                "extraction_method": ColumnContract("extraction_method", nullable=True),
                "error_message": ColumnContract("error_message", nullable=True),
                "created_at": ColumnContract("created_at", nullable=False),
                "completed_at": ColumnContract("completed_at", nullable=True),
            },
            foreign_keys=(ForeignKeyContract(("document_id",), "documents", ("id",)),),
            indexes=(IndexContract(columns=("document_id",)),),
        ),
        "document_pages": TableContract(
            name="document_pages",
            columns={
                "id": ColumnContract("id", primary_key=True, nullable=False),
                "document_id": ColumnContract("document_id", nullable=False),
                "page_number": ColumnContract("page_number", nullable=False),
                "width": ColumnContract("width", nullable=False),
                "height": ColumnContract("height", nullable=False),
                "extraction_method": ColumnContract("extraction_method", nullable=False),
                "text_hash": ColumnContract("text_hash", nullable=False),
                "encrypted_artifact_path": ColumnContract(
                    "encrypted_artifact_path", nullable=False
                ),
                "page_image_sha256": ColumnContract("page_image_sha256", nullable=True),
                "page_image_media_type": ColumnContract("page_image_media_type", nullable=True),
                "encrypted_page_image_path": ColumnContract(
                    "encrypted_page_image_path", nullable=True
                ),
                "masked_text": ColumnContract("masked_text", nullable=True),
                "masked_text_hash": ColumnContract("masked_text_hash", nullable=True),
                "created_at": ColumnContract("created_at", nullable=False),
            },
            foreign_keys=(ForeignKeyContract(("document_id",), "documents", ("id",)),),
            indexes=(IndexContract(columns=("document_id",)),),
        ),
        "document_visual_regions": TableContract(
            name="document_visual_regions",
            columns={
                "id": ColumnContract("id", primary_key=True, nullable=False),
                "document_id": ColumnContract("document_id", nullable=False),
                "page_number": ColumnContract("page_number", nullable=False),
                "region_sequence": ColumnContract("region_sequence", nullable=False),
                "region_type": ColumnContract("region_type", nullable=False),
                "source": ColumnContract("source", nullable=False),
                "bbox_json": ColumnContract("bbox_json", nullable=True),
                "caption_text": ColumnContract("caption_text", nullable=False),
                "caption_hash": ColumnContract("caption_hash", nullable=False),
                "image_sha256": ColumnContract("image_sha256", nullable=True),
                "created_at": ColumnContract("created_at", nullable=False),
            },
            foreign_keys=(ForeignKeyContract(("document_id",), "documents", ("id",)),),
            indexes=(IndexContract(columns=("document_id",)),),
        ),
        "redaction_findings": TableContract(
            name="redaction_findings",
            columns={
                "id": ColumnContract("id", primary_key=True, nullable=False),
                "document_id": ColumnContract("document_id", nullable=False),
                "page_number": ColumnContract("page_number", nullable=False),
                "finding_type": ColumnContract("finding_type", nullable=False),
                "confidence": ColumnContract("confidence", nullable=False),
                "detection_source": ColumnContract("detection_source", nullable=False),
                "start_offset": ColumnContract("start_offset", nullable=False),
                "end_offset": ColumnContract("end_offset", nullable=False),
                "bbox_json": ColumnContract("bbox_json", nullable=True),
                "salted_value_hash": ColumnContract("salted_value_hash", nullable=False),
                "masked_context": ColumnContract("masked_context", nullable=False),
                "status": ColumnContract("status", nullable=False),
                "reviewer_id": ColumnContract("reviewer_id", nullable=True),
                "version": ColumnContract("version", nullable=False),
                "created_at": ColumnContract("created_at", nullable=False),
                "updated_at": ColumnContract("updated_at", nullable=False),
            },
            foreign_keys=(ForeignKeyContract(("document_id",), "documents", ("id",)),),
            indexes=(
                IndexContract(columns=("document_id",)),
                IndexContract(columns=("finding_type",)),
            ),
        ),
        "audit_events": TableContract(
            name="audit_events",
            columns={
                "id": ColumnContract("id", primary_key=True, nullable=False),
                "actor_id": ColumnContract("actor_id", nullable=False),
                "event_type": ColumnContract("event_type", nullable=False),
                "resource_type": ColumnContract("resource_type", nullable=False),
                "resource_id": ColumnContract("resource_id", nullable=True),
                "details": ColumnContract("details", nullable=False),
                "previous_hash": ColumnContract("previous_hash", nullable=True),
                "event_hash": ColumnContract("event_hash", nullable=False),
                "created_at": ColumnContract("created_at", nullable=False),
            },
            forbidden_columns=("tenant_id", "hash_version"),
            foreign_keys=(ForeignKeyContract(("resource_id",), "documents", ("id",)),),
            indexes=(IndexContract(columns=("event_type",)),),
            unique_constraints=(("event_hash",),),
        ),
        "document_chunks": TableContract(
            name="document_chunks",
            columns={
                "id": ColumnContract("id", primary_key=True, nullable=False),
                "document_id": ColumnContract("document_id", nullable=False),
                "page_number": ColumnContract("page_number", nullable=False),
                "chunk_sequence": ColumnContract("chunk_sequence", nullable=False),
                "index_version": ColumnContract("index_version", nullable=None),
                "chunk_id": ColumnContract("chunk_id", nullable=False),
                "char_start": ColumnContract("char_start", nullable=False),
                "char_end": ColumnContract("char_end", nullable=False),
                "masked_content_hash": ColumnContract("masked_content_hash", nullable=False),
                "created_at": ColumnContract("created_at", nullable=False),
            },
            foreign_keys=(ForeignKeyContract(("document_id",), "documents", ("id",)),),
            indexes=(
                IndexContract(columns=("document_id",)),
                IndexContract(columns=("chunk_id",)),
            ),
        ),
        "agent_runs": TableContract(
            name="agent_runs",
            columns={
                "id": ColumnContract("id", primary_key=True, nullable=False),
                "tenant_id": ColumnContract("tenant_id", nullable=False),
                "actor_id": ColumnContract("actor_id", nullable=False),
                "tool_name": ColumnContract("tool_name", nullable=False),
                "classification": ColumnContract("classification", nullable=False),
                "arguments_json": ColumnContract("arguments_json", nullable=False),
                "arguments_hash": ColumnContract("arguments_hash", nullable=False),
                "argument_summary": ColumnContract("argument_summary", nullable=False),
                "status": ColumnContract("status", nullable=False),
                "state_history": ColumnContract("state_history", nullable=False),
                "policy_decision": ColumnContract("policy_decision", nullable=False),
                "policy_reason": ColumnContract("policy_reason", nullable=False),
                "approval_required": ColumnContract("approval_required", nullable=False),
                "result_json": ColumnContract("result_json", nullable=True),
                "result_hash": ColumnContract("result_hash", nullable=True),
                "failure_code": ColumnContract("failure_code", nullable=True),
                "version": ColumnContract("version", nullable=False),
                "created_at": ColumnContract("created_at", nullable=False),
                "updated_at": ColumnContract("updated_at", nullable=False),
            },
            indexes=(
                IndexContract(columns=("tenant_id",)),
                IndexContract(columns=("actor_id",)),
                IndexContract(columns=("tool_name",)),
                IndexContract(columns=("status",)),
            ),
        ),
        "agent_approvals": TableContract(
            name="agent_approvals",
            columns={
                "id": ColumnContract("id", primary_key=True, nullable=False),
                "agent_run_id": ColumnContract("agent_run_id", nullable=False),
                "decision": ColumnContract("decision", nullable=False),
                "reviewer_id": ColumnContract("reviewer_id", nullable=True),
                "version": ColumnContract("version", nullable=False),
                "created_at": ColumnContract("created_at", nullable=False),
                "decided_at": ColumnContract("decided_at", nullable=True),
            },
            foreign_keys=(ForeignKeyContract(("agent_run_id",), "agent_runs", ("id",)),),
            indexes=(IndexContract(columns=("agent_run_id",)),),
            unique_constraints=(("agent_run_id",),),
        ),
    }
    return SchemaContract(revision_id="0001_initial_schema", tables=tables)


def _build_0002_contract() -> SchemaContract:
    base = _build_0001_contract()
    tables = dict(base.tables)

    doc_cols = dict(tables["documents"].columns)
    doc_cols["tenant_id"] = ColumnContract("tenant_id", nullable=False)
    tables["documents"] = TableContract(
        name="documents",
        columns=doc_cols,
        forbidden_columns=(),
        foreign_keys=tables["documents"].foreign_keys,
        indexes=tables["documents"].indexes + (IndexContract(columns=("tenant_id",)),),
        unique_constraints=tables["documents"].unique_constraints,
    )

    audit_cols = dict(tables["audit_events"].columns)
    audit_cols["tenant_id"] = ColumnContract("tenant_id", nullable=False)
    audit_cols["hash_version"] = ColumnContract("hash_version", nullable=False)
    tables["audit_events"] = TableContract(
        name="audit_events",
        columns=audit_cols,
        forbidden_columns=(),
        foreign_keys=tables["audit_events"].foreign_keys,
        indexes=tables["audit_events"].indexes + (IndexContract(columns=("tenant_id",)),),
        unique_constraints=tables["audit_events"].unique_constraints,
    )

    return SchemaContract(revision_id="0002_add_tenant_ownership", tables=tables)


def _build_head_contract() -> SchemaContract:
    base = _build_0002_contract()
    tables = dict(base.tables)

    page_cols = dict(tables["document_pages"].columns)
    page_cols["encrypted_masked_page_image_path"] = ColumnContract(
        "encrypted_masked_page_image_path", nullable=True
    )
    page_cols["masked_page_image_sha256"] = ColumnContract(
        "masked_page_image_sha256", nullable=True
    )
    page_cols["masked_page_image_media_type"] = ColumnContract(
        "masked_page_image_media_type", nullable=True
    )
    page_cols["visual_privacy_status"] = ColumnContract("visual_privacy_status", nullable=False)
    page_cols["visual_redaction_version"] = ColumnContract(
        "visual_redaction_version", nullable=True
    )
    tables["document_pages"] = TableContract(
        name="document_pages",
        columns=page_cols,
        forbidden_columns=(),
        foreign_keys=tables["document_pages"].foreign_keys,
        indexes=tables["document_pages"].indexes,
        unique_constraints=tables["document_pages"].unique_constraints,
    )

    region_cols = dict(tables["document_visual_regions"].columns)
    region_cols["tenant_id"] = ColumnContract("tenant_id", nullable=False)
    region_cols["redaction_version"] = ColumnContract("redaction_version", nullable=False)
    region_cols["masked_image_sha256"] = ColumnContract("masked_image_sha256", nullable=True)
    tables["document_visual_regions"] = TableContract(
        name="document_visual_regions",
        columns=region_cols,
        forbidden_columns=(),
        foreign_keys=tables["document_visual_regions"].foreign_keys,
        indexes=tables["document_visual_regions"].indexes
        + (IndexContract(columns=("tenant_id",)),),
        unique_constraints=tables["document_visual_regions"].unique_constraints,
    )

    return SchemaContract(revision_id="0003_visual_evidence_derivatives", tables=tables)


SCHEMA_CONTRACT_0001 = _build_0001_contract()
SCHEMA_CONTRACT_0002 = _build_0002_contract()
SCHEMA_CONTRACT_HEAD = _build_head_contract()

RECOGNIZED_REVISIONS: dict[str, SchemaContract] = {
    "0001_initial_schema": SCHEMA_CONTRACT_0001,
    "0002_add_tenant_ownership": SCHEMA_CONTRACT_0002,
    "0003_visual_evidence_derivatives": SCHEMA_CONTRACT_HEAD,
}

RECOGNIZED_TABLES = set(SCHEMA_CONTRACT_HEAD.tables.keys())


def validate_schema_against_contract(
    conn: Connection,
    contract: SchemaContract,
) -> tuple[bool, str | None]:
    """Validates that the database schema strictly conforms to the given contract."""
    inspector = inspect(conn)
    all_tables = set(inspector.get_table_names())
    user_tables = {t for t in all_tables if not t.startswith("sqlite_") and t != "alembic_version"}

    # 1. Validate complete table set
    unrecognized_tables = user_tables - set(contract.tables.keys())
    if unrecognized_tables:
        return False, f"Unrecognized tables found in database: {sorted(unrecognized_tables)}"

    missing_tables = set(contract.tables.keys()) - user_tables
    if missing_tables:
        return False, f"Missing required tables: {sorted(missing_tables)}"

    # 2. Validate every table's structure
    for table_name, table_contract in contract.tables.items():
        columns_info = {c["name"]: c for c in inspector.get_columns(table_name)}
        existing_col_names = set(columns_info.keys())

        # Check missing columns
        missing_cols = set(table_contract.columns.keys()) - existing_col_names
        if missing_cols:
            return False, f"Table '{table_name}' missing required columns: {sorted(missing_cols)}"

        # Check forbidden columns
        forbidden = set(table_contract.forbidden_columns) & existing_col_names
        if forbidden:
            return False, f"Table '{table_name}' contains unexpected columns: {sorted(forbidden)}"

        # Check primary keys and nullability
        pk_constraint = inspector.get_pk_constraint(table_name)
        pk_columns = set(pk_constraint.get("constrained_columns") or [])
        for col_name, col_spec in table_contract.columns.items():
            col_info = columns_info[col_name]
            is_pk = (col_name in pk_columns) or bool(col_info.get("primary_key"))
            if col_spec.primary_key and not is_pk:
                return (
                    False,
                    f"Table '{table_name}' column '{col_name}' expected to be primary key",
                )
            if not col_spec.primary_key and is_pk:
                return (
                    False,
                    f"Table '{table_name}' column '{col_name}' unexpectedly primary key",
                )

            if not col_spec.primary_key and col_spec.nullable is not None:
                is_nullable = bool(col_info.get("nullable"))
                if is_nullable != col_spec.nullable:
                    return False, (
                        f"Table '{table_name}' column '{col_name}' nullability mismatch: "
                        f"expected nullable={col_spec.nullable}, got {is_nullable}"
                    )

        # Check foreign keys
        existing_fks = inspector.get_foreign_keys(table_name)
        for fk_spec in table_contract.foreign_keys:
            matched = False
            for fk in existing_fks:
                c_cols = tuple(fk.get("constrained_columns") or [])
                r_table = fk.get("referred_table")
                r_cols = tuple(fk.get("referred_columns") or [])
                if (
                    c_cols == fk_spec.constrained_columns
                    and r_table == fk_spec.referred_table
                    and r_cols == fk_spec.referred_columns
                ):
                    matched = True
                    break
            if not matched:
                return False, (
                    f"Table '{table_name}' missing foreign key on "
                    f"{list(fk_spec.constrained_columns)} -> "
                    f"{fk_spec.referred_table}.{list(fk_spec.referred_columns)}"
                )

        # Check unique constraints
        existing_uqs = inspector.get_unique_constraints(table_name)
        existing_indexes = inspector.get_indexes(table_name)
        found_unique_tuples = {
            tuple(uc.get("column_names") or []) for uc in existing_uqs if uc.get("column_names")
        }
        for idx in existing_indexes:
            if idx.get("unique") and idx.get("column_names"):
                found_unique_tuples.add(tuple(idx["column_names"]))

        if conn.dialect.name == "sqlite":
            try:
                pragma_indexes = conn.execute(text(f'PRAGMA index_list("{table_name}")')).fetchall()
                for row in pragma_indexes:
                    if bool(row[2]):  # is_unique
                        idx_name = row[1]
                        info_rows = conn.execute(
                            text(f'PRAGMA index_info("{idx_name}")')
                        ).fetchall()
                        cols = [c[2] for c in info_rows if c[2]]
                        if cols:
                            found_unique_tuples.add(tuple(cols))
            except Exception:
                pass

        for uq_cols in table_contract.unique_constraints:
            if uq_cols not in found_unique_tuples:
                return (
                    False,
                    f"Table '{table_name}' missing unique constraint on {list(uq_cols)}",
                )

        # Check required indexes
        for idx_spec in table_contract.indexes:
            matched = False
            for idx in existing_indexes:
                if tuple(idx.get("column_names") or []) == idx_spec.columns:
                    matched = True
                    break
            if not matched:
                return (
                    False,
                    f"Table '{table_name}' missing required index on {list(idx_spec.columns)}",
                )

    return True, None


def find_alembic_ini() -> Path:
    candidates = [
        Path("alembic.ini"),
        Path("apps/api/alembic.ini"),
        Path(__file__).resolve().parent.parent.parent.parent / "alembic.ini",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
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
    - ("managed", current_rev) if alembic_version exists and schema matches revision contract
    - ("empty", None) if no user tables exist
    - ("legacy_pre_alembic", "0001_initial_schema") if exactly matches full 0001 legacy schema
    - ("incompatible", reason) if unrecognized tables, missing tables/columns, or drift exist
    """
    inspector = inspect(conn)
    all_tables = set(inspector.get_table_names())
    user_tables = {t for t in all_tables if not t.startswith("sqlite_")}

    # Case 1: Database is completely empty
    if not user_tables:
        return "empty", None

    # Case 2: Database has alembic_version table
    if "alembic_version" in user_tables:
        rows = conn.execute(text("SELECT version_num FROM alembic_version")).fetchall()
        if not rows:
            return "incompatible", "alembic_version table exists but contains no revision record"
        if len(rows) > 1:
            revisions = [r[0] for r in rows]
            return (
                "incompatible",
                f"Multiple Alembic revisions found in alembic_version: {revisions}",
            )
        current_rev = rows[0][0]

        if current_rev not in RECOGNIZED_REVISIONS:
            return (
                "incompatible",
                f"Unknown Alembic revision in alembic_version: '{current_rev}'",
            )

        contract = RECOGNIZED_REVISIONS[current_rev]
        valid, error_msg = validate_schema_against_contract(conn, contract)
        if not valid:
            return (
                "incompatible",
                f"Schema drift detected for revision '{current_rev}': {error_msg}",
            )

        return "managed", current_rev

    # Case 3: Database has user tables but NO alembic_version table
    # Check if it matches genuine full legacy 0001 schema
    valid, error_msg = validate_schema_against_contract(conn, SCHEMA_CONTRACT_0001)
    if valid:
        return "legacy_pre_alembic", "0001_initial_schema"

    return (
        "incompatible",
        f"Unmanaged database does not match genuine legacy schema: {error_msg}",
    )


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
    """Validates that the database has been migrated to Alembic head and matches the head schema.
    Raises RuntimeError if uninitialized, behind head, corrupt, or drifted.
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
            f"Database schema is incompatible, corrupt, or drifted: {current_rev}. "
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
