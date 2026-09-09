import os
import tempfile
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from koshshield.database.migration import (
    SCHEMA_CONTRACT_0001,
    IncompatibleSchemaError,
    bootstrap_and_upgrade,
    check_schema_at_head,
)
from koshshield.models import AuditEvent, DocumentRecord
from koshshield.services.audit import (
    append_audit_event,
    calculate_event_hash_v1,
    verify_audit_chain,
)

LEGACY_TABLE_DDLS: dict[str, str] = {
    "documents": """CREATE TABLE documents (
        id VARCHAR(36) PRIMARY KEY,
        filename VARCHAR(255) NOT NULL,
        media_type VARCHAR(80) NOT NULL,
        size_bytes INTEGER NOT NULL,
        sha256 VARCHAR(64) NOT NULL,
        vault_path TEXT NOT NULL,
        status VARCHAR(40) DEFAULT 'ENCRYPTED' NOT NULL,
        version INTEGER DEFAULT 1 NOT NULL,
        active_index_version INTEGER,
        index_cleanup_pending BOOLEAN DEFAULT 0 NOT NULL,
        created_at DATETIME NOT NULL,
        updated_at DATETIME NOT NULL
    )""",
    "extraction_jobs": """CREATE TABLE extraction_jobs (
        id VARCHAR(36) PRIMARY KEY,
        document_id VARCHAR(36) NOT NULL,
        status VARCHAR(40) DEFAULT 'QUEUED' NOT NULL,
        pages_processed INTEGER DEFAULT 0 NOT NULL,
        total_pages INTEGER DEFAULT 0 NOT NULL,
        extraction_method VARCHAR(40),
        error_message TEXT,
        created_at DATETIME NOT NULL,
        completed_at DATETIME,
        FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
    )""",
    "document_pages": """CREATE TABLE document_pages (
        id VARCHAR(36) PRIMARY KEY,
        document_id VARCHAR(36) NOT NULL,
        page_number INTEGER NOT NULL,
        width FLOAT DEFAULT 0.0 NOT NULL,
        height FLOAT DEFAULT 0.0 NOT NULL,
        extraction_method VARCHAR(40) NOT NULL,
        text_hash VARCHAR(64) NOT NULL,
        encrypted_artifact_path TEXT NOT NULL,
        page_image_sha256 VARCHAR(64),
        page_image_media_type VARCHAR(80),
        encrypted_page_image_path TEXT,
        masked_text TEXT,
        masked_text_hash VARCHAR(64),
        created_at DATETIME NOT NULL,
        FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
    )""",
    "document_visual_regions": """CREATE TABLE document_visual_regions (
        id VARCHAR(36) PRIMARY KEY,
        document_id VARCHAR(36) NOT NULL,
        page_number INTEGER NOT NULL,
        region_sequence INTEGER NOT NULL,
        region_type VARCHAR(40) NOT NULL,
        source VARCHAR(80) NOT NULL,
        bbox_json JSON,
        caption_text TEXT NOT NULL,
        caption_hash VARCHAR(64) NOT NULL,
        image_sha256 VARCHAR(64),
        created_at DATETIME NOT NULL,
        FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
    )""",
    "redaction_findings": """CREATE TABLE redaction_findings (
        id VARCHAR(36) PRIMARY KEY,
        document_id VARCHAR(36) NOT NULL,
        page_number INTEGER NOT NULL,
        finding_type VARCHAR(60) NOT NULL,
        confidence FLOAT DEFAULT 1.0 NOT NULL,
        detection_source VARCHAR(80) NOT NULL,
        start_offset INTEGER NOT NULL,
        end_offset INTEGER NOT NULL,
        bbox_json JSON,
        salted_value_hash VARCHAR(64) NOT NULL,
        masked_context TEXT NOT NULL,
        status VARCHAR(40) DEFAULT 'PENDING' NOT NULL,
        reviewer_id VARCHAR(120),
        version INTEGER DEFAULT 1 NOT NULL,
        created_at DATETIME NOT NULL,
        updated_at DATETIME NOT NULL,
        FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
    )""",
    "audit_events": """CREATE TABLE audit_events (
        id VARCHAR(36) PRIMARY KEY,
        actor_id VARCHAR(120) NOT NULL,
        event_type VARCHAR(120) NOT NULL,
        resource_type VARCHAR(80) NOT NULL,
        resource_id VARCHAR(36),
        details JSON NOT NULL,
        previous_hash VARCHAR(64),
        event_hash VARCHAR(64) NOT NULL,
        created_at DATETIME NOT NULL,
        FOREIGN KEY(resource_id) REFERENCES documents(id) ON DELETE SET NULL,
        UNIQUE (event_hash)
    )""",
    "document_chunks": """CREATE TABLE document_chunks (
        id VARCHAR(36) PRIMARY KEY,
        document_id VARCHAR(36) NOT NULL,
        page_number INTEGER NOT NULL,
        chunk_sequence INTEGER NOT NULL,
        index_version INTEGER DEFAULT 1 NOT NULL,
        chunk_id VARCHAR(64) NOT NULL,
        char_start INTEGER NOT NULL,
        char_end INTEGER NOT NULL,
        masked_content_hash VARCHAR(64) NOT NULL,
        created_at DATETIME NOT NULL,
        FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
    )""",
    "agent_runs": """CREATE TABLE agent_runs (
        id VARCHAR(36) PRIMARY KEY,
        tenant_id VARCHAR(120) NOT NULL,
        actor_id VARCHAR(120) NOT NULL,
        tool_name VARCHAR(80) NOT NULL,
        classification VARCHAR(40) NOT NULL,
        arguments_json JSON NOT NULL,
        arguments_hash VARCHAR(64) NOT NULL,
        argument_summary VARCHAR(255) NOT NULL,
        status VARCHAR(40) NOT NULL,
        state_history JSON NOT NULL,
        policy_decision VARCHAR(20) NOT NULL,
        policy_reason VARCHAR(255) NOT NULL,
        approval_required BOOLEAN DEFAULT 1 NOT NULL,
        result_json JSON,
        result_hash VARCHAR(64),
        failure_code VARCHAR(80),
        version INTEGER DEFAULT 1 NOT NULL,
        created_at DATETIME NOT NULL,
        updated_at DATETIME NOT NULL
    )""",
    "agent_approvals": """CREATE TABLE agent_approvals (
        id VARCHAR(36) PRIMARY KEY,
        agent_run_id VARCHAR(36) NOT NULL UNIQUE,
        decision VARCHAR(20) DEFAULT 'PENDING' NOT NULL,
        reviewer_id VARCHAR(120),
        version INTEGER DEFAULT 1 NOT NULL,
        created_at DATETIME NOT NULL,
        decided_at DATETIME,
        FOREIGN KEY(agent_run_id) REFERENCES agent_runs(id) ON DELETE CASCADE
    )""",
}

LEGACY_INDEX_DDLS: list[str] = [
    "CREATE INDEX ix_documents_sha256 ON documents (sha256)",
    "CREATE INDEX ix_extraction_jobs_document_id ON extraction_jobs (document_id)",
    "CREATE INDEX ix_document_pages_document_id ON document_pages (document_id)",
    "CREATE INDEX ix_document_visual_regions_document_id ON document_visual_regions (document_id)",
    "CREATE INDEX ix_redaction_findings_document_id ON redaction_findings (document_id)",
    "CREATE INDEX ix_redaction_findings_finding_type ON redaction_findings (finding_type)",
    "CREATE INDEX ix_audit_events_event_type ON audit_events (event_type)",
    "CREATE INDEX ix_document_chunks_document_id ON document_chunks (document_id)",
    "CREATE INDEX ix_document_chunks_chunk_id ON document_chunks (chunk_id)",
    "CREATE INDEX ix_agent_runs_tenant_id ON agent_runs (tenant_id)",
    "CREATE INDEX ix_agent_runs_actor_id ON agent_runs (actor_id)",
    "CREATE INDEX ix_agent_runs_tool_name ON agent_runs (tool_name)",
    "CREATE INDEX ix_agent_runs_status ON agent_runs (status)",
    "CREATE INDEX ix_agent_approvals_agent_run_id ON agent_approvals (agent_run_id)",
]


def create_raw_legacy_pre_alembic_db(db_path: str) -> None:
    """Creates a genuine pre-Alembic database using raw SQL DDL without any Alembic metadata."""
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        for stmt in LEGACY_TABLE_DDLS.values():
            conn.execute(text(stmt))
        for stmt in LEGACY_INDEX_DDLS:
            conn.execute(text(stmt))


def test_empty_database_upgrade() -> None:
    """Proves that a completely empty database successfully upgrades to head."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        db_url = f"sqlite:///{db_path}"
        head_rev = bootstrap_and_upgrade(database_url=db_url)
        assert head_rev == "0002_add_tenant_ownership"

        engine = create_engine(db_url)
        check_schema_at_head(engine)

        with engine.connect() as conn:
            doc_cols = {c[1]: c for c in conn.execute(text("PRAGMA table_info(documents)"))}
            assert "tenant_id" in doc_cols
            assert doc_cols["tenant_id"][3] == 1  # notnull

            audit_cols = {c[1]: c for c in conn.execute(text("PRAGMA table_info(audit_events)"))}
            assert "tenant_id" in audit_cols
            assert "hash_version" in audit_cols
            assert audit_cols["tenant_id"][3] == 1  # notnull
            assert audit_cols["hash_version"][3] == 1  # notnull
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_genuine_pre_alembic_database_upgrade_with_data_preservation() -> None:
    """Proves genuine full legacy schema upgrades with data and audit-chain preservation."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        db_url = f"sqlite:///{db_path}"
        create_raw_legacy_pre_alembic_db(db_path)

        engine = create_engine(db_url)
        # Ensure startup check fails before migration with explicit pre-Alembic error
        with pytest.raises(RuntimeError, match="unmigrated pre-Alembic schema"):
            check_schema_at_head(engine)

        now_1 = datetime.now(UTC) - timedelta(seconds=10)
        now_2 = datetime.now(UTC) - timedelta(seconds=5)
        doc_id = "doc-legacy-1"
        audit_id_1 = "audit-legacy-1"
        audit_id_2 = "audit-legacy-2"

        hash_1 = calculate_event_hash_v1(
            event_id=audit_id_1,
            actor_id="legacy-officer",
            event_type="document.intake",
            resource_type="document",
            resource_id=doc_id,
            details={"action": "intake"},
            previous_hash=None,
            created_at=now_1,
        )
        hash_2 = calculate_event_hash_v1(
            event_id=audit_id_2,
            actor_id="legacy-officer",
            event_type="document.extracted",
            resource_type="document",
            resource_id=doc_id,
            details={"pages": 1},
            previous_hash=hash_1,
            created_at=now_2,
        )

        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO documents (
                        id, filename, media_type, size_bytes, sha256,
                        vault_path, status, version, created_at, updated_at
                    ) VALUES (
                        :id, 'legacy_tender.pdf', 'application/pdf', 1024, 'sha-123',
                        '/vault/legacy', 'ENCRYPTED', 1, :now, :now
                    )
                    """
                ),
                {"id": doc_id, "now": now_1},
            )
            conn.execute(
                text(
                    """
                    INSERT INTO audit_events (
                        id, actor_id, event_type, resource_type,
                        resource_id, details, previous_hash, event_hash, created_at
                    ) VALUES (
                        :id, 'legacy-officer', 'document.intake', 'document',
                        :doc_id, '{"action": "intake"}', NULL, :event_hash, :now
                    )
                    """
                ),
                {"id": audit_id_1, "doc_id": doc_id, "event_hash": hash_1, "now": now_1},
            )
            conn.execute(
                text(
                    """
                    INSERT INTO audit_events (
                        id, actor_id, event_type, resource_type,
                        resource_id, details, previous_hash, event_hash, created_at
                    ) VALUES (
                        :id, 'legacy-officer', 'document.extracted', 'document',
                        :doc_id, '{"pages": 1}', :prev_hash, :event_hash, :now
                    )
                    """
                ),
                {
                    "id": audit_id_2,
                    "doc_id": doc_id,
                    "prev_hash": hash_1,
                    "event_hash": hash_2,
                    "now": now_2,
                },
            )

        # Run safe bootstrap and upgrade
        head_rev = bootstrap_and_upgrade(database_url=db_url)
        assert head_rev == "0002_add_tenant_ownership"

        # Check startup validation now passes
        check_schema_at_head(engine)

        with engine.connect() as conn:
            # Verify document row preserved and backfilled
            doc = conn.execute(
                text("SELECT id, tenant_id, filename FROM documents WHERE id = :id"),
                {"id": doc_id},
            ).fetchone()
            assert doc is not None
            assert doc[0] == doc_id
            assert doc[1] == "default"
            assert doc[2] == "legacy_tender.pdf"

            # Verify audit event rows preserved and backfilled
            events = conn.execute(
                text(
                    "SELECT id, tenant_id, hash_version, event_hash "
                    "FROM audit_events ORDER BY created_at ASC"
                )
            ).fetchall()
            assert len(events) == 2
            assert events[0][1] == "default"
            assert events[0][2] == "v1"
            assert events[0][3] == hash_1
            assert events[1][1] == "default"
            assert events[1][2] == "v1"
            assert events[1][3] == hash_2

        # Verify audit chain validity after migration using the audit service
        with Session(engine) as session:
            valid, event_objs, invalid_id = verify_audit_chain(session, tenant_id="default")
            assert valid is True
            assert invalid_id is None
            assert len(event_objs) == 2

            # Now append a new v2 audit event and verify the chain stays valid
            new_event = append_audit_event(
                session=session,
                tenant_id="default",
                actor_id="new-officer",
                event_type="document.reviewed",
                resource_type="document",
                resource_id=doc_id,
                details={"status": "approved"},
            )
            session.commit()

            assert new_event.hash_version == "v2"
            assert new_event.previous_hash == hash_2

            valid, event_objs, invalid_id = verify_audit_chain(session, tenant_id="default")
            assert valid is True
            assert invalid_id is None
            assert len(event_objs) == 3

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_minimal_documents_and_audit_events_rejected_without_mutation() -> None:
    """Proves that a partial database with only documents and audit_events
    is rejected without schema mutation.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        db_url = f"sqlite:///{db_path}"
        engine = create_engine(db_url)
        with engine.begin() as conn:
            conn.execute(text(LEGACY_TABLE_DDLS["documents"]))
            conn.execute(text(LEGACY_TABLE_DDLS["audit_events"]))
            conn.execute(text("CREATE INDEX ix_documents_sha256 ON documents (sha256)"))
            conn.execute(
                text("CREATE INDEX ix_audit_events_event_type ON audit_events (event_type)")
            )

        # Startup check must reject it
        with pytest.raises(RuntimeError, match="Missing required tables"):
            check_schema_at_head(engine)

        # bootstrap_and_upgrade must reject it without stamping or mutating
        with pytest.raises(IncompatibleSchemaError, match="Missing required tables"):
            bootstrap_and_upgrade(database_url=db_url)

        # Verify zero mutation occurred
        with engine.connect() as conn:
            insp = inspect(conn)
            tables = set(insp.get_table_names())
            assert "alembic_version" not in tables
            assert tables == {"documents", "audit_events"}
            doc_cols = {c["name"] for c in insp.get_columns("documents")}
            assert "tenant_id" not in doc_cols
            audit_cols = {c["name"] for c in insp.get_columns("audit_events")}
            assert "tenant_id" not in audit_cols
            assert "hash_version" not in audit_cols
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_every_missing_required_table_detected() -> None:
    """Proves that omission of any required application table is detected and rejected."""
    required_tables = sorted(SCHEMA_CONTRACT_0001.tables.keys())

    for omitted_table in required_tables:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            db_url = f"sqlite:///{db_path}"
            engine = create_engine(db_url)
            with engine.begin() as conn:
                for tbl, ddl in LEGACY_TABLE_DDLS.items():
                    if tbl != omitted_table:
                        conn.execute(text(ddl))
                for idx_ddl in LEGACY_INDEX_DDLS:
                    # Execute index only if its target table exists
                    if f"ON {omitted_table} " not in idx_ddl:
                        conn.execute(text(idx_ddl))

            with pytest.raises(
                IncompatibleSchemaError, match=f"Missing required tables:.*'{omitted_table}'"
            ):
                bootstrap_and_upgrade(database_url=db_url)

            # Confirm alembic_version was not created
            with engine.connect() as conn:
                insp = inspect(conn)
                assert "alembic_version" not in insp.get_table_names()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


def test_missing_critical_columns_detected() -> None:
    """Proves that missing critical columns are detected and rejected."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        db_url = f"sqlite:///{db_path}"
        engine = create_engine(db_url)
        with engine.begin() as conn:
            for tbl, ddl in LEGACY_TABLE_DDLS.items():
                if tbl == "documents":
                    # Omit sha256 and vault_path
                    conn.execute(
                        text(
                            """CREATE TABLE documents (
                                id VARCHAR(36) PRIMARY KEY,
                                filename VARCHAR(255) NOT NULL,
                                media_type VARCHAR(80) NOT NULL,
                                size_bytes INTEGER NOT NULL,
                                status VARCHAR(40) DEFAULT 'ENCRYPTED' NOT NULL,
                                version INTEGER DEFAULT 1 NOT NULL,
                                created_at DATETIME NOT NULL,
                                updated_at DATETIME NOT NULL
                            )"""
                        )
                    )
                else:
                    conn.execute(text(ddl))
            for idx_ddl in LEGACY_INDEX_DDLS:
                if "ix_documents_sha256" not in idx_ddl:
                    conn.execute(text(idx_ddl))

        with pytest.raises(
            IncompatibleSchemaError, match="Table 'documents' missing required columns:"
        ):
            bootstrap_and_upgrade(database_url=db_url)
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_missing_foreign_keys_detected() -> None:
    """Proves that missing foreign key constraints are detected and rejected."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        db_url = f"sqlite:///{db_path}"
        engine = create_engine(db_url)
        with engine.begin() as conn:
            for tbl, ddl in LEGACY_TABLE_DDLS.items():
                if tbl == "extraction_jobs":
                    # Create extraction_jobs WITHOUT the foreign key constraint
                    conn.execute(
                        text(
                            """CREATE TABLE extraction_jobs (
                                id VARCHAR(36) PRIMARY KEY,
                                document_id VARCHAR(36) NOT NULL,
                                status VARCHAR(40) DEFAULT 'QUEUED' NOT NULL,
                                pages_processed INTEGER DEFAULT 0 NOT NULL,
                                total_pages INTEGER DEFAULT 0 NOT NULL,
                                extraction_method VARCHAR(40),
                                error_message TEXT,
                                created_at DATETIME NOT NULL,
                                completed_at DATETIME
                            )"""
                        )
                    )
                else:
                    conn.execute(text(ddl))
            for idx_ddl in LEGACY_INDEX_DDLS:
                conn.execute(text(idx_ddl))

        fk_match = (
            r"Table 'extraction_jobs' missing foreign key on \['document_id'\] -> "
            r"documents.\['id'\]"
        )
        with pytest.raises(IncompatibleSchemaError, match=fk_match):
            bootstrap_and_upgrade(database_url=db_url)
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_missing_indexes_detected() -> None:
    """Proves that missing required indexes are detected and rejected."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        db_url = f"sqlite:///{db_path}"
        engine = create_engine(db_url)
        with engine.begin() as conn:
            for ddl in LEGACY_TABLE_DDLS.values():
                conn.execute(text(ddl))
            # Omit ix_documents_sha256
            for idx_ddl in LEGACY_INDEX_DDLS:
                if "ix_documents_sha256" not in idx_ddl:
                    conn.execute(text(idx_ddl))

        with pytest.raises(
            IncompatibleSchemaError,
            match="Table 'documents' missing required index on \\['sha256'\\]",
        ):
            bootstrap_and_upgrade(database_url=db_url)
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_missing_unique_constraints_detected() -> None:
    """Proves that missing critical unique constraints are detected and rejected."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        db_url = f"sqlite:///{db_path}"
        engine = create_engine(db_url)
        with engine.begin() as conn:
            for tbl, ddl in LEGACY_TABLE_DDLS.items():
                if tbl == "agent_approvals":
                    # Omit UNIQUE constraint on agent_run_id
                    conn.execute(
                        text(
                            """CREATE TABLE agent_approvals (
                                id VARCHAR(36) PRIMARY KEY,
                                agent_run_id VARCHAR(36) NOT NULL,
                                decision VARCHAR(20) DEFAULT 'PENDING' NOT NULL,
                                reviewer_id VARCHAR(120),
                                version INTEGER DEFAULT 1 NOT NULL,
                                created_at DATETIME NOT NULL,
                                decided_at DATETIME,
                                FOREIGN KEY(agent_run_id)
                                    REFERENCES agent_runs(id) ON DELETE CASCADE
                            )"""
                        )
                    )
                else:
                    conn.execute(text(ddl))
            for idx_ddl in LEGACY_INDEX_DDLS:
                conn.execute(text(idx_ddl))

        with pytest.raises(
            IncompatibleSchemaError,
            match="Table 'agent_approvals' missing unique constraint on \\['agent_run_id'\\]",
        ):
            bootstrap_and_upgrade(database_url=db_url)
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_forged_head_revision_with_drift_rejected() -> None:
    """Proves that a forged alembic_version claiming head is rejected when schema has drift."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        db_url = f"sqlite:///{db_path}"
        engine = create_engine(db_url)
        # Create minimal tables + forged alembic_version table claiming 0002_add_tenant_ownership
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE documents (id VARCHAR(36) PRIMARY KEY)"))
            conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) PRIMARY KEY)"))
            conn.execute(text("INSERT INTO alembic_version VALUES ('0002_add_tenant_ownership')"))

        # Startup check must catch drift and reject
        with pytest.raises(
            RuntimeError, match="Schema drift detected for revision '0002_add_tenant_ownership'"
        ):
            check_schema_at_head(engine)

        # bootstrap_and_upgrade must also reject
        with pytest.raises(
            IncompatibleSchemaError,
            match="Schema drift detected for revision '0002_add_tenant_ownership'",
        ):
            bootstrap_and_upgrade(database_url=db_url)
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_unknown_and_multiple_alembic_revisions_rejected() -> None:
    """Proves that unknown, empty, or multiple Alembic revisions are detected and rejected."""
    # 1. Unknown revision
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    try:
        db_url = f"sqlite:///{db_path}"
        engine = create_engine(db_url)
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) PRIMARY KEY)"))
            conn.execute(text("INSERT INTO alembic_version VALUES ('9999_rogue_revision')"))

        with pytest.raises(
            RuntimeError, match="Unknown Alembic revision in alembic_version: '9999_rogue_revision'"
        ):
            check_schema_at_head(engine)

        with pytest.raises(
            IncompatibleSchemaError,
            match="Unknown Alembic revision in alembic_version: '9999_rogue_revision'",
        ):
            bootstrap_and_upgrade(database_url=db_url)
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

    # 2. Multiple revisions
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    try:
        db_url = f"sqlite:///{db_path}"
        engine = create_engine(db_url)
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) PRIMARY KEY)"))
            conn.execute(text("INSERT INTO alembic_version VALUES ('0001_initial_schema')"))
            conn.execute(text("INSERT INTO alembic_version VALUES ('0002_add_tenant_ownership')"))

        with pytest.raises(
            RuntimeError, match="Multiple Alembic revisions found in alembic_version"
        ):
            check_schema_at_head(engine)

        with pytest.raises(
            IncompatibleSchemaError, match="Multiple Alembic revisions found in alembic_version"
        ):
            bootstrap_and_upgrade(database_url=db_url)
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

    # 3. Empty alembic_version table
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    try:
        db_url = f"sqlite:///{db_path}"
        engine = create_engine(db_url)
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) PRIMARY KEY)"))

        with pytest.raises(
            RuntimeError, match="alembic_version table exists but contains no revision record"
        ):
            check_schema_at_head(engine)

        with pytest.raises(
            IncompatibleSchemaError,
            match="alembic_version table exists but contains no revision record",
        ):
            bootstrap_and_upgrade(database_url=db_url)
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_unknown_schema_rejection() -> None:
    """Proves that databases with unrecognized tables are rejected."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        db_url = f"sqlite:///{db_path}"
        engine = create_engine(db_url)
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE foreign_unrecognized_table (id TEXT PRIMARY KEY)"))

        with pytest.raises(IncompatibleSchemaError, match="Unrecognized tables found"):
            bootstrap_and_upgrade(database_url=db_url)

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_missing_tenant_insert_rejection() -> None:
    """Proves tenant constraint enforcement at the ORM/DB level."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        db_url = f"sqlite:///{db_path}"
        bootstrap_and_upgrade(database_url=db_url)
        engine = create_engine(db_url)

        with Session(engine) as session:
            # 1. DocumentRecord without tenant_id must fail
            now = datetime.now(UTC)
            doc_no_tenant = DocumentRecord(
                id=str(uuid4()),
                filename="no_tenant.pdf",
                media_type="application/pdf",
                size_bytes=100,
                sha256="s" * 64,
                vault_path="/v/path",
                status="ENCRYPTED",
                created_at=now,
                updated_at=now,
            )
            session.add(doc_no_tenant)
            with pytest.raises(IntegrityError):
                session.commit()
            session.rollback()

            # 2. AuditEvent without tenant_id must fail
            audit_no_tenant = AuditEvent(
                id=str(uuid4()),
                actor_id="officer",
                event_type="test.event",
                resource_type="document",
                details={},
                event_hash="h" * 64,
                created_at=now,
            )
            session.add(audit_no_tenant)
            with pytest.raises(IntegrityError):
                session.commit()
            session.rollback()

            # 3. append_audit_event without tenant_id must fail
            with pytest.raises((TypeError, ValueError)):
                append_audit_event(
                    session=session,
                    tenant_id="",  # empty tenant
                    actor_id="officer",
                    event_type="test.event",
                    resource_type="document",
                    resource_id=None,
                    details={},
                )
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)
