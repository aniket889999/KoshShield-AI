"""Add tenant ownership to documents and audit events with backfill and no server default

Revision ID: 0002_add_tenant_ownership
Revises: 0001_initial_schema
Create Date: 2026-09-09 21:05:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002_add_tenant_ownership"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Documents: add column, backfill existing, then enforce NOT NULL with server_default=None
    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.add_column(sa.Column("tenant_id", sa.String(length=120), nullable=True))

    op.execute("UPDATE documents SET tenant_id = 'default' WHERE tenant_id IS NULL")

    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.alter_column(
            "tenant_id",
            existing_type=sa.String(length=120),
            nullable=False,
            server_default=None,
        )
        batch_op.create_index("ix_documents_tenant_id", ["tenant_id"], unique=False)

    # 2. Audit Events: add tenant_id and hash_version, backfill existing, then enforce NOT NULL
    with op.batch_alter_table("audit_events", schema=None) as batch_op:
        batch_op.add_column(sa.Column("tenant_id", sa.String(length=120), nullable=True))
        batch_op.add_column(sa.Column("hash_version", sa.String(length=20), nullable=True))

    op.execute("UPDATE audit_events SET tenant_id = 'default' WHERE tenant_id IS NULL")
    op.execute("UPDATE audit_events SET hash_version = 'v1' WHERE hash_version IS NULL")

    with op.batch_alter_table("audit_events", schema=None) as batch_op:
        batch_op.alter_column(
            "tenant_id",
            existing_type=sa.String(length=120),
            nullable=False,
            server_default=None,
        )
        batch_op.alter_column(
            "hash_version",
            existing_type=sa.String(length=20),
            nullable=False,
            server_default=None,
        )
        batch_op.create_index("ix_audit_events_tenant_id", ["tenant_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("audit_events", schema=None) as batch_op:
        batch_op.drop_index("ix_audit_events_tenant_id")
        batch_op.drop_column("hash_version")
        batch_op.drop_column("tenant_id")

    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.drop_index("ix_documents_tenant_id")
        batch_op.drop_column("tenant_id")
