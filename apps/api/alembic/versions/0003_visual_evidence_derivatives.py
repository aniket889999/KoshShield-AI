"""Add visual evidence derivative fields and visual region provenance

Revision ID: 0003_visual_evidence_derivatives
Revises: 0002_add_tenant_ownership
Create Date: 2026-09-10 15:10:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003_visual_evidence_derivatives"
down_revision: str | None = "0002_add_tenant_ownership"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Document Pages: add masked image fields and visual privacy status
    with op.batch_alter_table("document_pages", schema=None) as batch_op:
        batch_op.add_column(sa.Column("encrypted_masked_page_image_path", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column("masked_page_image_sha256", sa.String(length=64), nullable=True)
        )
        batch_op.add_column(
            sa.Column("masked_page_image_media_type", sa.String(length=80), nullable=True)
        )
        batch_op.add_column(sa.Column("visual_privacy_status", sa.String(length=40), nullable=True))
        batch_op.add_column(sa.Column("visual_redaction_version", sa.Integer(), nullable=True))

    op.execute(
        "UPDATE document_pages SET visual_privacy_status = 'NOT_APPLICABLE' "
        "WHERE visual_privacy_status IS NULL"
    )

    with op.batch_alter_table("document_pages", schema=None) as batch_op:
        batch_op.alter_column(
            "visual_privacy_status",
            existing_type=sa.String(length=40),
            nullable=False,
            server_default=None,
        )

    # 2. Document Visual Regions: add tenant_id, redaction_version, and masked_image_sha256
    with op.batch_alter_table("document_visual_regions", schema=None) as batch_op:
        batch_op.add_column(sa.Column("tenant_id", sa.String(length=120), nullable=True))
        batch_op.add_column(sa.Column("redaction_version", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("masked_image_sha256", sa.String(length=64), nullable=True))

    op.execute(
        "UPDATE document_visual_regions "
        "SET tenant_id = ("
        "  SELECT tenant_id FROM documents "
        "  WHERE documents.id = document_visual_regions.document_id"
        ") "
        "WHERE tenant_id IS NULL"
    )
    op.execute("UPDATE document_visual_regions SET tenant_id = 'default' WHERE tenant_id IS NULL")
    op.execute(
        "UPDATE document_visual_regions "
        "SET redaction_version = ("
        "  SELECT version FROM documents "
        "  WHERE documents.id = document_visual_regions.document_id"
        ") "
        "WHERE redaction_version IS NULL"
    )
    op.execute(
        "UPDATE document_visual_regions SET redaction_version = 1 WHERE redaction_version IS NULL"
    )

    with op.batch_alter_table("document_visual_regions", schema=None) as batch_op:
        batch_op.alter_column(
            "tenant_id",
            existing_type=sa.String(length=120),
            nullable=False,
            server_default=None,
        )
        batch_op.alter_column(
            "redaction_version",
            existing_type=sa.Integer(),
            nullable=False,
            server_default=None,
        )
        batch_op.create_index("ix_document_visual_regions_tenant_id", ["tenant_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("document_visual_regions", schema=None) as batch_op:
        batch_op.drop_index("ix_document_visual_regions_tenant_id")
        batch_op.drop_column("masked_image_sha256")
        batch_op.drop_column("redaction_version")
        batch_op.drop_column("tenant_id")

    with op.batch_alter_table("document_pages", schema=None) as batch_op:
        batch_op.drop_column("visual_redaction_version")
        batch_op.drop_column("visual_privacy_status")
        batch_op.drop_column("masked_page_image_media_type")
        batch_op.drop_column("masked_page_image_sha256")
        batch_op.drop_column("encrypted_masked_page_image_path")
