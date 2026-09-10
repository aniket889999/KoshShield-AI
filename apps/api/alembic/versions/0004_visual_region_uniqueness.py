"""Add uniqueness constraint on document visual regions provenance

Revision ID: 0004_visual_region_uniqueness
Revises: 0003_visual_evidence_derivatives
Create Date: 2026-09-10 16:45:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0004_visual_region_uniqueness"
down_revision: str | None = "0003_visual_evidence_derivatives"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    conn = op.get_bind()

    # Detect existing duplicates before creating unique constraint
    duplicate_check_sql = sa.text("""
        SELECT tenant_id, document_id, page_number, region_sequence,
               redaction_version, COUNT(*) AS cnt
        FROM document_visual_regions
        GROUP BY tenant_id, document_id, page_number, region_sequence, redaction_version
        HAVING COUNT(*) > 1
    """)
    duplicates = conn.execute(duplicate_check_sql).fetchall()
    if duplicates:
        raise RuntimeError(
            f"Cannot apply migration 0004_visual_region_uniqueness: "
            f"found {len(duplicates)} duplicate group(s) in document_visual_regions. "
            f"Resolve duplicate visual regions before running this migration."
        )

    with op.batch_alter_table("document_visual_regions", schema=None) as batch_op:
        batch_op.create_unique_constraint(
            "uq_document_visual_regions_provenance",
            [
                "tenant_id",
                "document_id",
                "page_number",
                "region_sequence",
                "redaction_version",
            ],
        )


def downgrade() -> None:
    with op.batch_alter_table("document_visual_regions", schema=None) as batch_op:
        batch_op.drop_constraint(
            "uq_document_visual_regions_provenance",
            type_="unique",
        )
