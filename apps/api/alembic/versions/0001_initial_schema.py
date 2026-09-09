"""Initial schema baseline before tenant ownership

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-09-09 21:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("media_type", sa.String(length=80), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("vault_path", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False, server_default="ENCRYPTED"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("active_index_version", sa.Integer(), nullable=True),
        sa.Column("index_cleanup_pending", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_documents_sha256", "documents", ["sha256"])

    op.create_table(
        "extraction_jobs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(length=36),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=40), nullable=False, server_default="QUEUED"),
        sa.Column("pages_processed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_pages", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("extraction_method", sa.String(length=40), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_extraction_jobs_document_id", "extraction_jobs", ["document_id"])

    op.create_table(
        "document_pages",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(length=36),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("width", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("height", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("extraction_method", sa.String(length=40), nullable=False),
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        sa.Column("encrypted_artifact_path", sa.Text(), nullable=False),
        sa.Column("page_image_sha256", sa.String(length=64), nullable=True),
        sa.Column("page_image_media_type", sa.String(length=80), nullable=True),
        sa.Column("encrypted_page_image_path", sa.Text(), nullable=True),
        sa.Column("masked_text", sa.Text(), nullable=True),
        sa.Column("masked_text_hash", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_document_pages_document_id", "document_pages", ["document_id"])

    op.create_table(
        "document_visual_regions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(length=36),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("region_sequence", sa.Integer(), nullable=False),
        sa.Column("region_type", sa.String(length=40), nullable=False),
        sa.Column("source", sa.String(length=80), nullable=False),
        sa.Column("bbox_json", sa.JSON(), nullable=True),
        sa.Column("caption_text", sa.Text(), nullable=False),
        sa.Column("caption_hash", sa.String(length=64), nullable=False),
        sa.Column("image_sha256", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_document_visual_regions_document_id",
        "document_visual_regions",
        ["document_id"],
    )

    op.create_table(
        "redaction_findings",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(length=36),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("finding_type", sa.String(length=60), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("detection_source", sa.String(length=80), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.Column("bbox_json", sa.JSON(), nullable=True),
        sa.Column("salted_value_hash", sa.String(length=64), nullable=False),
        sa.Column("masked_context", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False, server_default="PENDING"),
        sa.Column("reviewer_id", sa.String(length=120), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_redaction_findings_document_id", "redaction_findings", ["document_id"])
    op.create_index("ix_redaction_findings_finding_type", "redaction_findings", ["finding_type"])

    op.create_table(
        "audit_events",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("actor_id", sa.String(length=120), nullable=False),
        sa.Column("event_type", sa.String(length=120), nullable=False),
        sa.Column("resource_type", sa.String(length=80), nullable=False),
        sa.Column(
            "resource_id",
            sa.String(length=36),
            sa.ForeignKey("documents.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("previous_hash", sa.String(length=64), nullable=True),
        sa.Column("event_hash", sa.String(length=64), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_audit_events_event_type", "audit_events", ["event_type"])

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(length=36),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("chunk_sequence", sa.Integer(), nullable=False),
        sa.Column("index_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("chunk_id", sa.String(length=64), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("masked_content_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])
    op.create_index("ix_document_chunks_chunk_id", "document_chunks", ["chunk_id"])

    op.create_table(
        "agent_runs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.String(length=120), nullable=False),
        sa.Column("actor_id", sa.String(length=120), nullable=False),
        sa.Column("tool_name", sa.String(length=80), nullable=False),
        sa.Column("classification", sa.String(length=40), nullable=False),
        sa.Column("arguments_json", sa.JSON(), nullable=False),
        sa.Column("arguments_hash", sa.String(length=64), nullable=False),
        sa.Column("argument_summary", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("state_history", sa.JSON(), nullable=False),
        sa.Column("policy_decision", sa.String(length=20), nullable=False),
        sa.Column("policy_reason", sa.String(length=255), nullable=False),
        sa.Column("approval_required", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column("result_json", sa.JSON(), nullable=True),
        sa.Column("result_hash", sa.String(length=64), nullable=True),
        sa.Column("failure_code", sa.String(length=80), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_agent_runs_tenant_id", "agent_runs", ["tenant_id"])
    op.create_index("ix_agent_runs_actor_id", "agent_runs", ["actor_id"])
    op.create_index("ix_agent_runs_tool_name", "agent_runs", ["tool_name"])
    op.create_index("ix_agent_runs_status", "agent_runs", ["status"])

    op.create_table(
        "agent_approvals",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "agent_run_id",
            sa.String(length=36),
            sa.ForeignKey("agent_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("decision", sa.String(length=20), nullable=False, server_default="PENDING"),
        sa.Column("reviewer_id", sa.String(length=120), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("agent_run_id", name="uq_agent_approval_run"),
    )
    op.create_index("ix_agent_approvals_agent_run_id", "agent_approvals", ["agent_run_id"])


def downgrade() -> None:
    op.drop_table("agent_approvals")
    op.drop_table("agent_runs")
    op.drop_table("document_chunks")
    op.drop_table("audit_events")
    op.drop_table("redaction_findings")
    op.drop_table("document_visual_regions")
    op.drop_table("document_pages")
    op.drop_table("extraction_jobs")
    op.drop_table("documents")
