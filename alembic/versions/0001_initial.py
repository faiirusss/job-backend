"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-05-27
"""

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "cv",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("filename", sa.Text, nullable=False),
        sa.Column("file_path", sa.Text, nullable=False),
        sa.Column("text_content", sa.Text, nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column("metadata", sa.JSON, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "search_query",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("raw_query", sa.Text, nullable=False),
        sa.Column("parsed_params", sa.JSON, nullable=True),
        sa.Column("cv_id", sa.Integer, sa.ForeignKey("cv.id", ondelete="SET NULL"), nullable=True),
        sa.Column("result_count", sa.Integer, server_default="0"),
        sa.Column("from_cache", sa.Boolean, nullable=True),
        sa.Column("duration_ms", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_search_query_created", "search_query", ["created_at"])

    op.create_table(
        "job_listing",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("external_id", sa.Text, nullable=False),
        sa.Column("portal", sa.Text, nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("company", sa.Text, nullable=False),
        sa.Column("location", sa.Text, nullable=True),
        sa.Column("work_type", sa.Text, nullable=True),
        sa.Column("seniority", sa.Text, nullable=True),
        sa.Column("salary_min", sa.BigInteger, nullable=True),
        sa.Column("salary_max", sa.BigInteger, nullable=True),
        sa.Column("salary_currency", sa.Text, server_default="IDR"),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("requirements", sa.Text, nullable=True),
        sa.Column("apply_url", sa.Text, nullable=False),
        sa.Column("posted_date", sa.Date, nullable=True),
        sa.Column("embedding", Vector(384), nullable=True),
        sa.Column("scraped_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("portal", "external_id", name="uq_job_listing_portal_extid"),
        sa.CheckConstraint(
            "portal IN ('linkedin','jobstreet','glints','kalibrr')",
            name="ck_job_listing_portal",
        ),
    )
    op.create_index("idx_job_listing_scraped", "job_listing", ["scraped_at"])
    op.execute(
        "CREATE INDEX idx_job_listing_embedding ON job_listing "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists=100)"
    )

    op.create_table(
        "match_result",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "query_id",
            sa.Integer,
            sa.ForeignKey("search_query.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "job_id",
            sa.Integer,
            sa.ForeignKey("job_listing.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("cv_id", sa.Integer, sa.ForeignKey("cv.id", ondelete="CASCADE"), nullable=False),
        sa.Column("match_score", sa.Integer, nullable=False),
        sa.Column("cosine_score", sa.Float, nullable=True),
        sa.Column("llm_score", sa.Integer, nullable=True),
        sa.Column("matched_skills", ARRAY(sa.Text), server_default="{}"),
        sa.Column("missing_skills", ARRAY(sa.Text), server_default="{}"),
        sa.Column("summary_id", sa.Text, nullable=True),
        sa.Column("summary_en", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("match_score BETWEEN 0 AND 100", name="ck_match_score_range"),
    )
    op.create_index("idx_match_query", "match_result", ["query_id", "match_score"])

    op.create_table(
        "cover_letter",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "job_id",
            sa.Integer,
            sa.ForeignKey("job_listing.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("cv_id", sa.Integer, sa.ForeignKey("cv.id", ondelete="CASCADE"), nullable=False),
        sa.Column("content_id", sa.Text, nullable=False),
        sa.Column("content_en", sa.Text, nullable=False),
        sa.Column("word_count_id", sa.Integer, nullable=True),
        sa.Column("word_count_en", sa.Integer, nullable=True),
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("job_id", "cv_id", name="uq_cover_letter_job_cv"),
    )

    op.create_table(
        "cache_entry",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("params_hash", sa.Text, unique=True, nullable=False),
        sa.Column("result_payload", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_cache_expires", "cache_entry", ["expires_at"])


def downgrade() -> None:
    op.drop_table("cache_entry")
    op.drop_table("cover_letter")
    op.drop_table("match_result")
    op.drop_table("job_listing")
    op.drop_table("search_query")
    op.drop_table("cv")
