"""add auth and persistent conversations

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-16
"""

import sqlalchemy as sa

from alembic import op


revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_account",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("email", name="uq_user_account_email"),
    )
    op.create_index("idx_user_account_email", "user_account", ["email"])

    op.create_table(
        "auth_session",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("token_hash", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["user_account.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("token_hash", name="uq_auth_session_token_hash"),
    )
    op.create_index("idx_auth_session_token", "auth_session", ["token_hash"])
    op.create_index("idx_auth_session_user", "auth_session", ["user_id"])

    op.create_table(
        "conversation",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["user_account.id"], ondelete="CASCADE"),
    )
    op.create_index("idx_conversation_user_updated", "conversation", ["user_id", "updated_at"])
    op.create_index("idx_conversation_user_deleted", "conversation", ["user_id", "deleted_at"])

    op.add_column("cv", sa.Column("user_id", sa.Integer(), nullable=True))
    op.create_foreign_key("fk_cv_user", "cv", "user_account", ["user_id"], ["id"], ondelete="CASCADE")

    op.add_column("search_query", sa.Column("user_id", sa.Integer(), nullable=True))
    op.add_column("search_query", sa.Column("conversation_id", sa.Integer(), nullable=True))
    op.add_column(
        "search_query",
        sa.Column("status", sa.Text(), nullable=False, server_default="queued"),
    )
    op.add_column("search_query", sa.Column("error_message", sa.Text(), nullable=True))
    op.create_foreign_key(
        "fk_search_query_user",
        "search_query",
        "user_account",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_search_query_conversation",
        "search_query",
        "conversation",
        ["conversation_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_check_constraint(
        "ck_search_query_status",
        "search_query",
        "status IN ('queued','running','completed','failed')",
    )
    op.create_index("idx_search_query_user_created", "search_query", ["user_id", "created_at"])
    op.create_index(
        "idx_search_query_conversation", "search_query", ["conversation_id", "created_at"]
    )

    op.create_table(
        "conversation_message",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("conversation_id", sa.Integer(), nullable=False),
        sa.Column("search_query_id", sa.Integer(), nullable=True),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint(
            "role IN ('user','assistant','system')", name="ck_conversation_message_role"
        ),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversation.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["search_query_id"], ["search_query.id"], ondelete="SET NULL"),
    )
    op.create_index(
        "idx_conversation_message_conversation_created",
        "conversation_message",
        ["conversation_id", "created_at"],
    )
    op.create_index("idx_conversation_message_query", "conversation_message", ["search_query_id"])

    op.add_column("match_result", sa.Column("user_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_match_result_user",
        "match_result",
        "user_account",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("idx_match_user_job", "match_result", ["user_id", "job_id", "created_at"])

    op.add_column("cover_letter", sa.Column("user_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_cover_letter_user",
        "cover_letter",
        "user_account",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("idx_cover_letter_user", "cover_letter", ["user_id", "generated_at"])

    op.create_table(
        "search_result",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("query_id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["query_id"], ["search_query.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["job_id"], ["job_listing.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("query_id", "job_id", name="uq_search_result_query_job"),
    )
    op.create_index("idx_search_result_query_position", "search_result", ["query_id", "position"])


def downgrade() -> None:
    op.drop_index("idx_search_result_query_position", table_name="search_result")
    op.drop_table("search_result")

    op.drop_index("idx_cover_letter_user", table_name="cover_letter")
    op.drop_constraint("fk_cover_letter_user", "cover_letter", type_="foreignkey")
    op.drop_column("cover_letter", "user_id")

    op.drop_index("idx_match_user_job", table_name="match_result")
    op.drop_constraint("fk_match_result_user", "match_result", type_="foreignkey")
    op.drop_column("match_result", "user_id")

    op.drop_index("idx_conversation_message_query", table_name="conversation_message")
    op.drop_index(
        "idx_conversation_message_conversation_created", table_name="conversation_message"
    )
    op.drop_table("conversation_message")

    op.drop_index("idx_search_query_conversation", table_name="search_query")
    op.drop_index("idx_search_query_user_created", table_name="search_query")
    op.drop_constraint("ck_search_query_status", "search_query", type_="check")
    op.drop_constraint("fk_search_query_conversation", "search_query", type_="foreignkey")
    op.drop_constraint("fk_search_query_user", "search_query", type_="foreignkey")
    op.drop_column("search_query", "error_message")
    op.drop_column("search_query", "status")
    op.drop_column("search_query", "conversation_id")
    op.drop_column("search_query", "user_id")

    op.drop_constraint("fk_cv_user", "cv", type_="foreignkey")
    op.drop_column("cv", "user_id")

    op.drop_index("idx_conversation_user_deleted", table_name="conversation")
    op.drop_index("idx_conversation_user_updated", table_name="conversation")
    op.drop_table("conversation")

    op.drop_index("idx_auth_session_user", table_name="auth_session")
    op.drop_index("idx_auth_session_token", table_name="auth_session")
    op.drop_table("auth_session")

    op.drop_index("idx_user_account_email", table_name="user_account")
    op.drop_table("user_account")
