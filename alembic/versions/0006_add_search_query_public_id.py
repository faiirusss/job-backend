"""add public search query uuid

Revision ID: 0006
Revises: 0005
Create Date: 2026-06-17
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.add_column(
        "search_query",
        sa.Column(
            "public_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
    )
    op.create_unique_constraint("uq_search_query_public_id", "search_query", ["public_id"])


def downgrade() -> None:
    op.drop_constraint("uq_search_query_public_id", "search_query", type_="unique")
    op.drop_column("search_query", "public_id")
