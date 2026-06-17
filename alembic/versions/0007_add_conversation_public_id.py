"""add public conversation uuid

Revision ID: 0007
Revises: 0006
Create Date: 2026-06-17
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.add_column(
        "conversation",
        sa.Column(
            "public_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
    )
    op.create_unique_constraint("uq_conversation_public_id", "conversation", ["public_id"])


def downgrade() -> None:
    op.drop_constraint("uq_conversation_public_id", "conversation", type_="unique")
    op.drop_column("conversation", "public_id")
