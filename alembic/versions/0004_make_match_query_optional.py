"""make match_result query optional

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-15
"""

import sqlalchemy as sa
from alembic import op


revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("match_result", "query_id", existing_type=sa.Integer(), nullable=True)


def downgrade() -> None:
    op.alter_column("match_result", "query_id", existing_type=sa.Integer(), nullable=False)
