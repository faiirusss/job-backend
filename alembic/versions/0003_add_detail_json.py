"""add detail_json to job_listing

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-30
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("job_listing", sa.Column("detail_json", JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("job_listing", "detail_json")
