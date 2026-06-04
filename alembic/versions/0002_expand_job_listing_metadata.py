"""expand job listing metadata

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-29
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("job_listing", sa.Column("responsibilities", ARRAY(sa.Text()), nullable=True))
    op.add_column(
        "job_listing", sa.Column("mandatory_requirements", ARRAY(sa.Text()), nullable=True)
    )
    op.add_column(
        "job_listing", sa.Column("nice_to_have_requirements", ARRAY(sa.Text()), nullable=True)
    )
    op.add_column("job_listing", sa.Column("skills_tags", ARRAY(sa.Text()), nullable=True))
    op.add_column("job_listing", sa.Column("benefits", ARRAY(sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column("job_listing", "benefits")
    op.drop_column("job_listing", "skills_tags")
    op.drop_column("job_listing", "nice_to_have_requirements")
    op.drop_column("job_listing", "mandatory_requirements")
    op.drop_column("job_listing", "responsibilities")
