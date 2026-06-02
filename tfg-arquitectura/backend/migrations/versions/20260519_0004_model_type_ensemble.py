"""Add 'ensemble' value to model_type enum

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-19
"""

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Postgres enums need ALTER TYPE ... ADD VALUE (transactional in PG 12+)
    op.execute("ALTER TYPE model_type ADD VALUE IF NOT EXISTS 'ensemble'")


def downgrade() -> None:
    # Postgres does not support removing enum values without dropping/recreating the type.
    # Leave downgrade as a no-op — removing this enum value would require dropping
    # all rows that use it first.
    pass
