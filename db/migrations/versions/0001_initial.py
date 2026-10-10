"""initial: people and faces tables

Revision ID: 0001
Revises:
Create Date: 2026-10-07
"""
from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "people",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False, unique=True),
    )
    op.create_table(
        "faces",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("person_id", sa.Integer(), sa.ForeignKey("people.id", ondelete="CASCADE"), nullable=False),
        sa.Column("image_path", sa.String(), nullable=False, unique=True),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(512), nullable=False),
    )
    op.create_index("ix_faces_person_id", "faces", ["person_id"])


def downgrade():
    op.drop_index("ix_faces_person_id", table_name="faces")
    op.drop_table("faces")
    op.drop_table("people")
