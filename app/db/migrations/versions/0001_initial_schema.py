"""initial schema

Revision ID: 0001
Revises: 
Create Date: 2026-03-31

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── datasets ─────────────────────────────────────────────────────────────
    op.create_table(
        "datasets",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "status",
            sa.Enum("uploading", "processing", "ready", "failed", name="dataset_status"),
            nullable=False,
        ),
        sa.Column("total_images", sa.Integer(), nullable=True),
        sa.Column("num_classes", sa.Integer(), nullable=True),
        sa.Column("class_names", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("train_count", sa.Integer(), nullable=True),
        sa.Column("val_count", sa.Integer(), nullable=True),
        sa.Column("s3_bucket", sa.String(255), nullable=True),
        sa.Column("s3_prefix", sa.String(512), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_datasets_name", "datasets", ["name"])
    op.create_index("ix_datasets_status", "datasets", ["status"])

    # ── ml_models ────────────────────────────────────────────────────────────
    op.create_table(
        "ml_models",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("version", sa.String(64), nullable=False),
        sa.Column(
            "status",
            sa.Enum("training", "ready", "failed", "archived", name="model_status"),
            nullable=False,
        ),
        sa.Column("base_model", sa.String(255), nullable=False),
        sa.Column("num_classes", sa.String(255), nullable=True),
        sa.Column("class_names", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("accuracy", sa.Float(), nullable=True),
        sa.Column("val_loss", sa.Float(), nullable=True),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("s3_bucket", sa.String(255), nullable=True),
        sa.Column("s3_key", sa.String(512), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ml_models_name", "ml_models", ["name"])
    op.create_index("ix_ml_models_status", "ml_models", ["status"])

    # ── training_jobs ────────────────────────────────────────────────────────
    op.create_table(
        "training_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("dataset_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("ml_model_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "status",
            sa.Enum("pending", "running", "completed", "failed", "cancelled", name="job_status"),
            nullable=False,
        ),
        sa.Column("celery_task_id", sa.String(255), nullable=True),
        sa.Column("base_model", sa.String(255), nullable=False),
        sa.Column("hyperparams", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("current_epoch", sa.Integer(), nullable=True),
        sa.Column("total_epochs", sa.Integer(), nullable=True),
        sa.Column("train_loss", sa.Float(), nullable=True),
        sa.Column("val_accuracy", sa.Float(), nullable=True),
        sa.Column("logs", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["ml_model_id"], ["ml_models.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("celery_task_id"),
    )
    op.create_index("ix_training_jobs_dataset_id", "training_jobs", ["dataset_id"])
    op.create_index("ix_training_jobs_ml_model_id", "training_jobs", ["ml_model_id"])
    op.create_index("ix_training_jobs_status", "training_jobs", ["status"])


def downgrade() -> None:
    op.drop_table("training_jobs")
    op.drop_index("ix_ml_models_status", table_name="ml_models")
    op.drop_index("ix_ml_models_name", table_name="ml_models")
    op.drop_table("ml_models")
    op.drop_index("ix_datasets_status", table_name="datasets")
    op.drop_index("ix_datasets_name", table_name="datasets")
    op.drop_table("datasets")
    op.execute("DROP TYPE IF EXISTS dataset_status")
    op.execute("DROP TYPE IF EXISTS model_status")
    op.execute("DROP TYPE IF EXISTS job_status")
