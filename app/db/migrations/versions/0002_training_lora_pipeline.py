"""training lora pipeline columns

Revision ID: 0002
Revises: 0001
Create Date: 2026-03-31

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- training_jobs ---
    # Expand enum values: job_status now uses queued/training/completed/failed/cancelled
    # (SQLite in tests ignores; Postgres needs type alter.)
    op.execute("ALTER TYPE job_status ADD VALUE IF NOT EXISTS 'queued'")
    op.execute("ALTER TYPE job_status ADD VALUE IF NOT EXISTS 'training'")

    op.add_column("training_jobs", sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("training_jobs", sa.Column("progress_pct", sa.Float(), nullable=True))
    op.add_column("training_jobs", sa.Column("epochs_completed", sa.Integer(), nullable=True))
    op.add_column("training_jobs", sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("training_jobs", sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True))

    # Keep legacy columns for now (hyperparams/current_epoch/total_epochs/train_loss/val_accuracy/logs/finished_at)

    # --- ml_models ---
    op.add_column("ml_models", sa.Column("dataset_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("ml_models", sa.Column("training_job_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("ml_models", sa.Column("training_time_seconds", sa.Float(), nullable=True))
    op.add_column("ml_models", sa.Column("onnx_s3_key", sa.String(length=512), nullable=True))
    op.add_column("ml_models", sa.Column("adapter_s3_key", sa.String(length=512), nullable=True))

    # Optional: keep old s3_key/metrics/val_loss/num_classes columns for backward compat.


def downgrade() -> None:
    op.drop_column("ml_models", "adapter_s3_key")
    op.drop_column("ml_models", "onnx_s3_key")
    op.drop_column("ml_models", "training_time_seconds")
    op.drop_column("ml_models", "training_job_id")
    op.drop_column("ml_models", "dataset_id")

    op.drop_column("training_jobs", "completed_at")
    op.drop_column("training_jobs", "metrics")
    op.drop_column("training_jobs", "epochs_completed")
    op.drop_column("training_jobs", "progress_pct")
    op.drop_column("training_jobs", "config")

