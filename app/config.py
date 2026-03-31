from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_name: str = "NeuroClassifier"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://neuroclassifier:secret@postgres:5432/neuroclassifier",
        env="DATABASE_URL",
    )
    database_url_sync: str = Field(
        default="postgresql+psycopg2://neuroclassifier:secret@postgres:5432/neuroclassifier",
        env="DATABASE_URL_SYNC",
    )

    # Redis / Celery
    redis_url: str = Field(default="redis://redis:6379/0", env="REDIS_URL")
    celery_broker_url: str = Field(default="redis://redis:6379/0", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://redis:6379/1", env="CELERY_RESULT_BACKEND")

    # MinIO / S3
    minio_endpoint: str = Field(default="minio:9000", env="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", env="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", env="MINIO_SECRET_KEY")
    minio_use_ssl: bool = Field(default=False, env="MINIO_USE_SSL")
    datasets_bucket: str = Field(default="datasets", env="DATASETS_BUCKET")
    models_bucket: str = Field(default="models", env="MODELS_BUCKET")

    # Upload limits
    max_upload_size_mb: int = Field(default=500, env="MAX_UPLOAD_SIZE_MB")
    min_classes: int = 2
    min_images_per_class: int = 10
    train_split_ratio: float = 0.8
    shared_tmp_dir: str = Field(default="/tmp/neuroclassifier", env="SHARED_TMP_DIR")

    # Prometheus
    metrics_enabled: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
