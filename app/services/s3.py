import logging
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from typing import BinaryIO, Optional

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


def get_s3_client():
    """Return a configured boto3 S3 client pointed at MinIO."""
    return boto3.client(
        "s3",
        endpoint_url=f"{'https' if settings.minio_use_ssl else 'http'}://{settings.minio_endpoint}",
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def ensure_bucket(bucket_name: str) -> None:
    """Create bucket if it doesn't exist."""
    client = get_s3_client()
    try:
        client.head_bucket(Bucket=bucket_name)
        logger.debug("Bucket %s already exists", bucket_name)
    except ClientError as e:
        error_code = int(e.response["Error"]["Code"])
        if error_code == 404:
            client.create_bucket(Bucket=bucket_name)
            logger.info("Created bucket: %s", bucket_name)
        else:
            raise


def upload_file(
    file_obj: BinaryIO,
    bucket: str,
    key: str,
    content_type: str = "application/octet-stream",
    extra_args: Optional[dict] = None,
) -> str:
    """Upload a file-like object to S3/MinIO. Returns the S3 key."""
    client = get_s3_client()
    ensure_bucket(bucket)

    upload_args = {"ContentType": content_type}
    if extra_args:
        upload_args.update(extra_args)

    client.upload_fileobj(file_obj, bucket, key, ExtraArgs=upload_args)
    logger.info("Uploaded s3://%s/%s", bucket, key)
    return key


def upload_bytes(data: bytes, bucket: str, key: str, content_type: str = "image/jpeg") -> str:
    """Upload raw bytes to S3/MinIO."""
    import io
    return upload_file(io.BytesIO(data), bucket, key, content_type)


def delete_prefix(bucket: str, prefix: str) -> int:
    """Delete all objects under a prefix. Returns count deleted."""
    client = get_s3_client()
    paginator = client.get_paginator("list_objects_v2")
    deleted = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects = page.get("Contents", [])
        if not objects:
            continue
        delete_payload = {"Objects": [{"Key": obj["Key"]} for obj in objects]}
        client.delete_objects(Bucket=bucket, Delete=delete_payload)
        deleted += len(objects)
    logger.info("Deleted %d objects from s3://%s/%s", deleted, bucket, prefix)
    return deleted


def list_objects(bucket: str, prefix: str) -> list[dict]:
    """List objects under a prefix."""
    client = get_s3_client()
    paginator = client.get_paginator("list_objects_v2")
    result = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        result.extend(page.get("Contents", []))
    return result


def generate_presigned_url(bucket: str, key: str, expires_in: int = 3600) -> str:
    """Generate a pre-signed URL for downloading an object."""
    client = get_s3_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in,
    )
