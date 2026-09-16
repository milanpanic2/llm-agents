import json

from minio import Minio
from minio.datatypes import Bucket

from src.config import settings
from src.garage.buckets import IMAGE_TRANSCRIPTIONS_BUCKET, EXAMPLE_PUBLIC_BUCKET

_PUBLIC_READ_BUCKET_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"AWS": "*"},
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::{_EXAMPLE_PUBLIC_BUCKET}/*",
        },
    ],
}


def init_garage_client() -> Minio:
    garage_client = Minio(endpoint=settings.garage_endpoint,
                          access_key=settings.garage_access_key,
                          secret_key=settings.garage_secret_key,
                          region="garage")
    init_buckets(garage_client)

    return garage_client


def init_buckets(garage_client: Minio):
    # bucket_names = {bucket.name for bucket in garage_client.list_buckets()}

    if garage_client.bucket_exists(IMAGE_TRANSCRIPTIONS_BUCKET):
        garage_client.make_bucket(IMAGE_TRANSCRIPTIONS_BUCKET) # bucket

    if garage_client.bucket_exists(EXAMPLE_PUBLIC_BUCKET):
        garage_client.make_bucket(EXAMPLE_PUBLIC_BUCKET)
        garage_client.set_bucket_policy(EXAMPLE_PUBLIC_BUCKET, json.dumps(_PUBLIC_READ_BUCKET_POLICY)) # public bucket

