import json

import urllib3
from minio import Minio

from src.config.settings import settings
from src.garage.buckets import EXAMPLE_PUBLIC_BUCKET, IMAGE_TRANSCRIPTIONS_BUCKET

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
    # One shared client serves every transcription worker (LICQ + FIFO engines,
    # so 2 x max_concurrent_transcriptions) plus concurrent HTTP uploads -- all
    # hitting the same garage host. urllib3 defaults to maxsize=1, so every call
    # past the first opens a fresh connection and discards it ("connection pool
    # is full"), churning TCP for nothing. Size the pool to the concurrency.
    pool_maxsize = settings.max_concurrent_transcriptions * 2 + 8
    # Fail fast instead of hanging forever if the endpoint is unreachable.
    http_client = urllib3.PoolManager(
        maxsize=pool_maxsize,
        block=True,  # wait for a free connection instead of opening+discarding one
        timeout=urllib3.Timeout(connect=5.0, read=10.0),
        retries=urllib3.Retry(total=2, backoff_factor=0.2),
    )
    garage_client = Minio(endpoint=settings.garage_endpoint,
                          access_key=settings.garage_access_key,
                          secret_key=settings.garage_secret_key,
                          region="garage",
                          secure=False,  # garage S3 API is plain HTTP inside the cluster
                          http_client=http_client)
    init_buckets(garage_client)

    return garage_client


def init_buckets(garage_client: Minio):
    if not garage_client.bucket_exists(IMAGE_TRANSCRIPTIONS_BUCKET):
        garage_client.make_bucket(IMAGE_TRANSCRIPTIONS_BUCKET) # bucket

    if not garage_client.bucket_exists(EXAMPLE_PUBLIC_BUCKET):
        garage_client.make_bucket(EXAMPLE_PUBLIC_BUCKET)
        garage_client.set_bucket_policy(EXAMPLE_PUBLIC_BUCKET, json.dumps(_PUBLIC_READ_BUCKET_POLICY)) # public bucket

