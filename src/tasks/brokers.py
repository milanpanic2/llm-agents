from typing import Annotated

from glide import GlideClient
from minio import Minio
from taskiq import TaskiqDepends
from taskiq_redis import RedisStreamBroker, RedisAsyncResultBackend
import taskiq_fastapi
from fastapi import Request

from src.config import settings


transcriptions_broker = (
    RedisStreamBroker(
        url=settings.valkey_url,
        queue_name="tasks:transcriptions").with_result_backend(RedisAsyncResultBackend(redis_url=settings.valkey_url)))

taskiq_fastapi.init(transcriptions_broker, "src.app:app")


async def get_taskiq_valkey_client(request: Annotated[Request, TaskiqDepends()]) -> GlideClient:
    return request.app.state.valkey_client


async def get_taskiq_garage_client(request: Annotated[Request, TaskiqDepends()]) -> Minio:
    return request.app.state.garage_client


async def get_taskiq_llm_model(request: Annotated[Request, TaskiqDepends()]) -> Minio:
    return request.app.state.model
