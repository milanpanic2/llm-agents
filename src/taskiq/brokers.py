from typing import Annotated

import taskiq_fastapi
from fastapi import Request
from glide import GlideClient
from minio import Minio
from pydantic_ai.models import Model
from taskiq_redis import RedisAsyncResultBackend, RedisStreamBroker

from src.config import settings
from taskiq import TaskiqDepends

transcriptions_broker = (
    RedisStreamBroker(
        url=settings.valkey_url,
        queue_name="taskiq:transcriptions").with_result_backend(RedisAsyncResultBackend(redis_url=settings.valkey_url)))

taskiq_fastapi.init(transcriptions_broker, "src.app:app")


async def get_taskiq_valkey_client(request: Annotated[Request, TaskiqDepends()]) -> GlideClient:
    return request.app.state.valkey_client


async def get_taskiq_garage_client(request: Annotated[Request, TaskiqDepends()]) -> Minio:
    return request.app.state.garage_client


async def get_taskiq_llm_model(request: Annotated[Request, TaskiqDepends()]) -> Model:
    return request.app.state.model
