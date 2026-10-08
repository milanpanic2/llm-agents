import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from functools import partial

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from minio import Minio
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from starlette.requests import Request

from src.agents import transcriptions_service
from src.clients.crw_client import CrwClient
from src.config.settings import settings
from src.config.telemetry import init_telemetry
from src.database.connection import init_db, sync_engine
from src.errors import AppError
from src.garage.garage_client import init_garage_client
from src.routes import router
from src.WFQEngine import wfq_queries
from src.WFQEngine.wfq_engine import WFQEngine

logger = logging.getLogger(__name__)


async def _cancel_workers(workers: list[asyncio.Task]):
    for worker in workers:
        worker.cancel()
    await asyncio.gather(*workers, return_exceptions=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_telemetry(app, sync_engine)

    logger.info("FastAPI started successfully. Starting init work...")
    init_db()

    async with AsyncExitStack() as stack:
        logger.info("Creating crawler client")
        crw_client = CrwClient()
        stack.push_async_callback(crw_client.aclose)
        app.state.crw_client = crw_client

        logger.info("Creating garage client")
        garage_client = init_garage_client()
        app.state.garage_client = garage_client

        logger.info("Connecting to OpenAIChatModel")
        openai_client = await stack.enter_async_context(
            AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
        )

        model = OpenAIChatModel(
            settings.llm_model,
            provider=OpenAIProvider(openai_client=openai_client),
        )
        app.state.model = model

        transcription_model = OpenAIChatModel(
            settings.llm_model,
            provider=OpenAIProvider(
                openai_client=openai_client.with_options(max_retries=1, timeout=120.0)),
            settings=OpenAIChatModelSettings(
                max_tokens=92, # TODO change
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}, # disable thinking of this model
            )
        )

        await init_wfq(garage_client, stack, transcription_model)

        # if not transcriptions_broker.is_worker_process:
        #     await transcriptions_broker.startup()
        #     stack.push_async_callback(transcriptions_broker.shutdown)

        logger.info("startup complete; serving requests")
        yield


async def init_wfq(garage_client: Minio, stack: AsyncExitStack[bool | None], transcription_model: OpenAIChatModel):
    # LICQ
    logger.info(f"Creating longest idle context engine, and starting {settings.max_licq_concurrent_transcriptions} "
                f"workers")
    licq_transcriptions = WFQEngine(
        transcriptions_service.TRANSCRIPTION_TASKS_TABLE_NAME,
        wfq_queries.WFQ_LONGEST_IDLE_CLAIM,
        handler_func=partial(transcriptions_service.transcription_worker_handler_func,
                             model=transcription_model, garage_client=garage_client),
        completion_func=partial(transcriptions_service.transcription_worker_completion_func,
                                garage_client=garage_client),
    )
    logger.info("ensuring task table exists")
    await licq_transcriptions.create_table()
    licq_workers = [
        asyncio.create_task(licq_transcriptions.worker(f"w{i}"))
        for i in range(settings.max_licq_concurrent_transcriptions)
    ]
    stack.push_async_callback(_cancel_workers, licq_workers)

    # FIFO
    logger.info(f"Creating first-in-first-out engine, and starting {settings.max_licq_concurrent_transcriptions} "
                f"workers")
    fifo_transcriptions = WFQEngine(
        transcriptions_service.TRANSCRIPTION_TASKS_TABLE_NAME,
        wfq_queries.WFQ_FIFO_CLAIM,
        handler_func=partial(transcriptions_service.transcription_worker_handler_func,
                             model=transcription_model, garage_client=garage_client),
        completion_func=partial(transcriptions_service.transcription_worker_completion_func,
                                garage_client=garage_client),
    )
    logger.info("ensuring task table exists")
    await fifo_transcriptions.create_table()
    fifo_workers = [
        asyncio.create_task(fifo_transcriptions.worker(f"w{i}"))
        for i in range(settings.max_fifo_concurrent_transcriptions)
    ]
    stack.push_async_callback(_cancel_workers, fifo_workers)


app = FastAPI(
    title="LLM Agents Service", description="LLM agents service for running llm taskiq with custom built tools", version="0.1.0", lifespan=lifespan
)


app.include_router(router)


@app.get("/api/v1/health", tags=["health"])
async def health():
    return {"status": "ok"}


@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError):
    logger.error("code=%s path=%s detail=%s", exc.code, request.url.path, exc.detail, exc_info=exc)
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.code, "detail": exc.detail},
    )
