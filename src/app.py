import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from starlette.requests import Request

from src.clients.crw_client import CrwClient
from src.config.settings import settings
from src.config.telemetry import init_telemetry
from src.database.connection import init_db, sync_engine
from src.errors import AppError
from src.garage.garage_client import init_garage_client
from src.routes import router
from src.services import transcriptions_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI started successfully. Starting init work...")
    init_db()

    async with AsyncExitStack() as stack:
        loop = asyncio.get_running_loop()
        max_blocking_io = (
                settings.max_licq_concurrent_transcriptions
                + settings.max_fifo_concurrent_transcriptions
                + 8
        )
        executor = ThreadPoolExecutor(max_workers=max_blocking_io, thread_name_prefix="to_thread")
        loop.set_default_executor(executor)
        stack.callback(executor.shutdown, wait=True, cancel_futures=True)

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
                max_tokens=600,
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        )

        logger.info("Creating transcriptions engines")
        engines = await transcriptions_service.init_transcription_engines(garage_client, transcription_model)
        for engine, worker_count in engines:
            await stack.enter_async_context(engine.run_workers(worker_count))

        logger.info("startup complete; serving requests")
        yield


app = FastAPI(
    title="LLM Agents Service", description="LLM agents service for running llm taskiq with custom built tools",
    version="0.1.0", lifespan=lifespan
)


init_telemetry(app, sync_engine)


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
