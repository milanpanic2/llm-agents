import logging
from contextlib import asynccontextmanager, AsyncExitStack

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.requests import Request
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from src.clients.crw_client import CrwClient
from src.config import settings
from src.config.telemetry import init_telemetry
from src.database.connection import init_db, sync_engine
from src.errors import AppError
from src.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    init_telemetry(app, sync_engine)

    async with AsyncExitStack() as stack:
        model = OpenAIChatModel(
            settings.llm_model,
            provider=OpenAIProvider(base_url=settings.llm_base_url, api_key=settings.llm_api_key),
        )
        app.state.model = model

        crw_client = CrwClient()
        stack.push_async_callback(crw_client.aclose)
        app.state.crw_client = crw_client

        # web_crawler_agent = WebCrawlerAgent()
        # stack.push_async_callback(web_crawler_agent.aclose)
        # app.state.web_crawler_agent = web_crawler_agent
        yield


app = FastAPI(
    title="LLM Agents Service", description="LLM agents service for running llm tasks with custom built tools", version="0.1.0", lifespan=lifespan
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
