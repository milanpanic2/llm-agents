import logging

import httpx
import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile as UF, File
from fastapi.responses import PlainTextResponse
from glide import GlideClient
from pydantic import BaseModel, WithJsonSchema
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIChatModel
from typing import Annotated

from starlette.responses import StreamingResponse

from src.agents import web_search_services
from src.clients.crw_client import CrwClient
from src.agents.web_crawler_agent import WebCrawlerAgent
from src.agents import text_image_to_image_services
from src.agents.text_image_to_image_services import Transcription
from src.config import settings

router = APIRouter(prefix="/llm-agents/v1", tags=["llm-agents"])

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    prompt: str

class CrawlResponse(BaseModel):
    result: str


def get_web_crawler(request: Request) -> WebCrawlerAgent:
    return request.app.state.agents.web_crawler

def get_crw_client(request: Request) -> CrwClient:
    return request.app.state.crw_client

@router.post("/crawl-web", response_model=CrawlResponse)
async def crawl_web(
    body: QueryRequest,
    web_crawler_agent: WebCrawlerAgent = Depends(get_web_crawler),
) -> CrawlResponse:
    try:
        answer = await web_crawler_agent.run(body.prompt)
    except (httpx.ConnectError, httpx.ReadTimeout, ModelHTTPError) as exc:
        # llama-server unreachable / slow / erroring — transient, retry later
        logger.warning("model backend unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Model backend unavailable") from exc
    except UnexpectedModelBehavior as exc:
        # model produced something unusable (e.g. exceeded retries)
        logger.error("model failed to complete: %s", exc)
        raise HTTPException(status_code=502, detail="Model failed to complete the request") from exc

    return CrawlResponse(result=answer)


def get_llm_model(request: Request) -> OpenAIChatModel:
    return request.app.state.model

@router.post("/open-webui/enhance-with-search")
async def enhance_prompt_with_search(
        body: QueryRequest,
        model: OpenAIChatModel = Depends(get_llm_model),
        crw_client: CrwClient = Depends(get_crw_client)) -> CrawlResponse:
    result = await web_search_services.enhance_prompt_with_search(body.prompt, model, crw_client)

    return CrawlResponse(result=result)


UploadFile = Annotated[UF, WithJsonSchema({"type": "string", "format": "binary"})] #todo, remove when swagger fixes files array

async def get_user_id(request: Request) -> str:
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing token")
    try:
        payload = jwt.decode(auth[7:], settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired") from None
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token") from None
    return payload["sub"]


def get_valkey_client(request: Request) -> GlideClient:
    return request.app.state.valkey_client


@router.post("/image-text-to-text/zip", response_model=list[Transcription])
async def image_text_to_text(
    zipfile: UploadFile,
    model: OpenAIChatModel = Depends(get_llm_model),
    valkey_client = Depends(get_valkey_client),
    user_id: str = Depends(get_user_id)
) -> StreamingResponse:


    async def stream_results():
        async for zf in await text_image_to_image_services.transcribe_uploads_from_zip(zipfile, model, valkey_client, user_id):
            yield zf

    return StreamingResponse(stream_results(), media_type="application/x-ndjson")


@router.post("/image-text-to-text/plain", response_class=PlainTextResponse)
async def image_text_to_text_plain(
    files: list[UploadFile] = File(...),
    model: OpenAIChatModel = Depends(get_llm_model),
) -> str:

    results = await text_image_to_image_services.transcribe_uploads(files, model)

    if not results:
        raise HTTPException(status_code=400, detail="No .jpg images found in upload")

    return "\n\n".join(f"=== {r.filename} ===\n{r.text}" for r in results)
