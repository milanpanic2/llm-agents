import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from pydantic import BaseModel
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIChatModel
from typing import Annotated

from src.agents import web_search_services
from src.clients.crw_client import CrwClient
from src.agents.web_crawler_agent import WebCrawlerAgent
from src.agents import text_image_to_image_services
from src.agents.text_image_to_image_services import Transcription

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


@router.post("/image-text-to-text", response_model=list[Transcription])
async def image_text_to_text(
    files: Annotated[list[UploadFile], File(description="JPEG images to transcribe")],
    model: OpenAIChatModel = Depends(get_llm_model),
) -> list[Transcription]:

    results = await text_image_to_image_services.transcribe_uploads(files, model)

    if not results:
        raise HTTPException(status_code=400, detail="No .jpg images found in upload")
    return results
