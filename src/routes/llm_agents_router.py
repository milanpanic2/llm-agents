import logging
from typing import Annotated

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi import UploadFile as UF
from glide import GlideClient
from minio import Minio
from pydantic import WithJsonSchema
from starlette.responses import StreamingResponse

from src.agents import transcriptions_service
from src.agents.transcriptions_service import TranscriptionResult
from src.config.settings import settings
from src.database.connection import get_db
from src.errors import BadRequestError

router = APIRouter(prefix="/llm-agents/v1", tags=["llm-agents"])


logger = logging.getLogger(__name__)


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


def get_garage_client(request: Request) -> Minio:
    return request.app.state.garage_client


@router.post("/transcription-agent/transcribe", response_model={"context_id": str})
async def image_text_to_text(
    uploads: list[UploadFile],
    psql_connection = Depends(get_db),
    garage_client = Depends(get_garage_client),
    user_id: str = Depends(get_user_id)) -> str:
    if len(uploads) > 5:
        raise BadRequestError("LAR-01",
                              "Max 5 transcriptions. Use zip file endpoint for more support")
    return await transcriptions_service.queue_images_for_transcript(uploads, psql_connection, garage_client,
                                                                    user_id)

@router.post("/transcription-agent/transcribe/zip", response_model={"context_id": str})
async def image_text_to_text_zip(
    uploads: UploadFile,
    psql_connection = Depends(get_db),
    garage_client = Depends(get_garage_client),
    user_id: str = Depends(get_user_id)) -> str:
    return await transcriptions_service.queue_images_for_transcript_zip(uploads, psql_connection, garage_client,
                                                                        user_id)


@router.get("/transcription-agent/{context_id}", response_model = list[str])
async def get_results(context_id: str,
                      session = Depends(get_db)) -> TranscriptionResult:
    return await transcriptions_service.get_results(context_id, session)


@router.get("/transcription-agent/{context_id}/download", response_model = StreamingResponse)
async def download_transcription_file(context_id: str,
                                      garage_client = Depends(get_garage_client)) -> StreamingResponse:
    return StreamingResponse(
        transcriptions_service.download_transcription_file(context_id, garage_client),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{context_id}.txt"'})


# @router.post("/image-text-to-text/plain", response_class=PlainTextResponse)
# async def image_text_to_text_plain(
#     files: list[UploadFile] = File(...),
#     model: OpenAIChatModel = Depends(get_llm_model),
# ) -> str:
#
#     results = await text_image_to_image_services.transcribe_uploads(files, model)
#
#     if not results:
#         raise HTTPException(status_code=400, detail="No .jpg images found in upload")
#
#     return "\n\n".join(f"=== {r.filename} ===\n{r.text}" for r in results)
