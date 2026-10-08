import logging

import jwt
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from glide import GlideClient
from minio import Minio
from starlette.responses import StreamingResponse

from src.agents import transcriptions_service
from src.agents.transcriptions_service import TranscriptionResult
from src.config.settings import settings
from src.database.connection import get_db
from src.errors import BadRequestError

router = APIRouter(prefix="/llm-agents/v1", tags=["llm-agents"])


logger = logging.getLogger(__name__)


bearer_scheme = HTTPBearer()


async def get_user_id(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> str:
    try:
        payload = jwt.decode(credentials.credentials, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired") from None
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token") from None
    return payload["sub"]


def get_valkey_client(request: Request) -> GlideClient:
    return request.app.state.valkey_client


def get_garage_client(request: Request) -> Minio:
    return request.app.state.garage_client


@router.post("/transcription-agent/transcribe")
async def image_text_to_text(
    uploads: list[UploadFile] = File(...),
    psql_connection = Depends(get_db),
    garage_client = Depends(get_garage_client),
    user_id: str = Depends(get_user_id)) -> dict[str, str]:
    if len(uploads) > 5:
        raise BadRequestError("LAR-01",
                              "Max 5 transcriptions. Use zip file endpoint for more support")
    context_id = await transcriptions_service.queue_images_for_transcript(uploads, psql_connection, garage_client,
                                                                          user_id)
    return {"context_id": context_id}

@router.post("/transcription-agent/transcribe/zip")
async def image_text_to_text_zip(
    upload: UploadFile = File(...),
    psql_connection = Depends(get_db),
    garage_client = Depends(get_garage_client),
    user_id = Depends(get_user_id)) -> dict[str, str]:
    context_id = await transcriptions_service.queue_images_for_transcript_zip(upload, psql_connection, garage_client,
                                                                              user_id)
    return {"context_id": context_id}


@router.get("/transcription-agent/{context_id}")
async def get_results(context_id: str,
                      session = Depends(get_db)) -> TranscriptionResult:
    return await transcriptions_service.get_results(context_id, session)


# TODO: add retry failed


@router.get("/transcription-agent/{context_id}/download")
async def download_transcription_file(context_id: str,
                                      garage_client = Depends(get_garage_client)) -> StreamingResponse:
    file_download = transcriptions_service.download_transcription_file(context_id, garage_client)
    return StreamingResponse(
        file_download,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{context_id}.txt"'})


@router.post("/transcription-agent/{context_id}/retry_failed")
async def retry_failed(context_id: str,
                       session = Depends(get_db)):
    return None


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
