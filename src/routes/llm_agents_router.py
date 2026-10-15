import asyncio
import logging

import jwt
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from minio import Minio
from starlette.responses import StreamingResponse

from src.agents import transcriptions_service
from src.agents.transcriptions_service import TranscriptionResult
from src.config.settings import settings
from src.database.connection import get_db


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


def get_garage_client(request: Request) -> Minio:
    return request.app.state.garage_client


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
                      session = Depends(get_db),
                      user_id = Depends(get_user_id)) -> TranscriptionResult:
    return await transcriptions_service.get_results(context_id, session, user_id)


@router.get("/transcription-agent/{context_id}/download")
async def download_transcription_file(context_id: str,
                                      garage_client = Depends(get_garage_client),
                                      session = Depends(get_db),
                                      user_id = Depends(get_user_id)) -> StreamingResponse:
    file_download = await transcriptions_service.download_transcription_file(session, context_id, garage_client, user_id)
    async def stream_chunks():
        try:
            while True:
                chunk = await asyncio.to_thread(file_download.read, 5 * 1024 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            await asyncio.to_thread(file_download.close)
            await asyncio.to_thread(file_download.release_conn)


    return StreamingResponse(
        stream_chunks(),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{context_id}.txt"'})


@router.patch("/transcription-agent/{context_id}/retry_failed")
async def retry_failed(context_id: str,
                       session = Depends(get_db),
                       user_id = Depends(get_user_id)):
    await transcriptions_service.retry_failed_transcriptions(session, context_id, user_id)
    return None
