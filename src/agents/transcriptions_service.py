import asyncio
import io
import logging
import shutil
import tempfile
import uuid
import zipfile
from collections.abc import AsyncIterator, Iterable, Iterator
from pathlib import Path

from fastapi import HTTPException, UploadFile
from minio import Minio, S3Error
from minio.deleteobjects import DeleteObject
from pydantic import BaseModel
from pydantic_ai import Agent, AgentRunError, BinaryContent
from pydantic_ai.models import Model
from sqlalchemy.ext.asyncio import AsyncSession
from urllib3 import BaseHTTPResponse

from src.config.settings import settings
from src.errors import AppError, BadRequestError
from src.garage.buckets import IMAGE_TRANSCRIPTIONS_BUCKET
from src.WFQEngine import wfq_utils
from src.WFQEngine.wfq_engine import WFQTaskData, WFQTaskStatus

logger = logging.getLogger(__name__)

TASK_TRANSCRIPTION = "taskiq:transcription"

TRANSCRIPTION_TASKS_TABLE_NAME = "transcription_tasks"

_ZIP_SUFIX = (".zip",)

_SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
_SUFFIX_TO_MEDIA = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}

_MIN_PART = 10 * 1024 * 1024

_CHUNK_SIZE = 2 * 1024 * 1024


class Transcription(BaseModel):
    filename: str
    text: str


class TranscriptionResult(BaseModel):
    context_id: str
    status: str
    total: int = 0
    done: int = 0
    failed: int = 0
    failure_reasons: list[str] = []
    download_link: str = ""


IMAGE_TEXT_TO_TEXT_AGENT_INSTRUCTIONS = (
    "You are an image-to-text transcription agent."
    "Given an image containing text (handwritten or printed), output exactly what is written as a string."
    "Output only the transcribed text, nothing else."
    "If the image contains no text, output an empty string."
)

# Timeout (15s) and retry budget live on this agent's model, built in app.py
# from a with_options() view of the shared client (see transcription_model).
transcriptions_agent = Agent(
    instructions=IMAGE_TEXT_TO_TEXT_AGENT_INSTRUCTIONS,
    output_type=str,
)

async def convert_to_text(model: Model, image_bytes: bytes, media_type: str) -> str:
    result = await transcriptions_agent.run(
        ["Transcribe the text in this image.",
         BinaryContent(data=image_bytes, media_type=media_type)],
        model=model,
    )

    return result.output


class TranscriptionTaskData(BaseModel):
    object_name: str
    media_type: str
    file_path: str
    original_file_name: str


async def queue_images_for_transcript(uploads: Iterable[UploadFile],
                                      session: AsyncSession,
                                      garage_client: Minio,
                                      user_id: str) -> str:
    """Upload file list to garage, and add transcription taskiq for workers with needed jsonb data for handler function"""
    context_id = str(uuid.uuid4())
    logger.info("queueing images for transcription context_id=%s user_id=%s", context_id, user_id)

    # media_type = _SUFFIX_TO_MEDIA[Path(processing_task.object_name).suffix.lower()] TODO, add validate
    # validate(uploads)

    wfq_tasks = []
    for index, upload in enumerate(uploads):
        filename = upload.filename or "unnamed"
        object_name = f"{index:04d}_{filename}"
        logger.debug("uploading context_id=%s object_name=%s", context_id, object_name)
        try:
            await asyncio.to_thread(garage_client.put_object, IMAGE_TRANSCRIPTIONS_BUCKET,
                                    f"{context_id}/{object_name}", upload.file,
                                    content_type="application/octet-stream",
                                    part_size=_MIN_PART, length=-1)
        except (OSError, ValueError, S3Error) as exc:
            logger.exception("upload failed context_id=%s object_name=%s; cleaning up", context_id, object_name)
            await clean_context_data(context_id, IMAGE_TRANSCRIPTIONS_BUCKET, garage_client)
            raise AppError("TS-03", f"Failed to upload file {object_name}") from exc
        suffix = Path(filename).suffix.lower()
        if suffix not in _SUFFIX_TO_MEDIA:
            logger.warning("unsupported file type context_id=%s filename=%s; cleaning up", context_id, filename)
            await clean_context_data(context_id, IMAGE_TRANSCRIPTIONS_BUCKET, garage_client)
            raise BadRequestError("TS-05", f"Unsupported file type: {filename}")
        media_type = _SUFFIX_TO_MEDIA[suffix]
        wfq_tasks.append(WFQTaskData(context_id=context_id,
                                     payload=TranscriptionTaskData(object_name=object_name, media_type=media_type,
                                                                   original_file_name=filename,
                                                                   file_path=f"{context_id}/{object_name}").model_dump()))

    await wfq_utils.add_tasks_bulk(session, TRANSCRIPTION_TASKS_TABLE_NAME, wfq_tasks)
    logger.info("queued context_id=%s tasks=%d", context_id, len(wfq_tasks))

    return context_id


async def queue_images_for_transcript_zip(zip_upload: UploadFile,
                                          session: AsyncSession,
                                          garage_client: Minio,
                                          user_id: str) -> str:
    """Upload file list to garage, and add transcription taskiq for workers with needed jsonb data for handler function"""
    logger.info("opening zip for transcription filename=%s user_id=%s", zip_upload.filename, user_id)
    zf = await asyncio.to_thread(zipfile.ZipFile, zip_upload.file)

    infos = zf.infolist()
    if len(infos) > 100:  # TODO - add max tokens spent for free users
        logger.warning("zip rejected: too many items filename=%s count=%d", zip_upload.filename, len(infos))
        zf.close()
        raise BadRequestError("TS-06", "Too many items in zip. 100 supported for free.")

    def zip_files_generator() -> Iterator[UploadFile]:
        try:
            for info in infos:
                if info.is_dir():
                    continue
                if Path(info.filename).suffix.lower() not in _SUPPORTED_IMAGE_SUFFIXES:
                    logger.debug("skipping non-image zip entry filename=%s", info.filename)
                    continue
                yield UploadFile(file=zf.open(info), size=info.file_size, filename=info.filename)
        finally:
            zf.close()

    context_id = await queue_images_for_transcript(zip_files_generator(), session, garage_client, user_id)

    return context_id


async def transcription_worker_handler_func(wfq_task_data: WFQTaskData,
                                            model: Model,
                                            garage_client: Minio):
    """Handler function called from workers dedicated to transcription taskiq.
    It gets an image object from garage base on given file_path, transcribes it with llm,
    and writes to :context_id/outputs/:object_name (0000_..., 0001_...)
    In the end it marks the task as 'done' or 'failed' depending on the worker outcome."""
    request_data: TranscriptionTaskData = TranscriptionTaskData.model_validate(wfq_task_data.payload)
    logger.info("transcribing context_id=%s object_name=%s", wfq_task_data.context_id, request_data.object_name)

    try:
        image_object: BaseHTTPResponse = await asyncio.to_thread(garage_client.get_object,
                                             IMAGE_TRANSCRIPTIONS_BUCKET,
                                             request_data.file_path)
    except S3Error as exc:
        raise AppError("TS-08", f"Failed to read image from s3. Reason: {exc}") from exc

    try:
        text = await convert_to_text(model, await asyncio.to_thread(image_object.read), request_data.media_type)
    except AgentRunError as err:
        raise AppError("TS-01", f"Agent failed to transcribe image to text. Reason: {err.message}") from err
    finally:
        image_object.close()
        image_object.release_conn()

    logger.debug("transcribed context_id=%s object_name=%s chars=%d",
                 wfq_task_data.context_id, request_data.object_name, len(text))

    full_text = f"=== {request_data.original_file_name} ===\n{text}\n\n"
    text_bytes = full_text.encode("utf-8")

    try:
        await asyncio.to_thread(garage_client.put_object, IMAGE_TRANSCRIPTIONS_BUCKET,
                                                         f"{wfq_task_data.context_id}/outputs/{request_data.object_name}.txt",
                                                         io.BytesIO(text_bytes),
                                                         length=len(text_bytes),
                                                         content_type="text/plain; charset=utf-8")
    except (ValueError, S3Error) as exc:
        raise AppError("TS-02", f"Failed to write transcribed text as s3 file. Reason: {exc}") from exc


async def transcription_worker_completion_func(wfq_task_data: WFQTaskData, garage_client: Minio):
    logger.info("context complete, merging transcripts context_id=%s", wfq_task_data.context_id)
    await asyncio.to_thread(_merge_and_upload_transcripts, wfq_task_data.context_id, garage_client)


def _merge_and_upload_transcripts(context_id: str, garage_client: Minio):
    workdir = tempfile.mkdtemp()
    file_path = f"{workdir}/output.txt"
    try:
        objects = garage_client.list_objects(IMAGE_TRANSCRIPTIONS_BUCKET, prefix=f"{context_id}/outputs/")
        with open(file_path, "ab") as f:
            for obj in objects:
                obj_to_write = garage_client.get_object(IMAGE_TRANSCRIPTIONS_BUCKET, obj.object_name)
                try:
                    f.write(obj_to_write.read())
                finally:
                    obj_to_write.close()
                    obj_to_write.release_conn()

        garage_client.fput_object(IMAGE_TRANSCRIPTIONS_BUCKET,
                                  f"{context_id}/output.txt",
                                  file_path,
                                  content_type="text/plain; charset=utf-8")
        logger.info("merged transcripts uploaded context_id=%s", context_id)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


async def clean_context_data(context_id: str, bucket: str, garage_client: Minio):
    logger.info("cleaning context data context_id=%s bucket=%s", context_id, bucket)
    objects = await asyncio.to_thread(lambda: list(garage_client.list_objects(bucket, prefix=context_id)))
    await asyncio.to_thread(
        garage_client.remove_objects,bucket, (DeleteObject(obj.object_name) for obj in objects))


async def get_results(context_id: str,
                      session: AsyncSession)-> TranscriptionResult:
    progress = await wfq_utils.get_progress(session, TRANSCRIPTION_TASKS_TABLE_NAME, context_id)
    if progress.total == 0:
        raise BadRequestError("TS-04", f"Data not found for context_id: {context_id}")

    if progress.done + progress.failed == progress.total:
        return TranscriptionResult(context_id=context_id,
                                   status=WFQTaskStatus.DONE,
                                   total=progress.total,
                                   done=progress.done,
                                   failed=progress.failed,
                                   download_link=f"{settings.public_base_url}/llm-agents/v1/transcription-agent/{context_id}/download")

    return TranscriptionResult(context_id=context_id,
                               status=WFQTaskStatus.RUNNING if
                                      progress.done > 0 or
                                      progress.failed > 0 else
                                      WFQTaskStatus.PENDING,
                               total=progress.total,
                               done=progress.done,
                               failed=progress.failed,
                               download_link="")


async def download_transcription_file(context_id: str, garage_client: Minio) -> AsyncIterator[bytes]:
    try:
        file_object: BaseHTTPResponse = await asyncio.to_thread(
            garage_client.get_object, IMAGE_TRANSCRIPTIONS_BUCKET, f"{context_id}/output.txt")
    except S3Error as exc:
        raise HTTPException(status_code=404, detail="Transcription not found") from exc

    try:
        while True:
            chunk = await asyncio.to_thread(file_object.read, 5 * 1024 * 1024)
            if not chunk:
                break
            yield chunk
    finally:
        await asyncio.to_thread(file_object.close)
        await asyncio.to_thread(file_object.release_conn)


async def save_file_to_disk(zip_upload: UploadFile, work_dir: str) -> str:
    dest = Path(work_dir) / (zip_upload.filename or "upload.zip")
    await zip_upload.seek(0)

    with open(dest, "wb") as out:
        while chunk := await zip_upload.read(_CHUNK_SIZE):
            out.write(chunk)

    return str(dest)
