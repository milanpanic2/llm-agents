import asyncio
import io
import logging
import shutil
import tempfile
import uuid
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from fastapi import UploadFile
from minio import Minio, S3Error
from minio.deleteobjects import DeleteObject
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent, AgentRunResult
from pydantic_ai.models import Model
from sqlalchemy.ext.asyncio import AsyncSession
from urllib3 import BaseHTTPResponse
from PIL import Image

from src.config import metrics
from src.config.settings import settings
from src.errors import AppError, BadRequestError, NotFoundError
from src.garage.buckets import IMAGE_TRANSCRIPTIONS_BUCKET
from src.WFQEngine import wfq_utils
from src.WFQEngine.wfq_engine import FailureReasons, TaskContextProgress
from src.WFQEngine.wfq_models import WFQTask, WFQTaskStatus

logger = logging.getLogger(__name__)

TRANSCRIPTION_TASKS_TABLE_NAME = "transcription_tasks"

_ZIP_SUFIX = (".zip",)

_SUPPORTED_IMAGE_FORMATS = ["JPEG", "PNG"]
_FORMAT_TO_MEDIA_TYPE = {"JPEG": "image/jpeg", "PNG": "image/png"}

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
    failure_reasons: list[FailureReasons] = []
    download_link: str = ""


IMAGE_TEXT_TO_TEXT_AGENT_INSTRUCTIONS = (
    "You are an image-to-text transcription agent."
    "Given an image containing text (handwritten or printed), output exactly what is written as a string."
    "Output only the transcribed text, nothing else."
    "If the image contains no text, output an empty string."
)


transcriptions_agent = Agent(
    instructions=IMAGE_TEXT_TO_TEXT_AGENT_INSTRUCTIONS,
    output_type=str,
)


async def convert_to_text(model: Model, image_bytes: bytes) -> str:
    with Image.open(io.BytesIO(image_bytes)) as i:
        media_type = _FORMAT_TO_MEDIA_TYPE[i.format]
    model_name = getattr(model, "model_name", settings.llm_model)
    with metrics.record_duration(metrics.LLM_REQUEST_DURATION, {"model": model_name}):
        result = await transcriptions_agent.run(
            ["Transcribe the text in this image.",
             BinaryContent(data=image_bytes, media_type=media_type)],
            model=model,
        )

    usage = result.usage
    metrics.LLM_TOKENS_TOTAL.add(usage.input_tokens or 0, {"model": model_name, "direction": "input"})
    metrics.LLM_TOKENS_TOTAL.add(usage.output_tokens or 0, {"model": model_name, "direction": "output"})

    await check_model_response_reason(result)

    return result.output


async def check_model_response_reason(result: AgentRunResult[Any]):
    finish_reason = result.response.finish_reason
    if finish_reason == "length":
        raise AppError("TS-03",
                       f"Transcription truncated (finish_reason=length, output={result.usage.output_tokens})")


class TranscriptionTaskData(BaseModel):
    object_name: str
    file_path: str
    original_file_name: str


async def queue_images_for_transcript_zip(zip_upload: UploadFile,
                                          session: AsyncSession,
                                          garage_client: Minio,
                                          user_id: str) -> str:
    """Upload file list to garage, and add transcription taskiq for workers with needed jsonb data for handler function"""
    logger.info("opening zip for transcription filename=%s user_id=%s", zip_upload.filename, user_id)
    try:
        zf = await asyncio.to_thread(zipfile.ZipFile, zip_upload.file)
    except (zipfile.BadZipFile, OSError) as exc:
        logger.warning("zip rejected: not a valid zip filename=%s", zip_upload.filename)
        raise BadRequestError("TS-13", "Uploaded file is not a valid zip archive") from exc

    images = [image for image in zf.infolist() if not _is_zipping_artifact(image.filename)]

    if len(images) == 0:
        logger.warning("zip rejected: no valid files found in zip for user_id=%s upload", user_id)
        zf.close()
        raise BadRequestError("TS-18", "No valid files found, check the submitted zip.")

    if len(images) > 100:  # TODO - add max tokens per user to 15 a day. after stripe integration add 2eur = 100 images
        logger.warning("zip rejected: too many items filename=%s count=%d", zip_upload.filename, len(images))
        zf.close()
        raise BadRequestError("TS-06", "Too many items in zip. 100 supported.")

    try:
        await asyncio.to_thread(_validate_images_in_zip_and_get_content_types, zf, images)
    except BadRequestError:
        zf.close()
        raise

    def generate_upload_files_from_zip() -> Iterator[UploadFile]:
        try:
            for image in images:
                yield UploadFile(file=zf.open(image), size=image.file_size, filename=image.filename)
        finally:
            zf.close()

    context_id = await queue_images_for_transcript(generate_upload_files_from_zip(), session, garage_client, user_id)
    return context_id


def _validate_images_in_zip_and_get_content_types(zf: ZipFile, images: list[zipfile.ZipInfo]):
    for image in images:
        if image.is_dir():
            raise BadRequestError("TS-14", "Zip invalid. Contains directory")

        if image.file_size > 40 * 1024 * 1024:
            raise BadRequestError("TS-15",  "Zip invalid. Contains image > 40MiB")

        with zf.open(image) as i:
            try:
                fmt = Image.open(i).format
            except Exception as e:
                raise BadRequestError("TS-16", "Zip invalid. Contains fake image files") from e
        if fmt not in _SUPPORTED_IMAGE_FORMATS:
            raise BadRequestError("TS-17", f"Zip invalid. Contains unsupported format: {fmt}")


async def queue_images_for_transcript(uploads: Iterable[UploadFile],
                                      session: AsyncSession,
                                      garage_client: Minio,
                                      user_id: str) -> str:
    """Upload file list to garage, and add transcription taskiq for workers with needed jsonb data for handler function"""
    context_id = str(uuid.uuid4())
    logger.info("queueing images for transcription context_id=%s user_id=%s", context_id, user_id)

    wfq_tasks = []
    for index, upload in enumerate(uploads):
        filename = Path(upload.filename).name or "unnamed"
        object_name = f"{index:04d}_{filename}"

        await _upload_image_to_object_storage(context_id, garage_client, object_name, upload)

        wfq_tasks.append(WFQTask(context_id=context_id,
                                 created_by=user_id,
                                 payload=TranscriptionTaskData(object_name=object_name,
                                                               original_file_name=filename,
                                                               file_path=f"{context_id}/{object_name}").model_dump()))

    await wfq_utils.add_tasks_bulk(session, TRANSCRIPTION_TASKS_TABLE_NAME, wfq_tasks, user_id)
    logger.info("queued context_id=%s tasks=%d", context_id, len(wfq_tasks))

    return context_id


async def _upload_image_to_object_storage(context_id: str, garage_client: Minio, object_name: str, upload: UploadFile):
    logger.debug("uploading context_id=%s object_name=%s", context_id, object_name)
    try:
        await asyncio.to_thread(garage_client.put_object, IMAGE_TRANSCRIPTIONS_BUCKET,
                                f"{context_id}/{object_name}", upload.file,
                                content_type="application/octet-stream",
                                length=upload.size)
    except (OSError, ValueError, S3Error) as exc:
        logger.exception("upload failed context_id=%s object_name=%s; cleaning up", context_id, object_name)
        await clean_context_data(context_id, IMAGE_TRANSCRIPTIONS_BUCKET, garage_client)
        raise AppError("TS-01", f"Failed to upload file {object_name}") from exc


async def transcription_worker_handler_func(wfq_task_data: WFQTask,
                                            model: Model,
                                            garage_client: Minio):
    """Handler function called from workers dedicated to transcription taskiq.
    It gets an image object from garage base on given file_path, transcribes it with llm,
    and writes to :context_id/outputs/:object_name (0000_..., 0001_...)
    In the end it marks the task as 'done' or 'failed' depending on the worker outcome."""
    request_data = TranscriptionTaskData.model_validate(wfq_task_data.payload)
    logger.info("transcribing context_id=%s object_name=%s", wfq_task_data.context_id, request_data.object_name)

    image_object: BaseHTTPResponse = await asyncio.to_thread(garage_client.get_object,
                                             IMAGE_TRANSCRIPTIONS_BUCKET,
                                             request_data.file_path)

    try:
        text = await convert_to_text(model, await asyncio.to_thread(image_object.read))
    finally:
        image_object.close()
        image_object.release_conn()

    logger.debug("transcribed context_id=%s object_name=%s chars=%d",
                 wfq_task_data.context_id, request_data.object_name, len(text))

    full_text = f"=== {request_data.original_file_name} ===\n{text}\n\n"
    text_bytes = full_text.encode("utf-8")

    await asyncio.to_thread(garage_client.put_object, IMAGE_TRANSCRIPTIONS_BUCKET,
                                                         f"{wfq_task_data.context_id}/outputs/{request_data.object_name}.txt",
                                                         io.BytesIO(text_bytes),
                                                         length=len(text_bytes),
                                                         content_type="text/plain; charset=utf-8")


async def transcription_worker_completion_func(wfq_task_data: WFQTask, garage_client: Minio):
    logger.info("context complete, merging transcripts context_id=%s", wfq_task_data.context_id)
    await asyncio.to_thread(_merge_and_upload_transcripts, wfq_task_data.context_id, garage_client)


def _merge_and_upload_transcripts(context_id: str, garage_client: Minio):
    workdir = tempfile.mkdtemp()
    file_path = f"{workdir}/output.txt"
    objects = garage_client.list_objects(IMAGE_TRANSCRIPTIONS_BUCKET, prefix=f"{context_id}/outputs/")
    try:
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
    objects = await asyncio.to_thread(
        lambda: list(garage_client.list_objects(bucket, prefix=context_id, recursive=True))
    )
    await asyncio.to_thread(
        lambda: list(garage_client.remove_objects(
            bucket, (DeleteObject(obj.object_name) for obj in objects)
        ))
    )


async def get_results(context_id: str,
                      session: AsyncSession,
                      user_id: str)-> TranscriptionResult:
    await validate_context_creator(session, context_id, user_id)

    progress = await get_progress_for_context(context_id, session)

    failure_reasons = []
    if progress.failed > 0:
        failure_reasons = await wfq_utils.get_failure_reasons(session, TRANSCRIPTION_TASKS_TABLE_NAME, context_id)

    finished = await wfq_utils.check_if_context_finished(progress)
    if finished:
        return TranscriptionResult(context_id=context_id,
                                   status=WFQTaskStatus.DONE,
                                   total=progress.total,
                                   done=progress.done,
                                   failed=progress.failed,
                                   download_link=f"{settings.public_base_url}/llm-agents/v1/transcription-agent/{context_id}/download",
                                   failure_reasons=failure_reasons)

    return TranscriptionResult(context_id=context_id,
                               status=WFQTaskStatus.RUNNING if
                                      progress.done > 0 or
                                      progress.failed > 0 else
                                      WFQTaskStatus.PENDING,
                               total=progress.total,
                               done=progress.done,
                               failed=progress.failed,
                               download_link="",
                               failure_reasons=failure_reasons)


async def download_transcription_file(session: AsyncSession,
                                      context_id: str,
                                      garage_client: Minio,
                                      user_id: str) -> BaseHTTPResponse:
    await validate_context_creator(session, context_id, user_id)
    try:
        file_object: BaseHTTPResponse = await asyncio.to_thread(
            garage_client.get_object, IMAGE_TRANSCRIPTIONS_BUCKET, f"{context_id}/output.txt")
    except S3Error as exc:
        raise NotFoundError("TS-08", "Transcription not found") from exc

    return file_object


async def retry_failed_transcriptions(session: AsyncSession, context_id: str, user_id: str):
    await validate_context_creator(session, context_id, user_id)
    progress = await wfq_utils.get_progress(session, TRANSCRIPTION_TASKS_TABLE_NAME, context_id)
    finished = await wfq_utils.check_if_context_finished(progress)
    if finished and progress.failed > 0:
        await wfq_utils.retry_all_failed_tasks(session, TRANSCRIPTION_TASKS_TABLE_NAME, context_id)
    else:
        raise BadRequestError("TS-05", f"Context is not in a retriable state. Progress: {progress}")


async def validate_context_creator(session: AsyncSession, context_id: str, user_id: str):
    context_created_by = await wfq_utils.get_context_created_by(session, TRANSCRIPTION_TASKS_TABLE_NAME, context_id)
    if context_created_by is None or user_id != context_created_by:
        raise BadRequestError("TS-07", "Cannot access context if you are not the creator")


async def get_progress_for_context(context_id: str, session: AsyncSession) -> TaskContextProgress:
    progress = await wfq_utils.get_progress(session, TRANSCRIPTION_TASKS_TABLE_NAME, context_id)
    if progress.total == 0:
        raise BadRequestError("TS-04", f"Data not found for context_id: {context_id}")
    return progress


def _is_zipping_artifact(info: zipfile.ZipInfo) -> bool:
    name = Path(info.filename).name
    return (info.is_dir()
            or info.filename.startswith("__MACOSX/")
            or name == ".DS_Store"
            or name.startswith("._"))
