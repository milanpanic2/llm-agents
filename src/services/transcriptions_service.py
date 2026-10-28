import asyncio
import io
import logging
import shutil
import tempfile
import uuid
from collections.abc import Iterable
from functools import partial
from pathlib import Path

from fastapi import UploadFile
from minio import Minio, S3Error
from minio.deleteobjects import DeleteObject
from pydantic import BaseModel
from pydantic_ai.models import Model
from sqlalchemy.ext.asyncio import AsyncSession
from urllib3 import BaseHTTPResponse

from src.agents.transcriptions_agents import convert_to_text
from src.config.settings import settings
from src.errors import AppError, BadRequestError, NotFoundError
from src.garage.buckets import IMAGE_TRANSCRIPTIONS_BUCKET
from src.services.zip_image_ingestion import images_from_zip
from src.WFQEngine import wfq_queries
from src.WFQEngine.wfq_engine import FailureReasons, TaskContextProgress, WFQEngine
from src.WFQEngine.wfq_models import WFQTask, WFQTaskStatus
from src.WFQEngine.wfq_queue_facade import WFQTaskQueue

_WFQ_TRANSCRIPTION_TASKS_TABLE_NAME = "transcription_tasks"

_CHUNK_SIZE = 2 * 1024 * 1024

transcription_queue = WFQTaskQueue(_WFQ_TRANSCRIPTION_TASKS_TABLE_NAME)

logger = logging.getLogger(__name__)


class TranscriptionResult(BaseModel):
    context_id: str
    status: str
    total: int = 0
    done: int = 0
    failed: int = 0
    failure_reasons: list[FailureReasons] = []
    download_link: str = ""


class TranscriptionTaskData(BaseModel):
    object_name: str
    file_path: str
    original_file_name: str


async def queue_images_for_transcript_zip(zip_upload: UploadFile,
                                          session: AsyncSession,
                                          garage_client: Minio,
                                          user_id: str) -> str:
    """Upload file list to garage, and add transcription taskiq for workers with needed jsonb data for handler
    function"""
    logger.info("opening zip for transcription filename=%s user_id=%s", zip_upload.filename, user_id)
    async with images_from_zip(zip_upload) as uploads:
        return await queue_images_for_transcript(uploads, session, garage_client, user_id)


async def queue_images_for_transcript(uploads: Iterable[UploadFile],
                                      session: AsyncSession,
                                      garage_client: Minio,
                                      user_id: str) -> str:
    """Upload file list to garage, and add transcription taskiq for workers with needed jsonb data for handler
    function"""
    context_id = str(uuid.uuid4())
    logger.info("queueing images for transcription context_id=%s user_id=%s", context_id, user_id)

    wfq_tasks = []
    for index, upload in enumerate(uploads):
        if upload.filename is None:
            raise AppError("TS-06", "Failed to get constructed image filename")
        filename = Path(upload.filename).name or "unnamed"
        object_name = f"{index:04d}_{filename}"

        await _upload_image_to_object_storage(context_id, garage_client, object_name, upload)

        wfq_tasks.append(WFQTask(context_id=context_id,
                                 created_by=user_id,
                                 payload=TranscriptionTaskData(object_name=object_name,
                                                               original_file_name=filename,
                                                               file_path=f"{context_id}/{object_name}").model_dump()))

    await transcription_queue.add_tasks_bulk(session, wfq_tasks, user_id)
    logger.info("queued context_id=%s tasks=%d", context_id, len(wfq_tasks))

    return context_id


async def _upload_image_to_object_storage(context_id: str, garage_client: Minio, object_name: str, upload: UploadFile):
    logger.debug("uploading context_id=%s object_name=%s", context_id, object_name)
    try:
        if upload.size is None:
            raise AppError("TS-09", "Failed to get constructed image file size")
        await asyncio.to_thread(garage_client.put_object, IMAGE_TRANSCRIPTIONS_BUCKET,
                                f"{context_id}/{object_name}", upload.file,
                                content_type="application/octet-stream",
                                length=upload.size)
    except (OSError, ValueError, S3Error) as exc:
        logger.exception("upload failed context_id=%s object_name=%s; cleaning up", context_id, object_name)
        await _clean_context_data(context_id, IMAGE_TRANSCRIPTIONS_BUCKET, garage_client)
        raise AppError("TS-01", f"Failed to upload file {object_name}") from exc


async def transcription_worker_handler_func(wfq_task_data: WFQTask,
                                            model: Model,
                                            garage_client: Minio):
    """Handler function called from workers dedicated to transcription taskiq.
    It gets an image object from garage base on given file_path, transcribes it with llm,
    and writes to :context_id/outputs/:object_name (0000_..., 0001_...)
    In the end it marks the task as 'done' or 'failed' depending on the worker outcome."""
    request_data = TranscriptionTaskData.model_validate(wfq_task_data.payload)
    logger.info("transcribing context_id=%s object_name=%s", wfq_task_data.context_id,
                request_data.object_name)

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
                                                         f"{wfq_task_data.context_id}/outputs/"
                                                         f"{request_data.object_name}.txt",
                                                         io.BytesIO(text_bytes),
                                                         length=len(text_bytes),
                                                         content_type="text/plain; charset=utf-8")


async def transcription_worker_completion_func(wfq_task_data: WFQTask, garage_client: Minio):
    """Handler function called from workers dedicated to run after the last task of given transcriptions job
    has been completed used to tie all the transcriptions in one file the user can later download."""
    logger.info("context complete, merging transcripts context_id=%s", wfq_task_data.context_id)
    await asyncio.to_thread(_merge_and_upload_transcripts, wfq_task_data.context_id, garage_client)


async def init_transcription_engines(garage_client: Minio, model: Model) -> list[tuple[WFQEngine, int]]:
    """Assemble the WFQ engines that process transcription work, each paired with its worker count,
    and ensure their task table exists. Both run over the same table with the same handler; they
    differ only in claim strategy: LICQ (longest-idle-context, fair across users) and FIFO (oldest-first)."""
    handler = partial(transcription_worker_handler_func, model=model, garage_client=garage_client)
    completion = partial(transcription_worker_completion_func, garage_client=garage_client)

    licq = WFQEngine(_WFQ_TRANSCRIPTION_TASKS_TABLE_NAME, wfq_queries.WFQ_LONGEST_IDLE_CLAIM,
                     handler_func=handler, completion_func=completion)
    fifo = WFQEngine(_WFQ_TRANSCRIPTION_TASKS_TABLE_NAME, wfq_queries.WFQ_FIFO_CLAIM,
                     handler_func=handler, completion_func=completion)

    # both engines share one table; materialize it once
    await licq.create_table()

    return [
        (licq, settings.max_licq_concurrent_transcriptions),
        (fifo, settings.max_fifo_concurrent_transcriptions),
    ]


def _merge_and_upload_transcripts(context_id: str, garage_client: Minio):
    workdir = tempfile.mkdtemp()
    file_path = f"{workdir}/output.txt"
    objects = garage_client.list_objects(IMAGE_TRANSCRIPTIONS_BUCKET, prefix=f"{context_id}/outputs/")
    try:
        with open(file_path, "ab") as f:
            for obj in objects:
                if obj.object_name is None:
                    continue # invariant
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


async def _clean_context_data(context_id: str, bucket: str, garage_client: Minio):
    logger.info("cleaning context data context_id=%s bucket=%s", context_id, bucket)
    objects = await asyncio.to_thread(
        lambda: list(garage_client.list_objects(bucket, prefix=context_id, recursive=True))
    )
    await asyncio.to_thread(
        lambda: list(garage_client.remove_objects(
            bucket, (DeleteObject(obj.object_name) for obj in objects if obj.object_name is not None)
        ))
    )


async def get_results(context_id: str,
                      session: AsyncSession,
                      user_id: str)-> TranscriptionResult:
    await _validate_context_creator(session, context_id, user_id)

    progress = await _get_transcriptions_progress(context_id, session)

    failure_reasons = []
    if progress.failed > 0:
        failure_reasons = await transcription_queue.get_failure_reasons(session, context_id)

    finished = await transcription_queue.check_if_context_finished(progress)
    if finished:
        return TranscriptionResult(context_id=context_id,
                                   status=WFQTaskStatus.DONE,
                                   total=progress.total,
                                   done=progress.done,
                                   failed=progress.failed,
                                   download_link=f"{settings.public_base_url}/llm-agents/v1/transcription-agent/"
                                                 f"{context_id}/download",
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


async def download_transcriptions_file(session: AsyncSession,
                                       context_id: str,
                                       garage_client: Minio,
                                       user_id: str) -> BaseHTTPResponse:
    await _validate_context_creator(session, context_id, user_id)
    try:
        file_object: BaseHTTPResponse = await asyncio.to_thread(
            garage_client.get_object, IMAGE_TRANSCRIPTIONS_BUCKET, f"{context_id}/output.txt")
    except S3Error as exc:
        raise NotFoundError("TS-08", "Transcription not found") from exc

    return file_object


async def retry_failed_transcriptions(session: AsyncSession, context_id: str, user_id: str):
    await _validate_context_creator(session, context_id, user_id)
    progress = await transcription_queue.get_progress(session, context_id)
    finished = await transcription_queue.check_if_context_finished(progress)
    if finished and progress.failed > 0:
        await transcription_queue.retry_all_failed_tasks(session, context_id)
    else:
        raise BadRequestError("TS-05", f"Context is not in a retriable state. Progress: {progress}")


async def _validate_context_creator(session: AsyncSession, context_id: str, user_id: str):
    context_created_by = await transcription_queue.get_context_created_by(session, context_id)
    if context_created_by is None or user_id != context_created_by:
        raise BadRequestError("TS-07", "Cannot access context if you are not the creator")


async def _get_transcriptions_progress(context_id: str, session: AsyncSession) -> TaskContextProgress:
    progress = await transcription_queue.get_progress(session, context_id)
    if progress.total == 0:
        raise BadRequestError("TS-04", f"Data not found for context_id: {context_id}")
    return progress
