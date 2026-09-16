import asyncio
import io
import shutil
import tempfile
import uuid
import zipfile
from collections.abc import AsyncIterator
from operator import length_hint
from pathlib import Path
from zipfile import ZipFile

from aiohttp.web_fileresponse import content_type
from fastapi import UploadFile, HTTPException
from fastmcp.utilities.skills import download_skill
from glide import GlideClient
from glide_shared import ListDirection
from minio import Minio, S3Error
from minio.deleteobjects import DeleteObject
from minio.helpers import ObjectWriteResult
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models import Model
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse
from taskiq import TaskiqDepends
from urllib3 import BaseHTTPResponse

from src.WFQEngine import wfq_transcriptions
from src.WFQEngine.wfq_engine import ADD_TASKS_TO_QUEUE
from src.WFQEngine.wfq_transcriptions import TranscriptionWFQTask, TranscriptionWFQTaskStatus
from src.config import settings
from src.errors import BadRequestError, AppError
from src.garage.buckets import IMAGE_TRANSCRIPTIONS_BUCKET

from src.tasks.brokers import transcriptions_broker, get_taskiq_garage_client, get_taskiq_llm_model, get_taskiq_valkey_client

TASK_TRANSCRIPTION = "tasks:transcription"

_ZIP_SUFIX = (".zip",)

_SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
_SUFFIX_TO_MEDIA = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}

_CONCURRENCY = 2

_MIN_PART = 10 * 1024 * 1024

_CHUNK_SIZE = 2 * 1024 * 1024


class Transcription(BaseModel):
    filename: str
    text: str


class TranscriptionResult(BaseModel):
    context_id: str
    status: str
    total: int
    completed: int
    control_transcript: str
    download_link: str


IMAGE_TEXT_TO_TEXT_AGENT_INSTRUCTIONS = (
    "You are an image-to-text transcription agent."
    "Given an image containing text (handwritten or printed), output exactly what is written as a string."
    "Output only the transcribed text, nothing else."
    "If the image contains no text, output an empty string."
)

image_text_to_text_agent = Agent(
    instructions=IMAGE_TEXT_TO_TEXT_AGENT_INSTRUCTIONS,
    output_type=str,
)

async def convert_to_text(model: Model, image_bytes: bytes, media_type: str) -> str:
    result = await image_text_to_text_agent.run(
        ["Transcribe the text in this image.",
         BinaryContent(data=image_bytes, media_type=media_type)],
        model=model,
    )

    return result.output


class TranscriptionTask(BaseModel):
    context_id: str
    status: str
    file_path: str
    file_index: str


async def transcribe_with_wfo(uploads: list[UploadFile],
                              psql_connection: AsyncSession,
                              garage_client: Minio):
    context_id = str(uuid.uuid4())

    await upload_files(context_id, garage_client, uploads)

    for index, upload in enumerate(uploads):
        object_name = f"{context_id}/{index:04d}_{upload.filename}"
        await wfq_transcriptions.add_transcription_task(psql_connection,
                                                  TranscriptionWFQTask(context_id,
                                                                       TranscriptionWFQTaskStatus.PENDING,
                                                                       object_name,
                                                                       index))


# TODO - add rate limiter, important !!!!!
# also, document somewhere:
# raw = await valkey_client.get(transcription_request_key)
# if raw:
#     found = TranscriptionWorkflow.model_validate_json(raw) -document somewhere
async def transcribe_uploads_fifo(uploads: list[UploadFile],
                                  valkey_client: GlideClient,
                                  garage_client: Minio,
                                  user_id: str) -> str:
    """Save zip to disk. Expand, and call transcribe for 2 image files at a time, and save to a file. Return file."""
    context_id = str(uuid.uuid4())

    await upload_files(context_id, garage_client, uploads)

    await valkey_client.hset(context_id, {"status": "running", "completed": 0, "failed": 0, "total": len(uploads)})

    for index, upload in enumerate(uploads):
        object_name = f"{context_id}/{upload.filename}"
        await transcribe_as_queued_broker.kiq(context_id=context_id, object_name=object_name, media_type=upload.content_type,
                                              index=index)

    return context_id

class ProcessingTask(BaseModel):
    object_name: str
    media_type: str
    original_file_name: str

async def transcribe_uploads_fair(uploads: list[UploadFile],
                                  valkey_client: GlideClient,
                                  garage_client: Minio,
                                  user_id: str) -> str:
    """Save zip to disk. Expand, and call transcribe for 2 image files at a time, and save to a file. Return file."""
    context_id = str(uuid.uuid4())

    # media_type = _SUFFIX_TO_MEDIA[Path(processing_task.object_name).suffix.lower()] TODO, add to validate
    validate(uploads)

    object_names = await upload_files(context_id, garage_client, uploads)

    await valkey_client.hset(context_id, {"status": "running", "completed": 0, "failed": 0, "total": len(uploads)})

    processing_tasks = [
        ProcessingTask(object_name=object_name,
                       media_type=upload.content_type,
                       original_file_name=upload.filename)
            .model_dump_json()
        for (object_name, upload) in zip(object_names, uploads)
    ]
    # queue tasks from users
    await valkey_client.lpush(f"{context_id}:pending", *processing_tasks)

    await transcribe_images_fair_broker.kiq(context_id, IMAGE_TRANSCRIPTIONS_BUCKET)

    return context_id


@transcriptions_broker.task
async def transcribe_images_fair_broker(context_id: str,
                                 bucket_name: str,
                                 valkey_client: GlideClient = TaskiqDepends(get_taskiq_valkey_client),
                                 garage_client: Minio = TaskiqDepends(get_taskiq_garage_client),
                                 model: Model = TaskiqDepends(get_taskiq_llm_model)):
    processing_task_b = await valkey_client.lmove(
        f"{context_id}:pending", f"{context_id}:processing", ListDirection.LEFT, ListDirection.RIGHT)
    if processing_task_b is None:
        return
    processing_task = ProcessingTask.model_validate_json(processing_task_b)

    image_object: BaseHTTPResponse = await asyncio.to_thread(
        garage_client.get_object,bucket_name, processing_task.object_name)

    try:
        image_text = await convert_to_text(
            model, await asyncio.to_thread(image_object.read), processing_task.media_type)
    finally:
        image_object.close(); image_object.release_conn()

    full_text = f"=== {processing_task.original_file_name} ===\n{image_text}\n\n"
    text_bytes = full_text.encode("utf-8")

    try:
        await asyncio.to_thread(garage_client.put_object, bucket_name,
                                f"{context_id}/outputs/{processing_task.object_name}",
                                io.BytesIO(text_bytes), len(text_bytes))
    except ValueError as exc:
        await valkey_client.hincrby(context_id, "failed", 1)
        return # TODO add log

    completed = await valkey_client.hincrby(context_id, "completed", 1)
    total = int(await valkey_client.hget(context_id, "total"))

    if completed == total:
        await asyncio.to_thread(_finalize_garage_file, context_id, bucket_name, garage_client)
        await valkey_client.hset(context_id, {"status": "completed"})
        return

    has_next = await valkey_client.llen(f"{context_id}:pending") > 0
    if has_next:
        await transcribe_images_fair_broker.kiq(context_id, bucket_name)


async def upload_files(context_id: str, garage_client: Minio, uploads: list[UploadFile]) -> list[str]:
    object_names = []
    for index, upload in enumerate(uploads):
        object_name = f"{context_id}/{index:04d}_{upload.filename}"
        try:
            await asyncio.to_thread(garage_client.put_object,
                                    IMAGE_TRANSCRIPTIONS_BUCKET,
                                    object_name,
                                    upload.file,
                                    content_type="application/octet-stream",
                                    part_size=_MIN_PART,
                                    length=-1)
            object_names.append(object_name)
        except (ValueError, IOError) as exc:
            await clean_context_data(context_id, IMAGE_TRANSCRIPTIONS_BUCKET, garage_client)
            raise AppError("TII-01", f"Failed to upload file {upload.file}") from exc

    return object_names


@transcriptions_broker.task
async def transcribe_as_queued_broker(context_id: str,
                                      object_name: str,
                                      media_type: str,
                                      index: int,
                                      garage_client: Minio = TaskiqDepends(get_taskiq_garage_client),
                                      valkey_client: GlideClient = TaskiqDepends(get_taskiq_valkey_client),
                                      model: Model = TaskiqDepends(get_taskiq_llm_model)) -> str:
    uploaded_image_object: BaseHTTPResponse = await asyncio.to_thread(garage_client.get_object,
                                                             IMAGE_TRANSCRIPTIONS_BUCKET,
                                                             object_name)
    try:
        image_text = await convert_to_text(model, await asyncio.to_thread(uploaded_image_object.read), media_type)
    finally:
        uploaded_image_object.close(); uploaded_image_object.release_conn()

    byte_data = f"==={object_name.split("/")[-1]}===\n{image_text}\n".encode("utf-8")

    output_name = f"{context_id}/outputs/{index:04d}_{Path(object_name).stem}.txt" # :04f - padding format 0000, 0001..

    try:
        await asyncio.to_thread(garage_client.put_object,
                          IMAGE_TRANSCRIPTIONS_BUCKET,
                                output_name,
                                io.BytesIO(byte_data),
                                content_type="application/octet-stream",
                                length=len(byte_data))
    except ValueError as exc:
        await valkey_client.hincrby(str(context_id), "failed", 1)
        # TODO add log

    completed = await valkey_client.hincrby(str(context_id), "completed", 1)

    total = int(await valkey_client.hget(context_id, "total"))
    if completed == total:
        await asyncio.to_thread(_finalize_garage_file, context_id, IMAGE_TRANSCRIPTIONS_BUCKET, garage_client)
        await valkey_client.hset(context_id, {"status": "completed"})


def _finalize_garage_file(context_id: str, bucket_name: str, garage_client: Minio):
    workdir = tempfile.mkdtemp()
    file_path = f"{workdir}/output.txt"

    output_objects = garage_client.list_objects(bucket_name, prefix=f"{context_id}/outputs/")
    with open(file_path, "ab") as f:
        for obj in output_objects:
            obj_to_write = garage_client.get_object(bucket_name, obj.object_name)
            try:
                f.write(obj_to_write.read())
            finally:
                obj_to_write.close(); obj_to_write.release_conn()

    garage_client.fput_object(bucket_name,
                              f"{context_id}/output.txt",
                              file_path,
                              content_type="text/plain; charset=utf-8")




async def clean_context_data(context_id: str, bucket: str, garage_client: Minio):
    objects = await asyncio.to_thread(lambda: list(garage_client.list_objects(bucket, prefix=context_id)))
    await asyncio.to_thread(
        garage_client.remove_objects,bucket, (DeleteObject(obj.object_name) for obj in objects))


async def get_results(context_id: str,
                      valkey_client: GlideClient,
                      garage_client: Minio)-> TranscriptionResult:
    status_b, total_b, completed_b = await valkey_client.hmget(
        context_id, ["status", "total", "completed"])
    status = status_b.decode() if status_b else ""
    total = int(total_b) if total_b else 0
    completed = int(completed_b) if completed_b else 0

    if len(status) > 0 and status == "running":
        control_transcript = await get_control_transcript(garage_client, IMAGE_TRANSCRIPTIONS_BUCKET, context_id)
        return TranscriptionResult(context_id=context_id,
                                   status=status,
                                   total=total,
                                   completed=completed,
                                   control_transcript=control_transcript,
                                   download_link="")
    elif len(status) > 0 and status == "completed" and total == completed:
        return TranscriptionResult(context_id=context_id,
                                   status=status,
                                   total=total,
                                   completed=completed,
                                   download_link=f"{settings.public_base_url}/llm-agents/v1/transcription-agent/{context_id}/download",
                                   control_transcript="")
    elif len(status) == 0:
        raise AppError("TII-3", f"Something wrong with task tracking: "
                                f"status={status}, total={total}, completed={completed}")
    else:
        raise BadRequestError("TII-02", f"Data not found for context_id: {context_id}")


async def get_control_transcript(garage_client: Minio, bucket: str, context_id: str) -> str:
    names = await asyncio.to_thread(
        lambda: list(garage_client.list_objects(bucket, prefix=f"{context_id}/outputs")))

    first = names[0]

    response: BaseHTTPResponse = await asyncio.to_thread(garage_client.get_object, bucket, first.object_name)

    try:
        return (await asyncio.to_thread(response.read)).decode("utf-8")
    finally:
        response.close(); response.release_conn()


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


async def _iter_images(upload_files: list[UploadFile]) -> AsyncIterator[tuple[str, bytes]]:
    """Yield (name, jpg_bytes) one at a time from uploaded files or zips.

    A .zip is read from its spooled temp file (on disk for > 1MB uploads),
    decompressing one image_file at a time; a .jpg is passed through.
    macOS zip junk (__MACOSX/, ._*) is skipped.
    """
    for f in upload_files:
        name = f.filename or "upload"
        lower = name.lower()
        if lower.endswith(_ZIP_SUFIX):
            await f.seek(0)
            zf = await asyncio.to_thread(zipfile.ZipFile, f.file)
            try:
                image_files = [
                    n for n in zf.namelist()
                    if n.lower().endswith(_SUPPORTED_IMAGE_SUFFIXES)
                    and not n.startswith("__MACOSX/")
                    and not n.rsplit("/", 1)[-1].startswith("._")
                ]
                for image_file in image_files:
                    data = await asyncio.to_thread(zf.read, image_file)
                    yield image_file, data
            finally:
                zf.close()
        elif lower.endswith(_SUPPORTED_IMAGE_SUFFIXES):
            yield name, await f.read()  # already runs in a threadpool
