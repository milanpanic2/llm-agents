import asyncio
import shutil
import tempfile
import uuid
import zipfile
from collections.abc import AsyncIterator
from pathlib import Path
from zipfile import ZipFile

from fastapi import UploadFile
from glide import GlideClient
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models import Model

from src.errors import BadRequestError

_ZIP_SUFIX = (".zip",)

_SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

_CONCURRENCY = 2

_CHUNK_SIZE = 1024 * 1024  # 1 MiB, for reading/writing zips without loading them whole


class Transcription(BaseModel):
    filename: str
    text: str

class TranscriptionResponse(BaseModel):
    transcription: Transcription
    context_id: str



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


class TranscriptionWorkflow(BaseModel):
    status: str
    work_dir: str
    context_id: str


async def transcribe_uploads_from_zip(zip_upload: UploadFile, model: Model, valkey_client: GlideClient, user_id: str) -> AsyncIterator[UploadFile]:
    """Save zip to disk. Expand, and call transcribe for 2 image files at a time, and save to a file. Return file."""
    transcription_request_key = f"transcription_{user_id}"
    context_id = uuid.uuid4()
    work_dir = tempfile.mkdtemp()
    transcription_wf = TranscriptionWorkflow(status="in_progress", work_dir=work_dir, context_id=str(context_id))

    await validate(transcription_request_key, valkey_client, zip_upload)
    await valkey_client.set(transcription_request_key, transcription_wf.model_dump_json())

    try:
        zip_file_path = await save_file_to_disk(zip_upload, work_dir)
        async for file_name, image in _iter_zip_images(zip_file_path):

    except:
        await valkey_client.delete([transcription_request_key])



async def _iter_zip_images(zip_file_path: str) -> AsyncIterator[tuple[str, bytes]]:
    zf = await asyncio.to_thread(zipfile.ZipFile, zip_file_path)

    try:
        image_files_names = [n for n in zf.namelist()
                             if n.lower().endswith(_SUPPORTED_IMAGE_SUFFIXES)
                             and not n.startswith("__MACOSX/")
                             and not n.rsplit("/", 1)[-1].startswith("._")]

        for image_file_name in image_files_names:
            image = await asyncio.to_thread(zf.read, image_file_name)
            yield image_file_name, image
    finally:
        zf.close()


async def validate(transcription_request_key: str, valkey_client: GlideClient, zip_upload: UploadFile):
    raw = await valkey_client.get(transcription_request_key)

    if raw:
        found = TranscriptionWorkflow.model_validate_json(raw)
        if found.status == "in_progress":
            raise BadRequestError("TII-01", f"An in_progress workflow exists. Use get by context_id with:"
                                            f" {found.context_id}")

    if not zip_upload.filename.endswith(_ZIP_SUFIX):
        raise BadRequestError("TII-02", "The file sent is not .zip format")


async def save_file_to_disk(zip_upload: UploadFile, work_dir: str) -> str:
    dest = Path(work_dir) / (zip_upload.filename or "upload.zip")
    await zip_upload.seek(0)

    with open(dest, "wb") as out:
        while chunk := await zip_upload.read(_CHUNK_SIZE):
            out.write(chunk)

    return str(dest)


async def transcribe_uploads(files: list[UploadFile], model: Model) -> list[Transcription]:
    """Expand any zips, transcribe each image one at a time, return results in order."""
    sem = asyncio.Semaphore(_CONCURRENCY)

    async def worker(_filename, _image_bytes, _media_type) -> Transcription:
        try:
            _text = await convert_to_text(model, _image_bytes, _media_type)

            return Transcription(filename=_filename, text=_text)
        finally:
            sem.release()

    tasks = []
    workflow_id = uuid.uuid4()

    async for filename, image_bytes in _iter_images(files):
        await sem.acquire()
        media_type = "image/jpeg"  # TODO
        tasks.append(asyncio.create_task(worker(filename, image_bytes, media_type)))

    results = await asyncio.gather(*tasks)

    return list(results)



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
