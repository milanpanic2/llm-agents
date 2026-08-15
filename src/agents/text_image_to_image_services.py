import asyncio
import zipfile
from collections.abc import AsyncIterator

from fastapi import UploadFile
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models import Model

_ZIP_SUFIX = (".zip",)

_JPG_SUFFIXES = (".jpg", ".jpeg")


class Transcription(BaseModel):
    filename: str
    text: str


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

async def convert_to_text(image_bytes: bytes, model: Model, media_type: str = "image/jpeg") -> str:
    result = await image_text_to_text_agent.run(
        ["Transcribe the text in this image.",
         BinaryContent(data=image_bytes, media_type=media_type)],
        model=model,
    )


    return result.output


async def transcribe_uploads(files: list[UploadFile], model: Model) -> list[Transcription]:
    """Expand any zips, transcribe each image one at a time, return results in order."""
    results: list[Transcription] = []
    async for name, image_bytes in _iter_images(files):
        text = await convert_to_text(image_bytes, model)
        if not text or len(text) == 0:
            continue
        results.append(Transcription(filename=name, text=text))
    return results


async def _iter_images(upload_files: list[UploadFile]) -> AsyncIterator[tuple[str, bytes]]:
    """Yield (name, jpg_bytes) one at a time from uploaded files or zips.

    A .zip is read from its spooled temp file (on disk for > 1MB uploads),
    decompressing one entry at a time; a .jpg is passed through.
    macOS zip junk (__MACOSX/, ._*) is skipped.
    """
    for f in upload_files:
        name = f.filename or "upload"
        lower = name.lower()
        if lower.endswith(_ZIP_SUFIX):
            await f.seek(0)
            zf = await asyncio.to_thread(zipfile.ZipFile, f.file)
            try:
                entries = [
                    n for n in zf.namelist()
                    if n.lower().endswith(_JPG_SUFFIXES)
                    and not n.startswith("__MACOSX/")
                    and not n.rsplit("/", 1)[-1].startswith("._")
                ]
                for entry in entries:
                    data = await asyncio.to_thread(zf.read, entry)
                    yield entry, data
            finally:
                zf.close()
        elif lower.endswith(_JPG_SUFFIXES):
            yield name, await f.read()  # already runs in a threadpool
