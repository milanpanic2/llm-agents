import asyncio
import logging
import zipfile
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import BinaryIO, cast
from zipfile import ZipFile

from fastapi import UploadFile
from PIL import Image

from src.errors import BadRequestError

logger = logging.getLogger(__name__)

# MPO is a JPEG container with extra embedded frames (iPhone HDR/portrait shots);
# its primary frame is a normal JPEG, so we accept it and treat it as JPEG.
_SUPPORTED_IMAGE_FORMATS = ["JPEG", "PNG", "MPO"]
_MAX_IMAGES = 100  # TODO - gate per user/day after stripe integration (2eur = 100 images)
_MAX_IMAGE_BYTES = 40 * 1024 * 1024


@asynccontextmanager
async def images_from_zip(zip_upload: UploadFile) -> AsyncIterator[Iterator[UploadFile]]:
    """Open an untrusted uploaded zip, validate it, and yield `UploadFile`s for the real images.
    Owns the `ZipFile` lifetime (closes it on exit), so callers must consume the iterator inside
    the `async with` block. Raises `BadRequestError` for any invalid input. Blocking zip/PIL work
    runs in a thread so the event loop stays free."""
    try:
        zf = await asyncio.to_thread(zipfile.ZipFile, zip_upload.file)
    except (zipfile.BadZipFile, OSError) as exc:
        logger.warning("zip rejected: not a valid zip filename=%s", zip_upload.filename)
        raise BadRequestError("ZIP-01", "Uploaded file is not a valid zip archive") from exc

    try:
        images = [image for image in zf.infolist() if not _is_zipping_artifact(image)]

        if len(images) == 0:
            logger.warning("zip rejected: no valid files found filename=%s", zip_upload.filename)
            raise BadRequestError("ZIP-02", "No valid files found, check the submitted zip.")

        if len(images) > _MAX_IMAGES:
            logger.warning("zip rejected: too many items filename=%s count=%d", zip_upload.filename, len(images))
            raise BadRequestError("ZIP-03", f"Too many items in zip. {_MAX_IMAGES} supported.")

        await asyncio.to_thread(_validate_images, zf, images)

        def generate_upload_files() -> Iterator[UploadFile]:
            for image in images:
                yield UploadFile(file=cast(BinaryIO, zf.open(image)), size=image.file_size, filename=image.filename)

        yield generate_upload_files()
    finally:
        zf.close()


def _validate_images(zf: ZipFile, images: list[zipfile.ZipInfo]):
    for image in images:
        if image.file_size > _MAX_IMAGE_BYTES:
            raise BadRequestError("ZIP-04", "Zip invalid. Contains image > 40MiB")

        with zf.open(image) as i:
            try:
                fmt = Image.open(i).format
            except Exception as e:
                raise BadRequestError("ZIP-05", "Zip invalid. Contains fake image files") from e
        if fmt not in _SUPPORTED_IMAGE_FORMATS:
            raise BadRequestError("ZIP-06", f"Zip invalid. Contains unsupported format: {fmt}")


def _is_zipping_artifact(info: zipfile.ZipInfo) -> bool:
    name = Path(info.filename).name
    return (info.is_dir()
            or info.filename.startswith("__MACOSX/")
            or name == ".DS_Store"
            or name.startswith("._"))


def _sniff_jpeg_png_media_type(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return None
