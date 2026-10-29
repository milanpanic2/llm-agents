import asyncio
import io
from typing import Any

from PIL import Image, ImageOps
from PIL.Image import Resampling
from pydantic_ai import Agent, AgentRunResult, BinaryContent
from pydantic_ai.models import Model

from src.config import metrics
from src.config.settings import settings
from src.errors import AppError

_FORMAT_TO_MEDIA_TYPE = {"JPEG": "image/jpeg", "PNG": "image/png", "MPO": "image/jpeg"}

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
    resized_image_bytes, media_type = await asyncio.to_thread(resize_image_and_get_media_type, image_bytes)
    
    model_name = getattr(model, "model_name", settings.llm_model)
    with metrics.record_duration(metrics.LLM_REQUEST_DURATION, {"model": model_name}):
        result = await transcriptions_agent.run(
            ["Transcribe the text in this image.",
             BinaryContent(data=resized_image_bytes, media_type=media_type)],
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


def resize_image_and_get_media_type(image_bytes: bytes) -> tuple[bytes, str]:
    with Image.open(io.BytesIO(image_bytes)) as i:
        image_format = i.format
        ImageOps.exif_transpose(i, in_place=True)
        i.thumbnail((settings.transcription_image_max_dim, settings.transcription_image_max_dim),
                    resample=Resampling.LANCZOS)
        img_buf = io.BytesIO()
        i.save(img_buf, format=image_format)

        media_type = _FORMAT_TO_MEDIA_TYPE.get(image_format or "")
        if media_type is None:
            raise AppError("TS-02", "Failed image media_type parse for llm")
        
    return img_buf.getvalue(), media_type
