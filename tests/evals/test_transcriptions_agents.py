import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from src.agents import transcriptions_agents

TRANSCRIPTION_OUTPUT_JSON = Path(__file__).parent / "transcription_output.json"


_GROUND_TRUTH = Path(__file__).parent / "gold" / "ground_truth.json"
_IMAGES_DIR = Path(__file__).parent / "gold" / "images"


_NUMBER_OF_CONCURRENT_TRANSCRIPTIONS = 4


async def _run_transcription_agent(truth, transcription_model):
    sem = asyncio.Semaphore(_NUMBER_OF_CONCURRENT_TRANSCRIPTIONS)
    async def _run_one(key, value):
        async with sem:
            text = await transcriptions_agents.convert_to_text(
                model=transcription_model,
                image_bytes=(_IMAGES_DIR / value["image"]).read_bytes()
            )
            return key, value["image"], text

    results = await asyncio.gather(*[_run_one(key, value) for key, value in truth.items()])

    return {key: {"image": image_name, "text": text} for key, image_name, text in results}


@pytest.mark.eval
def test_transcription_with_model_and_save_output(build_transcription_model):
    with open(_GROUND_TRUTH) as f:
        truth: dict[str, Any] = json.load(f)

    results = asyncio.run(_run_transcription_agent(truth, build_transcription_model))

    with open(TRANSCRIPTION_OUTPUT_JSON, "w") as o:
        json.dump(results, o, indent=2, ensure_ascii=False)
