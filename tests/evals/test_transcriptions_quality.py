import json
import string
from pathlib import Path

import pytest
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from tests.evals.test_transcriptions_agents import TRANSCRIPTION_OUTPUT_JSON

_GROUND_TRUTH_JSON = Path(__file__).parent / "gold" / "ground_truth.json"

_KEYS = list(json.loads(_GROUND_TRUTH_JSON.read_text()).keys())


@pytest.mark.parametrize("key", _KEYS)
def test_transcriptions_output_text_overall_quality(key):
    actual_transcription = json.loads(TRANSCRIPTION_OUTPUT_JSON.read_text())[key]["text"]
    expected_transcription = json.loads(_GROUND_TRUTH_JSON.read_text())[key]["text"]

    ratio = fuzz.ratio(actual_transcription, expected_transcription)
    print(f"{key}: {ratio}")

    assert ratio >= 99


@pytest.mark.parametrize("key", _KEYS)
def test_transcriptions_output_text_punctuation_quality(key):
    punct = set(string.punctuation) | {"\u2014"}  # em dash

    actual_transcription = json.loads(TRANSCRIPTION_OUTPUT_JSON.read_text())[key]["text"]
    expected_transcription = json.loads(_GROUND_TRUTH_JSON.read_text())[key]["text"]

    total = sum(1 for c in expected_transcription if c in punct)

    changed_positions = {op.src_pos for op in Levenshtein.editops(expected_transcription, actual_transcription)
               if op.tag in ("delete", "replace")}
    kept = sum(1 for i, c in enumerate(expected_transcription) if c in punct and i not in changed_positions)

    ratio = (kept / total) * 100

    print(f"{key}: {ratio}")

    assert ratio > 98