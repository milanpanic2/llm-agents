import json
import string
from pathlib import Path

import pytest
from rapidfuzz.distance import Levenshtein

from tests.evals.test_transcriptions_agents import TRANSCRIPTION_OUTPUT_JSON

_GROUND_TRUTH_JSON = Path(__file__).parent / "gold" / "ground_truth.json"

_KEYS = list(json.loads(_GROUND_TRUTH_JSON.read_text()).keys())


@pytest.mark.parametrize("key", _KEYS)
def test_transcriptions_output_text_overall_quality(key):
    actual_transcription = _normalize(json.loads(TRANSCRIPTION_OUTPUT_JSON.read_text())[key]["text"])
    expected_transcription = _normalize(json.loads(_GROUND_TRUTH_JSON.read_text())[key]["text"])

    distance = Levenshtein.distance(expected_transcription, actual_transcription)
    cer = distance / len(expected_transcription)
    accuracy = (1 - cer) * 100

    print(f"{key}: {accuracy}")

    assert accuracy >= 99


@pytest.mark.parametrize("key", _KEYS)
def test_transcriptions_output_text_punctuation_quality(key):
    punct = set(string.punctuation) | {"\u2014"}  # em dash

    actual_transcription = json.loads(TRANSCRIPTION_OUTPUT_JSON.read_text())[key]["text"]
    expected_transcription = json.loads(_GROUND_TRUTH_JSON.read_text())[key]["text"]
    
    ops = Levenshtein.editops(expected_transcription, actual_transcription)
    changed_src = {op.src_pos for op in ops if op.tag in ("delete", "replace")}
    added_dest = {op.dest_pos for op in ops if op.tag in ("insert", "replace")}

    kept = sum(1 for i, c in enumerate(expected_transcription) if c in punct and i not in changed_src)
    missing = sum(1 for i, c in enumerate(expected_transcription) if c in punct and i in changed_src)
    added = sum(1 for i, c in enumerate(actual_transcription) if c in punct and i in added_dest)
    
    recall = kept / (kept + missing) * 100
    precision = kept / (kept + added) * 100

    print(f"{key}: recall={recall:.2f} precision={precision:.2f} added={added} missing={missing}")

    assert recall > 95
    assert precision > 95

def _normalize(s: str) -> str:
     # collapse blank lines, trim trailing whitespace per line
    return "\n".join(line.rstrip() for line in s.splitlines() if line.strip())
