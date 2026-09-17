import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from src.config import INTENTS
from src.dataset import validate_record


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
SOURCE_INTENT_COUNTS = {
    "train": {
        "GET_TOP_CELLS": 5398,
        "GET_CELL_DETAIL": 1888,
        "GET_CELL_COUNT": 683,
        "UNKNOWN": 31,
    },
    "validation": {
        "GET_TOP_CELLS": 688,
        "GET_CELL_DETAIL": 223,
        "GET_CELL_COUNT": 85,
        "UNKNOWN": 4,
    },
    "test": {
        "GET_TOP_CELLS": 698,
        "GET_CELL_DETAIL": 220,
        "GET_CELL_COUNT": 78,
        "UNKNOWN": 4,
    },
}


@pytest.fixture(scope="module")
def split_records():
    return {
        split: [
            json.loads(line)
            for line in (ROOT / "data" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for split in SPLITS
    }


def test_check_is_byte_stable():
    paths = [ROOT / "data" / f"{split}.jsonl" for split in SPLITS]
    before = {path: path.read_bytes() for path in paths}

    command = [sys.executable, "-m", "src.generate_multilingual_data", "--check"]
    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert (first.stdout, first.stderr) == (second.stdout, second.stderr)
    assert before == {path: path.read_bytes() for path in paths}


def test_each_group_has_one_pair_in_one_split(split_records):
    groups = defaultdict(list)
    for split, records in split_records.items():
        for record in records:
            groups[record["group_id"]].append((split, record))

    assert len(groups) == 10000
    for entries in groups.values():
        assert len(entries) == 2
        assert len({split for split, _ in entries}) == 1
        assert {record["language"] for _, record in entries} == {"id", "en"}
        assert len({record["id"] for _, record in entries}) == 2
        assert len({record["intent"] for _, record in entries}) == 1
        source = next(record for _, record in entries if record["language"] == "id")
        canonical = json.dumps(
            {key: source[key] for key in ("intent", "slots", "tokens")},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        assert source["group_id"] == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_all_records_validate_and_metadata_is_consistent(split_records):
    ids = set()
    for split, records in split_records.items():
        for line_no, record in enumerate(records, 1):
            validate_record(record, line_no)
            assert record["language"] in {"id", "en"}
            assert record["source"] == "synthetic-template"
            assert record["text"] == " ".join(record["tokens"])
            assert record["id"].endswith(f"-{record['language']}")
            if record["intent"] == "UNKNOWN":
                assert set(record["slots"]) == {"O"}
            assert record["id"] not in ids, (split, line_no, record["id"])
            ids.add(record["id"])
    assert len(ids) == 20000


def test_intent_counts_are_doubled_and_each_intent_has_english(split_records):
    for split, records in split_records.items():
        expected = Counter({intent: count * 2 for intent, count in SOURCE_INTENT_COUNTS[split].items()})
        assert Counter(record["intent"] for record in records) == expected
        half = len(records) // 2
        assert Counter(record["language"] for record in records) == {"id": half, "en": half}
        for intent in INTENTS:
            assert any(record["language"] == "en" and record["intent"] == intent for record in records)


def test_pairs_keep_intent_equal(split_records):
    pairs = defaultdict(dict)
    for records in split_records.values():
        for record in records:
            pairs[record["group_id"]][record["language"]] = record
    for pair in pairs.values():
        assert pair["id"]["intent"] == pair["en"]["intent"]
