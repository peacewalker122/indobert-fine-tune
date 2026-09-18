import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from src.config import INTENTS
from src.dataset import decode_spans, validate_record
from src.generate_multilingual_data import MIXED_SOURCE_NAME


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
        assert len(entries) == 3
        assert len({split for split, _ in entries}) == 1
        assert {record["language"] for _, record in entries} == {"id", "en", "mixed"}
        assert len({record["id"] for _, record in entries}) == 3
        assert len({record["intent"] for _, record in entries}) == 1
        source = next(record for _, record in entries if record["language"] == "id")
        canonical = json.dumps(
            {key: source[key] for key in ("intent", "slots", "tokens")},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        assert source["group_id"] == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        mixed = next(record for _, record in entries if record["language"] == "mixed")
        assert mixed["source"] == MIXED_SOURCE_NAME


def test_all_records_validate_and_metadata_is_consistent(split_records):
    ids = set()
    for split, records in split_records.items():
        for line_no, record in enumerate(records, 1):
            validate_record(record, line_no)
            assert record["language"] in {"id", "en", "mixed"}
            expected_source = MIXED_SOURCE_NAME if record["language"] == "mixed" else "synthetic-template"
            assert record["source"] == expected_source
            assert record["text"] == " ".join(record["tokens"])
            assert record["id"].endswith(f"-{record['language']}")
            if record["intent"] == "UNKNOWN":
                assert set(record["slots"]) == {"O"}
            assert record["id"] not in ids, (split, line_no, record["id"])
            ids.add(record["id"])
    assert len(ids) == 30000


def test_intent_counts_are_doubled_and_each_intent_has_english(split_records):
    for split, records in split_records.items():
        expected = Counter({intent: count * 3 for intent, count in SOURCE_INTENT_COUNTS[split].items()})
        assert Counter(record["intent"] for record in records) == expected
        source_count = sum(SOURCE_INTENT_COUNTS[split].values())
        assert Counter(record["language"] for record in records) == {
            "id": source_count,
            "en": source_count,
            "mixed": source_count,
        }
        for intent in INTENTS:
            assert any(record["language"] == "en" and record["intent"] == intent for record in records)
            assert any(record["language"] == "mixed" and record["intent"] == intent for record in records)


def test_pairs_keep_intent_equal(split_records):
    pairs = defaultdict(dict)
    for records in split_records.values():
        for record in records:
            pairs[record["group_id"]][record["language"]] = record
    for pair in pairs.values():
        assert set(pair) == {"id", "en", "mixed"}
        assert len({record["intent"] for record in pair.values()}) == 1


def test_mixed_records_are_deterministic_code_switches(split_records):
    english_markers = {
        "I", "want", "need", "to", "please", "show", "how", "many", "what's",
        "how's", "details", "check", "inspect", "display", "give", "bring", "up",
        "find", "count", "list",
    }
    by_group = {
        record["group_id"]: record
        for records in split_records.values()
        for record in records
        if record["language"] == "id"
    }
    mixed_texts = {}
    source_to_mixed = {}
    for split, records in split_records.items():
        texts = set()
        for mixed in (record for record in records if record["language"] == "mixed"):
            source = by_group[mixed["group_id"]]
            source_to_mixed[source["text"]] = mixed["text"]
            assert mixed["tokens"] != source["tokens"]
            assert any(token in english_markers for token in mixed["tokens"])
            assert any(token not in source["tokens"] for token in mixed["tokens"])
            assert len(mixed["tokens"]) == len(mixed["slots"])
            assert {
                (kind, text)
                for kind, _, _, text in decode_spans(mixed["slots"], mixed["tokens"])
            } == {
                (kind, text)
                for kind, _, _, text in decode_spans(source["slots"], source["tokens"])
            }
            assert "paling paling" not in mixed["text"]
            assert mixed["text"] not in texts
            texts.add(mixed["text"])
        mixed_texts[split] = texts

    assert mixed_texts["train"].isdisjoint(mixed_texts["validation"])
    assert mixed_texts["train"].isdisjoint(mixed_texts["test"])
    assert mixed_texts["validation"].isdisjoint(mixed_texts["test"])
    assert source_to_mixed["tampilkan 50 cell dengan dl throughput paling rendah di Jakarta"] == (
        "show 50 cell dengan dl throughput paling rendah di Jakarta"
    )
    assert source_to_mixed["saya mau lihat 25 site dengan detractor tertinggi pada Medan"] == (
        "I want to lihat 25 site dengan detractor tertinggi pada Medan"
    )
    assert source_to_mixed["berapa lama penerbangan ke bali"] == (
        "please berapa lama penerbangan ke bali"
    )
