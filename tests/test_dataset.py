import json

import pytest

from src.dataset import load_jsonl, tokenize_and_align, validate_record
from transformers import AutoTokenizer

from src.config import MODEL_NAME


def write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


VALID = {
    "tokens": ["tampilkan", "10", "cell"],
    "intent": "GET_TOP_CELLS",
    "slots": ["O", "B-LIMIT", "O"],
}


def test_valid_record_passes():
    validate_record(VALID, 1)


def test_load_jsonl(tmp_path):
    p = tmp_path / "ok.jsonl"
    write_jsonl(p, [VALID])
    records = load_jsonl(p)
    assert len(records) == 1
    assert records[0][0] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        {"tokens": []},
        {"slots": []},
        {"slots": ["O", "B-LIMIT"]},
        {"intent": "NOPE"},
        {"slots": ["O", "B-LIMIT", "B-NOPE"]},
    ],
)
def test_malformed_records_raise(tmp_path, mutation):
    bad = {**VALID, **mutation}
    p = tmp_path / "bad.jsonl"
    write_jsonl(p, [VALID, bad])
    records = load_jsonl(p)
    with pytest.raises(ValueError, match="line 2"):
        for line_no, record in records:
            validate_record(record, line_no)


def test_tokenize_and_align_end_to_end(tmp_path):
    p = tmp_path / "ok.jsonl"
    write_jsonl(p, [VALID])
    records = load_jsonl(p)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = tokenize_and_align([r for _, r in records], tokenizer)
    assert len(ds) == 1
    assert len(ds[0]["slot_labels"]) == len(ds[0]["input_ids"])
