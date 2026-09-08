import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from .config import (
    ID_TO_SLOT,
    IGNORE_INDEX,
    INTENT_TO_ID,
    MAX_LENGTH,
    MODEL_NAME,
    SLOT_TO_ID,
)

LANGUAGES = ("id", "en", "mixed")

ENTITY_TYPES = sorted({s[2:] for s in SLOT_TO_ID if s != "O"})


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            records.append((line_no, json.loads(line)))
    return records


def validate_record(record, line_no=0):
    tokens = record.get("tokens")
    slots = record.get("slots")
    intent = record.get("intent")
    if not tokens or not isinstance(tokens, list):
        raise ValueError(f"line {line_no}: missing or empty 'tokens'")
    if not slots or not isinstance(slots, list):
        raise ValueError(f"line {line_no}: missing or empty 'slots'")
    if len(tokens) != len(slots):
        raise ValueError(
            f"line {line_no}: len(tokens)={len(tokens)} != len(slots)={len(slots)}"
        )
    if intent not in INTENT_TO_ID:
        raise ValueError(f"line {line_no}: unknown intent {intent!r}")
    for slot in slots:
        if slot not in SLOT_TO_ID:
            raise ValueError(f"line {line_no}: unknown slot {slot!r}")
    prev = "O"
    for slot in slots:
        if slot.startswith("I-"):
            want = slot[2:]
            if prev not in (f"B-{want}", f"I-{want}"):
                raise ValueError(f"line {line_no}: illegal BIO transition {prev} -> {slot}")
        prev = slot
    lang = record.get("language", None)
    if lang is not None and lang not in LANGUAGES:
        raise ValueError(f"line {line_no}: unknown language {lang!r}")
    for key in ("id", "group_id", "source"):
        if key in record and not isinstance(record[key], str):
            raise ValueError(f"line {line_no}: {key!r} must be a string")


def count_illegal_bio(labels):
    """Count I- tags not preceded by B-/I- of same type (before repair)."""
    n, prev = 0, "O"
    for lab in labels:
        if lab.startswith("I-"):
            want = lab[2:]
            if prev not in (f"B-{want}", f"I-{want}"):
                n += 1
        prev = lab
    return n


def decode_spans(labels, tokens=None):
    """BIO labels -> [(type, start, end_excl, text)]; illegal I- repaired as B-."""
    spans, cur, start = [], None, 0
    for i, lab in enumerate(list(labels) + ["O"]):
        tag = lab if lab in SLOT_TO_ID else "O"
        if tag.startswith("B-") or (tag.startswith("I-") and tag[2:] != cur):
            if cur is not None:
                words = (tokens or [])[start:i]
                spans.append((cur, start, i, " ".join(words) if words else ""))
            cur = tag[2:] if tag != "O" else None
            start = i
        elif tag == "O":
            if cur is not None:
                words = (tokens or [])[start:i]
                spans.append((cur, start, i, " ".join(words) if words else ""))
                cur = None
    return spans


def spans_from_ids(ids, tokens=None):
    return decode_spans([ID_TO_SLOT[int(i)] for i in ids], tokens)


def tokenize_and_align(records, tokenizer):
    encodings = tokenizer(
        [r["tokens"] for r in records],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    features = []
    for i in range(len(records)):
        enc = encodings.encodings[i]
        input_ids = encodings["input_ids"][i]
        attention_mask = encodings["attention_mask"][i]
        word_ids = enc.word_ids
        slot_labels = []
        prev_word = None
        for wid in word_ids:
            if wid is None:
                slot_labels.append(IGNORE_INDEX)
            elif wid != prev_word:
                slot_labels.append(SLOT_TO_ID[records[i]["slots"][wid]])
            else:
                slot_labels.append(IGNORE_INDEX)
            prev_word = wid
        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "intent_labels": INTENT_TO_ID[records[i]["intent"]],
                "slot_labels": slot_labels,
            }
        )
    ds = Dataset.from_list(features)
    return ds


class PadCollator:
    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, slot_labels, intent_labels = [], [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(list(f["input_ids"]) + [0] * pad)
            attention_mask.append(list(f["attention_mask"]) + [0] * pad)
            slot_labels.append(list(f["slot_labels"]) + [IGNORE_INDEX] * pad)
            intent_labels.append(f["intent_labels"])
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "intent_labels": torch.tensor(intent_labels),
            "slot_labels": torch.tensor(slot_labels),
        }


def load_split(data_dir, name, tokenizer):
    path = Path(data_dir) / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `uv run python -m src.split` first"
        )
    records = load_jsonl(path)
    for line_no, record in records:
        validate_record(record, line_no)
    ds = tokenize_and_align([r for _, r in records], tokenizer)
    ds.set_format("torch")
    return ds


def load_datasets(data_dir="data"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return (
        load_split(data_dir, "train", tokenizer),
        load_split(data_dir, "validation", tokenizer),
        load_split(data_dir, "test", tokenizer),
        tokenizer,
    )
