"""Generate deterministic Indonesian/English pairs and synthetic code-switches."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TypedDict, cast

from .config import INTENTS, SLOTS


SPLITS = ("train", "validation", "test")
CORE_KEYS = frozenset(("tokens", "intent", "slots"))
METADATA_KEYS = frozenset(("id", "language", "group_id", "source", "text"))
OUTPUT_KEYS = CORE_KEYS | METADATA_KEYS
SOURCE_NAME = "synthetic-template"
MIXED_SOURCE_NAME = "synthetic-code-switch"
MIXED_PREFIX_REPLACEMENTS = (
    (("saya", "mau", "lihat"), ("I", "want", "to", "lihat")),
    (("saya", "mau", "jumlah"), ("I", "need", "jumlah")),
    (("tolong", "tampilkan"), ("tolong", "show")),
    (("mohon", "tampilkan"), ("please", "tampilkan")),
    (("coba", "tampilkan"), ("coba", "show")),
    (("berapa", "banyak"), ("how", "many")),
    (("ada", "berapa"), ("ada", "how", "many")),
    (("berapa", "jumlah"), ("what's", "jumlah")),
    (("bagaimana", "status"), ("how's", "status")),
    (("detail", "untuk"), ("details", "untuk")),
    (("lihat",), ("check",)),
    (("cek",), ("inspect",)),
    (("tampilkan",), ("show",)),
    (("tunjukkan",), ("display",)),
    (("berikan",), ("give",)),
    (("munculkan",), ("bring", "up")),
    (("carikan",), ("find",)),
    (("hitung",), ("count",)),
    (("sebutkan",), ("list",)),
)
ORDER_DIRECTIONS = (
    ("ascending", ("rendah", "lowest", "buruk")),
    ("descending", ("tinggi", "highest", "baik")),
)

EXPECTED_INTENTS = (
    "GET_TOP_CELLS",
    "GET_CELL_DETAIL",
    "GET_CELL_COUNT",
    "UNKNOWN",
)
EXPECTED_SLOTS = (
    "O",
    "B-LIMIT",
    "B-CELL_ID",
    "B-LOCATION",
    "I-LOCATION",
    "B-METRIC",
    "I-METRIC",
    "B-ORDER",
    "I-ORDER",
)


class Record(TypedDict):
    tokens: list[str]
    intent: str
    slots: list[str]


class InputRecord(Record, total=False):
    id: str
    language: str
    group_id: str
    source: str
    text: str


class GeneratedRecord(Record):
    id: str
    language: str
    group_id: str
    source: str
    text: str


class GenerationError(ValueError):
    """A source or rendered record cannot be handled safely."""


@dataclass(frozen=True)
class SourceRecord:
    split: str
    index: int
    tokens: tuple[str, ...]
    intent: str
    slots: tuple[str, ...]


@dataclass(frozen=True)
class Span:
    kind: str
    tokens: tuple[str, ...]

    @property
    def value(self) -> str:
        return " ".join(self.tokens)


class EnglishBuilder:
    """Build English tokens and BIO labels without altering typed span values."""

    def __init__(self, split: str, index: int) -> None:
        self.split = split
        self.index = index
        self.tokens: list[str] = []
        self.slots: list[str] = []

    def literal(self, *tokens: str) -> None:
        for token in tokens:
            if not isinstance(token, str) or not token:
                raise self.error(f"invalid English template token {token!r}")
            self.tokens.append(token)
            self.slots.append("O")

    def span(self, span: Span, replacement: Iterable[str] | None = None) -> None:
        tokens = tuple(span.tokens if replacement is None else replacement)
        if not tokens or any(not isinstance(token, str) or not token for token in tokens):
            raise self.error(f"empty English replacement for {span.kind}")
        self.tokens.extend(tokens)
        self.slots.extend(
            f"{'B' if offset == 0 else 'I'}-{span.kind}"
            for offset in range(len(tokens))
        )

    def error(self, message: str) -> GenerationError:
        return GenerationError(f"{self.split} index {self.index}: {message}")

    def record(self, intent: str) -> Record:
        if not self.tokens or len(self.tokens) != len(self.slots):
            raise self.error("English token/slot mismatch")
        return {
            "tokens": self.tokens,
            "intent": intent,
            "slots": self.slots,
        }


UNKNOWN_TRANSLATIONS: dict[tuple[str, ...], tuple[str, ...]] = {
    ("berapa", "lama", "penerbangan", "ke", "bali"): (
        "how",
        "long",
        "is",
        "the",
        "flight",
        "to",
        "Bali",
    ),
    ("nyalakan", "lampu", "ruang", "tamu"): (
        "turn",
        "on",
        "the",
        "living",
        "room",
        "light",
    ),
    ("berapa", "hasil", "bagi", "144", "dengan", "12"): (
        "what",
        "is",
        "144",
        "divided",
        "by",
        "12",
    ),
    ("tampilkan", "berita", "terbaru"): ("show", "the", "latest", "news"),
    ("tampilkan", "wallpaper", "baru"): ("show", "a", "new", "wallpaper"),
    ("berapa", "target", "langkah", "harian", "saya"): (
        "what",
        "is",
        "my",
        "daily",
        "step",
        "goal",
    ),
    ("cara", "membuat", "kue", "coklat"): (
        "how",
        "do",
        "I",
        "make",
        "chocolate",
        "cake",
    ),
    ("kirim", "pesan", "ke", "budi"): ("send", "a", "message", "to", "Budi"),
    ("tampilkan", "jadwal", "kereta", "api"): (
        "show",
        "the",
        "train",
        "schedule",
    ),
    ("berapa", "biaya", "ganti", "oli", "motor"): (
        "how",
        "much",
        "does",
        "a",
        "motorcycle",
        "oil",
        "change",
        "cost",
    ),
    ("rekomendasi", "restoran", "terdekat"): (
        "recommend",
        "a",
        "nearby",
        "restaurant",
    ),
    ("terjemahkan", "kalimat", "ini", "ke", "inggris"): (
        "translate",
        "this",
        "sentence",
        "to",
        "English",
    ),
    ("berapa", "umur", "presiden", "pertama"): (
        "how",
        "old",
        "was",
        "the",
        "first",
        "president",
    ),
    ("apa", "kabar", "cuaca", "besok", "di", "surabaya"): (
        "what",
        "is",
        "the",
        "weather",
        "in",
        "Surabaya",
        "tomorrow",
    ),
    ("tolong", "ingatkan", "saya", "minum", "obat"): (
        "remind",
        "me",
        "to",
        "take",
        "my",
        "medicine",
    ),
    ("berapa", "berat", "ideal", "untuk", "tinggi", "170"): (
        "what",
        "is",
        "the",
        "ideal",
        "weight",
        "for",
        "height",
        "170",
    ),
    ("tampilkan", "foto", "keluarga", "saya"): (
        "show",
        "my",
        "family",
        "photos",
    ),
    ("berapa", "nilai", "tukar", "dolar", "hari", "ini"): (
        "what",
        "is",
        "the",
        "dollar",
        "exchange",
        "rate",
        "today",
    ),
    ("berapa", "kalori", "dalam", "nasi", "goreng"): (
        "how",
        "many",
        "calories",
        "are",
        "in",
        "fried",
        "rice",
    ),
    ("cara", "reset", "password", "akun", "saya"): (
        "how",
        "do",
        "I",
        "reset",
        "my",
        "account",
        "password",
    ),
    ("resep", "masakan", "ayam", "geprek"): (
        "recipe",
        "for",
        "ayam",
        "geprek",
    ),
    ("halo", "selamat", "pagi"): ("hello", "good", "morning"),
    ("berapa", "suhu", "di", "luar", "ruangan"): (
        "what",
        "is",
        "the",
        "temperature",
        "outside",
    ),
    ("tampilkan", "kalender", "bulan", "depan"): (
        "show",
        "the",
        "calendar",
        "for",
        "next",
        "month",
    ),
    ("apa", "itu", "kecerdasan", "buatan"): (
        "what",
        "is",
        "artificial",
        "intelligence",
    ),
    ("buat", "catatan", "belanja", "mingguan"): (
        "create",
        "a",
        "weekly",
        "shopping",
        "list",
    ),
    ("berapa", "jarak", "jakarta", "ke", "bandung"): (
        "how",
        "far",
        "is",
        "Jakarta",
        "from",
        "Bandung",
    ),
    ("apa", "itu", "machine", "learning"): (
        "what",
        "is",
        "machine",
        "learning",
    ),
    ("putar", "musik", "favorit", "saya"): (
        "play",
        "my",
        "favorite",
        "music",
    ),
    ("berapa", "hari", "lagi", "lebaran"): (
        "how",
        "many",
        "days",
        "until",
        "Eid",
    ),
    ("tampilkan", "rute", "paling", "cepat", "ke", "bandara"): (
        "show",
        "the",
        "fastest",
        "route",
        "to",
        "the",
        "airport",
    ),
    ("buatkan", "janji", "temu", "besok"): (
        "schedule",
        "an",
        "appointment",
        "for",
        "tomorrow",
    ),
    ("atur", "alarm", "jam", "5", "pagi"): (
        "set",
        "an",
        "alarm",
        "for",
        "5",
        "AM",
    ),
    ("alamat", "kantor", "pos", "terdekat"): (
        "nearest",
        "post",
        "office",
        "address",
    ),
    ("berapa", "harga", "paket", "internet"): (
        "how",
        "much",
        "does",
        "an",
        "internet",
        "package",
        "cost",
    ),
    ("buka", "kalkulator", "untuk", "saya"): (
        "open",
        "the",
        "calculator",
        "for",
        "me",
    ),
    ("tampilkan", "video", "lucu"): ("show", "a", "funny", "video"),
    ("berapa", "sisa", "kuota", "internet", "saya"): (
        "how",
        "much",
        "internet",
        "data",
        "do",
        "I",
        "have",
        "left",
    ),
    ("mainkan", "podcast", "teknologi"): (
        "play",
        "a",
        "technology",
        "podcast",
    ),
}


def _check_taxonomy() -> None:
    if tuple(INTENTS) != EXPECTED_INTENTS:
        raise RuntimeError("src.config.INTENTS no longer matches the fixed taxonomy")
    if tuple(SLOTS) != EXPECTED_SLOTS:
        raise RuntimeError("src.config.SLOTS no longer matches the fixed taxonomy")
    if len(UNKNOWN_TRANSLATIONS) != 39:
        raise RuntimeError("UNKNOWN translation table must contain exactly 39 entries")
    for source_tokens, english_tokens in UNKNOWN_TRANSLATIONS.items():
        if not source_tokens or not english_tokens:
            raise RuntimeError("UNKNOWN translation table contains an empty entry")
        if any(not isinstance(token, str) or not token for token in source_tokens + english_tokens):
            raise RuntimeError("UNKNOWN translation table contains an invalid token")
        if source_tokens == english_tokens:
            raise RuntimeError("UNKNOWN translation table copies an Indonesian entry")


def _normalize_order(text: str) -> str | None:
    """Mirror the finite-state order semantics used by ``src.dataset``."""
    lowered = text.lower()
    for direction, patterns in ORDER_DIRECTIONS:
        if any(pattern in lowered for pattern in patterns):
            return direction
    return None


def _error(split: str, index: int, message: str) -> GenerationError:
    return GenerationError(f"{split} index {index}: {message}")


def _validate_record_shape(record: object, split: str, index: int) -> None:
    if not isinstance(record, dict):
        raise _error(split, index, "record must be a JSON object")
    keys = set(record)
    if not CORE_KEYS.issubset(keys):
        missing = sorted(CORE_KEYS - keys)
        raise _error(split, index, f"missing source fields: {missing}")
    unexpected = sorted(keys - OUTPUT_KEYS)
    if unexpected:
        raise _error(split, index, f"unexpected fields: {unexpected}")
    tokens = record["tokens"]
    slots = record["slots"]
    intent = record["intent"]
    if not isinstance(tokens, list) or not tokens:
        raise _error(split, index, "missing or empty 'tokens'")
    if not isinstance(slots, list) or not slots:
        raise _error(split, index, "missing or empty 'slots'")
    if len(tokens) != len(slots):
        raise _error(split, index, f"len(tokens)={len(tokens)} != len(slots)={len(slots)}")
    if intent not in INTENTS:
        raise _error(split, index, f"unknown intent {intent!r}")
    if any(not isinstance(token, str) or not token for token in tokens):
        raise _error(split, index, "tokens must be non-empty strings")
    if any(not isinstance(slot, str) or not slot for slot in slots):
        raise _error(split, index, "slots must be non-empty strings")
    previous = "O"
    for slot in slots:
        if slot not in SLOTS:
            raise _error(split, index, f"unknown slot {slot!r}")
        if slot.startswith("I-"):
            kind = slot[2:]
            if previous not in (f"B-{kind}", f"I-{kind}"):
                raise _error(split, index, f"illegal BIO transition {previous} -> {slot}")
        previous = slot


def _read_jsonl(path: Path, split: str) -> list[InputRecord]:
    try:
        contents = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GenerationError(f"{split}: cannot read {path}: {exc}") from exc
    records: list[InputRecord] = []
    for line_no, line in enumerate(contents.splitlines(), 1):
        if not line.strip():
            raise _error(split, line_no - 1, "blank JSONL line")
        try:
            record: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _error(split, line_no - 1, f"invalid JSON: {exc.msg}") from exc
        _validate_record_shape(record, split, line_no - 1)
        records.append(cast(InputRecord, record))
    if not records:
        raise GenerationError(f"{split}: JSONL file contains no records")
    return records


def _select_id_source(records: list[InputRecord], split: str) -> list[InputRecord]:
    metadata_presence = [set(record) - CORE_KEYS for record in records]
    if not any(metadata_presence):
        return records
    if any(presence != METADATA_KEYS for presence in metadata_presence):
        for index, presence in enumerate(metadata_presence):
            if presence != METADATA_KEYS:
                raise _error(
                    split,
                    index,
                    "partial bilingual metadata; expected either legacy source fields "
                    "or all generated metadata",
                )
    for index, record in enumerate(records):
        language = record.get("language")
        if language not in ("id", "en", "mixed"):
            raise _error(
                split,
                index,
                f"generated language must be 'id', 'en', or 'mixed', got {language!r}",
            )
        for field in METADATA_KEYS:
            if not isinstance(record.get(field), str):
                raise _error(split, index, f"generated field {field!r} must be a string")
    sources = [record for record in records if record.get("language") == "id"]
    if not sources:
        raise GenerationError(f"{split}: bilingual data has no language=id source records")
    return sources


def _source_record(record: Record, split: str, index: int) -> SourceRecord:
    return SourceRecord(
        split=split,
        index=index,
        tokens=tuple(record["tokens"]),
        intent=record["intent"],
        slots=tuple(record["slots"]),
    )


def _decode_spans(source: SourceRecord) -> tuple[Span, ...]:
    spans: list[Span] = []
    active_kind: str | None = None
    active_tokens: list[str] = []

    def close() -> None:
        nonlocal active_kind, active_tokens
        if active_kind is not None:
            if not active_tokens:
                raise _error(source.split, source.index, f"empty {active_kind} span")
            spans.append(Span(active_kind, tuple(active_tokens)))
        active_kind = None
        active_tokens = []

    for token, tag in zip(source.tokens, source.slots):
        if tag == "O":
            close()
            continue
        prefix, kind = tag.split("-", 1)
        if prefix == "B":
            close()
            active_kind = kind
            active_tokens = [token]
        elif prefix == "I":
            if active_kind != kind:
                raise _error(
                    source.split,
                    source.index,
                    f"illegal BIO transition into I-{kind}",
                )
            active_tokens.append(token)
        else:
            raise _error(source.split, source.index, f"unhandled BIO tag {tag!r}")
    close()
    return tuple(spans)


def _spans_by_kind(source: SourceRecord, spans: tuple[Span, ...]) -> dict[str, Span]:
    found: dict[str, Span] = {}
    for span in spans:
        if span.kind in found:
            raise _error(source.split, source.index, f"duplicate {span.kind} span")
        found[span.kind] = span

    expected: dict[str, tuple[set[str], set[str]]] = {
        "GET_TOP_CELLS": ({"LIMIT", "METRIC", "ORDER", "LOCATION"}, set()),
        "GET_CELL_DETAIL": ({"CELL_ID"}, {"LOCATION", "METRIC"}),
        "GET_CELL_COUNT": ({"LOCATION"}, {"METRIC"}),
        "UNKNOWN": (set(), set()),
    }
    required, optional = expected[source.intent]
    allowed = required | optional
    missing = sorted(required - found.keys())
    extra = sorted(found.keys() - allowed)
    if missing:
        raise _error(source.split, source.index, f"missing required spans: {missing}")
    if extra:
        raise _error(source.split, source.index, f"unhandled spans for {source.intent}: {extra}")
    if source.intent == "UNKNOWN" and any(tag != "O" for tag in source.slots):
        raise _error(source.split, source.index, "UNKNOWN records must use only O slots")
    if "LIMIT" in found and (
        len(found["LIMIT"].tokens) != 1 or not found["LIMIT"].tokens[0].isdigit()
    ):
        raise _error(source.split, source.index, "LIMIT must be one numeric token")
    if "CELL_ID" in found and len(found["CELL_ID"].tokens) != 1:
        raise _error(source.split, source.index, "CELL_ID must be one token")
    return found


def _render_top_cells(source: SourceRecord, spans: dict[str, Span]) -> Record:
    builder = EnglishBuilder(source.split, source.index)
    order = _normalize_order(spans["ORDER"].value)
    if order == "ascending":
        order_word = "lowest"
    elif order == "descending":
        order_word = "highest"
    else:
        raise _error(source.split, source.index, f"translation miss for ORDER {spans['ORDER'].value!r}")
    builder.literal("show")
    builder.span(spans["LIMIT"])
    builder.literal("cells", "with")
    builder.span(spans["METRIC"])
    builder.literal("ranked")
    builder.span(spans["ORDER"], (order_word,))
    builder.literal("in")
    builder.span(spans["LOCATION"])
    return builder.record(source.intent)


def _render_cell_detail(source: SourceRecord, spans: dict[str, Span]) -> Record:
    builder = EnglishBuilder(source.split, source.index)
    if "METRIC" in spans:
        builder.span(spans["METRIC"])
        builder.literal("details", "for", "cell")
    else:
        builder.literal("show", "details", "for", "cell")
    builder.span(spans["CELL_ID"])
    if "LOCATION" in spans:
        builder.literal("in")
        builder.span(spans["LOCATION"])
    return builder.record(source.intent)


def _render_cell_count(source: SourceRecord, spans: dict[str, Span]) -> Record:
    builder = EnglishBuilder(source.split, source.index)
    builder.literal("how", "many", "cells")
    if "METRIC" in spans:
        builder.literal("with", "poor")
        builder.span(spans["METRIC"])
    builder.literal("are", "in")
    builder.span(spans["LOCATION"])
    return builder.record(source.intent)


def _render_unknown(source: SourceRecord) -> Record:
    english_tokens = UNKNOWN_TRANSLATIONS.get(source.tokens)
    if english_tokens is None:
        raise _error(source.split, source.index, f"translation miss for UNKNOWN {source.tokens!r}")
    builder = EnglishBuilder(source.split, source.index)
    builder.literal(*english_tokens)
    return builder.record(source.intent)


def _render_mixed(record: Record, split: str, index: int) -> GeneratedRecord:
    source = _source_record(record, split, index)
    if source.intent == "UNKNOWN":
        tokens = ["please", *source.tokens]
        slots = ["O", *source.slots]
    else:
        for prefix, replacement in MIXED_PREFIX_REPLACEMENTS:
            width = len(prefix)
            if source.tokens[:width] == prefix and set(source.slots[:width]) == {"O"}:
                tokens = [*replacement, *source.tokens[width:]]
                slots = [*("O" for _ in replacement), *source.slots[width:]]
                break
        else:
            raise _error(split, index, f"mixed prefix miss for {source.tokens!r}")

    group_id = _group_id(source)
    mixed_record = _with_metadata(
        source,
        "mixed",
        group_id,
        tokens,
        slots,
        source_name=MIXED_SOURCE_NAME,
    )
    _validate_generated(
        mixed_record,
        source,
        "mixed",
        group_id,
        source_name=MIXED_SOURCE_NAME,
    )
    return mixed_record


def _canonical_source(source: SourceRecord) -> str:
    return json.dumps(
        {
            "intent": source.intent,
            "slots": list(source.slots),
            "tokens": list(source.tokens),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _group_id(source: SourceRecord) -> str:
    return hashlib.sha256(_canonical_source(source).encode("utf-8")).hexdigest()


def _with_metadata(
    source: SourceRecord,
    language: str,
    group_id: str,
    tokens: Iterable[str],
    slots: Iterable[str],
    source_name: str = SOURCE_NAME,
) -> GeneratedRecord:
    token_list = list(tokens)
    slot_list = list(slots)
    if len(token_list) != len(slot_list) or not token_list:
        raise _error(source.split, source.index, f"{language} token/slot mismatch")
    return {
        "tokens": token_list,
        "intent": source.intent,
        "slots": slot_list,
        "id": f"{group_id}-{language}",
        "language": language,
        "group_id": group_id,
        "source": source_name,
        "text": " ".join(token_list),
    }


def _validate_generated(
    record: GeneratedRecord,
    source: SourceRecord,
    language: str,
    group_id: str,
    source_name: str = SOURCE_NAME,
) -> None:
    _validate_record_shape(record, source.split, source.index)
    if set(record) != OUTPUT_KEYS:
        raise _error(source.split, source.index, f"{language} output fields are not canonical")
    if record["language"] != language:
        raise _error(source.split, source.index, f"wrong language metadata for {language}")
    if record["source"] != source_name:
        raise _error(source.split, source.index, "wrong source metadata")
    if record["group_id"] != group_id:
        raise _error(source.split, source.index, "wrong group_id metadata")
    if record["id"] != f"{group_id}-{language}":
        raise _error(source.split, source.index, f"ID must end in -{language}")
    if record["text"] != " ".join(record["tokens"]):
        raise _error(source.split, source.index, "text does not match tokens")


def _render_pair(record: Record, split: str, index: int) -> tuple[GeneratedRecord, GeneratedRecord]:
    source = _source_record(record, split, index)
    spans = _decode_spans(source)
    by_kind = _spans_by_kind(source, spans)
    if source.intent == "UNKNOWN":
        english = _render_unknown(source)
    elif source.intent == "GET_TOP_CELLS":
        english = _render_top_cells(source, by_kind)
    elif source.intent == "GET_CELL_DETAIL":
        english = _render_cell_detail(source, by_kind)
    elif source.intent == "GET_CELL_COUNT":
        english = _render_cell_count(source, by_kind)
    else:
        raise _error(split, index, f"unhandled intent {source.intent!r}")
    if english["tokens"] == list(source.tokens):
        raise _error(split, index, "English renderer copied Indonesian tokens")
    if len(english["tokens"]) != len(english["slots"]):
        raise _error(split, index, "English token/slot mismatch")
    group_id = _group_id(source)
    id_record = _with_metadata(source, "id", group_id, source.tokens, source.slots)
    en_record = _with_metadata(source, "en", group_id, english["tokens"], english["slots"])
    _validate_generated(id_record, source, "id", group_id)
    _validate_generated(en_record, source, "en", group_id)
    return id_record, en_record


def _jsonl(records: Iterable[GeneratedRecord]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )


def generate(data_dir: Path, check: bool = False) -> dict[str, Counter[tuple[str, str]]]:
    _check_taxonomy()
    generated: dict[str, str] = {}
    counts: dict[str, Counter[tuple[str, str]]] = {}
    groups: dict[str, str] = {}
    ids: dict[str, str] = {}
    unknown_sources: set[tuple[str, ...]] = set()

    for split in SPLITS:
        path = data_dir / f"{split}.jsonl"
        records = _read_jsonl(path, split)
        sources = _select_id_source(records, split)
        output: list[GeneratedRecord] = []
        split_counts: Counter[tuple[str, str]] = Counter()
        for source_index, record in enumerate(sources):
            source = _source_record(record, split, source_index)
            if source.intent == "UNKNOWN":
                unknown_sources.add(source.tokens)
            pair = _render_pair(record, split, source_index)
            group_id = pair[0]["group_id"]
            if group_id in groups:
                raise _error(
                    split,
                    source_index,
                    f"group_id crosses or duplicates split {groups[group_id]!r}",
                )
            groups[group_id] = split
            mixed = _render_mixed(record, split, source_index)
            for language, rendered in (("id", pair[0]), ("en", pair[1]), ("mixed", mixed)):
                record_id = rendered["id"]
                if record_id in ids:
                    raise _error(split, source_index, f"duplicate ID also used by {ids[record_id]!r}")
                ids[record_id] = split
                output.append(rendered)
                split_counts[(language, rendered["intent"])] += 1
        generated[split] = _jsonl(output)
        counts[split] = split_counts

    missing_unknown = sorted(set(UNKNOWN_TRANSLATIONS) - unknown_sources)
    extra_unknown = sorted(unknown_sources - set(UNKNOWN_TRANSLATIONS))
    if missing_unknown or extra_unknown:
        details = []
        if missing_unknown:
            details.append(f"table entries absent from data: {missing_unknown}")
        if extra_unknown:
            details.append(f"data entries absent from table: {extra_unknown}")
        raise GenerationError("UNKNOWN translation table/data mismatch; " + "; ".join(details))

    for split in SPLITS:
        path = data_dir / f"{split}.jsonl"
        if check:
            try:
                current = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise GenerationError(f"{split}: cannot read {path}: {exc}") from exc
            if current != generated[split]:
                raise GenerationError(f"{split}: generated JSONL drift under --check")
        else:
            try:
                path.write_text(generated[split], encoding="utf-8")
            except OSError as exc:
                raise GenerationError(f"{split}: cannot write {path}: {exc}") from exc
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--check", action="store_true", help="fail if generated files would change")
    args = parser.parse_args(argv)
    try:
        counts = generate(args.data_dir, check=args.check)
    except (GenerationError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    mode = "checked" if args.check else "generated"
    for split in SPLITS:
        language_counts = Counter()
        for (language, _), count in counts[split].items():
            language_counts[language] += count
        summary = " + ".join(
            f"{language_counts[language]} {language}" for language in ("id", "en", "mixed")
        )
        print(f"{split}: {summary} ({mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
