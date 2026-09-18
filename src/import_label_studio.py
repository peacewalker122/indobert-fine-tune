"""Convert validated Label Studio JSON exports into project JSONL records.

The importer treats the JSON export as untrusted input. It validates the
untyped JSON at the boundary, then converts it into typed dataclasses before
building BIO records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


INTENTS = (
    "GET_TOP_CELLS",
    "GET_CELL_DETAIL",
    "GET_CELL_COUNT",
    "UNKNOWN",
)
SLOT_LABELS = ("LIMIT", "CELL_ID", "LOCATION", "METRIC", "ORDER")
LANGUAGES = ("id", "en", "mixed")
DEFAULT_LANGUAGE = "mixed"
DEFAULT_SOURCE = "human-label-studio"
_PROTECTED_SPLITS = frozenset(("train.jsonl", "validation.jsonl", "test.jsonl"))

_INTENT_SET = frozenset(INTENTS)
_SLOT_SET = frozenset(SLOT_LABELS)
_LANGUAGE_SET = frozenset(LANGUAGES)
_SLOT_CONTRACT = {
    "GET_TOP_CELLS": (frozenset(("LIMIT", "METRIC", "ORDER", "LOCATION")), frozenset()),
    "GET_CELL_DETAIL": (frozenset(("CELL_ID",)), frozenset(("LOCATION", "METRIC"))),
    "GET_CELL_COUNT": (frozenset(("LOCATION",)), frozenset(("METRIC",))),
    "UNKNOWN": (frozenset(), frozenset()),
}
_WORD_PATTERN = re.compile(r"\S+")


class LabelStudioValidationError(ValueError):
    """A Label Studio export violates the annotation/data contract."""


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Span:
    label: str
    start: int
    end: int
    start_token: int
    end_token: int


@dataclass(frozen=True)
class ParsedAnnotation:
    intent: str
    spans: tuple[Span, ...]


def _object(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise LabelStudioValidationError(f"{location}: expected a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise LabelStudioValidationError(f"{location}: object keys must be strings")
    return {key: item for key, item in value.items()}


def _task_id(task: dict[str, object], index: int) -> str:
    raw_id = task.get("id")
    if raw_id is None:
        return f"index {index}"
    if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)):
        raise LabelStudioValidationError(
            f"task index {index}: 'id' must be a string or integer when present"
        )
    return str(raw_id)


def _required_string(
    obj: dict[str, object], key: str, location: str, *, allow_empty: bool = False
) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or (not allow_empty and not value):
        requirement = "a non-empty string" if not allow_empty else "a string"
        raise LabelStudioValidationError(f"{location}: '{key}' must be {requirement}")
    return value


def _optional_case_id(
    task: dict[str, object], data: dict[str, object], location: str
) -> str | None:
    if "case_id" in data:
        value = data["case_id"]
    elif "case_id" in task:
        value = task["case_id"]
    else:
        return None
    if not isinstance(value, str) or not value:
        raise LabelStudioValidationError(f"{location}: 'case_id' must be a non-empty string")
    return value


def _metadata_string(
    value: object, key: str, default: str, location: str, allowed: frozenset[str] | None = None
) -> str:
    result = default if value is None else value
    if not isinstance(result, str) or not result:
        raise LabelStudioValidationError(f"{location}: '{key}' must be a non-empty string")
    if allowed is not None and result not in allowed:
        choices = ", ".join(sorted(allowed))
        raise LabelStudioValidationError(
            f"{location}: unknown {key} {result!r}; expected one of {choices}"
        )
    return result


def _tokens(text: str) -> tuple[Token, ...]:
    tokens = tuple(
        Token(match.group(), match.start(), match.end())
        for match in _WORD_PATTERN.finditer(text)
    )
    if not tokens:
        raise LabelStudioValidationError("task text must contain at least one non-whitespace token")
    return tokens


def _integer(value: object, field: str, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LabelStudioValidationError(f"{location}: '{field}' must be an integer character offset")
    return value


def _check_control_names(
    result: dict[str, object], expected_from_name: str, location: str
) -> None:
    from_name = result.get("from_name")
    if from_name is not None and from_name != expected_from_name:
        raise LabelStudioValidationError(
            f"{location}: expected from_name {expected_from_name!r}, got {from_name!r}"
        )
    to_name = result.get("to_name")
    if to_name is not None and to_name != "text":
        raise LabelStudioValidationError(f"{location}: expected to_name 'text', got {to_name!r}")


def _parse_intent(value: dict[str, object], location: str) -> str:
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], str):
        raise LabelStudioValidationError(
            f"{location}: intent must contain exactly one string choice"
        )
    intent = choices[0]
    if intent not in _INTENT_SET:
        raise LabelStudioValidationError(
            f"{location}: unknown intent {intent!r}; expected one of {', '.join(INTENTS)}"
        )
    return intent


def _parse_span(
    value: dict[str, object], text: str, tokens: tuple[Token, ...], location: str
) -> Span:
    labels = value.get("labels")
    if not isinstance(labels, list) or len(labels) != 1 or not isinstance(labels[0], str):
        raise LabelStudioValidationError(
            f"{location}: slot result must contain exactly one string label"
        )
    label = labels[0]
    if label not in _SLOT_SET:
        raise LabelStudioValidationError(
            f"{location}: unknown slot label {label!r}; expected one of {', '.join(SLOT_LABELS)}"
        )

    start = _integer(value.get("start"), "start", location)
    end = _integer(value.get("end"), "end", location)
    if start < 0 or end > len(text) or start >= end:
        raise LabelStudioValidationError(
            f"{location}: invalid end-exclusive offsets start={start}, end={end} for text length {len(text)}"
        )

    selected_text = text[start:end]
    reported_text = value.get("text")
    if reported_text is not None and reported_text != selected_text:
        raise LabelStudioValidationError(
            f"{location}: value.text does not match text[start:end] ({reported_text!r} != {selected_text!r})"
        )

    starts = {token.start: index for index, token in enumerate(tokens)}
    ends = {token.end: index + 1 for index, token in enumerate(tokens)}
    if start not in starts or end not in ends:
        raise LabelStudioValidationError(
            f"{location}: span {start}:{end} is not aligned to complete whitespace-token boundaries"
        )
    start_token = starts[start]
    end_token = ends[end]
    if start_token >= end_token:
        raise LabelStudioValidationError(f"{location}: span does not cover a token")
    if label in {"LIMIT", "CELL_ID"} and end_token - start_token != 1:
        raise LabelStudioValidationError(
            f"{location}: {label} must cover exactly one token because the project BIO taxonomy has no I-{label}"
        )
    return Span(label, start, end, start_token, end_token)


def _validate_slot_contract(
    intent: str, spans: tuple[Span, ...], location: str
) -> None:
    required, optional = _SLOT_CONTRACT[intent]
    labels = [span.label for span in spans]
    present = set(labels)
    missing = sorted(required - present)
    unsupported = sorted(present - required - optional)
    duplicates = sorted(label for label in present if labels.count(label) > 1)
    if missing:
        raise LabelStudioValidationError(
            f"{location}: intent {intent} is missing required slot(s): {', '.join(missing)}"
        )
    if unsupported:
        raise LabelStudioValidationError(
            f"{location}: intent {intent} does not support slot(s): {', '.join(unsupported)}"
        )
    if duplicates:
        raise LabelStudioValidationError(
            f"{location}: slot type(s) may occur only once: {', '.join(duplicates)}"
        )


def _parse_annotation(
    annotation: object,
    text: str,
    tokens: tuple[Token, ...],
    task_id: str,
    annotation_index: int,
) -> ParsedAnnotation:
    location = f"task {task_id!r} annotation {annotation_index}"
    parsed = _object(annotation, location)
    raw_results = parsed.get("result")
    if not isinstance(raw_results, list) or not raw_results:
        raise LabelStudioValidationError(f"{location}: result must be a non-empty list")

    intent: str | None = None
    spans: list[Span] = []
    for result_index, raw_result in enumerate(raw_results):
        result_location = f"{location} result {result_index}"
        result = _object(raw_result, result_location)
        result_type = result.get("type")
        if result_type == "choices":
            value = _object(result.get("value"), result_location)
            _check_control_names(result, "intent", result_location)
            if intent is not None:
                raise LabelStudioValidationError(
                    f"{location}: multiple intent results are not allowed"
                )
            intent = _parse_intent(value, result_location)
        elif result_type == "labels":
            value = _object(result.get("value"), result_location)
            _check_control_names(result, "slot", result_location)
            spans.append(_parse_span(value, text, tokens, result_location))
        else:
            raise LabelStudioValidationError(
                f"{result_location}: unsupported result type {result_type!r}; expected 'choices' or 'labels'"
            )

    if intent is None:
        raise LabelStudioValidationError(f"{location}: missing intent choice")

    ordered_spans = tuple(sorted(spans, key=lambda span: (span.start, span.end, span.label)))
    for previous, current in zip(ordered_spans, ordered_spans[1:]):
        if current.start < previous.end:
            raise LabelStudioValidationError(
                f"{location}: overlapping slot spans {previous.start}:{previous.end} and {current.start}:{current.end}"
            )
    _validate_slot_contract(intent, ordered_spans, location)
    return ParsedAnnotation(intent, ordered_spans)


def _bio_labels(tokens: tuple[Token, ...], spans: tuple[Span, ...], task_id: str) -> tuple[str, ...]:
    labels = ["O"] * len(tokens)
    for span in spans:
        for token_index in range(span.start_token, span.end_token):
            if labels[token_index] != "O":
                raise LabelStudioValidationError(
                    f"task {task_id!r}: overlapping slot spans cannot produce BIO labels"
                )
            prefix = "B" if token_index == span.start_token else "I"
            labels[token_index] = f"{prefix}-{span.label}"
    return tuple(labels)


def _canonical(intent: str, slots: tuple[str, ...], tokens: tuple[Token, ...]) -> str:
    return json.dumps(
        {
            "intent": intent,
            "slots": list(slots),
            "tokens": [token.text for token in tokens],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _convert_task(
    raw_task: object,
    index: int,
    default_language: str,
    default_source: str,
) -> dict[str, object]:
    task = _object(raw_task, f"task index {index}")
    task_id = _task_id(task, index)
    location = f"task {task_id!r}"
    data = _object(task.get("data"), location)
    text = _required_string(data, "text", f"{location} data")
    if not text.strip():
        raise LabelStudioValidationError(f"{location}: data.text must contain non-whitespace text")
    tokens = _tokens(text)

    raw_annotations = task.get("annotations")
    if not isinstance(raw_annotations, list) or not raw_annotations:
        raise LabelStudioValidationError(f"{location}: annotations must be a non-empty list")
    annotations = tuple(
        _parse_annotation(annotation, text, tokens, task_id, annotation_index)
        for annotation_index, annotation in enumerate(raw_annotations)
    )
    consensus = annotations[0]
    for annotation_index, annotation in enumerate(annotations[1:], 1):
        if annotation != consensus:
            raise LabelStudioValidationError(
                f"{location}: annotation {annotation_index} disagrees with annotation 0; all annotations must reach consensus"
            )

    slots = _bio_labels(tokens, consensus.spans, task_id)
    language = _metadata_string(
        data.get("language"), "language", default_language, location, _LANGUAGE_SET
    )
    source = _metadata_string(data.get("source"), "source", default_source, location)
    group_id = hashlib.sha256(
        _canonical(consensus.intent, slots, tokens).encode("utf-8")
    ).hexdigest()
    record: dict[str, object] = {
        "tokens": [token.text for token in tokens],
        "intent": consensus.intent,
        "slots": list(slots),
        "id": f"{group_id}-{language}",
        "language": language,
        "group_id": group_id,
        "source": source,
        "text": text,
    }
    case_id = _optional_case_id(task, data, location)
    if case_id is not None:
        record["case_id"] = case_id
    return record


def convert_tasks(
    export: object,
    *,
    language: str = DEFAULT_LANGUAGE,
    source: str = DEFAULT_SOURCE,
) -> list[dict[str, object]]:
    """Validate and convert a Label Studio task-array export."""

    if not isinstance(language, str) or language not in _LANGUAGE_SET:
        raise LabelStudioValidationError(
            f"language {language!r} is unsupported; expected one of {', '.join(LANGUAGES)}"
        )
    if not isinstance(source, str) or not source:
        raise LabelStudioValidationError("source must be a non-empty string")
    if not isinstance(export, list):
        raise LabelStudioValidationError("Label Studio export must be a JSON array of tasks")
    if not export:
        raise LabelStudioValidationError("Label Studio export contains no tasks")
    return [
        _convert_task(task, index, language, source) for index, task in enumerate(export)
    ]


def _load_export(path: Path) -> object:
    try:
        contents = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LabelStudioValidationError(f"cannot read export {path}: {exc}") from exc
    try:
        return json.loads(contents)
    except json.JSONDecodeError as exc:
        raise LabelStudioValidationError(
            f"cannot parse export {path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc


def _ensure_safe_output(output_path: Path) -> None:
    resolved = output_path.resolve()
    project_data = Path(__file__).resolve().parents[1] / "data"
    protected = {
        (project_data / filename).resolve() for filename in _PROTECTED_SPLITS
    }
    if resolved in protected:
        raise LabelStudioValidationError(
            f"refusing to overwrite generated split {resolved}; use data/challenge/mixed_human.jsonl"
        )


def convert_file(
    input_path: Path,
    output_path: Path,
    *,
    language: str = DEFAULT_LANGUAGE,
    source: str = DEFAULT_SOURCE,
) -> int:
    """Convert one Label Studio export file into JSONL and return its row count."""

    _ensure_safe_output(output_path)
    records = convert_tasks(_load_export(input_path), language=language, source=source)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        raise LabelStudioValidationError(f"cannot write JSONL output {output_path}: {exc}") from exc
    return len(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a Label Studio JSON export into validated project JSONL."
    )
    parser.add_argument("export", type=Path, help="Label Studio JSON export")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/challenge/mixed_human.jsonl"),
        help="JSONL output path (default: data/challenge/mixed_human.jsonl)",
    )
    parser.add_argument("--language", choices=LANGUAGES, default=DEFAULT_LANGUAGE)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    args = parser.parse_args(argv)

    try:
        count = convert_file(
            args.export,
            args.output,
            language=args.language,
            source=args.source,
        )
    except LabelStudioValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {count} validated records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
