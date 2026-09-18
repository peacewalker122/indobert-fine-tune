import hashlib
import json
from pathlib import Path

import pytest

from src.dataset import validate_record
from src.import_label_studio import (
    LabelStudioValidationError,
    convert_file,
    convert_tasks,
)
from src.prepare_label_studio_tasks import prepare_tasks, seed_tasks


TEXT = "show 10 cells with throughput lowest in Jakarta"


def _span(label, selected):
    start = TEXT.index(selected)
    return {
        "type": "labels",
        "from_name": "slot",
        "to_name": "text",
        "value": {
            "start": start,
            "end": start + len(selected),
            "text": selected,
            "labels": [label],
        },
    }


def _intent(name="GET_TOP_CELLS"):
    return {
        "type": "choices",
        "from_name": "intent",
        "to_name": "text",
        "value": {"choices": [name]},
    }


def _task(results, *, task_id="case-1", case_id="human-001"):
    annotation = {"id": f"annotation-{task_id}", "result": results}
    return {
        "id": task_id,
        "data": {"text": TEXT, "case_id": case_id},
        "annotations": [annotation],
    }


def test_seeder_and_converter_happy_path_with_consensus(tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("show 10 cells\n\ncount cells\n", encoding="utf-8")
    task_path = tmp_path / "tasks.json"

    assert seed_tasks(prompts, task_path) == 2
    assert json.loads(task_path.read_text(encoding="utf-8")) == [
        {"data": {"text": "show 10 cells"}},
        {"data": {"text": "count cells"}},
    ]
    assert prepare_tasks(["  one prompt  "]) == [{"data": {"text": "one prompt"}}]

    results = [
        _intent(),
        _span("LIMIT", "10"),
        _span("METRIC", "throughput"),
        _span("ORDER", "lowest"),
        _span("LOCATION", "Jakarta"),
    ]
    task = _task(results)
    task["annotations"].append({"id": "annotation-2", "result": list(reversed(results))})
    records = convert_tasks([task])

    record = records[0]
    assert record["tokens"] == TEXT.split()
    assert record["slots"] == [
        "O",
        "B-LIMIT",
        "O",
        "O",
        "B-METRIC",
        "B-ORDER",
        "O",
        "B-LOCATION",
    ]
    assert record["intent"] == "GET_TOP_CELLS"
    assert record["language"] == "mixed"
    assert record["source"] == "human-label-studio"
    assert record["case_id"] == "human-001"
    canonical = json.dumps(
        {key: record[key] for key in ("intent", "slots", "tokens")},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected_group_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert record["group_id"] == expected_group_id
    assert record["id"] == f"{expected_group_id}-mixed"
    validate_record(record)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda task: task["annotations"][0]["result"].__setitem__(
                0, _span("LOCATION", "Jakarta")
            ),
            "missing intent",
        ),
        (
            lambda task: task["annotations"].append(
                {
                    "id": "annotation-2",
                    "result": [_intent("GET_CELL_COUNT"), _span("LOCATION", "Jakarta")],
                }
            ),
            "disagrees",
        ),
        (
            lambda task: task["annotations"][0]["result"].append(_span("NOPE", "10")),
            "unknown slot label",
        ),
        (
            lambda task: task["annotations"][0]["result"].append(
                {
                    "type": "labels",
                    "from_name": "slot",
                    "to_name": "text",
                    "value": {
                        "start": TEXT.index("10") + 1,
                        "end": TEXT.index("10") + 2,
                        "text": "0",
                        "labels": ["LIMIT"],
                    },
                }
            ),
            "token boundaries",
        ),
        (
            lambda task: task["annotations"][0]["result"].extend(
                [_span("LOCATION", "throughput lowest"), _span("METRIC", "throughput")]
            ),
            "overlapping",
        ),
    ],
)
def test_converter_rejects_malformed_or_non_consensus_exports(mutation, message):
    task = _task(
        [
            _intent(),
            _span("LIMIT", "10"),
            _span("METRIC", "throughput"),
            _span("ORDER", "lowest"),
            _span("LOCATION", "Jakarta"),
        ]
    )
    mutation(task)
    with pytest.raises(LabelStudioValidationError, match=message):
        convert_tasks([task])


def test_converter_refuses_to_overwrite_generated_splits():
    protected = Path(__file__).resolve().parents[1] / "data" / "train.jsonl"
    with pytest.raises(LabelStudioValidationError, match="refusing to overwrite"):
        convert_file(Path("missing-export.json"), protected)


def test_converter_rejects_intent_slot_contract_violations():
    with pytest.raises(LabelStudioValidationError, match="does not support"):
        convert_tasks([_task([_intent("UNKNOWN"), _span("LIMIT", "10")])])

    with pytest.raises(LabelStudioValidationError, match="missing required slot"):
        convert_tasks([_task([_intent(), _span("LIMIT", "10")])])


def test_converter_rejects_missing_annotations_with_task_id():
    task = _task([_intent()])
    task["annotations"] = []
    with pytest.raises(LabelStudioValidationError, match="case-1.*non-empty"):
        convert_tasks([task])


def test_converter_rejects_unknown_intent_and_unsupported_result_type():
    with pytest.raises(LabelStudioValidationError, match="unknown intent"):
        convert_tasks([_task([_intent("NOPE")])])

    task = _task([_intent()])
    task["annotations"][0]["result"].append({"type": "textarea", "value": {}})
    with pytest.raises(LabelStudioValidationError, match="unsupported result type"):
        convert_tasks([task])


def test_converter_rejects_multi_token_limit_for_existing_bio_contract():
    task = _task([_intent(), _span("LIMIT", "10 cells")])
    with pytest.raises(LabelStudioValidationError, match="LIMIT.*exactly one token"):
        convert_tasks([task])
