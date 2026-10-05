import pytest
from src.quality_gate import build_release_metadata, evaluate_reports


def _report(value=0.95):
    return {
        "status": "ok",
        "split": "test",
        "dataset_sha256": "a" * 64,
        "ood_held_out": True,
        "calibration": {"method": "maxprob-sweep-v1", "threshold": 0.5},
        "exact_command_accuracy": value,
        "coverage": value,
        "intent": {"macro": {"f1": value}},
        "entity_span": {"micro": {"f1": value}, "macro_f1": value},
        "unknown": {"f1": value},
    }


@pytest.fixture
def policy():
    return {
        "policy_version": "p1",
        "seed": 42,
        "metrics": {
            "intent_macro_f1": 0.9,
            "entity_micro_f1": 0.9,
            "entity_macro_f1": 0.85,
            "exact_command_accuracy": 0.8,
            "coverage": 0.95,
        },
        "ood": {"required": False, "unknown_f1": 0.8},
    }


def test_gate_requires_every_seed_and_raw_thresholds(policy):
    result = evaluate_reports({}, policy, "a" * 64)
    assert result["result"] == "failed"
    assert result["failures"] == ["seed 42: report missing"]
    result = evaluate_reports({42: _report(0.9499)}, policy, "a" * 64)
    assert result["result"] == "failed"
    assert any("coverage 0.9499 < 0.95" in failure for failure in result["failures"])


def test_gate_rejects_dataset_mismatch_and_enforces_conditional_ood(policy):
    policy["ood"]["required"] = True
    report = _report()
    report["dataset_sha256"] = "b" * 64
    report["ood_held_out"] = False
    result = evaluate_reports({42: report}, policy, "a" * 64)
    assert result["result"] == "failed"
    assert len(result["failures"]) == 2


def test_held_out_ood_is_gated_even_when_presence_is_optional(policy):
    report = _report()
    report["unknown"]["f1"] = 0.1
    result = evaluate_reports({42: report}, policy, "a" * 64)
    assert result["result"] == "failed"
    assert "seed 42: unknown_f1 0.1 < 0.8" in result["failures"]


def test_gate_rejects_wrong_split_count_and_hash_and_smoke(policy):
    report = _report()
    report.update(
        split="validation",
        n=9,
        test_sha256="b" * 64,
        artifact_sha256="c" * 64,
        calibration_sha256="d" * 64,
    )
    result = evaluate_reports(
        {42: report},
        policy,
        "a" * 64,
        expected_test_sha256="e" * 64,
        expected_test_count=10,
        expected_artifact_sha256="f" * 64,
        expected_calibration_sha256="1" * 64,
        smoke=True,
    )
    assert result["result"] == "failed"
    assert any("split is not frozen test" in failure for failure in result["failures"])
    assert any("test checksum mismatch" in failure for failure in result["failures"])
    assert any("test count mismatch" in failure for failure in result["failures"])
    assert any("artifact identity mismatch" in failure for failure in result["failures"])
    assert any("calibration identity mismatch" in failure for failure in result["failures"])
    assert any("smoke execution" in failure for failure in result["failures"])


def test_gate_accepts_exact_floor_boundaries(policy):
    report = _report()
    report["intent"]["macro"]["f1"] = 0.9
    report["entity_span"]["micro"]["f1"] = 0.9
    report["entity_span"]["macro_f1"] = 0.85
    report["exact_command_accuracy"] = 0.8
    report["coverage"] = 0.95
    report["unknown"]["f1"] = 0.8
    assert evaluate_reports({42: report}, policy, "a" * 64)["result"] == "passed"


def test_release_metadata_matches_serving_contract(policy):
    metadata = build_release_metadata(
        version="v1",
        dataset={"name": "d", "version": "1", "sha256": "a" * 64},
        source={"git_commit": "b" * 40, "trainer_image": "trainer:1", "base_revision": "c" * 40},
        policy=policy,
        seeds=[42],
        selected_seed=42,
        selection_value=0.9,
        metrics={"exact_command_accuracy": 0.9},
        reports=["report-42.json"],
        files={"model.safetensors": {"sha256": "d" * 64, "bytes": 1}},
    )
    assert metadata["selection"] == {
        "metric": "validation_exact_command_accuracy",
        "value": 0.9,
        "tie_break": "lowest_seed",
    }
    assert metadata["status"] == metadata["gate"]["result"] == "passed"
