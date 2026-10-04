import pytest

from src.quality_gate import build_release_metadata, evaluate_reports


def _report(value=0.95):
    return {"status": "ok", "dataset_sha256": "a" * 64, "ood_held_out": True,
            "calibration": {"method": "maxprob-sweep-v1", "threshold": 0.5},
            "exact_command_accuracy": value, "coverage": value,
            "intent": {"macro": {"f1": value}},
            "entity_span": {"micro": {"f1": value}, "macro_f1": value},
            "unknown": {"f1": value}}


@pytest.fixture
def policy():
    return {"policy_version": "p1", "seed": 42,
            "metrics": {"intent_macro_f1": .9, "entity_micro_f1": .9,
                        "entity_macro_f1": .85, "exact_command_accuracy": .8, "coverage": .95},
            "ood": {"required": False, "unknown_f1": .8}}


def test_gate_requires_every_seed_and_raw_thresholds(policy):
    result = evaluate_reports({}, policy, "a" * 64)
    assert result["result"] == "failed"
    assert result["failures"] == ["seed 42: report missing"]
    result = evaluate_reports({42: _report(.9499)}, policy, "a" * 64)
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


def test_release_metadata_matches_serving_contract(policy):
    metadata = build_release_metadata(
        version="v1", dataset={"name": "d", "version": "1", "sha256": "a" * 64},
        source={"git_commit": "b" * 40, "trainer_image": "trainer:1", "base_revision": "c" * 40},
        policy=policy, seeds=[42], selected_seed=42, selection_value=.9,
        metrics={"exact_command_accuracy": .9}, reports=["report-42.json"],
        files={"model.safetensors": {"sha256": "d" * 64, "bytes": 1}},
    )
    assert metadata["selection"] == {"metric": "validation_exact_command_accuracy", "value": .9,
                                      "tie_break": "lowest_seed"}
    assert metadata["status"] == metadata["gate"]["result"] == "passed"
