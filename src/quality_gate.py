"""RFC 001 per-seed model quality gate and release metadata helpers."""
from __future__ import annotations

import math
from typing import Any


def report_metrics(report: dict[str, Any]) -> dict[str, float]:
    return {
        "intent_macro_f1": report["intent"]["macro"]["f1"],
        "entity_micro_f1": report["entity_span"]["micro"]["f1"],
        "entity_macro_f1": report["entity_span"]["macro_f1"],
        "exact_command_accuracy": report["exact_command_accuracy"],
        "coverage": report["coverage"],
        "unknown_f1": report["unknown"]["f1"],
    }


def evaluate_reports(
    reports: dict[int, dict[str, Any]], policy: dict[str, Any], dataset_sha256: str
) -> dict[str, Any]:
    required = [policy["seed"]]
    failures: list[str] = []
    per_seed: dict[str, dict[str, float]] = {}
    for seed in required:
        report = reports.get(seed)
        if report is None:
            failures.append(f"seed {seed}: report missing")
            continue
        if report.get("status") != "ok":
            failures.append(f"seed {seed}: status is not ok")
            continue
        if report.get("dataset_sha256") != dataset_sha256:
            failures.append(f"seed {seed}: dataset identity mismatch")
        calibration = report.get("calibration")
        if not isinstance(calibration, dict) or calibration.get("method") != "maxprob-sweep-v1":
            failures.append(f"seed {seed}: validation calibration is not frozen")
        try:
            metrics = report_metrics(report)
        except (KeyError, TypeError):
            failures.append(f"seed {seed}: required metric missing")
            continue
        per_seed[str(seed)] = metrics
        floors = dict(policy["metrics"])
        if policy["ood"]["required"]:
            if not report.get("ood_held_out"):
                failures.append(f"seed {seed}: held-out OOD declaration missing")
            floors["unknown_f1"] = policy["ood"]["unknown_f1"]
        for name, floor in floors.items():
            value = metrics.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                failures.append(f"seed {seed}: {name} missing or non-finite")
            elif value < floor:
                failures.append(f"seed {seed}: {name} {value} < {floor}")
    return {"result": "passed" if not failures else "failed", "failures": failures, "per_seed": per_seed}


def build_release_metadata(
    *, version: str, dataset: dict[str, str], source: dict[str, str], policy: dict[str, Any],
    seeds: list[int], selected_seed: int, selection_value: float, metrics: dict[str, float],
    reports: list[str], files: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": "passed", "model": "multilingual-intent-slot", "version": version,
        "selected_seed": selected_seed,
        "selection": {"metric": "validation_exact_command_accuracy", "value": selection_value,
                      "tie_break": "lowest_seed"},
        "dataset": dataset, "source": source, "policy_version": policy["policy_version"],
        "seeds": seeds, "metrics": metrics, "reports": reports, "files": files,
        "gate": {"result": "passed", "reports": reports},
    }
