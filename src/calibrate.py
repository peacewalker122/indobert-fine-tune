"""UNKNOWN calibration. Thresholds fit on validation ONLY, frozen for test.

Method: sweep max-probability thresholds, pick the highest selective
accuracy keeping coverage >= min_coverage. Default threshold 0.0 (accept
all) when no threshold helps — recorded explicitly, never silent.
"""
import json
from pathlib import Path

import numpy as np

METHOD = "maxprob-sweep-v1"


def select_threshold(maxprobs, correct, min_coverage=0.95):
    maxprobs = np.asarray(maxprobs, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    best = {"threshold": 0.0, "coverage": 1.0,
            "selective": float(correct.mean()) if len(correct) else 0.0}
    table = []
    for t in [round(x * 0.05, 2) for x in range(20)]:
        acc = maxprobs >= t
        cov = float(acc.mean()) if len(acc) else 0.0
        sel = float(correct[acc].mean()) if acc.sum() else 0.0
        table.append({"threshold": t, "coverage": cov, "selective": sel})
        if cov >= min_coverage and sel > best["selective"]:
            best = {"threshold": t, "coverage": cov, "selective": sel}
    return best, table


def apply_threshold(intent_ids, maxprobs, threshold, unknown_id):
    out = np.asarray(intent_ids).copy()
    out[np.asarray(maxprobs) < threshold] = unknown_id
    return out


def save(path, threshold, min_coverage=0.95):
    payload = {"method": METHOD, "threshold": float(threshold),
               "min_coverage": float(min_coverage)}
    Path(path).write_text(json.dumps(payload, indent=2))
    return payload


def load(path):
    return json.loads(Path(path).read_text())
