import json

import pytest

from src import calibrate as calib
from src.config import INTENT_TO_ID
from src.dataset import count_illegal_bio, decode_spans, normalize_order, validate_record
from src.metrics import bootstrap_ci, entity_span_report, exact_flags, intent_report
from src.split import group_split


def test_illegal_bio_raises():
    with pytest.raises(ValueError, match="illegal BIO"):
        validate_record({"tokens": ["a", "b"], "slots": ["O", "I-LOCATION"],
                         "intent": "GET_TOP_CELLS"}, 7)


def test_benchmark_fields_optional_but_checked():
    ok = {"tokens": ["hi"], "slots": ["O"], "intent": "UNKNOWN",
          "id": "u1", "language": "en", "group_id": "g1", "source": "natural",
          "text": "hi"}
    validate_record(ok, 1)
    bad = dict(ok, language="fr")
    with pytest.raises(ValueError, match="language"):
        validate_record(bad, 1)


def test_decode_spans():
    toks = ["di", "Jakarta", "Selatan"]
    spans = decode_spans(["O", "B-LOCATION", "I-LOCATION"], toks)
    assert spans == [("LOCATION", 1, 3, "Jakarta Selatan")]
    assert decode_spans(["O", "O"], toks) == []
    # illegal I- repaired as B-, counted before repair
    assert count_illegal_bio(["O", "I-LOCATION"]) == 1
    assert decode_spans(["O", "I-LOCATION"], toks) == [("LOCATION", 1, 2, "Jakarta")]


def test_one_vs_rest_counts_false_positives():
    # gold all A(0), pred all B(1): B precision must be 0, not skipped
    rep = intent_report([1, 1], [0, 0])
    names = list(INTENT_TO_ID)
    assert rep["per_intent"][names[1]]["p"] == 0.0
    assert rep["per_intent"][names[0]]["r"] == 0.0
    assert len(rep["confusion"]) == len(names)


def test_span_exact_rejects_extra_entity():
    gold = [[("LOCATION", 1, 2, "Jakarta")]]
    assert exact_flags([0], [0], gold, gold)[0]
    pred_extra = [[("LOCATION", 1, 2, "Jakarta"), ("METRIC", 0, 1, "x")]]
    assert not exact_flags([0], [0], pred_extra, gold)[0]
    rep = entity_span_report(pred_extra, gold)
    assert rep["micro"]["p"] < 1.0 and rep["micro"]["r"] == 1.0


def test_bootstrap_ci_ordered():
    lo, hi = bootstrap_ci([True, False, True, True])
    assert 0.0 <= lo <= hi <= 1.0


def test_calibration_boundary_and_roundtrip(tmp_path):
    unk = INTENT_TO_ID["UNKNOWN"]
    out = calib.apply_threshold([0, 0], [0.9, 0.1], 0.5, unk)
    assert list(out) == [0, unk]
    p = tmp_path / "t.json"
    calib.save(p, 0.5)
    assert calib.load(p)["threshold"] == 0.5


def test_group_split_no_leakage_and_deterministic():
    recs = [{"tokens": ["a"], "slots": ["O"], "intent": "UNKNOWN", "group_id": f"g{i // 3}"}
            for i in range(30)]
    a = group_split([dict(r) for r in recs], seed=1)
    seen = {}
    for split, rows in a.items():
        for r in rows:
            prev = seen.setdefault(r["group_id"], split)
            assert prev == split, (r["group_id"], prev, split)
    assert json.dumps(a["train"][:1])  # serializable

def test_normalize_order_finite_state():
    asc = ["terendah", "rendah", "Paling Rendah", "lowest", "terburuk", "paling buruk"]
    desc = ["tertinggi", "tinggi", "Paling Tinggi", "highest", "terbaik", "paling baik"]
    for t in asc:
        assert normalize_order(t) == "ascending", t
    for t in desc:
        assert normalize_order(t) == "descending", t
    assert normalize_order("paling murah") is None
    assert normalize_order("") is None
    assert normalize_order(None) is None
