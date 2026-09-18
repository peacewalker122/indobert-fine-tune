import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

from . import calibrate as calib
from .dataset import (
    PadCollator,
    count_illegal_bio,
    decode_spans,
    load_jsonl,
    normalize_order,
    tokenize_and_align,
    validate_record,
)
from .config import (
    ARTIFACT_DIR,
    ARTIFACT_VERSION,
    ID_TO_SLOT,
    IGNORE_INDEX,
    INTENT_TO_ID,
)
from .metrics import bootstrap_ci, entity_span_report, exact_flags, intent_report
from .split import MANIFEST


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_artifact(model_dir):
    model_dir = Path(model_dir)
    cfg = AutoConfig.from_pretrained(model_dir)
    encoder = AutoModel.from_config(cfg)
    from .model import IntentSlotModel

    model = IntentSlotModel(encoder)
    state = load_file(str(model_dir / "model.safetensors"))
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict(model, records, tokenizer, batch_size=32):
    ds = tokenize_and_align([r for _, r in records], tokenizer)

    collator = PadCollator()
    P, G, PS, GS, MP, MG = [], [], [], [], [], []
    for i in range(0, len(ds), batch_size):
        batch = collator([ds[j] for j in range(i, min(i + batch_size, len(ds)))])
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        probs = torch.softmax(out["intent_logits"], dim=-1)
        top2 = probs.topk(2, dim=-1)
        pred_intent = out["intent_logits"].argmax(-1)
        pred_slot = out["slot_logits"].argmax(-1)
        for b in range(len(pred_intent)):
            mask = batch["slot_labels"][b] != IGNORE_INDEX
            PS.append(pred_slot[b][mask].tolist())
            GS.append(batch["slot_labels"][b][mask].tolist())
        P.append(pred_intent.numpy())
        G.append(batch["intent_labels"].numpy())
        MP.append(top2.values[:, 0].numpy())
        MG.append((top2.values[:, 0] - top2.values[:, 1]).numpy())
    return (np.concatenate(P), PS, np.concatenate(G), GS,
            np.concatenate(MP), np.concatenate(MG))


def _spans(rows, recs):
    return [decode_spans([ID_TO_SLOT[i] for i in row], r["tokens"]) for row, (_, r) in zip(rows, recs)]


def report(model_dir, data_dir="data", thresholds=None, output=None,
           fit_calibration=None, min_coverage=0.95, split="test"):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        # B0 rule: missing artifact reported, never silently replaced.
        print(f"artifact {model_dir} unavailable — reported as unavailable, not replaced")
        res = {"status": "unavailable", "artifact": str(model_dir)}
        if output:
            Path(output).write_text(json.dumps(res, indent=2))
        return res

    records = load_jsonl(Path(data_dir) / f"{split}.jsonl")
    for line_no, record in records:
        validate_record(record, line_no)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = load_artifact(model_dir)
    pi, ps_rows, gi, gs_rows, maxprobs, margins = predict(model, records, tokenizer)

    unknown_id = INTENT_TO_ID["UNKNOWN"]
    threshold = 0.0
    method = "none"
    if thresholds:
        t = calib.load(thresholds)
        threshold, method = float(t["threshold"]), t.get("method", "file")
        pi = calib.apply_threshold(pi, maxprobs, threshold, unknown_id)
    if fit_calibration:
        ps = _spans(ps_rows, records)
        gs = _spans(gs_rows, records)
        flags = exact_flags(pi, gi, ps, gs)
        best, _ = calib.select_threshold(maxprobs, flags, min_coverage)
        calib.save(fit_calibration, best["threshold"], min_coverage)
        print(f"calibration saved to {fit_calibration}: {best}")

    pred_spans = _spans(ps_rows, records)
    gold_spans = _spans(gs_rows, records)
    flags = exact_flags(pi, gi, pred_spans, gold_spans)
    lo, hi = bootstrap_ci(flags)

    langs = [r.get("language", "unlabeled") for _, r in records]
    by_lang = {}
    for lang in sorted(set(langs)):
        sel = np.array([l == lang for l in langs])
        by_lang[lang] = {
            "n": int(sel.sum()),
            "exact": float(flags[sel].mean()) if sel.sum() else 0.0,
        }
    by_intent = {}
    for name, iid in INTENT_TO_ID.items():
        sel = gi == iid
        by_intent[name] = {
            "n": int(sel.sum()),
            "exact": float(flags[sel].mean()) if sel.sum() else 0.0,
        }

    intent = intent_report(pi, gi)
    entity = entity_span_report(pred_spans, gold_spans)
    unk = intent["per_intent"]["UNKNOWN"]
    gold_unk = gi == unknown_id
    pred_unk = pi == unknown_id
    # false-accept: gold UNKNOWN predicted known; false-reject: gold known predicted UNKNOWN
    ood_far = float(((gold_unk) & (~pred_unk)).sum() / max(gold_unk.sum(), 1)) if gold_unk.sum() else 0.0
    known_fr = float(((~gold_unk) & (pred_unk)).sum() / max((~gold_unk).sum(), 1))
    coverage = float((~pred_unk).mean())
    accepted = ~pred_unk
    selective = float(flags[accepted].mean()) if accepted.sum() else 0.0

    def order_text(spans):
        return " ".join(t for ty, _, _, t in spans if ty == "ORDER")

    dir_ok, dir_n, dir_unmapped_gold = 0, 0, 0
    for ps, gs in zip(pred_spans, gold_spans):
        g, p = normalize_order(order_text(gs)), normalize_order(order_text(ps))
        if order_text(gs):
            dir_n += 1
            dir_unmapped_gold += g is None
        dir_ok += p == g
    direction = {"accuracy": dir_ok / len(records) if records else 0.0,
                 "n_gold": dir_n,
                 "unmapped_gold": dir_unmapped_gold}
    test_path = Path(data_dir) / f"{split}.jsonl"
    man_path = Path(data_dir) / MANIFEST
    res = {
        "status": "ok",
        "artifact": str(model_dir),
        "split": split,
        "n": len(records),
        "test_sha256": sha256_file(test_path) if test_path.exists() else None,
        "manifest": json.loads(man_path.read_text()) if man_path.exists() else None,
        "calibration": {"method": method, "threshold": threshold},
        "exact_command_accuracy": float(flags.mean()) if len(flags) else 0.0,
        "exact_ci95": [lo, hi],
        "by_language": by_lang,
        "by_intent": by_intent,
        "intent": intent,
        "entity_span": entity,
        "unknown": {**unk, "false_accept_rate": ood_far, "known_false_reject_rate": known_fr},
        "coverage": coverage,
        "selective_accuracy": selective,
        "order_direction": direction,
        "margins": {"mean": float(margins.mean()), "p50": float(np.median(margins))},
        "illegal_bio_pred": int(sum(count_illegal_bio([ID_TO_SLOT[i] for i in r]) for r in ps_rows)),
        "misclassified": [
            r.get("id", f"line-{ln}") for f, ln, r in
            ((ok, ln, r) for ok, (ln, r) in zip(flags, records)) if not f
        ][:200],
    }

    print(f"{split} set: {len(records)} examples")
    print(f"exact command accuracy: {res['exact_command_accuracy']:.4f} "
          f"(95% CI {lo:.4f}-{hi:.4f})")
    print(f"intent macro F1: {intent['macro']['f1']:.4f} "
          f"(known-only {intent['known_macro_f1']:.4f}, acc {intent['accuracy']:.4f})")
    print(f"entity span micro/macro F1: {entity['micro']['f1']:.4f} / {entity['macro_f1']:.4f}")
    print(f"UNKNOWN F1: {unk['f1']:.3f} (n={unk['n']}) "
          f"false-accept {ood_far:.3f} known-false-reject {known_fr:.3f}")
    print(f"coverage {coverage:.3f} selective {selective:.3f} "
          f"threshold {threshold} ({method})")
    print(f"order direction acc {direction['accuracy']:.4f} "
          f"(gold {direction['n_gold']}, unmapped {direction['unmapped_gold']})")
    for lang, s in by_lang.items():
        print(f"  lang {lang}: exact {s['exact']:.4f} (n={s['n']})")

    if output:
        Path(output).write_text(json.dumps(res, indent=2))
        print(f"report written to {output}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=str(ARTIFACT_DIR / ARTIFACT_VERSION))
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--fit-calibration", default=None)
    ap.add_argument("--min-coverage", type=float, default=0.95)
    args = ap.parse_args()
    report(args.artifact, args.data_dir, args.thresholds, args.output,
           args.fit_calibration, args.min_coverage, args.split)
