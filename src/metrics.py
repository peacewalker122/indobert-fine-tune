import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .config import IGNORE_INDEX, INTENT_TO_ID, INTENTS, SLOTS
from .dataset import ENTITY_TYPES


def _as_int(a):
    return np.asarray(a, dtype=int)


# Trainer hook (subword/mask proxy). True span-exact lives in evaluate.report.
# ponytail: mask-exact approximates span-exact; full word rows unavailable here.
def compute_metrics(eval_pred):
    intent_logits, slot_logits = eval_pred.predictions
    intent_labels, slot_labels = eval_pred.label_ids

    pred_intent = np.asarray(intent_logits).argmax(-1)
    gold_intent = np.asarray(intent_labels)

    pred_slot = np.asarray(slot_logits).argmax(-1)
    gold_slot = np.asarray(slot_labels)

    intent_accuracy = float((pred_intent == gold_intent).mean())
    _, _, intent_macro_f1, _ = precision_recall_fscore_support(
        gold_intent, pred_intent, average="macro",
        labels=list(range(len(INTENTS))), zero_division=0,
    )

    mask = gold_slot != IGNORE_INDEX
    flat_pred = pred_slot[mask]
    flat_gold = gold_slot[mask]
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_gold, flat_pred, average="macro", labels=list(range(len(SLOTS))), zero_division=0
    )

    slot_correct = (pred_slot == gold_slot) | ~mask
    exact_command_accuracy = float(
        ((pred_intent == gold_intent) & slot_correct.all(axis=1)).mean()
    )

    return {
        "intent_accuracy": intent_accuracy,
        "intent_macro_f1": float(intent_macro_f1),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "exact_command_accuracy": exact_command_accuracy,
    }


def intent_report(pred, gold):
    """One-vs-rest over full set (filtering to gold rows would drop FPs)."""
    pi, gi = _as_int(pred), _as_int(gold)
    labels = list(range(len(INTENTS)))
    p, r, f, _ = precision_recall_fscore_support(
        gi, pi, labels=labels, zero_division=0,
    )
    macro = precision_recall_fscore_support(
        gi, pi, average="macro", labels=labels, zero_division=0,
    )
    known = [i for n, i in INTENT_TO_ID.items() if n != "UNKNOWN"]
    known_macro = precision_recall_fscore_support(
        gi, pi, average="macro", labels=known, zero_division=0,
    )
    return {
        "accuracy": float((pi == gi).mean()) if len(gi) else 0.0,
        "macro": {"p": float(macro[0]), "r": float(macro[1]), "f1": float(macro[2])},
        "known_macro_f1": float(known_macro[2]),
        "per_intent": {
            INTENTS[i]: {"p": float(p[k]), "r": float(r[k]), "f1": float(f[k]),
                         "n": int((gi == i).sum())}
            for k, i in enumerate(labels)
        },
        "confusion": confusion_matrix(gi, pi, labels=labels).tolist(),
    }


def entity_span_report(pred_spans, gold_spans):
    """Exact-span (type,start,end) micro/macro/per-type."""
    types = ENTITY_TYPES
    agg = {t: [0, 0, 0] for t in types}  # tp, fp, fn
    for ps, gs in zip(pred_spans, gold_spans):
        pk = {(t, s, e) for t, s, e, _ in ps}
        gk = {(t, s, e) for t, s, e, _ in gs}
        for t, s, e in pk & gk:
            agg[t][0] += 1
        for t, s, e in pk - gk:
            agg[t][1] += 1
        for t, s, e in gk - pk:
            agg[t][2] += 1
    per, f1s = {}, []
    for t in types:
        tp, fp, fn = agg[t]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per[t] = {"p": p, "r": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}
        f1s.append(f)
    TP = sum(v[0] for v in agg.values())
    FP = sum(v[1] for v in agg.values())
    FN = sum(v[2] for v in agg.values())
    mp = TP / (TP + FP) if TP + FP else 0.0
    mr = TP / (TP + FN) if TP + FN else 0.0
    mf = 2 * mp * mr / (mp + mr) if mp + mr else 0.0
    return {
        "micro": {"p": mp, "r": mr, "f1": mf},
        "macro_f1": float(sum(f1s) / len(f1s)) if f1s else 0.0,
        "per_type": per,
    }


def exact_flags(pred_intent, gold_intent, pred_spans, gold_spans):
    out = []
    for i in range(len(gold_intent)):
        pk = {(t, s, e) for t, s, e, _ in pred_spans[i]}
        gk = {(t, s, e) for t, s, e, _ in gold_spans[i]}
        out.append(bool(pred_intent[i] == gold_intent[i] and pk == gk))
    return np.array(out, dtype=bool)


def bootstrap_ci(flags, n=2000, seed=42):
    """Paired bootstrap 95% CI for a mean over examples."""
    rng = np.random.default_rng(seed)
    flags = np.asarray(flags, dtype=float)
    if not len(flags):
        return (0.0, 0.0)
    means = [flags[rng.integers(0, len(flags), len(flags))].mean() for _ in range(n)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))
