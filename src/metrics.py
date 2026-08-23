import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from .config import IGNORE_INDEX, SLOTS


# ponytail: assumes BIO well-formedness in gold (generator enforces it);
# upgrade path = continuity check in validate_record
def compute_metrics(eval_pred):
    intent_logits, slot_logits = eval_pred.predictions
    intent_labels, slot_labels = eval_pred.label_ids

    pred_intent = np.asarray(intent_logits).argmax(-1)
    gold_intent = np.asarray(intent_labels)

    pred_slot = np.asarray(slot_logits).argmax(-1)
    gold_slot = np.asarray(slot_labels)

    intent_accuracy = float((pred_intent == gold_intent).mean())

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
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "exact_command_accuracy": exact_command_accuracy,
    }
