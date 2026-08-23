import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support

from .config import IGNORE_INDEX, SLOTS
from .dataset import PadCollator, load_jsonl, tokenize_and_align, validate_record
from .model import IntentSlotModel


def load_artifact(model_dir):
    model_dir = Path(model_dir)
    cfg = AutoConfig.from_pretrained(model_dir)
    encoder = AutoModel.from_config(cfg)
    model = IntentSlotModel(encoder)
    state = load_file(str(model_dir / "model.safetensors"))
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict(model, records, tokenizer, batch_size=32):
    ds = tokenize_and_align([r for _, r in records], tokenizer)
    collator = PadCollator()
    pred_intents, gold_intents = [], []
    pred_slot_rows, gold_slot_rows = [], []
    for i in range(0, len(ds), batch_size):
        batch = collator([ds[j] for j in range(i, min(i + batch_size, len(ds)))])
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        pred_intent = out["intent_logits"].argmax(-1)
        pred_slot = out["slot_logits"].argmax(-1)
        for b in range(len(pred_intent)):
            mask = batch["slot_labels"][b] != IGNORE_INDEX
            pred_slot_rows.append(pred_slot[b][mask].numpy())
            gold_slot_rows.append(batch["slot_labels"][b][mask].numpy())
        pred_intents.append(pred_intent.numpy())
        gold_intents.append(batch["intent_labels"].numpy())
    return np.concatenate(pred_intents), pred_slot_rows, np.concatenate(gold_intents), gold_slot_rows


def report(model_dir, data_dir="data"):
    records = []
    with open(Path(data_dir) / "test.jsonl") as f:
        for line_no, line in enumerate(f, 1):
            record = json.loads(line)
            validate_record(record, line_no)
            records.append((line_no, record))

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = load_artifact(model_dir)

    pi, ps_rows, gi, gs_rows = predict(model, records, tokenizer)

    intent_accuracy = float((pi == gi).mean())
    flat_pred = np.concatenate(ps_rows)
    flat_gold = np.concatenate(gs_rows)
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_gold, flat_pred, average="macro", labels=list(range(len(SLOTS))), zero_division=0
    )
    exact_command_accuracy = float(
        sum((pi[i] == gi[i]) and np.array_equal(ps_rows[i], gs_rows[i]) for i in range(len(gi))) / len(gi)
    )

    print(f"test set: {len(records)} examples")
    print(f"intent accuracy:          {intent_accuracy:.4f}")
    print(f"slot token P/R/F1 (macro): {precision:.4f} / {recall:.4f} / {f1:.4f}")
    print(f"exact command accuracy:   {exact_command_accuracy:.4f}")

    from .config import ID_TO_INTENT

    print("\nper-intent:")
    for name in sorted(set(ID_TO_INTENT[i] for i in gi)):
        ids = [i for i, n in ID_TO_INTENT.items() if n == name]
        sel = gi == ids[0]
        p, r, f, _ = precision_recall_fscore_support(
            gi[sel], pi[sel], labels=[ids[0]], average="macro", zero_division=0
        )
        print(f"  {name}: P={p:.3f} R={r:.3f} F1={f:.3f} (n={int(sel.sum())})")

    from .config import INTENT_TO_ID

    unknown_id = INTENT_TO_ID.get("UNKNOWN")
    if unknown_id is not None:
        sel = gi == unknown_id
        print(f"\nUNKNOWN recall: {float((pi[sel] == unknown_id).mean()):.3f}")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="artifacts/intent-slot-v1")
    args = ap.parse_args()
    report(args.artifact)
