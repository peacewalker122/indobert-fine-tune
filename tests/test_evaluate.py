import torch

from src import evaluate
from src.config import INTENT_TO_ID, SLOT_TO_ID, SLOTS


def test_predict_masks_ignored_tokens(monkeypatch):
    records = [
        (
            1,
            {
                "tokens": ["10"],
                "intent": "GET_TOP_CELLS",
                "slots": ["B-LIMIT"],
            },
        )
    ]
    features = [
        {
            "input_ids": [101, 10, 102],
            "attention_mask": [1, 1, 1],
            "intent_labels": INTENT_TO_ID["GET_TOP_CELLS"],
            "slot_labels": [-100, SLOT_TO_ID["B-LIMIT"], -100],
        }
    ]
    monkeypatch.setattr(evaluate, "tokenize_and_align", lambda *_: features)

    class Model:
        def __call__(self, **_):
            intent_logits = torch.zeros(1, len(INTENT_TO_ID))
            intent_logits[0, INTENT_TO_ID["GET_TOP_CELLS"]] = 1
            slot_logits = torch.zeros(1, 3, len(SLOTS))
            slot_logits[0, 1, SLOT_TO_ID["B-LIMIT"]] = 1
            return {"intent_logits": intent_logits, "slot_logits": slot_logits}

    predicted_intents, predicted_slots, gold_intents, gold_slots, *_ = evaluate.predict(
        Model(), records, tokenizer=None
    )

    assert predicted_intents.tolist() == [INTENT_TO_ID["GET_TOP_CELLS"]]
    assert gold_intents.tolist() == [INTENT_TO_ID["GET_TOP_CELLS"]]
    assert predicted_slots == [[SLOT_TO_ID["B-LIMIT"]]]
    assert gold_slots == [[SLOT_TO_ID["B-LIMIT"]]]
    assert evaluate._spans(gold_slots, records) == [[("LIMIT", 0, 1, "10")]]
    assert evaluate.INTENT_TO_ID["UNKNOWN"] == INTENT_TO_ID["UNKNOWN"]
