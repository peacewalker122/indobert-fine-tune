from transformers import AutoTokenizer

from src.config import IGNORE_INDEX, MODEL_NAME, SLOT_TO_ID
from src.dataset import tokenize_and_align

RECORDS = [
    {
        "tokens": ["Kota", "Banjarmasin"],
        "intent": "GET_TOP_CELLS",
        "slots": ["B-LOCATION", "I-LOCATION"],
    }
]


def test_alignment():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = tokenize_and_align(RECORDS, tokenizer)
    example = ds[0]

    enc = tokenizer(["Kota", "Banjarmasin"], is_split_into_words=True).encodings[0]
    word_ids = enc.word_ids

    # both words present, no truncation
    assert set(w for w in word_ids if w is not None) == {0, 1}

    slot_labels = example["slot_labels"]
    for pos, wid in enumerate(word_ids):
        label = slot_labels[pos]
        if wid is None:
            # [CLS] / [SEP]
            assert label == IGNORE_INDEX
        elif pos == 0 or word_ids[pos - 1] != wid:
            # first subword of the word carries the label
            assert label == SLOT_TO_ID[RECORDS[0]["slots"][wid]]
        else:
            # continuation subword masked
            assert label == IGNORE_INDEX


def test_intent_label():
    from src.config import INTENT_TO_ID

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = tokenize_and_align(RECORDS, tokenizer)
    assert ds[0]["intent_labels"] == INTENT_TO_ID["GET_TOP_CELLS"]
