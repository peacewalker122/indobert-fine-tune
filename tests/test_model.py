import torch
from transformers import AutoModel, BertConfig

from src.config import IGNORE_INDEX, INTENTS, SLOTS
from src.model import IntentSlotModel

TINY_CONFIG = BertConfig(
    vocab_size=1000,
    hidden_size=32,
    num_hidden_layers=1,
    num_attention_heads=2,
    intermediate_size=64,
    max_position_embeddings=64,
)


def make_model():
    return IntentSlotModel(AutoModel.from_config(TINY_CONFIG))


def make_batch(batch_size=2, seq_len=8):
    return {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "intent_labels": torch.tensor([0, 1]),
        "slot_labels": torch.randint(
            0, len(SLOTS), (batch_size, seq_len)
        ).masked_fill(torch.rand(batch_size, seq_len) < 0.3, IGNORE_INDEX),
    }


def test_logit_shapes():
    model = make_model()
    out = model(**make_batch())
    assert out["intent_logits"].shape == (2, len(INTENTS))
    assert out["slot_logits"].shape == (2, 8, len(SLOTS))


def test_loss_none_without_labels():
    model = make_model()
    batch = make_batch()
    del batch["intent_labels"], batch["slot_labels"]
    out = model(**batch)
    assert out["loss"] is None


def test_forward_backward_smoke():
    model = make_model()
    out = model(**make_batch())
    loss = out["loss"]
    assert torch.isfinite(loss).item()
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name
