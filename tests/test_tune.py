from pathlib import Path

from transformers import AutoModel, BertConfig

from src.model import IntentSlotModel
from src.tune import SEARCH_SPACE, build_trial_dir, suggest_params


class FakeTrial:
    def __init__(self):
        self.params = {}

    def suggest_float(self, name, lo, hi, log=False):
        v = lo if log else (lo + hi) / 2
        self.params[name] = v
        return v

    def suggest_categorical(self, name, choices):
        self.params[name] = choices[0]
        return choices[0]


def test_search_space_bounds():
    assert SEARCH_SPACE["lr"][0] < SEARCH_SPACE["lr"][1]
    assert 16 in SEARCH_SPACE["batch_size"]
    assert SEARCH_SPACE["dropout"][0] >= 0.0


def test_suggest_params_in_range():
    p = suggest_params(FakeTrial())
    assert SEARCH_SPACE["lr"][0] <= p["lr"] <= SEARCH_SPACE["lr"][1]
    assert p["batch_size"] in SEARCH_SPACE["batch_size"]
    assert SEARCH_SPACE["dropout"][0] <= p["dropout"] <= SEARCH_SPACE["dropout"][1]


def test_build_trial_dir():
    assert build_trial_dir(Path("output/tune"), 3) == Path("output/tune/trial-3")


def test_model_dropout_param_backward_compatible():
    cfg = BertConfig(vocab_size=200, hidden_size=16, num_hidden_layers=1,
                     num_attention_heads=2, intermediate_size=32,
                     max_position_embeddings=32)
    default = IntentSlotModel(AutoModel.from_config(cfg))
    custom = IntentSlotModel(AutoModel.from_config(cfg), dropout=0.25)
    assert default.dropout.p == 0.1
    assert custom.dropout.p == 0.25
