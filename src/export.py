import argparse
import json
from pathlib import Path

from safetensors.torch import save_file

from .config import (
    ARTIFACT_DIR,
    INTENT_TO_ID,
    MODEL_NAME,
    SLOT_TO_ID,
)
from .dataset import PadCollator  # noqa: F401  (kept importable for consumers)


def save_artifact(model, tokenizer, version, metrics=None, hyperparams=None):
    out_dir = Path(ARTIFACT_DIR) / version
    if out_dir.exists():
        raise FileExistsError(f"artifact {out_dir} already exists — never overwrite a version")

    out_dir.mkdir(parents=True)

    model.encoder.config.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    save_file(dict(model.state_dict()), str(out_dir / "model.safetensors"))

    labels = {"intents": INTENT_TO_ID, "slots": SLOT_TO_ID}
    with open(out_dir / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    metadata = {
        "version": version,
        "base_model": MODEL_NAME,
        **(hyperparams or {}),
        "metrics": metrics or {},
    }
    with open(out_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"artifact saved to {out_dir}")
    return out_dir


def load_trained(model_dir):
    """Rebuild IntentSlotModel from a Trainer checkpoint dir."""
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    from .model import IntentSlotModel

    model_dir = Path(model_dir)
    encoder = AutoModel.from_pretrained(MODEL_NAME)
    model = IntentSlotModel(encoder)
    state = load_file(str(model_dir / "model.safetensors"))
    model.load_state_dict(state)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Trainer checkpoint dir")
    ap.add_argument("--version", default=ARTIFACT_VERSION)
    args = ap.parse_args()

    model, tokenizer = load_trained(args.model_dir)
    save_artifact(model, tokenizer, args.version)


if __name__ == "__main__":
    main()
