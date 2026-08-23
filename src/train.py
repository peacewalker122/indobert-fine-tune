import argparse
import json
from pathlib import Path

from transformers import Trainer, TrainingArguments, set_seed

from .config import ARTIFACT_VERSION, MODEL_NAME, SEED, TrainingConfig
from .dataset import PadCollator, load_split
from .export import save_artifact
from .metrics import compute_metrics
from .model import IntentSlotModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--output-dir", type=Path, default=Path("output"))
    ap.add_argument("--version", default=ARTIFACT_VERSION)
    args = ap.parse_args()

    cfg = TrainingConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs

    set_seed(SEED)

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = load_split("data", "train", tokenizer)
    val_ds = load_split("data", "validation", tokenizer)
    if args.max_samples:
        train_ds = train_ds.select(range(args.max_samples))
        val_ds = val_ds.select(range(min(args.max_samples, len(val_ds))))

    model = IntentSlotModel(AutoModel.from_pretrained(MODEL_NAME))

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    final_metrics = trainer.evaluate()

    hyperparams = {
        "learning_rate": cfg.learning_rate,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "max_length": 64,
        "seed": SEED,
        "train_size": len(train_ds),
        "validation_size": len(val_ds),
    }
    clean_metrics = {k: v for k, v in final_metrics.items() if isinstance(v, float)}
    save_artifact(trainer.model, tokenizer, args.version, metrics=clean_metrics, hyperparams=hyperparams)


if __name__ == "__main__":
    main()
