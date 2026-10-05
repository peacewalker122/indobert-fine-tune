import argparse
import json
import math
import numbers
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
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--model-name", default=MODEL_NAME)
    ap.add_argument("--model-revision", required=False)
    args = ap.parse_args()

    cfg = TrainingConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    cfg.seed = args.seed
    if (
        not math.isfinite(cfg.learning_rate)
        or cfg.learning_rate <= 0
        or cfg.epochs < 1
        or cfg.batch_size < 1
    ):
        raise ValueError("learning rate, epochs, and batch size must be positive")

    set_seed(args.seed)

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.model_revision)

    train_ds = load_split(args.data_dir, "train", tokenizer)
    val_ds = load_split(args.data_dir, "validation", tokenizer)
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        train_ds = train_ds.select(range(min(args.max_samples, len(train_ds))))
        val_ds = val_ds.select(range(min(args.max_samples, len(val_ds))))

    model = IntentSlotModel(
        AutoModel.from_pretrained(args.model_name, revision=args.model_revision)
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        seed=args.seed,
        data_seed=args.seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="exact_command_accuracy",
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
    history_path = Path(args.output_dir) / "trainer_log_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(trainer.state.log_history, indent=2))

    eval_history = [
        entry for entry in trainer.state.log_history if "eval_exact_command_accuracy" in entry
    ]
    best_checkpoint = Path(trainer.state.best_model_checkpoint or "").name
    best_step = (
        int(best_checkpoint.removeprefix("checkpoint-"))
        if best_checkpoint.startswith("checkpoint-")
        else None
    )
    best_record = next(
        (entry for entry in eval_history if entry.get("step") == best_step),
        max(eval_history, key=lambda entry: entry["eval_exact_command_accuracy"], default={}),
    )
    best_epoch = best_record.get("epoch")
    if (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, numbers.Real)
        or not math.isfinite(best_epoch)
        or not float(best_epoch).is_integer()
    ):
        best_epoch = None
    else:
        best_epoch = int(best_epoch)
    hyperparams = {
        "learning_rate": cfg.learning_rate,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "max_length": 64,
        "seed": args.seed,
        "train_size": len(train_ds),
        "validation_size": len(val_ds),
        "log_history": trainer.state.log_history,
        "best_epoch": best_epoch,
        "best_step": best_step or best_record.get("step"),
        "best_validation_exact_command_accuracy": trainer.state.best_metric,
    }
    clean_metrics = {k: v for k, v in final_metrics.items() if isinstance(v, float)}
    man = args.data_dir / "split_manifest.json"
    if man.exists():
        hyperparams["split_manifest"] = json.loads(man.read_text())
    save_artifact(
        trainer.model,
        tokenizer,
        args.version,
        metrics=clean_metrics,
        hyperparams=hyperparams,
        base_model=args.model_name,
        base_revision=args.model_revision,
    )


if __name__ == "__main__":
    main()
