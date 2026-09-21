"""Optuna HPO for IntentSlotModel: TPE search + median pruning.

Reuses train/evaluate contracts (same heads, loss, splits, metric).
Objective maximizes validation `exact_command_accuracy`.

Usage (GPU):
  uv run python -m src.tune --trials 15 --seed 42
Smoke (CPU-safe, no full train):
  uv run python -m src.tune --trials 2 --epochs 1 --max-samples 32
"""
import argparse
import json
from pathlib import Path

from transformers import set_seed

from .config import ARTIFACT_VERSION, MODEL_NAME, SEED, TrainingConfig

SEARCH_SPACE = {
    "lr": (1e-5, 5e-5),  # log scale
    "batch_size": [8, 16, 32],
    "weight_decay": (0.0, 0.1),
    "warmup_ratio": (0.0, 0.1),
    "dropout": (0.05, 0.3),
}

BEST_FILENAME = "tune_best.json"


def suggest_params(trial):
    """Sample one trial config. `trial` is optuna.Trial (duck-typed in tests)."""
    return {
        "lr": trial.suggest_float("lr", *SEARCH_SPACE["lr"], log=True),
        "batch_size": trial.suggest_categorical("batch_size", SEARCH_SPACE["batch_size"]),
        "weight_decay": trial.suggest_float("weight_decay", *SEARCH_SPACE["weight_decay"]),
        "warmup_ratio": trial.suggest_float("warmup_ratio", *SEARCH_SPACE["warmup_ratio"]),
        "dropout": trial.suggest_float("dropout", *SEARCH_SPACE["dropout"]),
    }


def build_trial_dir(output_dir: Path, trial_number: int) -> Path:
    return Path(output_dir) / f"trial-{trial_number}"


def objective(trial, train_ds, val_ds, tokenizer, base_cfg: TrainingConfig,
              output_dir: Path, epochs_override=None):
    import optuna
    from transformers import AutoModel, Trainer, TrainingArguments

    from .dataset import PadCollator
    from .metrics import compute_metrics
    from .model import IntentSlotModel

    params = suggest_params(trial)
    set_seed(base_cfg.seed + trial.number)

    def model_init():
        return IntentSlotModel(AutoModel.from_pretrained(MODEL_NAME),
                               dropout=params["dropout"])

    epochs = epochs_override if epochs_override is not None else base_cfg.epochs
    args = TrainingArguments(
        output_dir=str(build_trial_dir(output_dir, trial.number)),
        learning_rate=params["lr"],
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        num_train_epochs=epochs,
        weight_decay=params["weight_decay"],
        warmup_ratio=params["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="exact_command_accuracy",
        save_total_limit=1,
        remove_unused_columns=False,
        report_to=[],
    )
    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(),
        compute_metrics=compute_metrics,
    )
    # Enable Optuna pruning on intermediate eval_exact_command_accuracy.
    try:
        from transformers.integrations import OptunaCallback

        trainer.add_callback(OptunaCallback(trial, "eval_exact_command_accuracy"))
    except Exception:
        pass

    trainer.train()
    metrics = trainer.evaluate()
    score = float(metrics.get("eval_exact_command_accuracy", 0.0))
    for k in ("eval_intent_macro_f1", "eval_f1", "eval_loss"):
        if k in metrics and isinstance(metrics[k], float):
            trial.set_user_attr(k, metrics[k])
    # Pruning check for backends without callback support.
    trial.report(score, epochs)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return score


def run_study(train_ds, val_ds, tokenizer, base_cfg: TrainingConfig,
              output_dir: Path, n_trials=15, seed=SEED, timeout=None,
              storage=None, study_name=None, no_prune=False,
              epochs_override=None):
    import optuna

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = (optuna.pruners.NopPruner() if no_prune
              else optuna.pruners.MedianPruner(n_warmup_steps=1))
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                pruner=pruner, storage=storage,
                                study_name=study_name, load_if_exists=bool(storage))
    study.optimize(lambda t: objective(t, train_ds, val_ds, tokenizer,
                                       base_cfg, output_dir, epochs_override),
                   n_trials=n_trials, timeout=timeout)
    return study


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--output-dir", type=Path, default=Path("output/tune"))
    ap.add_argument("--timeout", type=float, default=None, help="seconds")
    ap.add_argument("--storage", default=None, help="optuna storage URL")
    ap.add_argument("--study-name", default=None)
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--version", default=ARTIFACT_VERSION)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from .dataset import load_split

    base_cfg = TrainingConfig(seed=args.seed)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = load_split("data", "train", tokenizer)
    val_ds = load_split("data", "validation", tokenizer)
    if args.max_samples:
        train_ds = train_ds.select(range(args.max_samples))
        val_ds = val_ds.select(range(min(args.max_samples, len(val_ds))))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    study = run_study(train_ds, val_ds, tokenizer, base_cfg, args.output_dir,
                      n_trials=args.trials, seed=args.seed, timeout=args.timeout,
                      storage=args.storage, study_name=args.study_name,
                      no_prune=args.no_prune, epochs_override=args.epochs)
    best = {"value": study.best_value, "params": study.best_params,
            "trial": study.best_trial.number,
            "user_attrs": study.best_trial.user_attrs}
    out = args.output_dir / BEST_FILENAME
    out.write_text(json.dumps(best, indent=2))
    print(f"best {study.best_value:.4f} params={study.best_params} -> {out}")
    print("Retrain winner via: uv run python -m src.train "
          f"--version {args.version} --seed {args.seed} "
          f"(apply best lr/batch manually; see tune_best.json)")


if __name__ == "__main__":
    main()
