# multilingual-bert-command-model

Fine-tuned `google-bert/bert-base-multilingual-cased` for Indonesian and English
cell-network commands: a cased multilingual BERT encoder + intent head (4 intents)
and slot head (9 BIO labels).

The target corpus is a paired Indonesian/English corpus: each command meaning has
aligned intent and BIO slot labels in both languages. Code-switched (`mixed`)
input is part of the multilingual goal.

Training runs on **Google Colab / Kaggle** (GPU). Local machine is for code + tests only.

## Labels

- Intents: `GET_TOP_CELLS`, `GET_CELL_DETAIL`, `GET_CELL_COUNT`, `UNKNOWN`
- Slots: `O`, `B-LIMIT`, `B-CELL_ID`, `B-LOCATION`, `I-LOCATION`, `B-METRIC`, `I-METRIC`, `B-ORDER`, `I-ORDER`

Splits in `data/` are generated (`uv run python -m src.split`) from
`../nlu-gen/dataset.jsonl` — group-aware when records carry `group_id`
(translations/paraphrases stay in one split), with `data/split_manifest.json`
(seed + source sha256). Same-dataset runs reuse committed splits.

## Local

```bash
uv sync
uv run pytest
```

## Colab / Kaggle

```bash
git clone <this-repo-url> && cd multilingual-bert-command-model
pip install -U uv
uv sync
uv run python -m src.train                # full: 5 epochs, ~30-60 min on T4
uv run python -m src.evaluate --artifact artifacts/multilingual-intent-slot-v2
```

Then download `artifacts/multilingual-intent-slot-v2/` (self-contained: safetensors + tokenizer +
`labels.json` + `training_metadata.json`).

The base model is cased, so capitalization is preserved by tokenization and training.

Useful flags:

```bash
uv run python -m src.train --epochs 1 --max-samples 32 --version smoke-test   # smoke
uv run python -m src.export --model-dir output/checkpoint-XXXX               # re-export a checkpoint
```

Artifact versions are never overwritten — export fails if the target dir exists.

## Metrics

- `exact_command_accuracy` — **primary metric**: intent correct AND decoded entity
  span sets equal (no missing/extra span). Reported overall, per language
  (`id`/`en`/`mixed`), per intent, with 95% bootstrap CI.
- intent macro F1 (one-vs-rest) + known-only macro + per-intent + confusion matrix.
  Accuracy is secondary.
- entity span micro/macro/per-type F1. Token-level slot F1 is debug-only.
- UNKNOWN P/R/F1, OOD false-accept, known false-reject, coverage, selective
  accuracy. Thresholds fit on validation only (`--fit-calibration`).
- ORDER direction accuracy: finite-state `normalize_order` maps span text →
  `ascending`/`descending`/`None` (substring families: rendah/lowest/buruk,
  tinggi/highest/baik). SQL reads the enum, never raw text. Unmapped-gold
  count must be 0 — nonzero means vocab gap. Mirrored in `DIRECTION_HINTS`
  (`ai/src/application/intent.ts`).
- CPU: artifact size, load time, p50/p95/p99 batch-1 latency, throughput
  (`uv run python -m src.bench --artifact ...`).

```bash
uv run python -m src.split
uv run python -m src.sweep --version-base multilingual-intent-slot-v2 --seeds 42 43 44
# per seed: train → fit thresholds on validation → frozen test report;
# writes reports/report-<seed>.json + reports/scorecard.json (mean/std).
# Smoke first: add --epochs 1 --max-samples 32 to sweep. Manual equivalent:
uv run python -m src.evaluate --artifact artifacts/multilingual-intent-slot-v2 --split validation \
  --fit-calibration thresholds.json   # validation only
uv run python -m src.evaluate --artifact artifacts/multilingual-intent-slot-v2 --output report.json \
  --thresholds thresholds.json        # frozen test run
```

Checkpoint selection uses validation `exact_command_accuracy`. Missing artifact
(e.g. B0) reports `{"status": "unavailable"}` instead of failing silently.

## Dataset notes

- Paired Indonesian/English records share the same four-intent and nine-slot
  contract; language slices are reported as `id`, `en`, and `mixed`.
- Template-generated examples are useful for repeatable evaluation but may
  underrepresent informal, abbreviated, typo-heavy, or genuinely out-of-domain input.
- UNKNOWN performance must be checked on held-out out-of-domain data, not only on
  the generated corpus.

## Artifact layout

```
artifacts/multilingual-intent-slot-v2/
├── config.json              # encoder config (BERT)
├── model.safetensors        # full IntentSlotModel state_dict
├── tokenizer.json           # + tokenizer_config.json
├── labels.json              # intent/slot name→id maps
└── training_metadata.json   # hyperparams + final validation metrics
```

Loading (no custom code needed):

```python
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer
from src.model import IntentSlotModel

cfg = AutoConfig.from_pretrained("artifacts/multilingual-intent-slot-v2")
model = IntentSlotModel(AutoModel.from_config(cfg))
model.load_state_dict(load_file("artifacts/multilingual-intent-slot-v2/model.safetensors"))
tokenizer = AutoTokenizer.from_pretrained("artifacts/multilingual-intent-slot-v2")
```
