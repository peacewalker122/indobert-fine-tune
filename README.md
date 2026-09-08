# indobert-command-model

Fine-tuned `indobenchmark/indobert-base-p1` for Indonesian cell-network commands:
shared IndoBERT encoder + intent head (4 intents) + slot head (9 BIO labels).

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
git clone <this-repo-url> && cd indobert-command-model
pip install -U uv
uv sync
uv run python -m src.train                # full: 5 epochs, ~30-60 min on T4
uv run python -m src.evaluate --artifact artifacts/intent-slot-v1
```

Then download `artifacts/intent-slot-v1/` (self-contained: safetensors + tokenizer +
`labels.json` + `training_metadata.json`).

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
- CPU: artifact size, load time, p50/p95/p99 batch-1 latency, throughput
  (`uv run python -m src.bench --artifact ...`).

```bash
uv run python -m src.split
uv run python -m src.sweep --version-base intent-slot-v2 --seeds 42 43 44
# per seed: train → fit thresholds on validation → frozen test report;
# writes reports/report-<seed>.json + reports/scorecard.json (mean/std).
# Smoke first: add --epochs 1 --max-samples 32 to sweep. Manual equivalent:
uv run python -m src.evaluate --artifact artifacts/intent-slot-v1 --split validation \
  --fit-calibration thresholds.json   # validation only
uv run python -m src.evaluate --artifact artifacts/intent-slot-v1 --output report.json \
  --thresholds thresholds.json        # frozen test run
```

Checkpoint selection uses validation `exact_command_accuracy`. Missing artifact
(e.g. B0) reports `{"status": "unavailable"}` instead of failing silently.

## Known dataset limitations

- Intent imbalance (~65% GET_TOP_CELLS) → check per-intent metrics, not just averages.
- UNKNOWN share is 0.4% (39 fixed sentences) → off-domain detection is essentially
  memorization; expect near-zero real-world UNKNOWN recall. Accepted for v1.
- Data is template-generated; informal real-user phrasing may underperform.

## Artifact layout

```
artifacts/intent-slot-v1/
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

cfg = AutoConfig.from_pretrained("artifacts/intent-slot-v1")
model = IntentSlotModel(AutoModel.from_config(cfg))
model.load_state_dict(load_file("artifacts/intent-slot-v1/model.safetensors"))
tokenizer = AutoTokenizer.from_pretrained("artifacts/intent-slot-v1")
```
