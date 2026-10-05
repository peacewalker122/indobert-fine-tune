# multilingual-bert-command-model

Fine-tuned `google-bert/bert-base-multilingual-cased` for Indonesian and English
cell-network commands: a cased multilingual BERT encoder + intent head (4 intents)
and slot head (9 BIO labels).

The target corpus contains a paired Indonesian/English corpus plus one
deterministic synthetic code-switched (`mixed`) record for each source command.
Every record keeps aligned intent and BIO slot labels. The mixed renderer keeps
the original Indonesian structure and switches command phrases at natural
boundaries; it is generated augmentation, not a human-collected corpus.

Direct experiments can run on Colab/Kaggle; the SmartCare gateway trainer image is
CPU-only and runs the same `src.train` implementation with frozen dataset and
policy inputs. Training artifacts are not quality claims until evaluated on the
frozen test set and accepted by the policy gate.

## Labels

- Intents: `GET_TOP_CELLS`, `GET_CELL_DETAIL`, `GET_CELL_COUNT`, `UNKNOWN`
- Slots: `O`, `B-LIMIT`, `B-CELL_ID`, `B-LOCATION`, `I-LOCATION`, `B-METRIC`, `I-METRIC`, `B-ORDER`, `I-ORDER`

Splits in `data/` are generated (`uv run python -m src.split`) from
`../nlu-gen/dataset.jsonl` — group-aware when records carry `group_id`
(translations/paraphrases stay in one split), with `data/split_manifest.json`
(seed + source sha256). Same-dataset runs reuse committed splits. The
multilingual generator then preserves each source split and emits three records
per group (`id`, `en`, `mixed`): 24,000 train records, 3,000 validation
records, and 3,000 test records.

Regenerate or verify the committed JSONL with:

```bash
uv run python -m src.generate_multilingual_data
uv run python -m src.generate_multilingual_data --check
```

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

## SmartCare gateway (no MLflow)

The orchestrator consumes an immutable RunSpec and has separate stages:

```bash
uv run python -m src.orchestrate --mode prepare --run-spec-json /work/run-spec.json \
  --attempt 1 --work-dir /work
uv run python -m src.orchestrate --mode publish --run-spec-json /work/run-spec.json \
  --attempt 1 --work-dir /work
uv run python -m src.orchestrate --mode recover --run-spec-json /work/run-spec.json \
  --attempt 1 --work-dir /work
```

`prepare` downloads and verifies pinned MinIO dataset files and exact metadata
bytes, trains, fits calibration on validation, evaluates the test split once,
and stores checksummed evidence. It does not publish models. `publish` verifies
that durable evidence and reruns the gate against the frozen policy before
writing model objects and committing `metadata.json` last. `recover` validates an
existing committed release or resumes publication from the staged attempt
without training or reevaluating. A smoke RunSpec (`max_samples`) is rejected as
non-promotable. The container runs as UID 10001 in `/work`; publisher credentials
must only be supplied to publish/recover stages.

To create an immutable dataset release, provide provenance JSON (for example,
source revision, annotation process, and generation code revision) and use the
explicit publisher:

```bash
uv run python -m src.publish_dataset --data-dir /work/data --name telecom-intent \
  --version d1 --provenance-json provenance.json
```

The publisher validates canonical BIO records, nonempty train/validation/test
splits, unique record IDs, and group isolation, checks uploaded file hashes, and
writes pinned metadata last. Dataset quality and model quality still require
independent review; fixture tests only verify lifecycle behavior.

## Human challenge data

Synthetic `mixed` examples are not a substitute for real operator language. Use the
offline Label Studio workflow in [`annotation/label-studio/README.md`](annotation/label-studio/README.md)
to author and export a separate `data/challenge/mixed_human.jsonl` holdout. The
converter owns BIO labels, `id`, and `group_id`; annotators label only intent and
highlighted spans.

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

- Indonesian/English pairs and their mixed companion share the same four-intent
  and nine-slot contract; language slices are reported as `id`, `en`, and
  `mixed`. All three records retain one `group_id` and one source split.
- Template-generated examples are useful for repeatable evaluation but may
  underrepresent informal, abbreviated, typo-heavy, or genuinely out-of-domain input.
  Treat `mixed` metrics as synthetic-regression results until a human-authored
  code-switch challenge set exists.
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
