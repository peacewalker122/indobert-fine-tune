# SPEC — indobert-command-model (benchmark-aligned)

## 1. Goal

Fine-tune `indobenchmark/indobert-base-p1` for Indonesian cell-network commands:
joint intent classification + slot filling. Single deployable artifact.
Benchmark companion: frozen B0 eval, reproducible B1, multilingual-data B2 —
same heads, loss, splits, decoder, evaluator for every candidate.

Non-goals: second extraction model, RAG/vector DB, LLM rescoring, API serving.

## 2. Shapes

```ts
Record { tokens: string[]; slots: Slot[]; intent: Intent;
         id?: string; text?: string; language?: "id"|"en"|"mixed";
         group_id?: string; source?: string }
Intent = "GET_TOP_CELLS" | "GET_CELL_DETAIL" | "GET_CELL_COUNT" | "UNKNOWN"
Slot   = "O" | "B-LIMIT" | "B-CELL_ID" | "B-LOCATION" | "I-LOCATION"
       | "B-METRIC" | "I-METRIC" | "B-ORDER" | "I-ORDER"
Span = (type, start_word, end_excl, text)
Feature { input_ids: int[]; attention_mask: int[]; intent_labels: int; slot_labels: int[] }
Artifact { config.json; model.safetensors; tokenizer{,_config}.json; labels.json; training_metadata.json }
Calibration { method: "maxprob-sweep-v1"; threshold: float; min_coverage: float }
```

Constants (`src/config.py`): `MAX_LENGTH=64`, `SEED=42`, `IGNORE_INDEX=-100`,
`TrainingConfig{lr=2e-5, epochs=5, batch=16}`.

## 3. Call graphs

```ts
Data prep
  → split.load(source=../nlu-gen/dataset.jsonl, --seed)
    → dataset.validate_record (empty / length-mismatch / unknown label /
       illegal BIO I-without-B → ValueError w/ line no; language/group validated if present)
    → split.group_split (records with group_id: whole group → one split, 80/10/10;
       without: seeded shuffle) → data/{train,validation,test}.jsonl + split_manifest.json
       (seed, source_sha256, counts, grouped flag)

Training (run ×3 seeds for B1/B2)
  → train.main --seed S --version <v>-sS
    → dataset.load_split + tokenize_and_align (first-subword label, rest + specials → -100)
      → dataset.PadCollator
    → model.IntentSlotModel → forward: intent=CLS head, slot=per-token head,
       loss=CE(intent)+CE(slot,ignore -100)
    → Trainer(eval/save per epoch, best by exact_command_accuracy, grad-clip 1.0, keep 2)
    → export.save_artifact (+ split_manifest digest in hyperparams; never overwrite)

Calibration (validation ONLY)
  → evaluate --split validation --fit-calibration thresholds.json
    → calibrate.select_threshold (maxprob sweep 0..0.95, best selective @ coverage>=0.95)
    → thresholds.json {method, threshold, min_coverage}

Eval (frozen artifact + thresholds, test once)
  → evaluate.report(artifact, data/test.jsonl)
    → missing dir → {"status":"unavailable"} (B0 rule: never silently replace)
    → evaluate.predict (batch 32, softmax maxprob + top1-top2 margin, strip -100 → word rows)
    → calibrate.apply_threshold (maxprob < t → UNKNOWN)
    → dataset.decode_spans (illegal I- repaired as B-; counted pre-repair)
    → metrics: exact(span) + CI + slices / intent_report / entity_span_report / UNKNOWN-OOD
    → JSON (--output) + stdout summary

CPU bench
  → bench --artifact (warmup 100, runs 1000, batch 1) → load_s, artifact_mb, p50/p95/p99, throughput
```

```ts
Tests
  → test_dataset (valid passes, malformed raise w/ "line 2", align len match)
  → test_label_alignment (first subword labeled, continuation + specials -100)
  → test_model (logit shapes (B,4)/(B,T,9), loss None w/o labels, backward finite)
  → test_benchmark (illegal-BIO raises, optional-field checks, span decode+repair,
     one-vs-rest FP counted, extra-entity rejects exact, CI ordered,
     threshold boundary + roundtrip, group no-leakage + determinism)
```

## 4. Metrics (contracts)

| Metric | Def | Role |
|---|---|---|
| `exact_command_accuracy` | intent correct AND decoded span sets equal (no missing/extra) | primary; checkpoint selection; reported overall + per-language + per-intent + 95% bootstrap CI |
| intent macro F1 / known-macro (excl UNKNOWN) / per-intent P-R-F1 / confusion | one-vs-rest over FULL set (never filter to gold rows first) | guardrail under imbalance; accuracy secondary |
| entity span micro + macro + per-type P-R-F1 | exact `(type,start,end)` sets | extraction guardrail; token-F1 debug only, never selects |
| UNKNOWN P-R-F1, false-accept (OOD→known), known false-reject, coverage, selective accuracy, margin stats | thresholds from validation only | safety guardrail |
| illegal BIO count (pre-repair) | predicted word labels | decoder health |
| CPU: artifact MB, load s, p50/p95/p99 (batch 1), throughput | fixed threads, pinned stack | operational gate |

Trainer `compute_metrics` keeps mask-based exact as a selection proxy (word rows
unavailable at that level); true span-exact is computed in `evaluate.report`.

## 5. Artifact contract

`artifacts/<version>/` self-contained (config + safetensors + tokenizer +
`labels.json` + `training_metadata.json` incl. seed, hyperparams, val metrics,
split manifest digest, optional calibration/threshold refs via `extra`).
Load per README. Missing artifact → `status: unavailable`.

## 6. Env split

Local: code + `uv run pytest` only. GPU (Colab/Kaggle T4, ~30-60 min per seed):
train × seeds → calibrate on validation → evaluate test once → bench on target CPU.
Smoke: `--epochs 1 --max-samples 32 --version smoke-test`.
`torch==2.13.0` pinned w/ cu126 index; `python>=3.10`.

## 7. Verify

```bash
uv sync && uv run pytest
uv run python -m src.split && cat data/split_manifest.json
uv run python -m src.train --epochs 1 --max-samples 32 --version smoke-test --seed 42
uv run python -m src.evaluate --artifact artifacts/smoke-test --split validation \
  --fit-calibration thresholds.json
uv run python -m src.evaluate --artifact artifacts/smoke-test --output report.json \
  --thresholds thresholds.json
uv run python -m src.evaluate --artifact artifacts/missing --output b0.json  # unavailable path
```

Multi-seed: repeat train with `--seed 42/43/44 --version <v>-s{42,43,44}`;
report mean/std + individual seeds. Test set untouched until thresholds frozen.

## 8. Known limits

- Current `data/` has Indonesian-template records only: no `language`/`group_id`
  yet → slices report `unlabeled`, splits fall back to seeded shuffle until
  multilingual data lands.
- UNKNOWN 39 fixed sentences → memorization, ~zero real-world recall.
- Template phrasing → informal/abbrev/typo slices pending; calibration threshold
  defaults to 0.0 (accept all) until validation OOD exists.
