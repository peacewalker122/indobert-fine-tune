# Label Studio human annotation

This directory contains the fixed Label Studio OSS interface for a small human-authored challenge set. The resulting records are a separate holdout: write them to `data/challenge/mixed_human.jsonl`, and never copy them into the generated `data/train.jsonl`, `data/validation.jsonl`, or `data/test.jsonl` files.

## Air-gapped setup

On a connected machine, pull and save the pinned image:

```bash
docker pull heartexlabs/label-studio:1.23.0
docker save --output label-studio-1.23.0.tar heartexlabs/label-studio:1.23.0
```

Transfer `label-studio-1.23.0.tar` through the approved air-gap process. On the offline machine, import it and verify the local image:

```bash
docker load --input label-studio-1.23.0.tar
docker image inspect heartexlabs/label-studio:1.23.0 --format '{{.Id}}'
```

After the image import, no network is needed. `pull_policy: never` makes Compose use only the local image and fail clearly if it was not loaded. Start the service from the repository root:

```bash
docker compose -f annotation/label-studio/docker-compose.yml up -d --pull never
```

Open <http://localhost:8080>, or use `LABEL_STUDIO_PORT` to bind another host port:

```bash
LABEL_STUDIO_PORT=18080 docker compose -f annotation/label-studio/docker-compose.yml up -d --pull never
```

Label Studio persists its local database and uploaded project data in the Compose named volume `label-studio_label-studio-data`. Do not run `docker compose down -v` unless deleting the project is intentional. Back up the volume before moving hosts; `docker volume inspect label-studio_label-studio-data` shows its location.

## Configure and annotate

1. Create a local Label Studio project.
2. In the project settings, replace the labeling interface with [`config.xml`](config.xml).
3. Prepare one task per prompt:

   ```bash
   uv run python -m src.prepare_label_studio_tasks \
     annotation/label-studio/prompts.example.txt \
     --output /tmp/label-studio-tasks.json
   ```

4. Import `/tmp/label-studio-tasks.json` into the project.
5. Annotate exactly one intent for every task. Select complete words for slots; do not select whitespace or partial words. Use only `LIMIT`, `CELL_ID`, `LOCATION`, `METRIC`, and `ORDER`.
6. For consensus, have every annotator submit a completed annotation. The importer rejects missing intent, unknown labels, overlapping spans, non-word-aligned offsets, and disagreement between annotations.
7. Export the completed project as the normal Label Studio JSON export and keep that export as the audit source.

## Convert and evaluate

Convert the export into the separate human holdout. The converter uses end-exclusive character offsets from Label Studio, derives `\S+` whitespace tokens with exact offsets, requires both span boundaries to equal token boundaries, and then builds BIO labels. `LIMIT` and `CELL_ID` must be single-token spans because the existing model taxonomy defines no `I-LIMIT` or `I-CELL_ID` labels.

```bash
uv run python -m src.import_label_studio \
  /path/to/label-studio-export.json \
  --output data/challenge/mixed_human.jsonl
```

Defaults are `--language mixed` and `--source human-label-studio`. A task's `case_id` is copied to the output when present. Each output row has a group ID equal to the SHA-256 of canonical `intent`/`slots`/`tokens` JSON, and `id=<group_id>-<language>`.

Validate the holdout without changing generated data:

```bash
uv run pytest -q tests/test_label_studio.py
uv run python -m src.generate_multilingual_data --check
```

Evaluate a trained artifact against the holdout only when one is available:

```bash
uv run python -m src.evaluate \
  --artifact artifacts/multilingual-intent-slot-v2 \
  --data-dir data/challenge \
  --split mixed_human \
  --output reports/mixed_human.json
```

Do not run `src.split` over the human holdout: it is intentionally not part of the train/validation/test corpus.
