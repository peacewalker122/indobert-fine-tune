"""Target-CPU inference harness. Batch-1 latency + throughput, pinned config."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import ARTIFACT_DIR, ARTIFACT_VERSION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=str(ARTIFACT_DIR / ARTIFACT_VERSION))
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    t0 = time.perf_counter()
    from transformers import AutoTokenizer

    from .evaluate import load_artifact

    model = load_artifact(args.artifact)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.artifact)
    load_s = time.perf_counter() - t0

    texts = []
    with open(Path(args.data_dir) / f"{args.split}.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["tokens"])
    if not texts:
        raise ValueError("no texts to benchmark")

    def once(tokens):
        enc = tok(tokens, is_split_into_words=True, return_tensors="pt",
                  truncation=True, max_length=64)
        with torch.no_grad():
            model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

    for i in range(args.warmup):
        once(texts[i % len(texts)])
    ts = []
    for i in range(args.runs):
        s = time.perf_counter()
        once(texts[i % len(texts)])
        ts.append((time.perf_counter() - s) * 1000)
    arr = np.array(ts)
    size_mb = sum(p.stat().st_size for p in Path(args.artifact).rglob("*") if p.is_file()) / 1e6
    res = {
        "artifact": args.artifact,
        "threads": args.threads,
        "torch": torch.__version__,
        "load_time_s": round(load_s, 3),
        "artifact_mb": round(size_mb, 1),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "throughput_s": round(1000 / float(arr.mean()), 1),
    }
    print(json.dumps(res, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
