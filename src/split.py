import argparse
import hashlib
import json
import random
from pathlib import Path

from .config import SEED
from .dataset import validate_record

DEFAULT_SOURCE = Path(__file__).resolve().parent.parent.parent / "nlu-gen" / "dataset.jsonl"
MANIFEST = "split_manifest.json"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def group_split(records, seed=SEED):
    """Group-aware 80/10/10: whole group_id stays in one split.

    Records without group_id fall back to seeded shuffle. Returns dict.
    """
    rng = random.Random(seed)
    groups = {}
    singles = []
    for r in records:
        g = r.get("group_id")
        if g:
            groups.setdefault(g, []).append(r)
        else:
            singles.append(r)
    if not groups:
        rng.shuffle(records)
        n = len(records)
        te = int(n * 0.8)
        ve = te + int(n * 0.1)
        return {"train": records[:te], "validation": records[te:ve], "test": records[ve:]}
    gkeys = list(groups)
    rng.shuffle(gkeys)
    rng.shuffle(singles)
    n = len(records)
    targets = {"train": int(n * 0.8), "validation": int(n * 0.1)}
    splits = {"train": [], "validation": [], "test": []}
    for g in gkeys:
        dest = "train" if len(splits["train"]) < targets["train"] else (
            "validation" if len(splits["validation"]) < targets["validation"] else "test")
        splits[dest].extend(groups[g])
    # distribute singles to smallest split first for balance
    for r in singles:
        dest = min(splits, key=lambda k: len(splits[k]))
        splits[dest].append(r)
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    records = []
    with open(args.source) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            validate_record(record, line_no)
            records.append(record)

    splits = group_split(records, args.seed)

    args.data_dir.mkdir(exist_ok=True)
    counts = {}
    for name, subset in splits.items():
        with open(args.data_dir / f"{name}.jsonl", "w") as f:
            for r in subset:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[name] = len(subset)
        print(f"{name}: {len(subset)}")

    manifest = {
        "seed": args.seed,
        "source": str(args.source),
        "source_sha256": sha256_file(args.source),
        "counts": counts,
        "grouped": any("group_id" in r for r in records),
    }
    with open(args.data_dir / MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    total = len(records)
    unknown = sum(1 for r in records if r["intent"] == "UNKNOWN")
    print(f"total: {total}, UNKNOWN: {unknown} ({unknown / total:.1%})")


if __name__ == "__main__":
    main()
