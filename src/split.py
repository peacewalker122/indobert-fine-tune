import argparse
import json
import random
from pathlib import Path

from .config import SEED
from .dataset import validate_record

DEFAULT_SOURCE = Path(__file__).resolve().parent.parent.parent / "nlu-gen" / "dataset.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
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

    rng = random.Random(SEED)
    rng.shuffle(records)
    n = len(records)
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)
    splits = {
        "train": records[:train_end],
        "validation": records[train_end:val_end],
        "test": records[val_end:],
    }

    args.data_dir.mkdir(exist_ok=True)
    for name, subset in splits.items():
        with open(args.data_dir / f"{name}.jsonl", "w") as f:
            for r in subset:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(subset)}")

    total = len(records)
    unknown = sum(1 for r in records if r["intent"] == "UNKNOWN")
    print(f"total: {total}, UNKNOWN: {unknown} ({unknown / total:.1%})")


if __name__ == "__main__":
    main()
