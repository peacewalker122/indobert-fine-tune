"""Multi-seed sweep: train × seeds → calibrate on validation → frozen test eval → aggregate.

Fail-fast: first seed error aborts; never silently averages fewer seeds.
Same inputs + seeds = same scorecard. Test set touched once per artifact,
after thresholds frozen.

Usage (Kaggle/Colab GPU):
  uv run python -m src.sweep --version-base intent-slot-v2 --seeds 42 43 44
Outputs: <reports-dir>/thresholds-<seed>.json, report-<seed>.json, scorecard.json
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

METRICS = [
    "exact_command_accuracy",
    "coverage",
    "selective_accuracy",
]


def run(*cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def build_scorecard(reports_dir, version_base, seeds):
    """Aggregate per-seed report JSONs → scorecard dict (mean/std, sample std)."""
    reports = [Path(reports_dir) / f"report-{s}.json" for s in seeds]
    rows = [json.loads(p.read_text()) for p in reports]
    assert all(r.get("status") == "ok" for r in rows), "a seed report is not ok"

    def agg(vals):
        return {"mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "seeds": vals}

    summary = {m: agg([r[m] for r in rows]) for m in METRICS}
    summary["intent_macro_f1"] = agg([r["intent"]["macro"]["f1"] for r in rows])
    summary["entity_micro_f1"] = agg([r["entity_span"]["micro"]["f1"] for r in rows])
    summary["entity_macro_f1"] = agg([r["entity_span"]["macro_f1"] for r in rows])
    summary["unknown_f1"] = agg([r["unknown"]["f1"] for r in rows])
    return {"version_base": version_base, "seeds": list(seeds), "n": len(rows),
            "summary": summary, "reports": [str(p) for p in reports]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version-base", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--min-coverage", type=float, default=0.95)
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        ap.error("duplicate seeds — each seed must run exactly once")

    exe = sys.executable
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for seed in args.seeds:
        version = f"{args.version_base}-s{seed}"
        tpath = args.reports_dir / f"thresholds-{seed}.json"
        rpath = args.reports_dir / f"report-{seed}.json"
        train = [exe, "-m", "src.train", "--version", version, "--seed", str(seed)]
        if args.epochs is not None:
            train += ["--epochs", str(args.epochs)]
        if args.max_samples is not None:
            train += ["--max-samples", str(args.max_samples)]
        artifact = f"artifacts/{version}"
        fit = [exe, "-m", "src.evaluate", "--artifact", artifact,
               "--split", "validation", "--fit-calibration", str(tpath),
               "--min-coverage", str(args.min_coverage)]
        test = [exe, "-m", "src.evaluate", "--artifact", artifact,
                "--output", str(rpath), "--thresholds", str(tpath)]
        if args.dry_run:
            for cmd in (train, fit, test):
                print("+", " ".join(cmd))
        else:
            run(*train)
            run(*fit)
            run(*test)
        reports.append(rpath)

    if args.dry_run:
        return

    card = build_scorecard(args.reports_dir, args.version_base, args.seeds)
    out = args.reports_dir / "scorecard.json"
    out.write_text(json.dumps(card, indent=2))
    print(f"\nscorecard: {args.version_base} over seeds {args.seeds}")
    for m, s in card["summary"].items():
        print(f"  {m:24s} mean {s['mean']:.4f} std {s['std']:.4f} "
              f"seeds {[round(v, 4) for v in s['seeds']]}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
