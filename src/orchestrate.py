"""CPU trainer job: pinned MinIO dataset through gated candidate registration."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .quality_gate import build_release_metadata, evaluate_reports, report_metrics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _declared_files(metadata: dict[str, Any]) -> dict[str, str]:
    declared: dict[str, str] = {}
    files = metadata.get("files", {})
    if isinstance(files, dict):
        for name, value in files.items():
            digest = value.get("sha256") if isinstance(value, dict) else value
            if isinstance(digest, str):
                declared[name] = digest.lower()
    for split in metadata.get("splits", {}).values():
        if isinstance(split, dict) and isinstance(split.get("path"), str) and isinstance(split.get("sha256"), str):
            declared[split["path"]] = split["sha256"].lower()
    required = {"train.jsonl", "validation.jsonl", "test.jsonl"}
    if not required.issubset(declared):
        raise ValueError("dataset metadata must declare SHA256 for train, validation, and test")
    return declared


def fetch_dataset(client: Any, bucket: str, prefix: str, destination: Path,
                  expected_name: str, expected_version: str) -> tuple[dict[str, Any], str]:
    prefix = prefix.strip("/")
    response = client.get_object(bucket, f"{prefix}/metadata.json")
    try:
        raw = response.read()
    finally:
        response.close()
        response.release_conn()
    metadata = json.loads(raw)
    if metadata.get("name") != expected_name or metadata.get("version") != expected_version:
        raise ValueError("dataset metadata does not match pinned name/version")
    declared = _declared_files(metadata)
    destination.mkdir(parents=True)
    for name, digest in declared.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe dataset path: {name}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        client.fget_object(bucket, f"{prefix}/{name}", str(target))
        if len(digest) != 64 or sha256_file(target) != digest:
            raise ValueError(f"dataset checksum mismatch: {name}")
    identity = hashlib.sha256(
        json.dumps(declared, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (destination / "metadata.json").write_bytes(raw)
    return metadata, identity


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, cwd=cwd, check=True, env=env)


def manifest(root: Path) -> dict[str, dict[str, Any]]:
    return {str(path.relative_to(root)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(root.rglob("*")) if path.is_file() and path.name != "metadata.json"}


def upload_immutable(client: Any, bucket: str, prefix: str, release: Path,
                     metadata: dict[str, Any]) -> None:
    prefix = prefix.strip("/")
    existing: set[str] = set()
    for obj in client.list_objects(bucket, prefix=f"{prefix}/", recursive=True):
        if obj.is_dir:
            continue
        relative = obj.object_name.removeprefix(f"{prefix}/")
        if relative == "metadata.json":
            raise FileExistsError(f"model release is already committed: {prefix}")
        existing.add(relative)
    unexpected = existing - metadata["files"].keys()
    if unexpected:
        raise RuntimeError(f"unrecognized objects in incomplete model release: {sorted(unexpected)}")
    for name, expected in metadata["files"].items():
        path = release / name
        object_name = f"{prefix}/{name}"
        if name not in existing:
            client.fput_object(bucket, object_name, str(path))
        response = client.get_object(bucket, object_name)
        try:
            actual = sha256_stream(response)
        finally:
            response.close()
            response.release_conn()
        if actual != expected["sha256"]:
            raise RuntimeError(f"uploaded checksum mismatch: {name}")
    marker = release / "metadata.json"
    marker.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    client.fput_object(bucket, f"{prefix}/metadata.json", str(marker))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--model-name", required=True, help="Pinned Hugging Face model repository")
    parser.add_argument("--model-revision", required=True, help="Pinned immutable model commit")
    parser.add_argument("--release-version", required=True)
    parser.add_argument("--dataset-bucket", required=True)
    parser.add_argument("--model-bucket", required=True)
    parser.add_argument("--dataset-prefix")
    parser.add_argument("--model-prefix")
    parser.add_argument("--policy", type=Path, default=Path("ci/model-eval-policy.json"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--trainer-image", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--control-run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from minio import Minio
    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    endpoint = os.environ["MINIO_ENDPOINT"]
    client = Minio(endpoint, access_key=os.environ["MINIO_ACCESS_KEY"],
                   secret_key=os.environ["MINIO_SECRET_KEY"],
                   secure=os.getenv("MINIO_SECURE", "false").lower() in {"1", "true", "yes"})
    policy = json.loads(args.policy.read_text())
    owned_work = args.work_dir is None
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="trainer-"))
    work.mkdir(parents=True, exist_ok=True)
    try:
        data = work / "data"
        dataset_prefix = args.dataset_prefix or f"datasets/{args.dataset_name}/releases/{args.dataset_version}"
        dataset_meta, dataset_sha = fetch_dataset(
            client, args.dataset_bucket, dataset_prefix, data, args.dataset_name, args.dataset_version
        )
        reports_dir = work / "reports"
        reports_dir.mkdir()
        seed = policy["seed"]
        artifact_version = args.release_version
        train = [sys.executable, "-m", "src.train", "--version", artifact_version, "--seed", str(seed),
                 "--model-name", args.model_name, "--model-revision", args.model_revision,
                 "--output-dir", str(work / "output")]
        if args.epochs is not None:
            train += ["--epochs", str(args.epochs)]
        if args.max_samples is not None:
            train += ["--max-samples", str(args.max_samples)]
        run(train, work)
        artifact = work / "artifacts" / artifact_version
        thresholds = reports_dir / "thresholds.json"
        run([sys.executable, "-m", "src.evaluate", "--artifact", str(artifact), "--data-dir", str(data),
             "--split", "validation", "--fit-calibration", str(thresholds),
             "--min-coverage", str(policy["min_coverage"])], work)
        report_path = reports_dir / "report.json"
        run([sys.executable, "-m", "src.evaluate", "--artifact", str(artifact), "--data-dir", str(data),
             "--split", "test", "--thresholds", str(thresholds), "--output", str(report_path)], work)
        report = json.loads(report_path.read_text())
        report["dataset_sha256"] = dataset_sha
        report["ood_held_out"] = bool(dataset_meta.get("ood", {}).get("held_out", False))
        report_path.write_text(json.dumps(report, indent=2))
        reports = {seed: report}
        training = json.loads((artifact / "training_metadata.json").read_text())
        validation_score = float(training["metrics"]["eval_exact_command_accuracy"])
        gate = evaluate_reports(reports, policy, dataset_sha)
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        with mlflow.start_run(run_name=args.release_version) as tracking_run:
            mlflow.log_params({"dataset_name": args.dataset_name, "dataset_version": args.dataset_version,
                               "base_model": args.model_name, "base_revision": args.model_revision,
                               "seed": seed, "epochs": args.epochs or "configured"})
            mlflow.set_tag("smartcare_control_run_id", args.control_run_id)
            mlflow.log_metrics(gate["per_seed"][str(seed)])
            mlflow.log_artifact(str(report_path), artifact_path="evaluation")
            mlflow.log_artifact(str(thresholds), artifact_path="evaluation")
            mlflow.set_tag("quality_gate", gate["result"])
            if gate["result"] != "passed":
                raise RuntimeError("quality gate failed: " + "; ".join(gate["failures"]))

            release = work / "release"
            shutil.copytree(artifact, release)
            shutil.copy2(thresholds, release / "thresholds.json")
            report_names = ["report.json"]
            shutil.copy2(report_path, release / report_names[0])
            training_path = release / "training_metadata.json"
            training = json.loads(training_path.read_text())
            training.update({"model": "multilingual-intent-slot", "version": args.release_version,
                             "selected_seed": seed, "dataset_sha256": dataset_sha})
            training_path.write_text(json.dumps(training, indent=2))
            files = manifest(release)
            metadata = build_release_metadata(
                version=args.release_version,
                dataset={"name": args.dataset_name, "version": args.dataset_version, "sha256": dataset_sha},
                source={"git_commit": args.git_commit, "trainer_image": args.trainer_image,
                        "base_revision": args.model_revision}, policy=policy, seeds=[seed],
                selected_seed=seed, selection_value=validation_score,
                metrics=report_metrics(report), reports=report_names, files=files,
            )
            model_prefix = args.model_prefix or f"models/multilingual-intent-slot/{args.release_version}"
            upload_immutable(client, args.model_bucket, model_prefix, release, metadata)
            registry = MlflowClient()
            try:
                registry.get_registered_model("multilingual-intent-slot")
            except MlflowException as exc:
                if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
                    raise
                registry.create_registered_model("multilingual-intent-slot")
            model_version = registry.create_model_version(
                "multilingual-intent-slot", f"s3://{args.model_bucket}/{model_prefix}", tracking_run.info.run_id
            )
            registry.set_model_version_tag("multilingual-intent-slot", model_version.version,
                                           "lifecycle", "candidate")
            registry.set_model_version_tag("multilingual-intent-slot", model_version.version,
                                           "release_version", args.release_version)
            print("SMARTCARE_ML_RUN_RESULT=" + json.dumps({
                "mlflow_run_id": tracking_run.info.run_id,
                "model_release_uri": f"s3://{args.model_bucket}/{model_prefix}",
            }), flush=True)
    finally:
        if owned_work:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
