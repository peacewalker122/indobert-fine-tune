"""No-MLflow SmartCare trainer gateway CLI. Prepare has no model-write path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any
import uuid

from .quality_gate import evaluate_reports, report_metrics

MARKER = "SMARTCARE_ML_RUN_RESULT="
DEFAULT_POLICY = Path(__file__).resolve().parents[1] / "ci/model-eval-policy.json"


class ReleaseConflictError(ValueError):
    """The immutable model-release prefix contains conflicting objects."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_digest(client: Any, bucket: str, name: str) -> tuple[str, int]:
    response = client.get_object(bucket, name)
    digest = hashlib.sha256()
    size = 0
    try:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    finally:
        response.close()
        response.release_conn()
    return digest.hexdigest(), size


def _object(client: Any, bucket: str, name: str) -> bytes:
    response = client.get_object(bucket, name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def _declared_files(metadata: dict[str, Any]) -> dict[str, str]:
    declared: dict[str, str] = {}

    def add(name: Any, digest: Any) -> None:
        if (
            not isinstance(name, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError(f"invalid dataset file checksum declaration: {name!r}")
        if name in declared and declared[name] != digest:
            raise ValueError(f"conflicting dataset checksums for {name}")
        declared[name] = digest

    for name, value in metadata.get("files", {}).items():
        digest = value.get("sha256") if isinstance(value, dict) else value
        add(name, digest)
    for split in metadata.get("splits", {}).values():
        if (
            isinstance(split, dict)
            and isinstance(split.get("path"), str)
            and isinstance(split.get("sha256"), str)
        ):
            add(split["path"], split["sha256"])
    if not {"train.jsonl", "validation.jsonl", "test.jsonl"}.issubset(declared):
        raise ValueError("dataset metadata must declare train, validation, and test checksums")
    return declared


def fetch_dataset(
    client: Any,
    bucket: str,
    prefix: str,
    destination: Path,
    expected_name: str,
    expected_version: str,
    expected_manifest_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    prefix = prefix.strip("/")
    raw = _object(client, bucket, f"{prefix}/metadata.json")
    if expected_manifest_sha256 and hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError("dataset metadata manifest checksum mismatch")
    metadata = json.loads(raw)
    if metadata.get("name") != expected_name or metadata.get("version") != expected_version:
        raise ValueError("dataset metadata does not match pinned name/version")
    declared = _declared_files(metadata)
    destination.mkdir(parents=True)
    for name, digest in declared.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or len(digest) != 64:
            raise ValueError(f"unsafe dataset file declaration: {name}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        client.fget_object(bucket, f"{prefix}/{name}", str(target))
        if sha256_file(target) != digest:
            raise ValueError(f"dataset checksum mismatch: {name}")
    (destination / "metadata.json").write_bytes(raw)
    identity = hashlib.sha256(
        json.dumps(declared, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return metadata, identity


def manifest(root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_file() and str(relative) not in {
            "metadata.json",
            "result.json",
            "evidence.json",
        }:
            result[str(relative)] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return result


def _minio() -> Any:
    from minio import Minio

    return Minio(
        os.environ["MINIO_ENDPOINT"],
        access_key=os.environ["MINIO_ACCESS_KEY"],
        secret_key=os.environ["MINIO_SECRET_KEY"],
        secure=os.getenv("MINIO_SECURE", "false").lower() in {"1", "true", "yes"},
    )


def _emit(result: dict[str, Any], exit_code: int = 0) -> None:
    print(MARKER + json.dumps(result, sort_keys=True), flush=True)
    raise SystemExit(exit_code)


def _result(spec: dict[str, Any], attempt: int, outcome: str, **values: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": spec["run_id"],
        "attempt": attempt,
        "outcome": outcome,
        "evidence_uri": None,
        "evidence_sha256": None,
        "gate": {"result": "failed", "failures": []},
        "metrics": {},
        "parameters": {},
        "model_release_uri": None,
        "model_manifest_sha256": None,
        "release_metadata": None,
        "error_kind": None,
        "error": None,
        **values,
    }


def _load_spec(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    spec = json.loads(raw)
    required = {
        "schema_version",
        "run_id",
        "model_name",
        "model_version",
        "dataset",
        "base_model",
        "base_revision",
        "trainer_image",
        "git_commit",
        "policy_sha256",
        "parameters",
        "model_release_uri",
        "evidence_bucket",
    }
    if (
        not isinstance(spec, dict)
        or not required.issubset(spec)
        or type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
    ):
        raise ValueError("invalid RunSpec")
    dataset = spec["dataset"]
    params = spec["parameters"]
    if not isinstance(dataset, dict) or not {"name", "version", "uri", "manifest_sha256"}.issubset(
        dataset
    ):
        raise ValueError("invalid RunSpec dataset identity")
    if any(
        not isinstance(dataset[key], str) or not dataset[key] for key in ("name", "version", "uri")
    ):
        raise ValueError("invalid RunSpec dataset fields")
    if any(
        not isinstance(spec[key], str) or not spec[key]
        for key in (
            "run_id",
            "model_name",
            "model_version",
            "base_model",
            "base_revision",
            "trainer_image",
            "git_commit",
        )
    ):
        raise ValueError("invalid RunSpec identity fields")
    if any("/" in spec[key] for key in ("model_name", "model_version")):
        raise ValueError("model name and version must each be one URI path component")
    try:
        if str(uuid.UUID(spec["run_id"])) != spec["run_id"]:
            raise ValueError("run_id must be a canonical lowercase UUID")
    except ValueError as exc:
        raise ValueError("run_id must be a canonical lowercase UUID") from exc
    if not isinstance(params, dict) or not {
        "learning_rate",
        "epochs",
        "batch_size",
        "max_length",
        "seed",
    }.issubset(params):
        raise ValueError("invalid RunSpec training parameters")
    if params["max_length"] != 64:
        raise ValueError("trainer supports only the frozen max_length=64 contract")
    if (
        isinstance(params["epochs"], bool)
        or not isinstance(params["epochs"], int)
        or params["epochs"] < 1
        or isinstance(params["batch_size"], bool)
        or not isinstance(params["batch_size"], int)
        or params["batch_size"] < 1
        or isinstance(params["seed"], bool)
        or not isinstance(params["seed"], int)
        or isinstance(params["learning_rate"], bool)
        or not isinstance(params["learning_rate"], (float, int))
        or not math.isfinite(params["learning_rate"])
        or params["learning_rate"] <= 0
    ):
        raise ValueError("invalid RunSpec hyperparameters")
    for field in ("policy_sha256",):
        if (
            not isinstance(spec[field], str)
            or len(spec[field]) != 64
            or any(c not in "0123456789abcdef" for c in spec[field])
        ):
            raise ValueError(f"invalid RunSpec {field}")
    manifest_sha = dataset["manifest_sha256"]
    if (
        not isinstance(manifest_sha, str)
        or len(manifest_sha) != 64
        or any(c not in "0123456789abcdef" for c in manifest_sha)
    ):
        raise ValueError("invalid RunSpec dataset manifest_sha256")
    uri_parts = spec["model_release_uri"].removeprefix("s3://").split("/", 1)
    expected_release_path = f"models/{spec['model_name']}/{spec['model_version']}"
    dataset_uri_parts = dataset["uri"].removeprefix("s3://").split("/", 1)
    if (
        len(uri_parts) != 2
        or uri_parts[1] != expected_release_path
        or not uri_parts[0]
        or not dataset["uri"].startswith("s3://")
        or len(dataset_uri_parts) != 2
        or not dataset_uri_parts[0]
        or not isinstance(spec["evidence_bucket"], str)
        or not spec["evidence_bucket"]
    ):
        raise ValueError("RunSpec must pin model release URI and evidence bucket")
    return spec, raw


def _validate_evidence(
    root: Path,
    run_id: str,
    attempt: int,
    spec_raw: bytes,
    policy_raw: bytes,
    expected_spec_sha: str | None = None,
) -> dict[str, Any]:
    evidence_raw = (root / "evidence.json").read_bytes()
    evidence = json.loads(evidence_raw)
    if not isinstance(evidence, dict):
        raise ValueError("evidence record must be a JSON object")
    if evidence.get("run_id") != run_id or evidence.get("attempt") != attempt:
        raise ValueError("evidence run/attempt identity mismatch")
    if evidence.get("spec_sha256") != hashlib.sha256(spec_raw).hexdigest():
        raise ValueError("evidence RunSpec mismatch")
    if expected_spec_sha and evidence["spec_sha256"] != expected_spec_sha:
        raise ValueError("committed release RunSpec mismatch")
    if evidence.get("policy_sha256") != hashlib.sha256(policy_raw).hexdigest():
        raise ValueError("evidence policy mismatch")
    expected = evidence.get("files")
    if not isinstance(expected, dict) or expected != manifest(root):
        raise ValueError("evidence file manifest mismatch")
    for name, item in expected.items():
        path = root / name
        if (
            not path.is_file()
            or path.stat().st_size != item.get("bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise ValueError(f"evidence corruption: {name}")
    return evidence


def _evidence_uri(bucket: str, run_id: str, attempt: int) -> str:
    return f"s3://{bucket}/runs/{run_id}/attempts/{attempt}"


def _validate_staged_result(
    root: Path,
    spec: dict[str, Any],
    spec_raw: bytes,
    policy_raw: bytes,
    evidence_attempt: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if hashlib.sha256(policy_raw).hexdigest() != spec["policy_sha256"]:
        raise ValueError("policy checksum does not match frozen RunSpec")
    result_path = root / "result.json"
    evidence_path = root / "evidence.json"
    if not result_path.is_file() or not evidence_path.is_file():
        raise ValueError("durable attempt result or evidence missing")
    result = json.loads(result_path.read_text())
    if not isinstance(result, dict):
        raise ValueError("staged result must be a JSON object")
    if (
        result.get("run_id") != spec["run_id"]
        or result.get("attempt") != evidence_attempt
        or result.get("outcome") != "prepared"
        or result.get("evidence_uri")
        != _evidence_uri(spec["evidence_bucket"], spec["run_id"], evidence_attempt)
        or result.get("evidence_sha256") != sha256_file(evidence_path)
    ):
        raise ValueError("staged result/evidence identity mismatch")
    evidence = _validate_evidence(root, spec["run_id"], evidence_attempt, spec_raw, policy_raw)
    if evidence.get("gate", {}).get("result") != "passed":
        raise ValueError("evidence gate rejected")
    return result, evidence


def _attempts_with_evidence(client: Any, bucket: str, run_id: str, current: int) -> list[int]:
    prefix = f"runs/{run_id}/attempts/"
    attempts: set[int] = set()
    pattern = re.compile(rf"^{re.escape(prefix)}([1-9][0-9]*)/evidence\.json$")
    for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
        if obj.is_dir:
            continue
        match = pattern.fullmatch(obj.object_name)
        if match and int(match.group(1)) <= current:
            attempts.add(int(match.group(1)))
    return sorted(attempts, reverse=True)


def _stage_release_files(
    client: Any,
    bucket: str,
    prefix: str,
    release: Path,
    files: dict[str, dict[str, Any]],
) -> None:
    expected = set(files)
    existing: dict[str, str] = {}
    for obj in client.list_objects(bucket, prefix=f"{prefix}/", recursive=True):
        if obj.is_dir:
            continue
        name = obj.object_name.removeprefix(f"{prefix}/")
        if name == "metadata.json":
            raise ReleaseConflictError("model release already committed")
        if name not in expected:
            raise ReleaseConflictError(f"unmanifested object already exists: {name}")
        existing[name] = obj.object_name
    for name, object_name in existing.items():
        actual, size = _remote_digest(client, bucket, object_name)
        declared = files[name]
        if actual != declared["sha256"] or size != declared["bytes"]:
            raise ReleaseConflictError(f"existing staged object conflicts: {name}")
    for name, item in files.items():
        if name in existing:
            continue
        local = release / name
        object_name = f"{prefix}/{name}"
        client.fput_object(bucket, object_name, str(local))
        actual, size = _remote_digest(client, bucket, object_name)
        if actual != item["sha256"] or size != item["bytes"]:
            raise ValueError(f"uploaded model checksum mismatch: {name}")


def prepare(
    args: argparse.Namespace, spec: dict[str, Any], spec_raw: bytes, client: Any, work: Path
) -> None:
    ds = spec["dataset"]
    data = work / "data"
    uri_bucket = ds["uri"].split("/")[2]
    dataset_bucket = args.dataset_bucket or uri_bucket
    if dataset_bucket != uri_bucket:
        raise ValueError("runtime dataset bucket differs from frozen dataset URI")
    if args.evidence_bucket and args.evidence_bucket != spec["evidence_bucket"]:
        raise ValueError("evidence bucket differs from frozen RunSpec")
    args.evidence_bucket = spec["evidence_bucket"]
    prefix = "/".join(ds["uri"].split("/")[3:])
    dataset_meta, dataset_sha = fetch_dataset(
        client, dataset_bucket, prefix, data, ds["name"], ds["version"], ds["manifest_sha256"]
    )
    policy_raw = Path(args.policy).read_bytes()
    if hashlib.sha256(policy_raw).hexdigest() != spec["policy_sha256"]:
        raise ValueError("policy checksum does not match frozen RunSpec")
    policy = json.loads(policy_raw)
    params = spec["parameters"]
    artifact_version = f"run-{spec['run_id']}-{args.attempt}"
    cmd = [
        sys.executable,
        "-m",
        "src.train",
        "--version",
        artifact_version,
        "--seed",
        str(params["seed"]),
        "--model-name",
        spec["base_model"],
        "--model-revision",
        spec["base_revision"],
        "--data-dir",
        str(data),
        "--output-dir",
        str(work / "output"),
        "--learning-rate",
        str(params["learning_rate"]),
        "--batch-size",
        str(params["batch_size"]),
        "--epochs",
        str(params["epochs"]),
    ]
    if params.get("max_samples") is not None:
        cmd += ["--max-samples", str(params["max_samples"])]
    subprocess.run(
        cmd,
        cwd=work,
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )
    artifact = work / "artifacts" / artifact_version
    report_dir = work / "reports"
    report_dir.mkdir()
    thresholds = report_dir / "thresholds.json"
    run_eval = [sys.executable, "-m", "src.evaluate"]
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    subprocess.run(
        run_eval
        + [
            "--artifact",
            str(artifact),
            "--data-dir",
            str(data),
            "--split",
            "validation",
            "--fit-calibration",
            str(thresholds),
            "--min-coverage",
            str(policy["min_coverage"]),
        ],
        cwd=work,
        check=True,
        env=env,
    )
    report_path = report_dir / "report.json"
    subprocess.run(
        run_eval
        + [
            "--artifact",
            str(artifact),
            "--data-dir",
            str(data),
            "--split",
            "test",
            "--thresholds",
            str(thresholds),
            "--output",
            str(report_path),
        ],
        cwd=work,
        check=True,
        env=env,
    )
    report = json.loads(report_path.read_text())
    split_decl = next(
        (v for v in dataset_meta.get("splits", {}).values() if v.get("path") == "test.jsonl"), {}
    )
    declared_test_sha = _declared_files(dataset_meta)["test.jsonl"]
    test_path = data / "test.jsonl"
    with test_path.open(encoding="utf-8") as test_file:
        frozen_test_count = sum(1 for line in test_file if line.strip())
    if split_decl.get("sha256", declared_test_sha) != declared_test_sha:
        raise ValueError("dataset split and file checksum declarations conflict")
    if split_decl.get("count", frozen_test_count) != frozen_test_count:
        raise ValueError("dataset split count does not match frozen test file")
    artifact_manifest = manifest(artifact)
    artifact_sha = hashlib.sha256(
        json.dumps(artifact_manifest, sort_keys=True).encode()
    ).hexdigest()
    report.update(
        dataset_sha256=dataset_sha,
        ood_held_out=bool(dataset_meta.get("ood", {}).get("held_out", False)),
        artifact_sha256=artifact_sha,
        calibration_sha256=sha256_file(thresholds),
    )
    report_path.write_text(json.dumps(report, indent=2))
    gate = evaluate_reports(
        {params["seed"]: report},
        policy,
        dataset_sha,
        expected_test_sha256=declared_test_sha,
        expected_test_count=frozen_test_count,
        expected_artifact_sha256=artifact_sha,
        expected_calibration_sha256=sha256_file(thresholds),
        smoke=params.get("max_samples") is not None,
    )
    evidence_root = work / "evidence"
    evidence_root.mkdir()
    shutil.copytree(artifact, evidence_root / "artifact")
    shutil.copy2(report_path, evidence_root / "report.json")
    shutil.copy2(thresholds, evidence_root / "thresholds.json")
    shutil.copy2(
        work / "output" / "trainer_log_history.json", evidence_root / "training_history.json"
    )
    (evidence_root / "run_spec.json").write_bytes(spec_raw)
    (evidence_root / "policy.json").write_bytes(policy_raw)
    files = manifest(evidence_root)
    evidence = {
        "run_id": spec["run_id"],
        "attempt": args.attempt,
        "spec_sha256": hashlib.sha256(spec_raw).hexdigest(),
        "policy_sha256": hashlib.sha256(policy_raw).hexdigest(),
        "dataset_sha256": dataset_sha,
        "frozen_test": {
            "sha256": declared_test_sha,
            "count": frozen_test_count,
        },
        "gate": gate,
        "files": files,
    }
    evidence_bytes = json.dumps(evidence, sort_keys=True, indent=2).encode()
    (evidence_root / "evidence.json").write_bytes(evidence_bytes)
    result = _result(
        spec,
        args.attempt,
        "prepared" if gate["result"] == "passed" else "rejected",
        evidence_uri=f"s3://{args.evidence_bucket}/runs/{spec['run_id']}/attempts/{args.attempt}",
        evidence_sha256=hashlib.sha256(evidence_bytes).hexdigest(),
        gate=gate,
        metrics=report_metrics(report),
        parameters=params,
        error_kind=None if gate["result"] == "passed" else "quality_rejected",
        error=None if gate["result"] == "passed" else "; ".join(gate["failures"]),
    )
    # Evidence and result are durable before printing. Prepare has no model publisher operation.
    _upload_tree(
        client,
        args.evidence_bucket,
        f"runs/{spec['run_id']}/attempts/{args.attempt}",
        evidence_root,
        include_result=result,
    )
    _emit(result, 0 if gate["result"] == "passed" else 2)


def _upload_tree(
    client: Any, bucket: str, prefix: str, root: Path, include_result: dict[str, Any] | None = None
) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            client.fput_object(bucket, f"{prefix}/{path.relative_to(root)}", str(path))
    if include_result is not None:
        target = root / "result.json"
        target.write_text(json.dumps(include_result, sort_keys=True, indent=2))
        client.fput_object(bucket, f"{prefix}/result.json", str(target))


def _download_tree(client: Any, bucket: str, prefix: str, root: Path) -> bool:
    objects = list(client.list_objects(bucket, prefix=prefix.rstrip("/") + "/", recursive=True))
    if not objects:
        return False
    root.mkdir(parents=True, exist_ok=True)
    for obj in objects:
        if obj.is_dir:
            continue
        relative = obj.object_name.removeprefix(prefix.rstrip("/") + "/")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or not relative_path.parts:
            raise ValueError(f"unsafe evidence object path: {relative}")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        client.fget_object(bucket, obj.object_name, str(target))
    return True


def publish(
    args: argparse.Namespace,
    spec: dict[str, Any],
    spec_raw: bytes,
    client: Any,
    work: Path,
    *,
    evidence_root: Path | None = None,
    evidence_attempt: int | None = None,
) -> None:
    if args.evidence_bucket and args.evidence_bucket != spec["evidence_bucket"]:
        raise ValueError("evidence bucket differs from frozen RunSpec")
    args.evidence_bucket = spec["evidence_bucket"]
    evidence_attempt = args.attempt if evidence_attempt is None else evidence_attempt
    if evidence_root is None:
        evidence_root = work / "evidence"
        if evidence_root.exists():
            shutil.rmtree(evidence_root)
        prefix = f"runs/{spec['run_id']}/attempts/{evidence_attempt}"
        if not _download_tree(client, args.evidence_bucket, prefix, evidence_root):
            _emit(
                _result(
                    spec,
                    args.attempt,
                    "not_found",
                    error_kind="transient",
                    error="prepared evidence not found",
                )
            )
    assert evidence_root is not None
    policy_raw = Path(args.policy).read_bytes()
    try:
        staged_result, evidence = _validate_staged_result(
            evidence_root, spec, spec_raw, policy_raw, evidence_attempt
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        _emit(
            _result(
                spec,
                args.attempt,
                "rejected",
                error_kind="quality_rejected",
                error=str(exc),
            ),
            2,
        )
    report = json.loads((evidence_root / "report.json").read_text())
    policy = json.loads(policy_raw)
    artifact_files = {
        key.removeprefix("artifact/"): value
        for key, value in manifest(evidence_root).items()
        if key.startswith("artifact/")
    }
    artifact_sha = hashlib.sha256(json.dumps(artifact_files, sort_keys=True).encode()).hexdigest()
    calibration_sha = sha256_file(evidence_root / "thresholds.json")
    frozen_test = evidence["frozen_test"]
    recalculated = evaluate_reports(
        {spec["parameters"]["seed"]: report},
        policy,
        evidence["dataset_sha256"],
        expected_test_sha256=frozen_test["sha256"],
        expected_test_count=frozen_test["count"],
        expected_artifact_sha256=artifact_sha,
        expected_calibration_sha256=calibration_sha,
        smoke=spec["parameters"].get("max_samples") is not None,
    )
    if recalculated["result"] != "passed":
        _emit(
            _result(
                spec,
                args.attempt,
                "rejected",
                gate=recalculated,
                error_kind="quality_rejected",
                error="frozen-policy gate failed",
            ),
            2,
        )
    # Upload all release data before committing metadata last.
    release = work / "release"
    if release.exists():
        shutil.rmtree(release)
    shutil.copytree(evidence_root / "artifact", release)
    for name in ("report.json", "thresholds.json"):
        shutil.copy2(evidence_root / name, release / name)
    training_path = release / "training_metadata.json"
    training = json.loads(training_path.read_text())
    train_spec = spec["parameters"]
    best_epoch, best_step = training.get("best_epoch"), training.get("best_step")
    if (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, (int, float))
        or not math.isfinite(best_epoch)
        or not float(best_epoch).is_integer()
    ):
        _emit(
            _result(
                spec,
                args.attempt,
                "rejected",
                gate=recalculated,
                error_kind="quality_rejected",
                error="training metadata does not contain a complete integer best_epoch",
            ),
            2,
        )
    best_epoch = int(best_epoch)
    training["best_epoch"] = best_epoch
    training.update(
        {
            "model": spec["model_name"],
            "version": spec["model_version"],
            "schema_version": 2,
            "run_id": spec["run_id"],
            "spec_sha256": hashlib.sha256(spec_raw).hexdigest(),
            "dataset_sha256": evidence["dataset_sha256"],
            "selection": {
                "strategy": "best_epoch",
                "metric": "validation_exact_command_accuracy",
                "value": training.get("best_validation_exact_command_accuracy"),
                "best_epoch": best_epoch,
                "checkpoint_step": best_step,
            },
        }
    )
    training_path.write_text(json.dumps(training, indent=2))
    from .quality_gate import build_release_metadata

    files = manifest(release)
    metadata = build_release_metadata(
        version=spec["model_version"],
        dataset={
            "name": spec["dataset"]["name"],
            "version": spec["dataset"]["version"],
            "sha256": evidence["dataset_sha256"],
        },
        source={
            "git_commit": spec["git_commit"],
            "trainer_image": spec["trainer_image"],
            "base_model": spec["base_model"],
            "base_revision": spec["base_revision"],
        },
        policy=policy,
        seeds=[train_spec["seed"]],
        selected_seed=train_spec["seed"],
        selection_value=float(training["selection"]["value"]),
        metrics=report_metrics(report),
        reports=["report.json"],
        files=files,
    )
    metadata.update(
        schema_version=2,
        model=spec["model_name"],
        run_id=spec["run_id"],
        spec_sha256=hashlib.sha256(spec_raw).hexdigest(),
        dataset={**metadata["dataset"], "manifest_sha256": spec["dataset"]["manifest_sha256"]},
        policy_sha256=spec["policy_sha256"],
        parameters=train_spec,
        selection={
            "strategy": "best_epoch",
            "metric": "validation_exact_command_accuracy",
            "value": training["selection"]["value"],
            "best_epoch": best_epoch,
            "checkpoint_step": best_step,
        },
        gate=recalculated,
        metrics=report_metrics(report),
        evidence={
            "uri": staged_result["evidence_uri"],
            "sha256": staged_result["evidence_sha256"],
            "attempt": evidence_attempt,
        },
    )
    uri = spec["model_release_uri"]
    parts = uri.removeprefix("s3://").split("/", 1)
    if len(parts) != 2:
        raise ValueError("invalid frozen model_release_uri")
    bucket, model_prefix = parts
    if args.model_bucket and args.model_bucket != bucket:
        raise ValueError("runtime model bucket differs from frozen model_release_uri")
    committed = f"{model_prefix}/metadata.json"
    try:
        _stage_release_files(client, bucket, model_prefix, release, files)
    except ReleaseConflictError as exc:
        _emit(
            _result(
                spec,
                args.attempt,
                "failed",
                error_kind="conflict",
                error=str(exc),
            ),
            3,
        )
    meta_path = release / "metadata.json"
    meta_path.write_text(json.dumps(metadata, sort_keys=True, indent=2))
    client.fput_object(bucket, committed, str(meta_path))
    result = _result(
        spec,
        args.attempt,
        "passed",
        gate=recalculated,
        metrics=report_metrics(report),
        parameters=train_spec,
        evidence_uri=staged_result["evidence_uri"],
        evidence_sha256=staged_result["evidence_sha256"],
        model_release_uri=spec["model_release_uri"],
        model_manifest_sha256=sha256_file(meta_path),
        release_metadata=metadata,
    )
    _emit(result)


def recover(
    args: argparse.Namespace, spec: dict[str, Any], spec_raw: bytes, client: Any, work: Path
) -> None:
    uri = spec["model_release_uri"]
    parts = uri.removeprefix("s3://").split("/", 1)
    if len(parts) != 2:
        raise ValueError("invalid frozen model_release_uri")
    bucket, prefix = parts
    if args.model_bucket and args.model_bucket != bucket:
        raise ValueError("runtime model bucket differs from frozen model_release_uri")
    try:
        raw = _object(client, bucket, f"{prefix}/metadata.json")
    except Exception as exc:
        code = getattr(exc, "code", "")
        if code not in {"NoSuchKey", "NoSuchObject", "NotFound"} and not any(
            marker in str(exc) for marker in ("NoSuchKey", "NoSuchObject", "NotFound")
        ):
            raise
        policy_raw = Path(args.policy).read_bytes()
        evidence_root = work / "recovery-evidence"
        for evidence_attempt in _attempts_with_evidence(
            client, spec["evidence_bucket"], spec["run_id"], args.attempt
        ):
            if evidence_root.exists():
                shutil.rmtree(evidence_root)
            evidence_prefix = f"runs/{spec['run_id']}/attempts/{evidence_attempt}"
            if not _download_tree(client, spec["evidence_bucket"], evidence_prefix, evidence_root):
                continue
            try:
                _validate_staged_result(evidence_root, spec, spec_raw, policy_raw, evidence_attempt)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
            # The already downloaded evidence is passed through; publish must not download it again.
            publish(
                args,
                spec,
                spec_raw,
                client,
                work,
                evidence_root=evidence_root,
                evidence_attempt=evidence_attempt,
            )
        _emit(
            _result(
                spec,
                args.attempt,
                "not_found",
                error_kind="transient",
                error="committed release and verified prior-attempt evidence not found",
            )
        )
        raise
    metadata = json.loads(raw)
    if (
        metadata.get("schema_version") != 2
        or metadata.get("model") != spec["model_name"]
        or metadata.get("version") != spec["model_version"]
        or metadata.get("status") != "passed"
        or metadata.get("run_id") != spec["run_id"]
        or metadata.get("spec_sha256") != hashlib.sha256(spec_raw).hexdigest()
        or metadata.get("dataset", {}).get("manifest_sha256") != spec["dataset"]["manifest_sha256"]
        or metadata.get("policy_sha256") != spec["policy_sha256"]
        or metadata.get("parameters") != spec["parameters"]
    ):
        _emit(
            _result(
                spec,
                args.attempt,
                "failed",
                error_kind="conflict",
                error="committed release identity mismatch",
            ),
            3,
        )
    release_files = metadata.get("files")
    if not isinstance(release_files, dict) or not release_files:
        _emit(
            _result(
                spec,
                args.attempt,
                "failed",
                error_kind="conflict",
                error="committed release file manifest missing",
            ),
            3,
        )
    for name, item in release_files.items():
        if not isinstance(item, dict) or not isinstance(item.get("sha256"), str):
            _emit(
                _result(
                    spec,
                    args.attempt,
                    "failed",
                    error_kind="conflict",
                    error=f"invalid committed checksum: {name}",
                ),
                3,
            )
        actual, size = _remote_digest(client, bucket, f"{prefix}/{name}")
        if actual != item.get("sha256") or ("bytes" in item and size != item["bytes"]):
            _emit(
                _result(
                    spec,
                    args.attempt,
                    "failed",
                    error_kind="conflict",
                    error=f"committed checksum mismatch: {name}",
                ),
                3,
            )
    lineage = metadata.get("evidence")
    if (
        not isinstance(lineage, dict)
        or type(lineage.get("attempt")) is not int
        or lineage["attempt"] < 1
        or lineage.get("uri")
        != _evidence_uri(spec["evidence_bucket"], spec["run_id"], lineage["attempt"])
        or not isinstance(lineage.get("sha256"), str)
        or len(lineage["sha256"]) != 64
        or not isinstance(metadata.get("gate"), dict)
        or metadata["gate"].get("result") != "passed"
        or not isinstance(metadata.get("metrics"), dict)
        or not isinstance(metadata.get("parameters"), dict)
    ):
        _emit(
            _result(
                spec,
                args.attempt,
                "failed",
                error_kind="conflict",
                error="committed release lineage or RunResult fields missing",
            ),
            3,
        )
    evidence_raw = _object(
        client,
        spec["evidence_bucket"],
        f"runs/{spec['run_id']}/attempts/{lineage['attempt']}/evidence.json",
    )
    if hashlib.sha256(evidence_raw).hexdigest() != lineage["sha256"]:
        _emit(
            _result(
                spec,
                args.attempt,
                "failed",
                error_kind="conflict",
                error="committed evidence checksum mismatch",
            ),
            3,
        )
    try:
        evidence_record = json.loads(evidence_raw)
    except json.JSONDecodeError:
        evidence_record = None
    if (
        not isinstance(evidence_record, dict)
        or evidence_record.get("run_id") != spec["run_id"]
        or evidence_record.get("attempt") != lineage["attempt"]
        or evidence_record.get("spec_sha256") != hashlib.sha256(spec_raw).hexdigest()
        or evidence_record.get("policy_sha256") != spec["policy_sha256"]
    ):
        _emit(
            _result(
                spec,
                args.attempt,
                "failed",
                error_kind="conflict",
                error="committed evidence identity mismatch",
            ),
            3,
        )
    _emit(
        _result(
            spec,
            args.attempt,
            "passed",
            evidence_uri=lineage["uri"],
            evidence_sha256=lineage["sha256"],
            gate=metadata["gate"],
            metrics=metadata["metrics"],
            parameters=metadata["parameters"],
            model_release_uri=spec["model_release_uri"],
            model_manifest_sha256=hashlib.sha256(raw).hexdigest(),
            release_metadata=metadata,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "publish", "recover"), required=True)
    parser.add_argument("--run-spec-json", type=Path, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--evidence-bucket")
    parser.add_argument("--dataset-bucket")
    parser.add_argument("--model-bucket")
    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY,
    )
    parser.add_argument("--work-dir", type=Path, default=Path("/work"))
    args = parser.parse_args()
    spec, spec_raw = _load_spec(args.run_spec_json)
    if args.attempt < 1:
        raise ValueError("attempt must be positive")
    # Disposable scratch crosses trainer/worker UIDs; both must be able to remove it.
    previous_umask = os.umask(0)
    try:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        client = _minio()
        if args.mode == "prepare":
            if args.model_bucket:
                raise ValueError("prepare must not receive model-publisher target")
            prepare(args, spec, spec_raw, client, args.work_dir)
        elif args.mode == "publish":
            publish(args, spec, spec_raw, client, args.work_dir)
        else:
            recover(args, spec, spec_raw, client, args.work_dir)
    except SystemExit:
        raise
    except Exception as exc:
        if isinstance(exc, ReleaseConflictError):
            kind = "conflict"
        else:
            kind = (
                "invalid_input" if isinstance(exc, (ValueError, FileNotFoundError)) else "internal"
            )
        _emit(_result(spec, args.attempt, "failed", error_kind=kind, error=str(exc)), 1)
    finally:
        os.umask(previous_umask)


if __name__ == "__main__":
    main()
