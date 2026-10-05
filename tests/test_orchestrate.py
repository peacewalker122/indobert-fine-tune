import hashlib
import json
import os
from pathlib import Path

import pytest
from src import orchestrate

RUN_ID = "00000000-0000-4000-8000-000000000001"
POLICY_RAW = (Path(__file__).resolve().parents[1] / "ci/model-eval-policy.json").read_bytes()
POLICY = json.loads(POLICY_RAW)
RESULT_FIELDS = {
    "schema_version",
    "run_id",
    "attempt",
    "outcome",
    "evidence_uri",
    "evidence_sha256",
    "gate",
    "metrics",
    "parameters",
    "model_release_uri",
    "model_manifest_sha256",
    "release_metadata",
    "error_kind",
    "error",
}


def _spec():
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "model_name": "intent",
        "model_version": "v1",
        "dataset": {
            "name": "d",
            "version": "v1",
            "uri": "s3://data/datasets/d/v1",
            "manifest_sha256": "a" * 64,
        },
        "base_model": "base",
        "base_revision": "rev",
        "trainer_image": "trainer@sha256:x",
        "git_commit": "abc",
        "policy_sha256": hashlib.sha256(POLICY_RAW).hexdigest(),
        "parameters": {
            "seed": 42,
            "learning_rate": 2e-5,
            "epochs": 5,
            "batch_size": 16,
            "max_length": 64,
        },
        "model_release_uri": "s3://models/models/intent/v1",
        "evidence_bucket": "runs",
    }


def _stage(root: Path, outcome="prepared", attempt=1):
    root.mkdir(parents=True, exist_ok=True)
    (root / "artifact").mkdir(parents=True)
    (root / "artifact" / "model.safetensors").write_bytes(b"weights")
    (root / "artifact" / "training_metadata.json").write_text(
        json.dumps(
            {
                "best_epoch": 1.0,
                "best_step": 10,
                "best_validation_exact_command_accuracy": 0.95,
            }
        )
    )
    thresholds_raw = b'{"method":"maxprob-sweep-v1","threshold":0.5}'
    (root / "thresholds.json").write_bytes(thresholds_raw)
    spec_raw = json.dumps(_spec(), sort_keys=True).encode()
    policy_raw = POLICY_RAW
    (root / "run_spec.json").write_bytes(spec_raw)
    (root / "policy.json").write_bytes(policy_raw)
    artifact_manifest = orchestrate.manifest(root / "artifact")
    artifact_sha = hashlib.sha256(
        json.dumps(artifact_manifest, sort_keys=True).encode()
    ).hexdigest()
    test_sha = "c" * 64
    dataset_sha = "a" * 64
    report = {
        "status": "ok",
        "split": "test",
        "dataset_sha256": dataset_sha,
        "test_sha256": test_sha,
        "n": 3,
        "artifact_sha256": artifact_sha,
        "calibration_sha256": hashlib.sha256(thresholds_raw).hexdigest(),
        "ood_held_out": False,
        "calibration": {"method": "maxprob-sweep-v1", "threshold": 0.5},
        "exact_command_accuracy": 0.95,
        "coverage": 0.95,
        "intent": {"macro": {"f1": 0.95}},
        "entity_span": {"micro": {"f1": 0.95}, "macro_f1": 0.95},
        "unknown": {"f1": 0.95},
    }
    (root / "report.json").write_text(json.dumps(report))
    gate = orchestrate.evaluate_reports(
        {42: report},
        POLICY,
        dataset_sha,
        expected_test_sha256=test_sha,
        expected_test_count=3,
        expected_artifact_sha256=artifact_sha,
        expected_calibration_sha256=hashlib.sha256(thresholds_raw).hexdigest(),
    )
    evidence = {
        "run_id": RUN_ID,
        "attempt": attempt,
        "spec_sha256": hashlib.sha256(spec_raw).hexdigest(),
        "policy_sha256": hashlib.sha256(policy_raw).hexdigest(),
        "dataset_sha256": dataset_sha,
        "frozen_test": {"sha256": test_sha, "count": 3},
        "gate": gate,
        "files": orchestrate.manifest(root),
    }
    evidence_raw = json.dumps(evidence, sort_keys=True, indent=2).encode()
    (root / "evidence.json").write_bytes(evidence_raw)
    result = {
        **orchestrate._result(
            _spec(),
            attempt,
            outcome,
            evidence_uri=orchestrate._evidence_uri("runs", RUN_ID, attempt),
            evidence_sha256=hashlib.sha256(evidence_raw).hexdigest(),
            gate=gate,
            metrics=orchestrate.report_metrics(report),
            parameters=_spec()["parameters"],
        ),
    }
    result["evidence_uri"] = f"s3://runs/runs/{RUN_ID}/attempts/{attempt}"
    (root / "result.json").write_text(json.dumps(result))
    return spec_raw, policy_raw


def test_staged_evidence_integrity_and_corruption(tmp_path):
    spec_raw, policy_raw = _stage(tmp_path)
    evidence = orchestrate._validate_evidence(tmp_path, RUN_ID, 1, spec_raw, policy_raw)
    assert evidence["run_id"] == RUN_ID
    (tmp_path / "artifact" / "weights.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="manifest mismatch|corruption"):
        orchestrate._validate_evidence(tmp_path, RUN_ID, 1, spec_raw, policy_raw)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["parameters"].update(max_length=128),
        lambda value: value.update(model_release_uri="s3://models/wrong/version"),
        lambda value: value["dataset"].update(manifest_sha256="not-a-sha"),
        lambda value: value.update(run_id="../../escape"),
    ],
)
def test_run_spec_rejects_mutated_identity_or_unsupported_length(tmp_path, mutate):
    spec = _spec()
    mutate(spec)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        orchestrate._load_spec(path)


def test_publish_rejects_staged_rejected_result_before_upload(tmp_path, monkeypatch):
    spec_raw, policy_raw = _stage(tmp_path / "staged", outcome="rejected")
    args = type(
        "Args",
        (),
        {
            "evidence_bucket": "runs",
            "policy": tmp_path / "policy.json",
            "attempt": 1,
            "model_bucket": "models",
        },
    )()
    args.policy.write_bytes(policy_raw)

    class Client:
        def list_objects(self, *args, **kwargs):
            raise AssertionError("rejected evidence must not publish")

    def download(client, bucket, prefix, root):
        for path in (tmp_path / "staged").rglob("*"):
            if path.is_file():
                destination = root / path.relative_to(tmp_path / "staged")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(path.read_bytes())
        return True

    monkeypatch.setattr(orchestrate, "_download_tree", download)
    with pytest.raises(SystemExit) as exited:
        orchestrate.publish(args, _spec(), spec_raw, Client(), tmp_path)
    assert exited.value.code == 2


def test_recover_validates_existing_commit_without_training(tmp_path, capsys):
    spec = _spec()
    spec_raw = json.dumps(spec, sort_keys=True).encode()
    weight = b"weights"
    evidence_raw = json.dumps(
        {
            "run_id": RUN_ID,
            "attempt": 1,
            "spec_sha256": hashlib.sha256(spec_raw).hexdigest(),
            "policy_sha256": spec["policy_sha256"],
        }
    ).encode()
    metadata = {
        "schema_version": 2,
        "model": spec["model_name"],
        "version": spec["model_version"],
        "status": "passed",
        "run_id": spec["run_id"],
        "spec_sha256": hashlib.sha256(spec_raw).hexdigest(),
        "dataset": {"manifest_sha256": spec["dataset"]["manifest_sha256"]},
        "policy_sha256": spec["policy_sha256"],
        "gate": {"result": "passed", "failures": []},
        "metrics": {"exact_command_accuracy": 0.95},
        "parameters": spec["parameters"],
        "evidence": {
            "uri": f"s3://runs/runs/{RUN_ID}/attempts/1",
            "sha256": hashlib.sha256(evidence_raw).hexdigest(),
            "attempt": 1,
        },
        "files": {"model.bin": {"sha256": hashlib.sha256(weight).hexdigest()}},
    }
    raw_metadata = json.dumps(metadata).encode()

    class Response:
        def __init__(self, content):
            self.content = content
            self.closed = False

        def read(self, size=-1):
            if size < 0:
                value, self.content = self.content, b""
                return value
            value, self.content = self.content[:size], self.content[size:]
            return value

        def close(self):
            self.closed = True

        def release_conn(self):
            pass

    class Client:
        def get_object(self, bucket, name):
            if name.endswith("metadata.json"):
                return Response(raw_metadata)
            if name.endswith("evidence.json"):
                return Response(evidence_raw)
            return Response(weight)

    with pytest.raises(SystemExit) as exited:
        orchestrate.recover(
            type(
                "Args",
                (),
                {
                    "model_bucket": "models",
                    "attempt": 2,
                    "policy": tmp_path / "policy.json",
                },
            )(),
            spec,
            spec_raw,
            Client(),
            tmp_path,
        )
    assert exited.value.code == 0
    result = json.loads(capsys.readouterr().out.split(orchestrate.MARKER, 1)[1])
    assert result.keys() == RESULT_FIELDS
    assert result["outcome"] == "passed"
    assert result["attempt"] == 2
    assert result["evidence_uri"] == metadata["evidence"]["uri"]
    assert result["evidence_sha256"] == metadata["evidence"]["sha256"]
    assert result["gate"] == metadata["gate"]
    assert result["metrics"] == metadata["metrics"]
    assert result["parameters"] == metadata["parameters"]
    assert result["model_manifest_sha256"] == hashlib.sha256(raw_metadata).hexdigest()


def test_recover_rejects_corrupt_committed_checksum(tmp_path, capsys):
    spec = _spec()
    spec_raw = json.dumps(spec, sort_keys=True).encode()
    metadata = {
        "schema_version": 2,
        "model": "intent",
        "version": "v1",
        "status": "passed",
        "run_id": RUN_ID,
        "spec_sha256": hashlib.sha256(spec_raw).hexdigest(),
        "dataset": {"manifest_sha256": "a" * 64},
        "policy_sha256": spec["policy_sha256"],
        "parameters": spec["parameters"],
        "files": {"model.bin": {"sha256": hashlib.sha256(b"original").hexdigest()}},
    }
    raw_metadata = json.dumps(metadata).encode()

    class Response:
        def __init__(self, content):
            self.content = content

        def read(self, size=-1):
            if size < 0:
                value, self.content = self.content, b""
                return value
            value, self.content = self.content[:size], self.content[size:]
            return value

        def close(self):
            pass

        def release_conn(self):
            pass

    class Client:
        def get_object(self, bucket, name):
            return Response(raw_metadata if name.endswith("metadata.json") else b"tampered")

    with pytest.raises(SystemExit) as exited:
        orchestrate.recover(
            type("Args", (), {"model_bucket": "models", "attempt": 1})(),
            spec,
            spec_raw,
            Client(),
            tmp_path,
        )
    assert exited.value.code == 3
    result = json.loads(capsys.readouterr().out.split(orchestrate.MARKER, 1)[1])
    assert result["error_kind"] == "conflict"
    assert "checksum mismatch" in result["error"]


def test_recover_publishes_verified_prior_attempt_without_redownload_or_retrain(
    tmp_path, monkeypatch, capsys
):
    spec = _spec()
    spec_raw, policy_raw = _stage(tmp_path / "staged", attempt=1)
    (tmp_path / "policy.json").write_bytes(policy_raw)
    stored = {}
    downloaded = []

    class Missing(Exception):
        code = "NoSuchKey"

    class Response:
        def __init__(self, content):
            self.content = content

        def read(self, size=-1):
            if size < 0:
                value, self.content = self.content, b""
                return value
            value, self.content = self.content[:size], self.content[size:]
            return value

        def close(self):
            pass

        def release_conn(self):
            pass

    class Object:
        def __init__(self, name):
            self.object_name = name
            self.is_dir = False

    class Client:
        def list_objects(self, bucket, prefix, recursive):
            return [Object(key) for key in stored if key.startswith(prefix)]

        def get_object(self, bucket, name):
            if name not in stored:
                raise Missing("NoSuchKey")
            return Response(stored[name])

        def fget_object(self, bucket, name, destination):
            downloaded.append(name)
            Path(destination).write_bytes(stored[name])

        def fput_object(self, bucket, name, source):
            stored[name] = Path(source).read_bytes()

    evidence_prefix = f"runs/{RUN_ID}/attempts/1"
    for path in (tmp_path / "staged").rglob("*"):
        if path.is_file():
            stored[f"{evidence_prefix}/{path.relative_to(tmp_path / 'staged')}"] = path.read_bytes()

    def no_subprocess(*args, **kwargs):
        pytest.fail("recover/publish must not retrain or re-evaluate the model")

    monkeypatch.setattr(orchestrate.subprocess, "run", no_subprocess)
    with pytest.raises(SystemExit) as exited:
        orchestrate.recover(
            type(
                "Args",
                (),
                {
                    "model_bucket": "models",
                    "attempt": 2,
                    "evidence_bucket": "runs",
                    "policy": tmp_path / "policy.json",
                },
            )(),
            spec,
            spec_raw,
            Client(),
            tmp_path,
        )
    assert exited.value.code == 0
    result = json.loads(capsys.readouterr().out.split(orchestrate.MARKER, 1)[1])
    assert result.keys() == RESULT_FIELDS
    assert result["outcome"] == "passed"
    assert result["attempt"] == 2
    assert result["evidence_uri"] == f"s3://runs/runs/{RUN_ID}/attempts/1"
    assert (
        result["evidence_sha256"]
        == hashlib.sha256(stored[f"{evidence_prefix}/evidence.json"]).hexdigest()
    )
    assert result["release_metadata"]["evidence"]["attempt"] == 1
    release_training = json.loads(stored["models/intent/v1/training_metadata.json"])
    assert type(release_training["best_epoch"]) is int
    assert type(release_training["selection"]["best_epoch"]) is int
    assert downloaded and len(downloaded) == len(set(downloaded))
    assert all(key.startswith(evidence_prefix) for key in downloaded)


@pytest.mark.parametrize("partial", ["mismatched", "unmanifested"])
def test_publish_rejects_wrong_partial_release_without_overwrite(tmp_path, partial):
    release = tmp_path / "release"
    release.mkdir()
    local = release / "model.bin"
    local.write_bytes(b"expected")
    files = {
        "model.bin": {
            "sha256": hashlib.sha256(b"expected").hexdigest(),
            "bytes": len(b"expected"),
        }
    }
    existing_name = (
        "models/intent/v1/model.bin" if partial == "mismatched" else "models/intent/v1/orphan.bin"
    )
    existing_bytes = b"wrong" if partial == "mismatched" else b"orphan"
    uploaded = []

    class Object:
        object_name = existing_name
        is_dir = False

    class Response:
        def __init__(self):
            self.content = existing_bytes

        def read(self, size=-1):
            if size < 0:
                value, self.content = self.content, b""
                return value
            value, self.content = self.content[:size], self.content[size:]
            return value

        def close(self):
            pass

        def release_conn(self):
            pass

    class Client:
        def list_objects(self, bucket, prefix, recursive):
            return [Object()]

        def get_object(self, bucket, name):
            return Response()

        def fput_object(self, *args):
            uploaded.append(args)

    with pytest.raises(orchestrate.ReleaseConflictError):
        orchestrate._stage_release_files(Client(), "models", "models/intent/v1", release, files)
    assert uploaded == []


def test_trainer_image_keeps_code_and_venv_outside_scratch():
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile.trainer").read_text()
    assert "WORKDIR /app" in dockerfile
    assert "PATH=/app/.venv/bin:$PATH" in dockerfile
    assert "uv sync --frozen --no-dev --no-install-project" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert "HF_HOME=/work/cache" in dockerfile
    assert "WORKDIR /work" not in dockerfile
    assert "/work/.venv" not in dockerfile
    assert orchestrate.DEFAULT_POLICY.is_absolute()
    assert orchestrate.DEFAULT_POLICY.is_file()


def test_evidence_download_rejects_path_traversal(tmp_path):
    class Object:
        object_name = "attempts/1/../../outside"
        is_dir = False

    class Client:
        def list_objects(self, bucket, prefix, recursive):
            return [Object()]

        def fget_object(self, *args):
            pytest.fail("unsafe object path should be rejected before download")

    with pytest.raises(ValueError, match="unsafe evidence object path"):
        orchestrate._download_tree(Client(), "runs", "attempts/1", tmp_path)


def test_cli_scratch_permissions_allow_cross_uid_cleanup_and_restore_umask(tmp_path, monkeypatch):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(_spec()))
    work = tmp_path / "work"
    monkeypatch.setattr(
        orchestrate.sys,
        "argv",
        [
            "orchestrate",
            "--mode",
            "recover",
            "--run-spec-json",
            str(spec),
            "--attempt",
            "1",
            "--work-dir",
            str(work),
        ],
    )
    monkeypatch.setattr(orchestrate, "_minio", lambda: object())

    def recover(args, spec, spec_raw, client, scratch):
        nested = scratch / "evidence" / "artifact"
        nested.mkdir(parents=True)
        (nested / "weights.bin").write_bytes(b"fixture")
        raise SystemExit(3)

    monkeypatch.setattr(orchestrate, "recover", recover)
    previous = os.umask(0o027)
    try:
        with pytest.raises(SystemExit) as exited:
            orchestrate.main()
        assert exited.value.code == 3
        assert work.stat().st_mode & 0o777 == 0o777
        assert (work / "evidence" / "artifact").stat().st_mode & 0o777 == 0o777
        assert (work / "evidence" / "artifact" / "weights.bin").stat().st_mode & 0o777 == 0o666
        restored = os.umask(0o027)
        assert restored == 0o027
    finally:
        os.umask(previous)
