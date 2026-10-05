"""Validate and immutably publish a canonical BIO dataset release to MinIO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .dataset import load_jsonl, validate_record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1 << 20), b""):
        digest.update(chunk)
    return digest.hexdigest()


def validate_splits(data_dir: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    group_splits: dict[str, str] = {}
    record_ids: set[str] = set()
    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}.jsonl"
        if not path.is_file():
            raise ValueError(f"missing dataset split: {path.name}")
        records = load_jsonl(path)
        if not records:
            raise ValueError(f"empty dataset split: {split}")
        for line_no, record in records:
            validate_record(record, line_no)
            record_id = record.get("id")
            if record_id is not None:
                if record_id in record_ids:
                    raise ValueError(f"duplicate record id: {record_id}")
                record_ids.add(record_id)
            group = record.get("group_id")
            if not group:
                raise ValueError(f"record in {split} has no group_id")
            prior = group_splits.setdefault(group, split)
            if prior != split:
                raise ValueError(f"group_id {group} crosses {prior}/{split} split boundary")
        files[path.name] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "count": len(records),
        }
    return files


def publish(
    client: Any,
    bucket: str,
    name: str,
    version: str,
    data_dir: Path,
    provenance: dict[str, Any],
    ood: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    if not name or not version or "/" in name or "/" in version or not provenance:
        raise ValueError("dataset name/version and nonempty provenance are required")
    files = validate_splits(data_dir)
    prefix = f"datasets/{name}/releases/{version}"
    if any(
        not obj.is_dir for obj in client.list_objects(bucket, prefix=f"{prefix}/", recursive=True)
    ):
        raise FileExistsError(f"dataset release path is not empty: {prefix}")
    for filename, item in files.items():
        client.fput_object(bucket, f"{prefix}/{filename}", str(data_dir / filename))
        response = client.get_object(bucket, f"{prefix}/{filename}")
        try:
            digest = sha256_stream(response)
        finally:
            response.close()
            response.release_conn()
        if digest != item["sha256"]:
            raise ValueError(f"published dataset checksum mismatch: {filename}")
    metadata = {
        "schema_version": 1,
        "name": name,
        "version": version,
        "files": files,
        "splits": {
            split: {
                "path": f"{split}.jsonl",
                "sha256": files[f"{split}.jsonl"]["sha256"],
                "count": files[f"{split}.jsonl"]["count"],
            }
            for split in ("train", "validation", "test")
        },
        "provenance": provenance,
        "ood": ood or {"held_out": False},
    }
    raw = json.dumps(metadata, sort_keys=True, indent=2).encode()
    import io

    client.put_object(
        bucket,
        f"{prefix}/metadata.json",
        io.BytesIO(raw),
        length=len(raw),
        content_type="application/json",
    )
    return hashlib.sha256(raw).hexdigest(), metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--provenance-json", type=Path, required=True)
    parser.add_argument("--ood-held-out", action="store_true")
    args = parser.parse_args()
    from minio import Minio

    client = Minio(
        os.environ["MINIO_ENDPOINT"],
        access_key=os.environ["MINIO_ACCESS_KEY"],
        secret_key=os.environ["MINIO_SECRET_KEY"],
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
    )
    digest, metadata = publish(
        client,
        os.environ["DATASET_BUCKET"],
        args.name,
        args.version,
        args.data_dir,
        json.loads(args.provenance_json.read_text()),
        {"held_out": args.ood_held_out},
    )
    print(
        json.dumps(
            {
                "uri": f"s3://{os.environ['DATASET_BUCKET']}/datasets/{args.name}/releases/{args.version}",
                "manifest_sha256": digest,
                "metadata": metadata,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
