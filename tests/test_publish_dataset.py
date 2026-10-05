import hashlib
import json
from pathlib import Path

import pytest
from src.publish_dataset import publish, validate_splits


def _record(group: str, **changes):
    value = {
        "id": f"id-{group}",
        "group_id": group,
        "tokens": ["hello"],
        "intent": "UNKNOWN",
        "slots": ["O"],
        "language": "en",
    }
    value.update(changes)
    return value


def _splits(root: Path, train, validation, test):
    for name, records in (("train", train), ("validation", validation), ("test", test)):
        (root / f"{name}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )


def test_publisher_accepts_grouped_canonical_splits(tmp_path):
    _splits(tmp_path, [_record("g1")], [_record("g2")], [_record("g3")])
    files = validate_splits(tmp_path)
    assert set(files) == {"train.jsonl", "validation.jsonl", "test.jsonl"}
    assert files["test.jsonl"]["count"] == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"tokens": ["two", "tokens"], "slots": ["O"]},
        {"slots": ["I-LOCATION"]},
        {"group_id": ""},
        {"intent": "NOT_A_REAL_INTENT"},
    ],
)
def test_publisher_rejects_asymmetric_malformed_records(tmp_path, bad):
    invalid = _record("g1", **bad)
    _splits(tmp_path, [invalid], [_record("g2")], [_record("g3")])
    with pytest.raises((ValueError, KeyError)):
        validate_splits(tmp_path)


def test_publisher_rejects_group_leakage_between_splits(tmp_path):
    _splits(tmp_path, [_record("shared")], [_record("g2")], [_record("shared", id="second-shared")])
    with pytest.raises(ValueError, match="crosses"):
        validate_splits(tmp_path)


def test_publication_checksums_files_and_commits_metadata_last(tmp_path):
    _splits(tmp_path, [_record("g1")], [_record("g2")], [_record("g3")])

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def read(self, size=-1):
            payload, self.payload = self.payload[:size], self.payload[size:]
            return payload

        def close(self):
            pass

        def release_conn(self):
            pass

    class Client:
        def __init__(self):
            self.objects = {}
            self.put_order = []

        def list_objects(self, bucket, prefix, recursive):
            return iter(())

        def fput_object(self, bucket, name, filename):
            self.objects[name] = Path(filename).read_bytes()
            self.put_order.append(name)

        def get_object(self, bucket, name):
            return Response(self.objects[name])

        def put_object(self, bucket, name, stream, length, content_type):
            self.objects[name] = stream.read()
            self.put_order.append(name)

    client = Client()
    digest, metadata = publish(
        client, "datasets", "telecom", "d1", tmp_path, {"source_revision": "abc"}
    )
    manifest_raw = client.objects["datasets/telecom/releases/d1/metadata.json"]
    assert digest == hashlib.sha256(manifest_raw).hexdigest()
    assert metadata["files"]["test.jsonl"]["count"] == 1
    assert client.put_order[-1] == "datasets/telecom/releases/d1/metadata.json"


def test_publication_does_not_commit_after_staged_corruption(tmp_path):
    _splits(tmp_path, [_record("g1")], [_record("g2")], [_record("g3")])

    class Response:
        def __init__(self):
            self.payload = b"corrupted"

        def read(self, size=-1):
            payload, self.payload = self.payload[:size], self.payload[size:]
            return payload

        def close(self):
            pass

        def release_conn(self):
            pass

    class Client:
        committed = False

        def list_objects(self, bucket, prefix, recursive):
            return iter(())

        def fput_object(self, bucket, name, filename):
            pass

        def get_object(self, bucket, name):
            return Response()

        def put_object(self, *args, **kwargs):
            self.committed = True

    client = Client()
    with pytest.raises(ValueError, match="checksum mismatch"):
        publish(client, "datasets", "telecom", "d1", tmp_path, {"source_revision": "abc"})
    assert not client.committed
