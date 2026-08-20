from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from piginot.store import ContentCorruptionError, ObjectStore, StoreFormatError


@pytest.mark.parametrize(
    "document",
    [
        '{"schema_version":1,"schema_version":1}',
        '{"schema_version":1,"unexpected":true}',
        "not-json",
    ],
)
def test_content_metadata_is_strict_json(document: str, tmp_path: Path) -> None:
    store = ObjectStore.create(tmp_path / "store")
    digest = store.put(b"payload")
    store._content_metadata_path(digest).write_text(document, encoding="utf-8")

    with pytest.raises(StoreFormatError, match="metadata"):
        store.get(digest)


def test_content_metadata_must_match_its_address(tmp_path: Path) -> None:
    store = ObjectStore.create(tmp_path / "store")
    digest = store.put(b"payload")
    metadata = store._content_metadata_path(digest)
    metadata.write_text(
        metadata.read_text(encoding="utf-8").replace(digest, "f" * 64),
        encoding="utf-8",
    )

    with pytest.raises(StoreFormatError, match="address"):
        store.get(digest)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("size_bytes", -1),
        ("size_bytes", "7"),
        ("materialized", 1),
        ("retention", 1),
        ("retention", "UNKNOWN"),
    ],
)
def test_content_metadata_field_types_and_values_are_validated(
    field: str,
    value: object,
    tmp_path: Path,
) -> None:
    store = ObjectStore.create(tmp_path / "store")
    digest = store.put(b"payload")
    metadata = store._content_metadata_path(digest)
    document = json.loads(metadata.read_text(encoding="utf-8"))
    document[field] = value
    metadata.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(StoreFormatError, match="metadata"):
        store.info(digest)


def test_recorded_size_is_verified_on_read(tmp_path: Path) -> None:
    store = ObjectStore.create(tmp_path / "store")
    digest = store.put(b"payload")
    metadata = store._content_metadata_path(digest)
    document = json.loads(metadata.read_text(encoding="utf-8"))
    document["size_bytes"] = 999
    metadata.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ContentCorruptionError, match="size"):
        store.get(digest)


def test_store_marker_values_are_validated(tmp_path: Path) -> None:
    store = ObjectStore.create(tmp_path / "store")
    marker = store.root / "store.json"
    document = json.loads(marker.read_text(encoding="utf-8"))
    document["store_format"] = "future-format"
    marker.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(StoreFormatError, match="unsupported"):
        ObjectStore.open(store.root)


def test_object_symlink_is_never_followed(tmp_path: Path) -> None:
    store = ObjectStore.create(tmp_path / "store")
    digest = store.put(b"payload")
    destination = store.object_path(digest)
    target = tmp_path / "target"
    target.write_bytes(b"payload")
    destination.unlink()
    try:
        os.symlink(target, destination)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(ContentCorruptionError, match="symbolic"):
        store.get(digest)
