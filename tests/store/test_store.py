from __future__ import annotations

import random
from pathlib import Path

import pytest

from piginot.store import (
    ContentCorruptionError,
    ContentNotFoundError,
    InvalidDigestError,
    ObjectStore,
    RetentionTier,
    StoreFormatError,
    StoreValidationError,
    content_hash,
)


def test_content_hash_vectors_freeze_the_raw_byte_framing() -> None:
    vectors = (
        (b"", "164df0678594538ecfa5b083db7c65b974c7f67fd61cfdc03e7213be2a85694b"),
        (
            b"abc",
            "a3d30ba786d6e20d1b33d8e028f03fad7b2c4e91baa06b509836e94266ed8436",
        ),
        (
            bytes(range(256)),
            "d0dee553fc50e0a8c82a31c5c2b47d0fd9794f196813134ed9b91734ba53d0cf",
        ),
    )

    for payload, expected in vectors:
        assert content_hash(payload) == expected


def test_one_thousand_deterministic_blobs_roundtrip_byte_identically(
    tmp_path: Path,
) -> None:
    store = ObjectStore.create(tmp_path / "store")
    rng = random.Random(20260820)

    for index in range(1000):
        payload = rng.randbytes(index % 257)
        digest = store.put(payload)

        assert store.get(digest) == payload


def test_corrupt_object_is_detected_and_not_returned(tmp_path: Path) -> None:
    store = ObjectStore.create(tmp_path / "store")
    digest = store.put(b"trusted bytes")
    object_path = store.object_path(digest)

    raw = bytearray(object_path.read_bytes())
    raw[0] ^= 0xFF
    object_path.write_bytes(bytes(raw))

    with pytest.raises(ContentCorruptionError, match=digest):
        store.get(digest)


def test_open_requires_a_valid_store_marker(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(StoreFormatError, match="store"):
        ObjectStore.open(missing)

    unmarked = tmp_path / "unmarked"
    unmarked.mkdir()
    with pytest.raises(StoreFormatError, match="store"):
        ObjectStore.open(unmarked)
    with pytest.raises(StoreFormatError, match="store"):
        ObjectStore(unmarked)

    created = ObjectStore.create(tmp_path / "store")
    assert ObjectStore.open(created.root).root == created.root
    assert ObjectStore(created.root).root == created.root


def test_create_rejects_an_invalid_existing_marker(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    (root / "store.json").write_text("{}", encoding="utf-8")

    with pytest.raises(StoreFormatError, match="store"):
        ObjectStore.create(root)


def test_create_is_idempotent_but_rejects_unmarked_nonempty_roots(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    first = ObjectStore.create(root)
    assert ObjectStore.create(root).root == first.root

    unmarked = tmp_path / "unmarked"
    unmarked.mkdir()
    (unmarked / "foreign").write_bytes(b"x")
    with pytest.raises(StoreFormatError, match="non-empty"):
        ObjectStore.create(unmarked)

    not_a_directory = tmp_path / "file"
    not_a_directory.write_bytes(b"x")
    with pytest.raises(StoreFormatError, match="directory"):
        ObjectStore.create(not_a_directory)


def test_hash_paths_reject_malformed_or_traversal_input(tmp_path: Path) -> None:
    store = ObjectStore.create(tmp_path / "store")

    for digest in ("../escape", "g" * 64, "A" * 64, "f" * 63, "f" * 65):
        with pytest.raises(InvalidDigestError, match="hash"):
            store.object_path(digest)
        with pytest.raises(InvalidDigestError, match="hash"):
            store.get(digest)


def test_orphan_bytes_without_metadata_are_not_visible(tmp_path: Path) -> None:
    store = ObjectStore.create(tmp_path / "store")
    payload = b"complete but uncommitted"
    digest = content_hash(payload)
    destination = store.object_path(digest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)

    with pytest.raises(ContentNotFoundError, match=digest):
        store.get(digest)


def test_presence_and_argument_validation_are_fail_closed(tmp_path: Path) -> None:
    store = ObjectStore.create(tmp_path / "store")
    digest = store.put(b"present")

    assert store.has_object(digest)
    assert not store.has_object("f" * 64)
    with pytest.raises(ContentNotFoundError):
        store.info("e" * 64)
    with pytest.raises(StoreValidationError, match="bytes"):
        store.put(bytearray(b"not immutable"))  # type: ignore[arg-type]
    with pytest.raises(StoreValidationError, match="RetentionTier"):
        store.put(b"payload", tier="RETAINED")  # type: ignore[arg-type]


def test_missing_committed_bytes_are_reported_not_returned(tmp_path: Path) -> None:
    store = ObjectStore.create(tmp_path / "store")
    digest = store.put(b"payload")
    store.object_path(digest).unlink()

    with pytest.raises(ContentNotFoundError, match="missing"):
        store.get(digest)


def test_repeated_put_promotes_to_the_stronger_tier(tmp_path: Path) -> None:
    store = ObjectStore.create(tmp_path / "store")
    digest = store.put(b"tier", tier=RetentionTier.PRUNABLE)

    assert store.put(b"tier", tier=RetentionTier.PERMANENT) == digest
    assert store.info(digest).retention is RetentionTier.PERMANENT
