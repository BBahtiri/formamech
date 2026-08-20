from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from piginot.store import (
    CorruptObjectError,
    DivergentRecipeError,
    MissingObjectError,
    ObjectStore,
    ObjectStoreError,
    RetentionTier,
)


def test_random_blobs_roundtrip_byte_identically(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "store")

    for index in range(1000):
        payload = os.urandom(index % 257)
        digest = store.put(payload)

        assert store.get(digest) == payload


def test_corrupt_object_is_detected_and_not_returned(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "store")
    digest = store.put(b"trusted bytes")
    object_path = store.object_path(digest)

    raw = bytearray(object_path.read_bytes())
    raw[0] ^= 0xFF
    object_path.write_bytes(bytes(raw))

    with pytest.raises(CorruptObjectError, match=digest):
        store.get(digest)


def test_tmp_file_left_by_interrupted_write_is_not_visible(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "store")
    tmp_file = store.root / "objects" / ".tmp-interrupted"
    tmp_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file.write_bytes(b"partial")

    assert not store.has_object("f" * 64)
    with pytest.raises(MissingObjectError):
        store.get("f" * 64)


def test_killed_writer_leaves_no_partial_object_visible(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "store")
    script = """
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
tmp = root / "objects" / "f0" / ".tmp-killed-writer"
tmp.parent.mkdir(parents=True)
tmp.write_bytes(b"partial")
os._exit(17)
"""

    result = subprocess.run(
        [sys.executable, "-c", script, str(store.root)],
        check=False,
    )

    assert result.returncode == 17
    assert not store.has_object("f" * 64)
    with pytest.raises(MissingObjectError):
        store.get("f" * 64)


def test_hash_paths_reject_malformed_or_traversal_input(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "store")

    for digest in ("../escape", "g" * 64, "A" * 64, "f" * 63, "f" * 65):
        with pytest.raises(ObjectStoreError, match="hash"):
            store.object_path(digest)


def test_recipe_divergence_is_rejected(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "store")
    recipe_hash = "a" * 64

    store.put(b"first", recipe_hash=recipe_hash)

    with pytest.raises(DivergentRecipeError, match=recipe_hash):
        store.put(b"second", recipe_hash=recipe_hash)


def test_retention_never_downgrades_an_existing_object(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "store")
    digest = store.put(b"must keep", tier=RetentionTier.PERMANENT)

    assert store.put(b"must keep", tier=RetentionTier.PRUNABLE) == digest
    assert store.prune().pruned_hashes == ()
    assert store.get(digest) == b"must keep"


def test_prune_removes_prunable_bytes_but_keeps_lineage(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "store")
    retained_recipe = "b" * 64
    prunable_recipe = "c" * 64
    retained = store.put(
        b"keep", recipe_hash=retained_recipe, tier=RetentionTier.RETAINED
    )
    prunable = store.put(
        b"drop", recipe_hash=prunable_recipe, tier=RetentionTier.PRUNABLE
    )

    result = store.prune()

    assert result.pruned_hashes == (prunable,)
    assert result.pruned_bytes == 4
    assert store.get(retained) == b"keep"
    assert not store.object_path(prunable).exists()
    assert store.contents_for_recipe(prunable_recipe) == (prunable,)
    with pytest.raises(MissingObjectError, match="pruned"):
        store.get(prunable)


def test_put_tree_is_deterministic_and_rejects_symlinks(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        (root / "nested").mkdir(parents=True)
        (root / "nested" / "b.txt").write_bytes(b"b")
        (root / "a.txt").write_bytes(b"a")

    store = ObjectStore(tmp_path / "store")

    assert store.put_tree(first) == store.put_tree(second)

    if hasattr(os, "symlink"):
        try:
            os.symlink(first / "a.txt", first / "link.txt")
        except OSError as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")
        with pytest.raises(ObjectStoreError, match="symbolic link"):
            store.put_tree(first)


def test_put_tree_rejects_source_tree_that_contains_store_root(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "input.txt").write_bytes(b"input")
    store = ObjectStore(source / ".piginot-store")

    with pytest.raises(ObjectStoreError, match="store root"):
        store.put_tree(source)
