from __future__ import annotations

import os
from pathlib import Path

import pytest

from piginot.store import InvalidTreeError, ObjectStore


def _tree(root: Path) -> None:
    (root / "nested" / "empty").mkdir(parents=True)
    (root / "nested" / "b.bin").write_bytes(b"\x00\xff")
    (root / "a.txt").write_bytes(b"a")


def test_tree_hash_is_deterministic_and_ignores_root_name(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _tree(first)
    _tree(second)
    store = ObjectStore.create(tmp_path / "store")

    assert store.put_tree(first) == store.put_tree(second)


def test_empty_directories_and_entry_names_affect_tree_identity(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    (second / "empty").mkdir(parents=True)
    store = ObjectStore.create(tmp_path / "store")

    assert store.put_tree(first) != store.put_tree(second)

    (first / "x").write_bytes(b"same")
    (second / "y").write_bytes(b"same")
    assert store.put_tree(first) != store.put_tree(second)


def test_tree_requires_a_directory_outside_the_store(tmp_path: Path) -> None:
    store = ObjectStore.create(tmp_path / "store")
    regular_file = tmp_path / "file"
    regular_file.write_bytes(b"x")

    with pytest.raises(InvalidTreeError, match="directory"):
        store.put_tree(regular_file)
    with pytest.raises(InvalidTreeError, match="inspected"):
        store.put_tree(tmp_path / "missing")

    source_inside_store = store.root / "source"
    source_inside_store.mkdir()
    with pytest.raises(InvalidTreeError, match="contain"):
        store.put_tree(source_inside_store)


def test_tree_rejects_symlinks_and_a_store_nested_under_input(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    target = root / "target"
    target.write_bytes(b"target")
    store = ObjectStore.create(tmp_path / "store")

    try:
        os.symlink(target, root / "link")
    except (OSError, NotImplementedError):
        pass
    else:
        with pytest.raises(InvalidTreeError, match="link"):
            store.put_tree(root)

    nested_store = ObjectStore.create(root / "nested-store")
    with pytest.raises(InvalidTreeError, match="store"):
        nested_store.put_tree(root)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unavailable")
def test_tree_rejects_non_regular_files(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    os.mkfifo(root / "pipe")
    store = ObjectStore.create(tmp_path / "store")

    with pytest.raises(InvalidTreeError, match="regular"):
        store.put_tree(root)
