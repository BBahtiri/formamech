from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from piginot.store import ContentNotFoundError, ObjectStore, content_hash
from piginot.store import store as store_module


@pytest.mark.parametrize("install_then_kill", [False, True])
def test_killed_writer_never_makes_an_uncommitted_object_visible(
    tmp_path: Path,
    install_then_kill: bool,
) -> None:
    store = ObjectStore.create(tmp_path / "store")
    payload = b"atomic payload"
    digest = content_hash(payload)
    script = r"""
import os
import sys
from pathlib import Path

from piginot.store import ObjectStore

store = ObjectStore.open(Path(sys.argv[1]))
install_then_kill = sys.argv[2] == "true"
original_replace = os.replace

def interrupted_replace(source, destination):
    if "objects" in Path(destination).parts:
        if install_then_kill:
            original_replace(source, destination)
        os._exit(86)
    original_replace(source, destination)

os.replace = interrupted_replace
store.put(b"atomic payload")
"""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(store.root),
            str(install_then_kill).lower(),
        ],
        check=False,
    )

    assert result.returncode == 86
    with pytest.raises(ContentNotFoundError, match=digest):
        ObjectStore.open(store.root).get(digest)

    repaired = ObjectStore.open(store.root)
    assert repaired.put(payload) == digest
    assert repaired.get(digest) == payload


def test_failed_replace_cleans_the_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ObjectStore.create(tmp_path / "store")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError(f"cannot replace {source} with {destination}")

    monkeypatch.setattr(store_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="cannot replace"):
        store.put(b"payload")

    assert not tuple(store.root.rglob(".tmp-*"))


def test_directory_fsync_fallbacks_are_best_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_open(path: Path, flags: int) -> int:
        raise OSError(f"cannot open {path} with {flags}")

    monkeypatch.setattr(store_module.os, "open", fail_open)
    store_module._fsync_directory(tmp_path)

    closed: list[int] = []
    monkeypatch.setattr(store_module.os, "open", lambda path, flags: 17)
    monkeypatch.setattr(
        store_module.os,
        "fsync",
        lambda descriptor: (_ for _ in ()).throw(OSError("unsupported")),
    )
    monkeypatch.setattr(store_module.os, "close", closed.append)
    store_module._fsync_directory(tmp_path)

    assert closed == [17]
