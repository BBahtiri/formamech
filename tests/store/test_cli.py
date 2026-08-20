from __future__ import annotations

from pathlib import Path

from piginot import cli
from piginot.store import ObjectStore, RetentionTier


def test_store_prune_cli_removes_only_prunable_bytes(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = ObjectStore(root)
    retained = store.put(b"keep", tier=RetentionTier.RETAINED)
    prunable = store.put(b"drop", tier=RetentionTier.PRUNABLE)

    assert cli.main(["store", "prune", "--root", str(root)]) == 0

    assert store.object_path(retained).exists()
    assert not store.object_path(prunable).exists()
