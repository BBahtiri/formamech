from __future__ import annotations

import json
from pathlib import Path

import pytest

from piginot import cli
from piginot.store import ObjectStore, RetentionTier


def test_store_prune_cli_emits_deterministic_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "store"
    store = ObjectStore.create(root)
    retained = store.put(b"keep", tier=RetentionTier.RETAINED)
    prunable = store.put(b"drop", tier=RetentionTier.PRUNABLE)

    assert cli.main(["store", "prune", "--root", str(root)]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report == {
        "dry_run": False,
        "object_count": 1,
        "pruned_bytes": 4,
        "pruned_hashes": [prunable],
    }
    assert store.object_path(retained).exists()
    assert not store.object_path(prunable).exists()


def test_store_prune_cli_dry_run_preserves_bytes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "store"
    store = ObjectStore.create(root)
    digest = store.put(b"drop", tier=RetentionTier.PRUNABLE)

    assert cli.main(["store", "prune", "--root", str(root), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True
    assert store.get(digest) == b"drop"


def test_store_prune_cli_rejects_unmarked_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "not-a-store"
    root.mkdir()

    assert cli.main(["store", "prune", "--root", str(root)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "store" in captured.err.lower()


def test_store_prune_cli_requires_root() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["store", "prune"])
    assert error.value.code == 2


def test_store_prune_cli_turns_io_failures_into_exit_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_open(root: Path) -> ObjectStore:
        raise OSError(f"cannot read {root}")

    monkeypatch.setattr(ObjectStore, "open", fail_open)

    assert cli.main(["store", "prune", "--root", str(tmp_path)]) == 1
    assert "cannot read" in capsys.readouterr().err
