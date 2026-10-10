"""G19.manifests: the fork manifest tool detects every kind of byte difference."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "verify_fork", ROOT / "scripts" / "verify_fork.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(root: Path) -> None:
    (root / "runs" / "a" / "__pycache__").mkdir(parents=True)
    (root / "runs" / "a" / "x.json").write_text('{"x": 1}\n', encoding="utf-8")
    (root / "runs" / "b.bin").write_bytes(b"\x00\x01\x02")
    (root / "runs" / "a" / "__pycache__" / "skip.pyc").write_bytes(b"cache")
    (root / "runs" / "skip.pyc").write_bytes(b"cache")
    (root / "single.txt").write_text("one\n", encoding="utf-8")


def test_identical_trees_in_different_roots_compare_equal(tmp_path: Path) -> None:
    tool = _tool()
    _tree(tmp_path / "source")
    _tree(tmp_path / "fork")
    left = tool.build_manifest(tmp_path / "source", ["runs", "single.txt"])
    right = tool.build_manifest(tmp_path / "fork", ["single.txt", "runs"])
    assert left["files"] == right["files"]
    assert left["table_blake2b"] == right["table_blake2b"]
    assert sorted(left["files"]) == ["runs/a/x.json", "runs/b.bin", "single.txt"]
    assert tool.compare_manifests(left, right) == {
        "missing": [],
        "extra": [],
        "changed": [],
    }


def test_missing_extra_and_changed_files_are_reported(tmp_path: Path) -> None:
    tool = _tool()
    _tree(tmp_path / "source")
    _tree(tmp_path / "fork")
    (tmp_path / "fork" / "runs" / "b.bin").unlink()
    (tmp_path / "fork" / "runs" / "new.txt").write_text("new\n", encoding="utf-8")
    (tmp_path / "fork" / "single.txt").write_text("One\n", encoding="utf-8")
    left = tool.build_manifest(tmp_path / "source", ["runs", "single.txt"])
    right = tool.build_manifest(tmp_path / "fork", ["runs", "single.txt"])
    assert tool.compare_manifests(left, right) == {
        "missing": ["runs/b.bin"],
        "extra": ["runs/new.txt"],
        "changed": ["single.txt"],
    }
    assert left["table_blake2b"] != right["table_blake2b"]


def test_times_count_only_when_both_manifests_record_them(tmp_path: Path) -> None:
    tool = _tool()
    _tree(tmp_path / "source")
    _tree(tmp_path / "fork")
    moved = tmp_path / "fork" / "single.txt"
    os.utime(moved, ns=(1_000_000_000, 1_000_000_000))
    with_times = (
        tool.build_manifest(tmp_path / "source", ["single.txt"], mtime=True),
        tool.build_manifest(tmp_path / "fork", ["single.txt"], mtime=True),
    )
    assert tool.compare_manifests(*with_times)["changed"] == ["single.txt"]
    without = tool.build_manifest(tmp_path / "fork", ["single.txt"])
    assert tool.compare_manifests(with_times[0], without)["changed"] == []


def test_cli_round_trip_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = _tool()
    _tree(tmp_path / "source")
    _tree(tmp_path / "fork")
    for name in ("source", "fork"):
        code = tool.main(
            [
                "manifest",
                "--root",
                str(tmp_path / name),
                "--include",
                "runs",
                "--out",
                str(tmp_path / f"{name}.json"),
            ]
        )
        assert code == 0
    document = json.loads((tmp_path / "source.json").read_text("utf-8"))
    assert document["format"] == "piginot-fork-manifest-v1"
    assert document["file_count"] == 2
    assert (
        tool.main(
            ["compare", str(tmp_path / "source.json"), str(tmp_path / "fork.json")]
        )
        == 0
    )
    (tmp_path / "fork" / "runs" / "b.bin").write_bytes(b"\xff")
    tool.main(
        [
            "manifest",
            "--root",
            str(tmp_path / "fork"),
            "--include",
            "runs",
            "--out",
            str(tmp_path / "fork.json"),
        ]
    )
    assert (
        tool.main(
            ["compare", str(tmp_path / "source.json"), str(tmp_path / "fork.json")]
        )
        == 1
    )
    assert "changed: runs/b.bin" in capsys.readouterr().out


def test_bad_inputs_are_refused(tmp_path: Path) -> None:
    tool = _tool()
    with pytest.raises(FileNotFoundError):
        tool.build_manifest(tmp_path, ["absent"])
    with pytest.raises(ValueError):
        tool.compare_manifests({"format": "other"}, {"format": "other"})
