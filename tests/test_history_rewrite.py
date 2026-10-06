"""Recorded pre-rewrite revisions resolve; the removed files are gone from history."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from campaigns.history import MAP_PATH, UnknownRevisionError, resolve_revision
from piginot.git_environment import isolated_git_environment

ROOT = Path(__file__).resolve().parents[1]
REMOVED = "experiments/spike/upstream_pi_ginot/models/modules"


def _git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        env=isolated_git_environment(),
        capture_output=True,
        text=True,
        check=False,
    )


def _shallow() -> bool:
    return _git("rev-parse", "--is-shallow-repository").stdout.strip() == "true"


def _map() -> dict[str, object]:
    document: dict[str, object] = json.loads((ROOT / MAP_PATH).read_text("utf-8"))
    return document


def test_the_map_records_the_rewrite() -> None:
    document = _map()
    assert document["removed_path"] == f"{REMOVED}/"
    assert len(document["removed_blobs"]) == 5  # type: ignore[arg-type]
    mapping = document["commit_map"]
    assert isinstance(mapping, dict) and len(mapping) > 200
    assert all(len(old) == len(new) == 40 for old, new in mapping.items())


@pytest.mark.skipif(_shallow(), reason="needs the full history")
def test_recorded_revisions_resolve_to_rewritten_commits() -> None:
    mapping = _map()["commit_map"]
    assert isinstance(mapping, dict)
    for recorded in ("8a87cd5", "a126e3a", "8736c1a"):
        resolved = resolve_revision(recorded)
        (expected,) = {new for old, new in mapping.items() if old.startswith(recorded)}
        assert resolved == expected
    head = _git("rev-parse", "HEAD").stdout.strip()
    assert resolve_revision("HEAD") == head


def test_unknown_and_short_revisions_are_refused() -> None:
    for revision in ("deadbeefcafe0123", "abc"):
        with pytest.raises(UnknownRevisionError):
            resolve_revision(revision)


@pytest.mark.skipif(_shallow(), reason="needs the full history")
def test_no_commit_holds_the_removed_files_or_their_blobs() -> None:
    assert _git("log", "--all", "--format=%H", "--", REMOVED).stdout.strip() == ""
    blobs = _map()["removed_blobs"]
    assert isinstance(blobs, dict)
    reachable = {
        line.split()[0]
        for line in _git("rev-list", "--objects", "--all").stdout.splitlines()
    }
    assert not reachable & set(blobs.values())
