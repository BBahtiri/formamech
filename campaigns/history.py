"""Resolve commit ids recorded before the 2026-10-06 history rewrite.

Evidence written before the rewrite (gate artifacts, protocols, run bindings,
freezes, constants such as the S16 base) names the original commit ids. The
rewrite removed one path and changed no other file, so every original commit
has exactly one rewritten counterpart, recorded in
``artifacts/release/HISTORY-REWRITE.json``. Recorded evidence is never
edited; code that asks Git about a recorded revision resolves it here first.
"""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path

from piginot.git_environment import isolated_git_environment

REPOSITORY = Path(__file__).resolve().parents[1]
MAP_PATH = Path("artifacts/release/HISTORY-REWRITE.json")
MINIMUM_PREFIX = 7


class UnknownRevisionError(LookupError):
    """A revision is neither in this history nor in the rewrite map."""


@lru_cache(maxsize=1)
def _commit_map(root: Path = REPOSITORY) -> dict[str, str]:
    path = root / MAP_PATH
    if not path.is_file():
        return {}
    mapping: dict[str, str] = json.loads(path.read_text("utf-8"))["commit_map"]
    return mapping


def _verify(revision: str, root: Path) -> str | None:
    completed = subprocess.run(
        ("git", "rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}"),
        cwd=root,
        env=isolated_git_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def resolve_revision(revision: str, root: Path = REPOSITORY) -> str:
    """The full id of ``revision`` in this history.

    A revision that exists is returned as is. An original (pre-rewrite) id,
    full or abbreviated to at least seven characters, is mapped to its
    rewritten id; an ambiguous or unknown one raises.
    """

    found = _verify(revision, root)
    if found is not None:
        return found
    if len(revision) < MINIMUM_PREFIX:
        raise UnknownRevisionError(revision)
    mapping = _commit_map(root)
    matches = {new for old, new in mapping.items() if old.startswith(revision)}
    if len(matches) != 1:
        raise UnknownRevisionError(revision)
    (rewritten,) = matches
    found = _verify(rewritten, root)
    if found is None:
        raise UnknownRevisionError(revision)
    return found


__all__ = ("MAP_PATH", "UnknownRevisionError", "resolve_revision")
