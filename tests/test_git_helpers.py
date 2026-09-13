from __future__ import annotations

from pathlib import Path

import pytest

from piginot.git_environment import isolated_git_environment
from tests.git_helpers import run_git

_LOCAL_REPOSITORY_VARIABLES = (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CONFIG",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_PARAMETERS",
    "GIT_DIR",
    "GIT_GRAFT_FILE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_OBJECT_DIRECTORY",
    "GIT_PREFIX",
    "GIT_REPLACE_REF_BASE",
    "GIT_SHALLOW_FILE",
    "GIT_WORK_TREE",
)


def test_isolated_git_environment_removes_repository_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _LOCAL_REPOSITORY_VARIABLES:
        monkeypatch.setenv(name, f"sentinel-{name}")
    monkeypatch.setenv("GIT_PAGER", "cat")

    environment = isolated_git_environment()

    assert all(name not in environment for name in _LOCAL_REPOSITORY_VARIABLES)
    assert environment["GIT_PAGER"] == "cat"


def test_run_git_ignores_an_inherited_hook_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hook_repo = tmp_path / "hook-repository"
    hook_repo.mkdir()
    run_git(hook_repo, "init", "--quiet")
    target_repo = tmp_path / "target-repository"
    target_repo.mkdir()

    monkeypatch.setenv("GIT_DIR", str((hook_repo / ".git").resolve()))
    monkeypatch.setenv("GIT_WORK_TREE", str(hook_repo.resolve()))
    run_git(target_repo, "init", "--quiet")

    assert (target_repo / ".git").is_dir()
    assert Path(run_git(target_repo, "rev-parse", "--show-toplevel")) == (
        target_repo.resolve()
    )
