"""Environment isolation for Git commands targeting an explicit repository."""

from __future__ import annotations

import os

_GIT_LOCAL_ENVIRONMENT_VARIABLES = frozenset(
    {
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
    }
)


def isolated_git_environment() -> dict[str, str]:
    """Return the process environment without Git's repository-local bindings."""

    return {
        name: value
        for name, value in os.environ.items()
        if name not in _GIT_LOCAL_ENVIRONMENT_VARIABLES
    }
