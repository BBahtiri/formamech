from __future__ import annotations

import subprocess
from pathlib import Path

from piginot.git_environment import isolated_git_environment


def run_git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repo,
        env=isolated_git_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
