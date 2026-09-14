"""Command-line entry point for the preregistered S11 architecture screen."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from campaigns.dogbone_v1.s11_contingency import ArchitectureScreenProtocol
from campaigns.dogbone_v1.s11_contingency_artifacts import (
    DEVELOPMENT_ACCESS_NAME,
    ContingencyEvidence,
    persist_contingency_result,
    reserve_output_root,
)
from campaigns.dogbone_v1.s11_contingency_execute import (
    execute_architecture_contingency,
)
from campaigns.dogbone_v1.s11_contingency_run import load_committed_protocol
from campaigns.dogbone_v1.s11_contingency_source import open_selective_s10_source
from piginot.git_environment import isolated_git_environment

_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
Progress = Callable[[str], None]


def verified_clean_head(repo_root: Path, expected_revision: str) -> str:
    """Return HEAD only when `repo_root` is the exact clean expected checkout."""

    root = _absolute_directory(repo_root, "repo_root")
    revision = _revision(expected_revision)
    top_level = Path(_git(root, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if top_level != root:
        raise ValueError("repo_root must be the Git worktree top level")
    head = _git(root, "rev-parse", "HEAD")
    if head != revision:
        raise ValueError(
            f"Git HEAD {head} differs from implementation revision {revision}"
        )
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError("Git worktree must be clean before opening the S10 source")
    return head


def run_architecture_contingency(
    *,
    repo_root: Path,
    s10_run_root: Path,
    output_root: Path,
    implementation_revision: str,
    progress: Progress | None = None,
) -> ContingencyEvidence:
    """Preflight, reserve output, open S10 roles, execute, and persist once."""

    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    revision = _revision(implementation_revision)
    verified_clean_head(repo_root, revision)
    _emit(progress, f"phase=PREFLIGHT event=git_clean revision={revision}")

    protocol = load_committed_protocol()
    if type(protocol) is not ArchitectureScreenProtocol:
        raise TypeError("committed protocol loader returned a noncanonical value")
    s10_root = _absolute_directory(s10_run_root, "s10_run_root")
    output = _absolute_new_path(output_root, "output_root")
    if output == s10_root or _is_within(output, s10_root):
        raise ValueError("output_root must be outside the S10 source run root")

    reservation = reserve_output_root(
        output,
        protocol_content_hash=protocol.content_hash,
        implementation_revision=revision,
    )
    _emit(progress, f"phase=PREFLIGHT event=output_reserved path={output}")
    source = open_selective_s10_source(
        s10_root,
        development_access_record=reservation.output_root / DEVELOPMENT_ACCESS_NAME,
    )
    _emit(
        progress,
        f"phase=SOURCE event=verified bindings={_bindings_hash_for_progress(source.bindings)}",
    )
    result = execute_architecture_contingency(
        protocol,
        source,
        implementation_revision=revision,
        progress=progress,
    )
    evidence = persist_contingency_result(
        reservation,
        protocol=protocol,
        source_bindings=source.bindings,
        result=result,
    )
    _emit(
        progress,
        "phase=PERSIST event=complete "
        f"evidence={evidence.content_hash} "
        f"final={_final_candidate_for_progress(evidence)}",
    )
    return evidence


def build_parser() -> argparse.ArgumentParser:
    """Build the intentionally non-configurable production runner interface."""

    parser = argparse.ArgumentParser(
        description="Run the preregistered S11 architecture contingency exactly once."
    )
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--s10-run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--implementation-revision", required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the CLI while streaming phase and candidate progress immediately."""

    options = build_parser().parse_args(arguments)

    def print_progress(message: str) -> None:
        print(message, flush=True)

    evidence = run_architecture_contingency(
        repo_root=options.repo_root,
        s10_run_root=options.s10_run_root,
        output_root=options.output_root,
        implementation_revision=options.implementation_revision,
        progress=print_progress,
    )
    print(f"evidence_content_hash={evidence.content_hash}", flush=True)
    return 0


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        env=isolated_git_environment(),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Git preflight failed: {detail}")
    return completed.stdout.strip()


def _absolute_directory(path: Path, label: str) -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{label} must be an exact pathlib.Path")
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a directory")
    return resolved


def _absolute_new_path(path: Path, label: str) -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{label} must be an exact pathlib.Path")
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    return path.resolve(strict=False)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _revision(value: object) -> str:
    if type(value) is not str or _REVISION_PATTERN.fullmatch(value) is None:
        raise ValueError("implementation_revision must be a lowercase Git revision")
    return value


def _emit(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _bindings_hash_for_progress(bindings: object) -> str:
    value = getattr(bindings, "content_hash", None)
    return value if isinstance(value, str) else "unavailable"


def _final_candidate_for_progress(evidence: object) -> str:
    value = getattr(evidence, "final_candidate_id", None)
    return value if isinstance(value, str) else "unavailable"


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())


__all__ = (
    "build_parser",
    "main",
    "run_architecture_contingency",
    "verified_clean_head",
)
