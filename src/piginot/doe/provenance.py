"""Fail-closed filesystem and Git evidence for holdout commitment ordering."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from piginot.doe.roles import EvidenceRole


class CommitmentOrderError(ValueError):
    """Commitment provenance cannot prove strict pre-artifact ordering."""


def _relative_path(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("artifact path must be non-empty text")
    path = Path(value)
    if path.is_absolute() or path == Path("."):
        raise ValueError("artifact path must be relative and inside repository")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Declared training artifact; exclusion is based only on its evidence role."""

    path: str
    role: EvidenceRole

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path))
        try:
            role = EvidenceRole(self.role)
        except (TypeError, ValueError) as exc:
            raise ValueError("artifact role must be a declared evidence Role") from exc
        object.__setattr__(self, "role", role)


@dataclass(frozen=True, slots=True)
class LiveArtifactEvidence:
    """Pure filesystem observation for one non-legacy artifact."""

    path: str
    role: EvidenceRole
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class LiveCommitmentOrderEvidence:
    """Pure evidence for checks performed before Git artifacts are required."""

    commitment_path: str
    commitment_mtime_ns: int
    artifacts: tuple[LiveArtifactEvidence, ...]

    def verify(self) -> None:
        for artifact in self.artifacts:
            if self.commitment_mtime_ns >= artifact.mtime_ns:
                raise CommitmentOrderError(
                    f"commitment mtime must precede artifact mtime: {artifact.path}"
                )


@dataclass(frozen=True, slots=True)
class GitPathEvidence:
    """Pure Git and filesystem observations for one repository-relative path."""

    path: str
    exists: bool
    tracked: bool
    clean: bool
    mtime_ns: int | None
    first_add_commit: str | None
    history_commits: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CommittedArtifactEvidence:
    """Pure committed-order observation for a non-legacy training artifact."""

    path: str
    role: EvidenceRole
    mtime_ns: int
    first_add_commit: str
    commitment_is_strict_ancestor: bool


@dataclass(frozen=True, slots=True)
class CommittedCommitmentOrderEvidence:
    """Pure evidence that a clean immutable commitment predates artifacts."""

    commitment_path: str
    commitment_mtime_ns: int
    commitment_first_add_commit: str
    commitment_history_commits: tuple[str, ...]
    artifacts: tuple[CommittedArtifactEvidence, ...]
    reveal_verified: bool

    def verify(self) -> None:
        if not self.reveal_verified:
            raise CommitmentOrderError("holdout reveal has not been verified")
        if len(self.commitment_history_commits) != 1:
            raise CommitmentOrderError(
                "commitment history was modified after first add"
            )
        for artifact in self.artifacts:
            if self.commitment_mtime_ns >= artifact.mtime_ns:
                raise CommitmentOrderError(
                    f"commitment mtime must precede artifact mtime: {artifact.path}"
                )
            if not artifact.commitment_is_strict_ancestor:
                raise CommitmentOrderError(
                    "commitment first-add Git commit must strictly precede artifact "
                    f"first-add commit: {artifact.path}"
                )


class GitProbe:
    """Narrow, read-only subprocess adapter for repository provenance facts."""

    def __init__(self, repo_root: str | Path) -> None:
        root = Path(repo_root).resolve()
        if not root.is_dir():
            raise ValueError("repo_root must be an existing directory")
        self._root = root
        completed = self._run(("rev-parse", "--show-toplevel"), check=False)
        if completed.returncode != 0:
            raise ValueError("repo_root must be a Git working tree")
        reported = Path(completed.stdout.strip()).resolve()
        if reported != root:
            raise ValueError("repo_root must be the Git working-tree root")

    def _run(
        self,
        arguments: tuple[str, ...],
        *,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=self._root,
            check=False,
            capture_output=True,
            text=True,
        )
        if check and completed.returncode != 0:
            detail = completed.stderr.strip() or "Git provenance query failed"
            raise CommitmentOrderError(detail)
        return completed

    def _resolve(self, value: str) -> tuple[str, Path]:
        relative = _relative_path(value)
        candidate = (self._root / Path(relative)).resolve()
        try:
            candidate.relative_to(self._root)
        except ValueError as exc:
            raise ValueError("artifact path must resolve inside repository") from exc
        return relative, candidate

    def inspect(self, path: str) -> GitPathEvidence:
        """Observe existence, cleanliness, and first-add history for one path."""

        relative, absolute = self._resolve(path)
        exists = absolute.is_file()
        tracked_query = self._run(
            ("ls-files", "--error-unmatch", "--", relative),
            check=False,
        )
        tracked = tracked_query.returncode == 0
        status = self._run(
            ("status", "--porcelain=v1", "--untracked-files=all", "--", relative),
            check=True,
        ).stdout
        history_query = self._run(
            ("log", "--follow", "--format=%H", "--", relative),
            check=False,
        )
        history = tuple(
            line.strip() for line in history_query.stdout.splitlines() if line.strip()
        )
        additions_query = self._run(
            ("log", "--follow", "--diff-filter=A", "--format=%H", "--", relative),
            check=False,
        )
        additions = tuple(
            line.strip() for line in additions_query.stdout.splitlines() if line.strip()
        )
        return GitPathEvidence(
            path=relative,
            exists=exists,
            tracked=tracked,
            clean=status.strip() == "",
            mtime_ns=absolute.stat().st_mtime_ns if exists else None,
            first_add_commit=additions[-1] if additions else None,
            history_commits=history,
        )

    def is_strict_ancestor(self, ancestor: str, descendant: str) -> bool:
        """Return true only when two distinct commits have an ancestry relation."""

        if ancestor == descendant:
            return False
        result = self._run(
            ("merge-base", "--is-ancestor", ancestor, descendant),
            check=False,
        )
        if result.returncode not in (0, 1):
            raise CommitmentOrderError("Git ancestry query failed")
        return result.returncode == 0


def _active_records(
    artifacts: Iterable[ArtifactRecord],
    *,
    commitment_path: str,
) -> tuple[ArtifactRecord, ...]:
    records = tuple(artifacts)
    if any(not isinstance(record, ArtifactRecord) for record in records):
        raise ValueError("artifacts must contain ArtifactRecord values")
    paths = tuple(record.path for record in records)
    if len(paths) != len(set(paths)):
        raise CommitmentOrderError("artifact records must have unique paths")
    if commitment_path in paths:
        raise CommitmentOrderError("commitment path cannot also be an artifact")
    return tuple(record for record in records if record.role is not EvidenceRole.LEGACY)


def _live_path(repo_root: Path, relative: str) -> Path:
    absolute = (repo_root / Path(relative)).resolve()
    try:
        absolute.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("artifact path must resolve inside repository") from exc
    if not absolute.is_file():
        raise CommitmentOrderError(f"missing artifact record: {relative}")
    return absolute


def verify_live_pre_artifact_order(
    *,
    repo_root: str | Path,
    commitment_path: str,
    artifacts: Iterable[ArtifactRecord],
) -> LiveCommitmentOrderEvidence:
    """Check actual mtimes without claiming that artifacts are Git-committed."""

    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise ValueError("repo_root must be an existing directory")
    relative_commitment = _relative_path(commitment_path)
    commitment = _live_path(root, relative_commitment)
    records = _active_records(artifacts, commitment_path=relative_commitment)
    artifact_evidence = tuple(
        LiveArtifactEvidence(
            path=record.path,
            role=record.role,
            mtime_ns=_live_path(root, record.path).stat().st_mtime_ns,
        )
        for record in records
    )
    evidence = LiveCommitmentOrderEvidence(
        commitment_path=relative_commitment,
        commitment_mtime_ns=commitment.stat().st_mtime_ns,
        artifacts=artifact_evidence,
    )
    evidence.verify()
    return evidence


def _require_committed(record: GitPathEvidence, *, kind: str) -> None:
    if not record.exists:
        raise CommitmentOrderError(f"missing {kind} record: {record.path}")
    if not record.tracked:
        raise CommitmentOrderError(f"untracked {kind} record: {record.path}")
    if not record.clean:
        raise CommitmentOrderError(f"modified {kind} record: {record.path}")
    if record.mtime_ns is None or record.first_add_commit is None:
        raise CommitmentOrderError(f"missing Git first-add evidence: {record.path}")


def verify_committed_artifact_order(
    *,
    repo_root: str | Path,
    commitment_path: str,
    artifacts: Iterable[ArtifactRecord],
    reveal_verified: bool,
) -> CommittedCommitmentOrderEvidence:
    """Prove clean Git ancestry and actual mtime order for committed artifacts."""

    if not isinstance(reveal_verified, bool):
        raise ValueError("reveal_verified must be boolean")
    probe = GitProbe(repo_root)
    relative_commitment = _relative_path(commitment_path)
    records = _active_records(artifacts, commitment_path=relative_commitment)
    commitment = probe.inspect(relative_commitment)
    _require_committed(commitment, kind="commitment")
    if len(commitment.history_commits) != 1:
        raise CommitmentOrderError("commitment history was modified after first add")
    if not reveal_verified:
        raise CommitmentOrderError("holdout reveal has not been verified")
    if commitment.mtime_ns is None or commitment.first_add_commit is None:
        raise AssertionError("validated commitment evidence is incomplete")

    artifact_evidence: list[CommittedArtifactEvidence] = []
    for declared in records:
        observed = probe.inspect(declared.path)
        _require_committed(observed, kind="artifact")
        if observed.mtime_ns is None or observed.first_add_commit is None:
            raise AssertionError("validated artifact evidence is incomplete")
        artifact_evidence.append(
            CommittedArtifactEvidence(
                path=declared.path,
                role=declared.role,
                mtime_ns=observed.mtime_ns,
                first_add_commit=observed.first_add_commit,
                commitment_is_strict_ancestor=probe.is_strict_ancestor(
                    commitment.first_add_commit,
                    observed.first_add_commit,
                ),
            )
        )

    evidence = CommittedCommitmentOrderEvidence(
        commitment_path=relative_commitment,
        commitment_mtime_ns=commitment.mtime_ns,
        commitment_first_add_commit=commitment.first_add_commit,
        commitment_history_commits=commitment.history_commits,
        artifacts=tuple(artifact_evidence),
        reveal_verified=reveal_verified,
    )
    evidence.verify()
    return evidence


__all__ = (
    "ArtifactRecord",
    "CommitmentOrderError",
    "CommittedArtifactEvidence",
    "CommittedCommitmentOrderEvidence",
    "GitPathEvidence",
    "GitProbe",
    "LiveArtifactEvidence",
    "LiveCommitmentOrderEvidence",
    "verify_committed_artifact_order",
    "verify_live_pre_artifact_order",
)
