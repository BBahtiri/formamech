"""Fail-closed filesystem and Git evidence for holdout commitment ordering."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from piginot.contracts import canonical_json_bytes, strict_json_loads
from piginot.doe.roles import EvidenceRole
from piginot.git_environment import isolated_git_environment

_LIVE_EVIDENCE_SCHEMA = "piginot.s9.live-commitment-order-v1"
_DECIMAL_INTEGER = re.compile(r"^(?:0|[1-9][0-9]*)$")


class CommitmentOrderError(ValueError):
    """Commitment provenance cannot prove strict pre-artifact ordering."""


def _relative_path(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("artifact path must be non-empty text")
    path = Path(value)
    if path.is_absolute() or path == Path("."):
        raise ValueError("artifact path must be relative and inside repository")
    return path.as_posix()


def _mtime_ns(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("mtime_ns must be a non-negative integer")
    return value


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
    """Persistable creation-time observation for one non-legacy artifact."""

    path: str
    role: EvidenceRole
    mtime_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path))
        try:
            role = EvidenceRole(self.role)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "live artifact role must be a declared evidence Role"
            ) from exc
        if role is EvidenceRole.LEGACY:
            raise ValueError("live evidence must exclude LEGACY artifacts")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "mtime_ns", _mtime_ns(self.mtime_ns))


@dataclass(frozen=True, slots=True)
class LiveCommitmentOrderEvidence:
    """Canonical creation-time evidence persisted before checkout mtimes can change."""

    __canonical_namespace__: ClassVar[str] = "piginot.doe"

    commitment_path: str
    commitment_mtime_ns: int
    artifacts: tuple[LiveArtifactEvidence, ...]

    def __post_init__(self) -> None:
        commitment_path = _relative_path(self.commitment_path)
        commitment_mtime_ns = _mtime_ns(self.commitment_mtime_ns)
        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, LiveArtifactEvidence) for item in artifacts):
            raise ValueError("artifacts must contain LiveArtifactEvidence values")
        artifacts = tuple(sorted(artifacts, key=lambda item: item.path))
        paths = tuple(item.path for item in artifacts)
        if len(paths) != len(set(paths)):
            raise CommitmentOrderError("persisted artifact paths must be unique")
        if commitment_path in paths:
            raise CommitmentOrderError(
                "persisted commitment path cannot also be an artifact"
            )
        object.__setattr__(self, "commitment_path", commitment_path)
        object.__setattr__(self, "commitment_mtime_ns", commitment_mtime_ns)
        object.__setattr__(self, "artifacts", artifacts)

    def verify(self) -> None:
        for artifact in self.artifacts:
            if self.commitment_mtime_ns >= artifact.mtime_ns:
                raise CommitmentOrderError(
                    f"commitment mtime must precede artifact mtime: {artifact.path}"
                )

    def to_canonical_bytes(self) -> bytes:
        """Serialize the immutable live record as versioned RFC 8785 JSON."""

        record = {
            "schema": _LIVE_EVIDENCE_SCHEMA,
            "commitment_path": self.commitment_path,
            "commitment_mtime_ns": str(self.commitment_mtime_ns),
            "artifacts": [
                {
                    "path": artifact.path,
                    "role": artifact.role.value,
                    "mtime_ns": str(artifact.mtime_ns),
                }
                for artifact in self.artifacts
            ],
        }
        return canonical_json_bytes(record)

    @classmethod
    def from_canonical_bytes(cls, document: str | bytes) -> LiveCommitmentOrderEvidence:
        """Load an exactly canonical persisted record and reject schema drift."""

        if isinstance(document, str):
            encoded = document.encode("utf-8")
        elif isinstance(document, bytes):
            encoded = document
        else:
            raise ValueError("live evidence document must be UTF-8 text or bytes")
        parsed = strict_json_loads(encoded)
        if not isinstance(parsed, dict):
            raise ValueError("live evidence document must be a JSON object")
        if canonical_json_bytes(parsed) != encoded:
            raise ValueError("live evidence document must use canonical JSON bytes")
        expected_keys = {
            "schema",
            "commitment_path",
            "commitment_mtime_ns",
            "artifacts",
        }
        if (
            set(parsed) != expected_keys
            or parsed.get("schema") != _LIVE_EVIDENCE_SCHEMA
        ):
            raise ValueError("live evidence document has an unsupported schema")
        commitment_path = parsed["commitment_path"]
        artifacts_value = parsed["artifacts"]
        if not isinstance(commitment_path, str) or not isinstance(
            artifacts_value, list
        ):
            raise ValueError("live evidence document has invalid field types")
        artifacts = tuple(_parse_live_artifact(item) for item in artifacts_value)
        return cls(
            commitment_path=commitment_path,
            commitment_mtime_ns=_parse_mtime(parsed["commitment_mtime_ns"]),
            artifacts=artifacts,
        )


def _parse_mtime(value: object) -> int:
    if not isinstance(value, str) or _DECIMAL_INTEGER.fullmatch(value) is None:
        raise ValueError("persisted mtime_ns must be a canonical decimal string")
    return int(value)


def _parse_live_artifact(value: object) -> LiveArtifactEvidence:
    if not isinstance(value, Mapping) or set(value) != {"path", "role", "mtime_ns"}:
        raise ValueError("persisted artifact record has invalid fields")
    path = value["path"]
    role = value["role"]
    if not isinstance(path, str) or not isinstance(role, str):
        raise ValueError("persisted artifact record has invalid field types")
    return LiveArtifactEvidence(
        path=path,
        role=EvidenceRole(role),
        mtime_ns=_parse_mtime(value["mtime_ns"]),
    )


@dataclass(frozen=True, slots=True)
class GitPathEvidence:
    """Pure Git observations for one repository-relative path."""

    path: str
    exists: bool
    tracked: bool
    clean: bool
    first_add_commit: str | None
    history_commits: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CommittedArtifactEvidence:
    """Pure Git-order observation for a non-legacy training artifact."""

    path: str
    role: EvidenceRole
    first_add_commit: str
    commitment_is_strict_ancestor: bool


@dataclass(frozen=True, slots=True)
class CommittedCommitmentOrderEvidence:
    """Pure persisted-mtime and independent Git-order evidence."""

    commitment_path: str
    commitment_first_add_commit: str
    commitment_history_commits: tuple[str, ...]
    artifacts: tuple[CommittedArtifactEvidence, ...]
    live_evidence: LiveCommitmentOrderEvidence
    live_evidence_path: str
    live_evidence_first_add_commit: str
    live_evidence_history_commits: tuple[str, ...]
    commitment_is_strict_ancestor_of_live_evidence: bool
    reveal_verified: bool

    def verify(self) -> None:
        if not self.reveal_verified:
            raise CommitmentOrderError("holdout reveal has not been verified")
        if len(self.commitment_history_commits) != 1:
            raise CommitmentOrderError(
                "commitment history was modified after first add"
            )
        if len(self.live_evidence_history_commits) != 1:
            raise CommitmentOrderError(
                "persisted live evidence was modified after first add"
            )
        if not self.commitment_is_strict_ancestor_of_live_evidence:
            raise CommitmentOrderError(
                "commitment must strictly precede the persisted live evidence commit"
            )
        self.live_evidence.verify()
        if self.live_evidence.commitment_path != self.commitment_path:
            raise CommitmentOrderError(
                "persisted commitment path does not match declared commitment path"
            )
        persisted = tuple(
            (artifact.path, artifact.role) for artifact in self.live_evidence.artifacts
        )
        committed = tuple(
            sorted((artifact.path, artifact.role) for artifact in self.artifacts)
        )
        if persisted != committed:
            raise CommitmentOrderError(
                "persisted artifact paths and roles do not match declared artifacts"
            )
        for artifact in self.artifacts:
            if artifact.first_add_commit != self.live_evidence_first_add_commit:
                raise CommitmentOrderError(
                    "persisted live evidence and every witnessed artifact must be "
                    f"first-added in the same commit: {artifact.path}"
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
            env=isolated_git_environment(),
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
            first_add_commit=additions[-1] if additions else None,
            history_commits=history,
        )

    def read_bytes(self, path: str) -> bytes:
        """Read one repository-contained file without following an escape path."""

        relative, absolute = self._resolve(path)
        if not absolute.is_file():
            raise CommitmentOrderError(f"missing persisted evidence: {relative}")
        return absolute.read_bytes()

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
    return tuple(
        sorted(
            (record for record in records if record.role is not EvidenceRole.LEGACY),
            key=lambda record: record.path,
        )
    )


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
    """Capture creation-time mtimes for immediate canonical persistence."""

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
    if record.first_add_commit is None:
        raise CommitmentOrderError(f"missing Git first-add evidence: {record.path}")


def _verify_live_record_matches(
    live_evidence: LiveCommitmentOrderEvidence,
    *,
    commitment_path: str,
    records: tuple[ArtifactRecord, ...],
) -> None:
    if not isinstance(live_evidence, LiveCommitmentOrderEvidence):
        raise ValueError("live_evidence must be persisted LiveCommitmentOrderEvidence")
    live_evidence.verify()
    if live_evidence.commitment_path != commitment_path:
        raise CommitmentOrderError(
            "persisted commitment path does not match declared commitment path"
        )
    declared = tuple((record.path, record.role) for record in records)
    persisted = tuple(
        (artifact.path, artifact.role) for artifact in live_evidence.artifacts
    )
    if persisted != declared:
        raise CommitmentOrderError(
            "persisted artifact paths and roles do not match declared artifacts"
        )


def verify_committed_artifact_order(
    *,
    repo_root: str | Path,
    commitment_path: str,
    artifacts: Iterable[ArtifactRecord],
    live_evidence: LiveCommitmentOrderEvidence,
    live_evidence_path: str,
    reveal_verified: bool,
) -> CommittedCommitmentOrderEvidence:
    """Verify persisted creation mtimes and independent Git first-add ancestry."""

    if not isinstance(reveal_verified, bool):
        raise ValueError("reveal_verified must be boolean")
    probe = GitProbe(repo_root)
    relative_commitment = _relative_path(commitment_path)
    records = _active_records(artifacts, commitment_path=relative_commitment)
    relative_live_evidence = _relative_path(live_evidence_path)
    if relative_live_evidence == relative_commitment or relative_live_evidence in {
        record.path for record in records
    }:
        raise CommitmentOrderError(
            "persisted live evidence path must be distinct from commitment and artifacts"
        )
    _verify_live_record_matches(
        live_evidence,
        commitment_path=relative_commitment,
        records=records,
    )
    commitment = probe.inspect(relative_commitment)
    _require_committed(commitment, kind="commitment")
    if len(commitment.history_commits) != 1:
        raise CommitmentOrderError("commitment history was modified after first add")
    if not reveal_verified:
        raise CommitmentOrderError("holdout reveal has not been verified")
    if commitment.first_add_commit is None:
        raise AssertionError("validated commitment evidence is incomplete")
    persisted_record = probe.inspect(relative_live_evidence)
    _require_committed(persisted_record, kind="live evidence")
    if len(persisted_record.history_commits) != 1:
        raise CommitmentOrderError(
            "persisted live evidence was modified after first add"
        )
    if persisted_record.first_add_commit is None:
        raise AssertionError("validated live evidence is incomplete")
    if probe.read_bytes(relative_live_evidence) != live_evidence.to_canonical_bytes():
        raise CommitmentOrderError(
            "tracked live evidence file does not match the supplied canonical record"
        )
    commitment_precedes_live = probe.is_strict_ancestor(
        commitment.first_add_commit,
        persisted_record.first_add_commit,
    )
    if not commitment_precedes_live:
        raise CommitmentOrderError(
            "commitment must strictly precede the persisted live evidence commit"
        )

    artifact_evidence: list[CommittedArtifactEvidence] = []
    for declared in records:
        observed = probe.inspect(declared.path)
        _require_committed(observed, kind="artifact")
        if observed.first_add_commit is None:
            raise AssertionError("validated artifact evidence is incomplete")
        if observed.first_add_commit != persisted_record.first_add_commit:
            raise CommitmentOrderError(
                "persisted live evidence and every witnessed artifact must be "
                f"first-added in the same commit: {declared.path}"
            )
        artifact_evidence.append(
            CommittedArtifactEvidence(
                path=declared.path,
                role=declared.role,
                first_add_commit=observed.first_add_commit,
                commitment_is_strict_ancestor=probe.is_strict_ancestor(
                    commitment.first_add_commit,
                    observed.first_add_commit,
                ),
            )
        )

    evidence = CommittedCommitmentOrderEvidence(
        commitment_path=relative_commitment,
        commitment_first_add_commit=commitment.first_add_commit,
        commitment_history_commits=commitment.history_commits,
        artifacts=tuple(artifact_evidence),
        live_evidence=live_evidence,
        live_evidence_path=relative_live_evidence,
        live_evidence_first_add_commit=persisted_record.first_add_commit,
        live_evidence_history_commits=persisted_record.history_commits,
        commitment_is_strict_ancestor_of_live_evidence=commitment_precedes_live,
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
