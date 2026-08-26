from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from piginot.doe.provenance import (
    ArtifactRecord,
    CommitmentOrderError,
    CommittedArtifactEvidence,
    CommittedCommitmentOrderEvidence,
    GitProbe,
    LiveArtifactEvidence,
    LiveCommitmentOrderEvidence,
    verify_committed_artifact_order,
    verify_live_pre_artifact_order,
)
from piginot.doe.roles import EvidenceRole


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main")
    _git(repo, "config", "user.name", "S9 test")
    _git(repo, "config", "user.email", "s9@example.invalid")
    return repo


def _commit(repo: Path, path: str, text: str, message: str) -> str:
    _write(repo / path, text)
    _git(repo, "add", "--", path)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _set_mtime(path: Path, nanoseconds: int) -> None:
    os.utime(path, ns=(nanoseconds, nanoseconds))


def _live_evidence(
    artifacts: tuple[ArtifactRecord, ...] = (),
    *,
    commitment_path: str = "holdout.json",
    commitment_mtime_ns: int = 1,
    artifact_mtime_ns: int = 2,
) -> LiveCommitmentOrderEvidence:
    active = tuple(
        sorted(
            (
                artifact
                for artifact in artifacts
                if artifact.role is not EvidenceRole.LEGACY
            ),
            key=lambda artifact: artifact.path,
        )
    )
    return LiveCommitmentOrderEvidence(
        commitment_path=commitment_path,
        commitment_mtime_ns=commitment_mtime_ns,
        artifacts=tuple(
            LiveArtifactEvidence(
                path=artifact.path,
                role=artifact.role,
                mtime_ns=artifact_mtime_ns + index,
            )
            for index, artifact in enumerate(active)
        ),
    )


_LIVE_EVIDENCE_PATH = "audit/live-order.json"


def _commit_live_batch(
    repo: Path,
    live_evidence: LiveCommitmentOrderEvidence,
    *,
    artifact_paths: tuple[str, ...] = (),
    message: str = "artifact batch and live evidence",
) -> str:
    _write(
        repo / _LIVE_EVIDENCE_PATH,
        live_evidence.to_canonical_bytes().decode("utf-8"),
    )
    _git(repo, "add", "--", _LIVE_EVIDENCE_PATH, *artifact_paths)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def test_live_pre_artifact_check_uses_actual_file_mtimes(git_repo: Path) -> None:
    commitment = git_repo / "holdout.json"
    artifact = git_repo / "train" / "model.pt"
    _write(commitment, "commitment")
    _write(artifact, "weights")
    _set_mtime(commitment, 1_000_000_000)
    _set_mtime(artifact, 2_000_000_000)

    evidence = verify_live_pre_artifact_order(
        repo_root=git_repo,
        commitment_path="holdout.json",
        artifacts=(ArtifactRecord("train/model.pt", EvidenceRole.TRAIN),),
    )

    assert evidence.commitment_mtime_ns == 1_000_000_000
    assert evidence.artifacts[0].mtime_ns == 2_000_000_000

    _set_mtime(commitment, 3_000_000_000)
    with pytest.raises(CommitmentOrderError, match="mtime"):
        verify_live_pre_artifact_order(
            repo_root=git_repo,
            commitment_path="holdout.json",
            artifacts=(ArtifactRecord("train/model.pt", EvidenceRole.TRAIN),),
        )


def test_committed_check_proves_first_add_commit_ancestry_and_mtime(
    git_repo: Path,
) -> None:
    commitment_commit = _commit(git_repo, "holdout.json", "sealed", "commit")
    artifacts = (ArtifactRecord("train/model.pt", EvidenceRole.TRAIN),)
    live = _live_evidence(artifacts)
    _write(git_repo / "train/model.pt", "weights")
    artifact_commit = _commit_live_batch(
        git_repo,
        live,
        artifact_paths=("train/model.pt",),
    )
    evidence = verify_committed_artifact_order(
        repo_root=git_repo,
        commitment_path="holdout.json",
        artifacts=artifacts,
        live_evidence=live,
        live_evidence_path=_LIVE_EVIDENCE_PATH,
        reveal_verified=True,
    )

    assert evidence.commitment_first_add_commit == commitment_commit
    assert evidence.artifacts[0].first_add_commit == artifact_commit
    assert evidence.reveal_verified is True


def test_modified_commitment_history_is_rejected(git_repo: Path) -> None:
    _commit(git_repo, "holdout.json", "sealed", "commit")
    artifacts = (ArtifactRecord("train/model.pt", EvidenceRole.TRAIN),)
    live = _live_evidence(artifacts)
    _write(git_repo / "train/model.pt", "weights")
    _commit_live_batch(
        git_repo,
        live,
        artifact_paths=("train/model.pt",),
    )
    _commit(git_repo, "holdout.json", "rewritten", "rewrite commitment")

    with pytest.raises(CommitmentOrderError, match="modified"):
        verify_committed_artifact_order(
            repo_root=git_repo,
            commitment_path="holdout.json",
            artifacts=artifacts,
            live_evidence=live,
            live_evidence_path=_LIVE_EVIDENCE_PATH,
            reveal_verified=True,
        )


@pytest.mark.parametrize("kind", ["missing", "untracked", "modified"])
def test_committed_check_rejects_missing_untracked_or_modified_artifacts(
    git_repo: Path,
    kind: str,
) -> None:
    _commit(git_repo, "holdout.json", "sealed", "commit")
    if kind == "missing":
        record = ArtifactRecord("train/missing.pt", EvidenceRole.TRAIN)
        live = _live_evidence((record,))
        _commit_live_batch(git_repo, live)
    elif kind == "untracked":
        record = ArtifactRecord("train/model.pt", EvidenceRole.TRAIN)
        live = _live_evidence((record,))
        _commit_live_batch(git_repo, live)
        _write(git_repo / "train/model.pt", "weights")
    else:
        record = ArtifactRecord("train/model.pt", EvidenceRole.TRAIN)
        live = _live_evidence((record,))
        _write(git_repo / "train/model.pt", "weights")
        _commit_live_batch(
            git_repo,
            live,
            artifact_paths=("train/model.pt",),
        )
        _write(git_repo / "train/model.pt", "changed")

    with pytest.raises(CommitmentOrderError, match=kind):
        verify_committed_artifact_order(
            repo_root=git_repo,
            commitment_path="holdout.json",
            artifacts=(record,),
            live_evidence=live,
            live_evidence_path=_LIVE_EVIDENCE_PATH,
            reveal_verified=True,
        )


def test_only_explicit_legacy_role_is_excluded(git_repo: Path) -> None:
    _commit(git_repo, "holdout.json", "sealed", "commit")
    suspicious_path = git_repo / "experiments" / "legacy-looking.pt"
    _write(suspicious_path, "old")
    legacy = ArtifactRecord("arbitrary/absent.pt", EvidenceRole.LEGACY)
    empty_live = _live_evidence((legacy,))
    _commit_live_batch(git_repo, empty_live)

    verify_committed_artifact_order(
        repo_root=git_repo,
        commitment_path="holdout.json",
        artifacts=(legacy,),
        live_evidence=empty_live,
        live_evidence_path=_LIVE_EVIDENCE_PATH,
        reveal_verified=True,
    )

    suspicious = ArtifactRecord("experiments/legacy-looking.pt", EvidenceRole.TRAIN)
    with pytest.raises(CommitmentOrderError, match="persisted"):
        verify_committed_artifact_order(
            repo_root=git_repo,
            commitment_path="holdout.json",
            artifacts=(suspicious,),
            live_evidence=empty_live,
            live_evidence_path=_LIVE_EVIDENCE_PATH,
            reveal_verified=True,
        )


def test_reveal_must_be_verified_for_committed_artifacts(git_repo: Path) -> None:
    _commit(git_repo, "holdout.json", "sealed", "commit")
    live = _live_evidence()
    _commit_live_batch(git_repo, live)

    with pytest.raises(CommitmentOrderError, match="reveal"):
        verify_committed_artifact_order(
            repo_root=git_repo,
            commitment_path="holdout.json",
            artifacts=(),
            live_evidence=live,
            live_evidence_path=_LIVE_EVIDENCE_PATH,
            reveal_verified=False,
        )


def test_commitment_itself_must_be_tracked_clean_and_unmodified(git_repo: Path) -> None:
    _write(git_repo / "holdout.json", "sealed")
    with pytest.raises(CommitmentOrderError, match="untracked"):
        verify_committed_artifact_order(
            repo_root=git_repo,
            commitment_path="holdout.json",
            artifacts=(),
            live_evidence=_live_evidence(),
            live_evidence_path=_LIVE_EVIDENCE_PATH,
            reveal_verified=True,
        )

    _git(git_repo, "add", "--", "holdout.json")
    _git(git_repo, "commit", "-m", "commit")
    live = _live_evidence()
    _commit_live_batch(git_repo, live)
    _write(git_repo / "holdout.json", "dirty")
    with pytest.raises(CommitmentOrderError, match="modified"):
        verify_committed_artifact_order(
            repo_root=git_repo,
            commitment_path="holdout.json",
            artifacts=(),
            live_evidence=live,
            live_evidence_path=_LIVE_EVIDENCE_PATH,
            reveal_verified=True,
        )


def test_git_probe_rejects_paths_outside_repository(git_repo: Path) -> None:
    probe = GitProbe(git_repo)

    with pytest.raises(ValueError, match="inside repository"):
        probe.inspect("../escape.json")


@pytest.mark.parametrize(
    "path, role, message",
    [
        ("", EvidenceRole.TRAIN, "non-empty"),
        (".", EvidenceRole.TRAIN, "inside repository"),
        (str(Path.cwd().anchor), EvidenceRole.TRAIN, "inside repository"),
        ("model.pt", "UNKNOWN", "declared evidence Role"),
    ],
)
def test_artifact_record_rejects_invalid_path_and_role(
    path: str,
    role: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ArtifactRecord(path, role)  # type: ignore[arg-type]


def _pure_committed_evidence(
    *,
    reveal_verified: bool = True,
    history: tuple[str, ...] = ("commitment-add",),
    artifact_mtime_ns: int = 2,
    strict_ancestor: bool = True,
) -> CommittedCommitmentOrderEvidence:
    live = LiveCommitmentOrderEvidence(
        commitment_path="holdout.json",
        commitment_mtime_ns=1,
        artifacts=(
            LiveArtifactEvidence(
                path="train/model.pt",
                role=EvidenceRole.TRAIN,
                mtime_ns=artifact_mtime_ns,
            ),
        ),
    )
    return CommittedCommitmentOrderEvidence(
        commitment_path="holdout.json",
        commitment_first_add_commit="commitment-add",
        commitment_history_commits=history,
        artifacts=(
            CommittedArtifactEvidence(
                path="train/model.pt",
                role=EvidenceRole.TRAIN,
                first_add_commit="artifact-add",
                commitment_is_strict_ancestor=strict_ancestor,
            ),
        ),
        live_evidence=live,
        live_evidence_path=_LIVE_EVIDENCE_PATH,
        live_evidence_first_add_commit="artifact-add",
        live_evidence_history_commits=("artifact-add",),
        commitment_is_strict_ancestor_of_live_evidence=True,
        reveal_verified=reveal_verified,
    )


@pytest.mark.parametrize(
    "evidence, message",
    [
        (_pure_committed_evidence(reveal_verified=False), "reveal"),
        (
            _pure_committed_evidence(history=("rewrite", "commitment-add")),
            "modified",
        ),
        (_pure_committed_evidence(artifact_mtime_ns=1), "mtime"),
        (_pure_committed_evidence(strict_ancestor=False), "Git commit"),
    ],
)
def test_pure_committed_evidence_fails_closed(
    evidence: CommittedCommitmentOrderEvidence,
    message: str,
) -> None:
    with pytest.raises(CommitmentOrderError, match=message):
        evidence.verify()


def test_git_probe_rejects_non_repository_and_nested_roots(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="existing directory"):
        GitProbe(missing)

    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    with pytest.raises(ValueError, match="Git working tree"):
        GitProbe(ordinary)

    repo = tmp_path / "repo"
    nested = repo / "nested"
    nested.mkdir(parents=True)
    _git(repo, "init", "--initial-branch=main")
    with pytest.raises(ValueError, match="working-tree root"):
        GitProbe(nested)


def test_git_probe_fails_closed_on_git_query_errors(git_repo: Path) -> None:
    probe = GitProbe(git_repo)

    with pytest.raises(CommitmentOrderError, match="unknown option"):
        probe._run(("--definitely-not-a-git-option",), check=True)
    with pytest.raises(CommitmentOrderError, match="ancestry"):
        probe.is_strict_ancestor("not-a-commit", "also-not-a-commit")


def test_git_probe_treats_same_commit_as_not_strictly_earlier(
    git_repo: Path,
) -> None:
    commit = _commit(git_repo, "holdout.json", "sealed", "commit")

    assert not GitProbe(git_repo).is_strict_ancestor(commit, commit)


def test_duplicate_invalid_and_commitment_artifact_records_are_rejected(
    git_repo: Path,
) -> None:
    _write(git_repo / "holdout.json", "sealed")
    record = ArtifactRecord("train/model.pt", EvidenceRole.TRAIN)

    with pytest.raises(ValueError, match="ArtifactRecord"):
        verify_live_pre_artifact_order(
            repo_root=git_repo,
            commitment_path="holdout.json",
            artifacts=(object(),),  # type: ignore[arg-type]
        )
    with pytest.raises(CommitmentOrderError, match="unique"):
        verify_live_pre_artifact_order(
            repo_root=git_repo,
            commitment_path="holdout.json",
            artifacts=(record, record),
        )
    with pytest.raises(CommitmentOrderError, match="also be an artifact"):
        verify_live_pre_artifact_order(
            repo_root=git_repo,
            commitment_path="holdout.json",
            artifacts=(ArtifactRecord("holdout.json", EvidenceRole.TRAIN),),
        )


def test_live_check_rejects_invalid_root_missing_files_and_escaped_paths(
    tmp_path: Path,
    git_repo: Path,
) -> None:
    with pytest.raises(ValueError, match="existing directory"):
        verify_live_pre_artifact_order(
            repo_root=tmp_path / "missing",
            commitment_path="holdout.json",
            artifacts=(),
        )
    with pytest.raises(CommitmentOrderError, match="missing"):
        verify_live_pre_artifact_order(
            repo_root=git_repo,
            commitment_path="holdout.json",
            artifacts=(),
        )

    _write(git_repo / "holdout.json", "sealed")
    with pytest.raises(ValueError, match="inside repository"):
        verify_live_pre_artifact_order(
            repo_root=git_repo,
            commitment_path="holdout.json",
            artifacts=(ArtifactRecord("../escape.pt", EvidenceRole.TRAIN),),
        )
    with pytest.raises(CommitmentOrderError, match="missing"):
        verify_live_pre_artifact_order(
            repo_root=git_repo,
            commitment_path="holdout.json",
            artifacts=(ArtifactRecord("train/missing.pt", EvidenceRole.TRAIN),),
        )


def test_committed_check_rejects_non_boolean_reveal_flag(git_repo: Path) -> None:
    with pytest.raises(ValueError, match="boolean"):
        verify_committed_artifact_order(
            repo_root=git_repo,
            commitment_path="holdout.json",
            artifacts=(),
            live_evidence=_live_evidence(),
            live_evidence_path=_LIVE_EVIDENCE_PATH,
            reveal_verified=1,  # type: ignore[arg-type]
        )


def test_commitment_and_artifact_added_together_do_not_prove_order(
    git_repo: Path,
) -> None:
    _write(git_repo / "holdout.json", "sealed")
    _write(git_repo / "train/model.pt", "weights")
    artifacts = (ArtifactRecord("train/model.pt", EvidenceRole.TRAIN),)
    live = _live_evidence(artifacts)
    _write(
        git_repo / _LIVE_EVIDENCE_PATH,
        live.to_canonical_bytes().decode("utf-8"),
    )
    _git(
        git_repo,
        "add",
        "--",
        "holdout.json",
        "train/model.pt",
        _LIVE_EVIDENCE_PATH,
    )
    _git(git_repo, "commit", "-m", "added together")

    with pytest.raises(CommitmentOrderError, match="strictly precede"):
        verify_committed_artifact_order(
            repo_root=git_repo,
            commitment_path="holdout.json",
            artifacts=artifacts,
            live_evidence=live,
            live_evidence_path=_LIVE_EVIDENCE_PATH,
            reveal_verified=True,
        )


def test_fresh_clone_uses_persisted_live_mtimes_not_checkout_mtimes(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    _commit(git_repo, "holdout.json", "sealed", "commit")
    artifact = git_repo / "train/model.pt"
    _write(artifact, "weights")
    _set_mtime(git_repo / "holdout.json", 1_000_000_000)
    _set_mtime(artifact, 2_000_000_000)
    live = verify_live_pre_artifact_order(
        repo_root=git_repo,
        commitment_path="holdout.json",
        artifacts=(ArtifactRecord("train/model.pt", EvidenceRole.TRAIN),),
    )
    _write(
        git_repo / "audit/live-order.json",
        live.to_canonical_bytes().decode("utf-8"),
    )
    _git(git_repo, "add", "--", "train/model.pt", "audit/live-order.json")
    _git(git_repo, "commit", "-m", "training artifact")

    clone = tmp_path / "fresh-clone"
    _git(tmp_path, "clone", str(git_repo), str(clone))
    _set_mtime(clone / "holdout.json", 9_000_000_000)
    _set_mtime(clone / "train/model.pt", 8_000_000_000)
    persisted = LiveCommitmentOrderEvidence.from_canonical_bytes(
        (clone / "audit/live-order.json").read_bytes()
    )

    evidence = verify_committed_artifact_order(
        repo_root=clone,
        commitment_path="holdout.json",
        artifacts=(ArtifactRecord("train/model.pt", EvidenceRole.TRAIN),),
        live_evidence=persisted,
        live_evidence_path="audit/live-order.json",
        reveal_verified=True,
    )

    assert evidence.live_evidence == persisted


def test_persisted_record_must_exactly_match_declared_paths_and_roles(
    git_repo: Path,
) -> None:
    _commit(git_repo, "holdout.json", "sealed", "commit")
    _commit(git_repo, "train/model.pt", "weights", "training artifact")
    declared = (ArtifactRecord("train/model.pt", EvidenceRole.TRAIN),)
    mismatches = (
        _live_evidence(declared, commitment_path="other.json"),
        _live_evidence((ArtifactRecord("train/other.pt", EvidenceRole.TRAIN),)),
        _live_evidence((ArtifactRecord("train/model.pt", EvidenceRole.DEVELOPMENT),)),
        _live_evidence(),
    )

    for persisted in mismatches:
        with pytest.raises(CommitmentOrderError, match="persisted"):
            verify_committed_artifact_order(
                repo_root=git_repo,
                commitment_path="holdout.json",
                artifacts=declared,
                live_evidence=persisted,
                live_evidence_path=_LIVE_EVIDENCE_PATH,
                reveal_verified=True,
            )


def test_pure_committed_evidence_rechecks_persisted_identity() -> None:
    evidence = _pure_committed_evidence()
    wrong_commitment = CommittedCommitmentOrderEvidence(
        commitment_path="other.json",
        commitment_first_add_commit=evidence.commitment_first_add_commit,
        commitment_history_commits=evidence.commitment_history_commits,
        artifacts=evidence.artifacts,
        live_evidence=evidence.live_evidence,
        live_evidence_path=evidence.live_evidence_path,
        live_evidence_first_add_commit=evidence.live_evidence_first_add_commit,
        live_evidence_history_commits=evidence.live_evidence_history_commits,
        commitment_is_strict_ancestor_of_live_evidence=(
            evidence.commitment_is_strict_ancestor_of_live_evidence
        ),
        reveal_verified=True,
    )
    wrong_artifacts = CommittedCommitmentOrderEvidence(
        commitment_path=evidence.commitment_path,
        commitment_first_add_commit=evidence.commitment_first_add_commit,
        commitment_history_commits=evidence.commitment_history_commits,
        artifacts=(),
        live_evidence=evidence.live_evidence,
        live_evidence_path=evidence.live_evidence_path,
        live_evidence_first_add_commit=evidence.live_evidence_first_add_commit,
        live_evidence_history_commits=evidence.live_evidence_history_commits,
        commitment_is_strict_ancestor_of_live_evidence=(
            evidence.commitment_is_strict_ancestor_of_live_evidence
        ),
        reveal_verified=True,
    )

    with pytest.raises(CommitmentOrderError, match="commitment path"):
        wrong_commitment.verify()
    with pytest.raises(CommitmentOrderError, match="artifact paths and roles"):
        wrong_artifacts.verify()


def test_committed_check_requires_a_persisted_live_record(git_repo: Path) -> None:
    with pytest.raises(ValueError, match="persisted LiveCommitmentOrderEvidence"):
        verify_committed_artifact_order(
            repo_root=git_repo,
            commitment_path="holdout.json",
            artifacts=(),
            live_evidence=object(),  # type: ignore[arg-type]
            live_evidence_path=_LIVE_EVIDENCE_PATH,
            reveal_verified=True,
        )


def test_committed_check_rejects_untracked_and_wrong_live_evidence_files(
    git_repo: Path,
) -> None:
    _commit(git_repo, "holdout.json", "sealed", "commitment")
    live = _live_evidence()
    _write(git_repo / _LIVE_EVIDENCE_PATH, "not the supplied canonical record")

    with pytest.raises(CommitmentOrderError, match="untracked live evidence"):
        verify_committed_artifact_order(
            repo_root=git_repo,
            commitment_path="holdout.json",
            artifacts=(),
            live_evidence=live,
            live_evidence_path=_LIVE_EVIDENCE_PATH,
            reveal_verified=True,
        )

    _git(git_repo, "add", "--", _LIVE_EVIDENCE_PATH)
    _git(git_repo, "commit", "-m", "wrong witness file")
    with pytest.raises(CommitmentOrderError, match="does not match"):
        verify_committed_artifact_order(
            repo_root=git_repo,
            commitment_path="holdout.json",
            artifacts=(),
            live_evidence=live,
            live_evidence_path=_LIVE_EVIDENCE_PATH,
            reveal_verified=True,
        )


def test_committed_check_rejects_modified_live_evidence_file(git_repo: Path) -> None:
    _commit(git_repo, "holdout.json", "sealed", "commitment")
    live = _live_evidence()
    _commit_live_batch(git_repo, live)
    _write(git_repo / _LIVE_EVIDENCE_PATH, "working-tree tamper")

    with pytest.raises(CommitmentOrderError, match="modified live evidence"):
        verify_committed_artifact_order(
            repo_root=git_repo,
            commitment_path="holdout.json",
            artifacts=(),
            live_evidence=live,
            live_evidence_path=_LIVE_EVIDENCE_PATH,
            reveal_verified=True,
        )


def test_committed_check_rejects_witness_modified_in_git_history(
    git_repo: Path,
) -> None:
    _commit(git_repo, "holdout.json", "sealed", "commitment")
    live = _live_evidence()
    _commit_live_batch(git_repo, live)
    _commit(
        git_repo,
        _LIVE_EVIDENCE_PATH,
        live.to_canonical_bytes().decode("utf-8") + " ",
        "rewrite witness",
    )

    with pytest.raises(CommitmentOrderError, match="modified after first add"):
        verify_committed_artifact_order(
            repo_root=git_repo,
            commitment_path="holdout.json",
            artifacts=(),
            live_evidence=live,
            live_evidence_path=_LIVE_EVIDENCE_PATH,
            reveal_verified=True,
        )


def test_witness_and_artifact_must_share_their_first_add_commit(
    git_repo: Path,
) -> None:
    _commit(git_repo, "holdout.json", "sealed", "commitment")
    artifact = ArtifactRecord("train/model.pt", EvidenceRole.TRAIN)
    _commit(git_repo, artifact.path, "weights", "artifact without witness")
    live = _live_evidence((artifact,))
    _commit_live_batch(git_repo, live)

    with pytest.raises(CommitmentOrderError, match="same commit"):
        verify_committed_artifact_order(
            repo_root=git_repo,
            commitment_path="holdout.json",
            artifacts=(artifact,),
            live_evidence=live,
            live_evidence_path=_LIVE_EVIDENCE_PATH,
            reveal_verified=True,
        )


def test_commitment_must_strictly_precede_witness_commit(git_repo: Path) -> None:
    live = _live_evidence()
    _commit_live_batch(git_repo, live)
    _commit(git_repo, "holdout.json", "sealed", "late commitment")

    with pytest.raises(CommitmentOrderError, match="strictly precede"):
        verify_committed_artifact_order(
            repo_root=git_repo,
            commitment_path="holdout.json",
            artifacts=(),
            live_evidence=live,
            live_evidence_path=_LIVE_EVIDENCE_PATH,
            reveal_verified=True,
        )
