"""S9 ceremony and G9 evidence of the notched campaign, in a scratch repository."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from campaigns.notched_v1 import proposal, s9, s9_gate
from piginot.doe import PublishedCaseManifest, PublishedCaseStatus
from piginot.doe.roles import EvidenceRole
from tests.platform_support import posix_modes_supported


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=PI-GINOT test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-C",
            str(repo),
            *arguments,
        ],
        check=True,
        capture_output=True,
    )


def test_the_proposal_budget_matches_the_design_policy() -> None:
    plan = proposal.budget()
    spec = proposal.design_spec(seed="public-test-seed")
    assert (plan.pilot, plan.train, plan.development, plan.holdout) == (8, 128, 64, 64)
    assert (
        spec.sample_count == plan.pilot + plan.train + plan.development + plan.holdout
    )
    assert spec.reserve_count == plan.reserve == sum(s9._RESERVE_COUNTS.values())
    envelope = proposal.executable_envelope()
    axes = {axis.name for axis in spec.axes}
    assert {bound.name for bound in envelope.primitive_bounds} <= axes
    assert {"net_section_factor", "notch_sharpness"} <= axes


def test_the_ceremony_and_g9_evidence_in_a_scratch_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(s9, "FEM_PILOT_EVIDENCE_HASH", "f" * 64)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / ".gitignore").write_text(".piginot-private/\n")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-q", "-m", "init")
    backup = tmp_path / "backup" / "notched-v1" / "reveal-v1.json"

    with pytest.raises(RuntimeError, match="not pinned"):
        monkeypatch.setattr(s9, "FEM_PILOT_EVIDENCE_HASH", "PENDING")
        s9.execute_ceremony(repo_root=repo, backup_path=backup, created_utc="x")
    monkeypatch.setattr(s9, "FEM_PILOT_EVIDENCE_HASH", "f" * 64)
    with pytest.raises(ValueError, match="dedicated notched-v1"):
        s9.execute_ceremony(
            repo_root=repo,
            backup_path=tmp_path / "elsewhere" / "reveal-v1.json",
            created_utc="2026-10-05T00:00:00Z",
        )
    ceremony = s9.execute_ceremony(
        repo_root=repo, backup_path=backup, created_utc="2026-10-05T00:00:00Z"
    )
    primary = repo / s9.PRIVATE_REVEAL_PATH
    assert primary.read_bytes() == backup.read_bytes()
    if posix_modes_supported():  # else: POSIX_MODES_REASON (ADR 0100 findings)
        assert primary.stat().st_mode & 0o077 == 0
    manifest_bytes = (repo / s9.PUBLIC_MANIFEST_PATH).read_bytes()
    manifest = PublishedCaseManifest.from_bytes(manifest_bytes)
    assert b'"role":"HOLDOUT"' not in manifest_bytes
    active = [c for c in manifest.cases if c.status is PublishedCaseStatus.ACTIVE]
    roles = {role: sum(c.role is role for c in active) for role in EvidenceRole}
    assert len(active) == 200 and roles[EvidenceRole.TRAIN] == 128
    assert roles[EvidenceRole.DEVELOPMENT] == 64 and roles[EvidenceRole.PILOT] == 8
    assert len(manifest.cases) == 216
    with pytest.raises(FileExistsError):
        s9.execute_ceremony(repo_root=repo, backup_path=backup, created_utc="x")

    public = [str(p) for p in (s9.EXECUTABLE_ENVELOPE_PATH, s9.PUBLIC_MANIFEST_PATH)]
    public += [str(s9.REPLACEMENT_POLICY_PATH), str(s9.CP_ENVELOPE_PATH)]
    _git(repo, "add", *public)
    _git(repo, "commit", "-q", "-m", "commitment")
    _git(repo, "add", str(s9.LIVE_ORDER_PATH))
    _git(repo, "commit", "-q", "-m", "live order")

    reveal = json.loads(primary.read_bytes())
    assert reveal["campaign_hash"] == ceremony.result.campaign_hash
    design = s9_gate.production_design(repo, reveal)
    assert design["plan_hash_verified"] and design["seed_commitment_verified"]
    assert all(design["checks"].values())
    holdout = design.pop("holdout_points")
    assert len(holdout) == 64
    assert set(s9_gate.visibility_violations(repo, holdout).values()) == {0}
    assert s9_gate.commitment_order(repo, True) == (True, True, None)
    file_order, git_order, problem = s9_gate.commitment_order(repo, False)
    assert file_order and not git_order and problem is not None
    evidence = s9_gate.replacement_evidence(repo)
    assert (evidence.failures_injected, evidence.failures_recorded) == (3, 3)
    assert evidence.deterministic_rule_matches
