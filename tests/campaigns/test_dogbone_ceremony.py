from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import patch

import pytest

from campaigns.dogbone_v1.campaign_proposal import (
    LOAD_APPLICATION_POLICY_ID,
    proposed_budget,
    proposed_design_spec,
    proposed_executable_envelope,
)
from campaigns.dogbone_v1.ceremony import (
    CP_ENVELOPE_PATH,
    EXECUTABLE_ENVELOPE_PATH,
    PRIVATE_REVEAL_PATH,
    PUBLIC_MANIFEST_PATH,
    REPLACEMENT_POLICY_PATH,
    STRAIN_64_STEP_EVIDENCE_HASH,
    S9Ceremony,
    approved_pilot_evidence,
    build_s9_preregistration,
    build_s9_reserve_policy,
    build_s9_reveal_document,
    execute_s9_ceremony,
    write_private_reveal,
    write_public_preregistration_artifacts,
)
from piginot.contracts import canonical_json_bytes, hash_spec
from piginot.doe import EvidenceRole
from piginot.doe.design import generate_coupled_design

_CREATED_UTC = "2026-08-31T00:00:00Z"
_TEST_SECRET_DRAWS = (
    bytes(range(32)),
    bytes(range(32, 64)),
    bytes(range(64, 96)),
)


@pytest.fixture(scope="module")
def ceremony() -> S9Ceremony:
    with patch(
        "piginot.doe.secret_material.secrets.token_bytes",
        side_effect=_TEST_SECRET_DRAWS,
    ):
        return build_s9_preregistration(created_utc=_CREATED_UTC)


def _initialize_ignored_repo(repo_root: Path) -> None:
    repo_root.mkdir(parents=True)
    (repo_root / ".gitignore").write_text(".piginot-private/\n", encoding="ascii")
    import subprocess

    subprocess.run(
        ["git", "init", "--quiet", str(repo_root)],
        check=True,
        capture_output=True,
        text=True,
    )


def _backup_path(tmp_path: Path) -> Path:
    return (tmp_path / "backup" / "dogbone-v1" / "reveal-v1.json").resolve()


def _disable_private_acl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "campaigns.dogbone_v1.ceremony._restrict_private_directory_to_operator",
        lambda _path: None,
    )
    monkeypatch.setattr(
        "campaigns.dogbone_v1.ceremony._restrict_private_file_to_operator",
        lambda _path: None,
    )


def test_production_builder_has_no_caller_secret_injection_surface() -> None:
    assert (
        "campaign_secrets" not in inspect.signature(build_s9_preregistration).parameters
    )


def test_production_builder_rejects_a_replaced_nonfactory_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "campaigns.dogbone_v1.ceremony.generate_campaign_secrets",
        lambda: object(),
    )

    with pytest.raises(TypeError, match="generate_campaign_secrets"):
        build_s9_preregistration(created_utc=_CREATED_UTC)


def test_production_preregistration_binds_every_ratified_input(
    ceremony: S9Ceremony,
) -> None:
    checkpoint = ceremony.result.checkpoint

    assert ceremony.executable_envelope == proposed_executable_envelope()
    assert ceremony.spec == proposed_design_spec(
        seed=ceremony.campaign_secrets.doe_seed
    )
    assert checkpoint.executable_envelope_hash == hash_spec(
        ceremony.executable_envelope, "content"
    )
    assert checkpoint.private_doe_plan_hash == hash_spec(
        {"design": ceremony.design, "spec": ceremony.spec}, "content"
    )
    assert checkpoint.budget == proposed_budget()
    assert checkpoint.budget.holdout == 64
    assert checkpoint.derived_axes == (
        "arc_feasibility_ratio",
        "fillet_to_gauge_ratio",
        "grip_to_gauge_ratio",
        "length_to_gauge_ratio",
    )
    assert tuple(
        (reference.role, reference.content_hash)
        for reference in checkpoint.pilot_evidence
    ) == tuple(
        (reference.role, reference.content_hash)
        for reference in approved_pilot_evidence()
    )
    assert any(
        reference.content_hash == STRAIN_64_STEP_EVIDENCE_HASH
        and reference.role is EvidenceRole.PILOT
        for reference in checkpoint.pilot_evidence
    )


def test_reserve_policy_uses_ratified_role_counts_in_design_order() -> None:
    envelope = proposed_executable_envelope()
    spec = proposed_design_spec(seed="public-dogbone-s9-proposal-dry-run-v2")
    design = generate_coupled_design(envelope, spec)

    policy = build_s9_reserve_policy(design)

    assert tuple(
        len(policy.candidates_for_role(role))
        for role in (EvidenceRole.PILOT, EvidenceRole.TRAIN, EvidenceRole.DEVELOPMENT)
    ) == (4, 20, 8)
    assert len(policy.candidates_for_role(EvidenceRole.HOLDOUT)) == 0


def test_public_manifest_excludes_holdout_and_keeps_reserves_visible(
    ceremony: S9Ceremony,
) -> None:
    manifest = ceremony.result.public_manifest

    assert {case.role for case in manifest.cases}.isdisjoint({EvidenceRole.HOLDOUT})
    assert sum(case.status.value == "RESERVE" for case in manifest.cases) == 32
    assert len(manifest.cases) == 376


def test_private_reveal_recomputes_every_private_binding(
    ceremony: S9Ceremony,
) -> None:
    reveal = build_s9_reveal_document(ceremony)

    assert reveal["load_application_policy_id"] == LOAD_APPLICATION_POLICY_ID
    assert reveal["doe_seed_hex"] == ceremony.campaign_secrets.doe_seed.hex()
    assert (
        reveal["allocation_seed_hex"] == ceremony.campaign_secrets.allocation_seed.hex()
    )
    assert reveal["holdout_salt_hex"] == ceremony.campaign_secrets.holdout_salt.hex()
    assert len(reveal["holdout_coordinates"]) == 64
    assert hash_spec(reveal["doe_plan_preimage"], "content") == (
        ceremony.result.binding.private_doe_plan_hash
    )
    assert hash_spec(reveal["role_ledger_preimage"], "content") == (
        ceremony.result.binding.role_ledger_hash
    )
    assert hash_spec(reveal["replacement_policy_preimage"], "content") == (
        ceremony.result.binding.replacement_policy_hash
    )


def test_no_private_secret_appears_in_any_public_payload(ceremony: S9Ceremony) -> None:
    public_payloads = (
        ceremony.result.checkpoint.to_bytes(),
        ceremony.executable_envelope.to_bytes(),
        ceremony.result.public_manifest.to_bytes(),
        canonical_json_bytes(ceremony.replacement_policy),
    )

    for secret in (
        ceremony.campaign_secrets.holdout_salt.hex(),
        ceremony.campaign_secrets.doe_seed.hex(),
        ceremony.campaign_secrets.allocation_seed.hex(),
    ):
        assert all(secret.encode("ascii") not in payload for payload in public_payloads)


def test_ceremony_repr_cannot_expose_holdout_identity_or_coordinates(
    ceremony: S9Ceremony,
) -> None:
    rendered = repr(ceremony)
    assignments = ceremony.result.production_allocation.registry.assignments
    holdout = next(
        point
        for point in ceremony.design.selected_points
        if assignments[point.case_id].role is EvidenceRole.HOLDOUT
    )

    assert len(rendered) < 256
    assert holdout.case_id not in rendered
    assert canonical_json_bytes(holdout.all_coordinates).decode("utf-8") not in rendered
    assert ceremony.campaign_secrets.doe_seed.hex() not in rendered
    assert ceremony.campaign_secrets.allocation_seed.hex() not in rendered
    assert ceremony.campaign_secrets.holdout_salt.hex() not in rendered


def test_public_writer_is_canonical_idempotent_and_preflights_all_conflicts(
    ceremony: S9Ceremony,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    write_public_preregistration_artifacts(ceremony, repo_root)

    expected = {
        CP_ENVELOPE_PATH: ceremony.result.checkpoint.to_bytes(),
        EXECUTABLE_ENVELOPE_PATH: ceremony.executable_envelope.to_bytes(),
        PUBLIC_MANIFEST_PATH: ceremony.result.public_manifest.to_bytes(),
        REPLACEMENT_POLICY_PATH: canonical_json_bytes(ceremony.replacement_policy),
    }
    assert {
        relative: (repo_root / relative).read_bytes() for relative in expected
    } == expected
    write_public_preregistration_artifacts(ceremony, repo_root)

    conflicting_root = tmp_path / "conflicting"
    conflict_path = conflicting_root / REPLACEMENT_POLICY_PATH
    conflict_path.parent.mkdir(parents=True)
    conflict_path.write_bytes(b"different")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_public_preregistration_artifacts(ceremony, conflicting_root)
    assert not (conflicting_root / CP_ENVELOPE_PATH).exists()
    assert not (conflicting_root / EXECUTABLE_ENVELOPE_PATH).exists()
    assert not (conflicting_root / PUBLIC_MANIFEST_PATH).exists()


def test_public_writer_publishes_cp_last_and_rolls_back_on_failure(
    ceremony: S9Ceremony,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from campaigns.dogbone_v1 import ceremony as ceremony_module

    real_write = ceremony_module._write_exact
    successful_root = tmp_path / "successful"
    order: list[Path] = []

    def recording_write(path: Path, payload: bytes) -> bool:
        order.append(path)
        return real_write(path, payload)

    monkeypatch.setattr(ceremony_module, "_write_exact", recording_write)
    write_public_preregistration_artifacts(ceremony, successful_root)
    assert order[-1] == (successful_root / CP_ENVELOPE_PATH).resolve()

    failing_root = tmp_path / "failing"
    attempts = 0

    def fail_after_first(path: Path, payload: bytes) -> bool:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise OSError("simulated public write failure")
        return real_write(path, payload)

    monkeypatch.setattr(ceremony_module, "_write_exact", fail_after_first)
    with pytest.raises(OSError, match="simulated public write failure"):
        write_public_preregistration_artifacts(ceremony, failing_root)
    assert not any(
        (failing_root / relative).exists()
        for relative in (
            CP_ENVELOPE_PATH,
            EXECUTABLE_ENVELOPE_PATH,
            PUBLIC_MANIFEST_PATH,
            REPLACEMENT_POLICY_PATH,
        )
    )


def test_private_writer_requires_ignored_primary_and_external_backup(
    ceremony: S9Ceremony,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _initialize_ignored_repo(repo_root)
    backup = _backup_path(tmp_path)
    _disable_private_acl(monkeypatch)

    primary, written_backup = write_private_reveal(
        ceremony,
        repo_root=repo_root,
        backup_path=backup.resolve(),
    )
    payload = canonical_json_bytes(build_s9_reveal_document(ceremony))
    assert primary == (repo_root / PRIVATE_REVEAL_PATH).resolve()
    assert written_backup == backup.resolve()
    assert primary.read_bytes() == payload
    assert written_backup.read_bytes() == payload
    write_private_reveal(
        ceremony,
        repo_root=repo_root,
        backup_path=backup.resolve(),
    )

    with pytest.raises(ValueError, match="outside the repository"):
        write_private_reveal(
            ceremony,
            repo_root=repo_root,
            backup_path=(repo_root / "backup.json").resolve(),
        )
    with pytest.raises(ValueError, match="absolute path"):
        write_private_reveal(
            ceremony,
            repo_root=repo_root,
            backup_path=Path("relative-backup.json"),
        )


def test_private_writer_removes_new_files_if_acl_restriction_fails(
    ceremony: S9Ceremony,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _initialize_ignored_repo(repo_root)
    backup = _backup_path(tmp_path)
    calls = 0

    def fail_on_backup(_path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PermissionError("simulated ACL failure")

    monkeypatch.setattr(
        "campaigns.dogbone_v1.ceremony._restrict_private_file_to_operator",
        fail_on_backup,
    )
    monkeypatch.setattr(
        "campaigns.dogbone_v1.ceremony._restrict_private_directory_to_operator",
        lambda _path: None,
    )

    with pytest.raises(PermissionError, match="simulated ACL failure"):
        write_private_reveal(
            ceremony,
            repo_root=repo_root,
            backup_path=backup,
        )
    assert not (repo_root / PRIVATE_REVEAL_PATH).exists()
    assert not backup.exists()


def test_private_writer_restricts_empty_files_before_writing_secret_bytes(
    ceremony: S9Ceremony,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _initialize_ignored_repo(repo_root)
    backup = _backup_path(tmp_path)
    observed_contents: list[bytes] = []

    def inspect_before_secret_write(path: Path) -> None:
        observed_contents.append(path.read_bytes())

    monkeypatch.setattr(
        "campaigns.dogbone_v1.ceremony._restrict_private_file_to_operator",
        inspect_before_secret_write,
    )
    monkeypatch.setattr(
        "campaigns.dogbone_v1.ceremony._restrict_private_directory_to_operator",
        lambda _path: None,
    )

    write_private_reveal(
        ceremony,
        repo_root=repo_root,
        backup_path=backup,
    )

    assert observed_contents == [b"", b""]


def test_private_writer_refuses_when_primary_is_not_git_ignored(
    ceremony: S9Ceremony,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    import subprocess

    subprocess.run(
        ["git", "init", "--quiet", str(repo_root)],
        check=True,
        capture_output=True,
        text=True,
    )

    with pytest.raises(ValueError, match="must be ignored by Git"):
        write_private_reveal(
            ceremony,
            repo_root=repo_root,
            backup_path=_backup_path(tmp_path),
        )


def test_execution_preflights_before_generating_production_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _initialize_ignored_repo(repo_root)
    conflict = repo_root / CP_ENVELOPE_PATH
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b"existing")
    generated = False

    def unexpected_generation():
        nonlocal generated
        generated = True
        raise AssertionError("secret factory must not run after a preflight failure")

    monkeypatch.setattr(
        "campaigns.dogbone_v1.ceremony.generate_campaign_secrets",
        unexpected_generation,
    )

    with pytest.raises(FileExistsError, match="ceremony target already exists"):
        execute_s9_ceremony(
            repo_root=repo_root,
            backup_path=_backup_path(tmp_path),
            created_utc=_CREATED_UTC,
        )
    assert generated is False


def test_execution_persists_both_private_copies_before_publication(
    ceremony: S9Ceremony,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _initialize_ignored_repo(repo_root)
    backup = _backup_path(tmp_path)
    monkeypatch.setattr(
        "campaigns.dogbone_v1.ceremony.build_s9_preregistration",
        lambda *, created_utc: ceremony,
    )
    _disable_private_acl(monkeypatch)
    from campaigns.dogbone_v1 import ceremony as ceremony_module

    real_public_writer = ceremony_module.write_public_preregistration_artifacts

    def assert_private_first(value: S9Ceremony, root: Path) -> None:
        assert (root / PRIVATE_REVEAL_PATH).is_file()
        assert backup.is_file()
        real_public_writer(value, root)

    monkeypatch.setattr(
        ceremony_module,
        "write_public_preregistration_artifacts",
        assert_private_first,
    )

    result = execute_s9_ceremony(
        repo_root=repo_root,
        backup_path=backup,
        created_utc=_CREATED_UTC,
    )

    assert result is ceremony
    assert (repo_root / CP_ENVELOPE_PATH).is_file()


def test_execution_removes_private_copies_after_publication_failure(
    ceremony: S9Ceremony,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _initialize_ignored_repo(repo_root)
    backup = _backup_path(tmp_path)
    _disable_private_acl(monkeypatch)
    monkeypatch.setattr(
        "campaigns.dogbone_v1.ceremony.build_s9_preregistration",
        lambda *, created_utc: ceremony,
    )

    def fail_publication(_ceremony: S9Ceremony, _repo_root: Path) -> None:
        raise OSError("simulated publication failure")

    monkeypatch.setattr(
        "campaigns.dogbone_v1.ceremony.write_public_preregistration_artifacts",
        fail_publication,
    )

    with pytest.raises(OSError, match="simulated publication failure"):
        execute_s9_ceremony(
            repo_root=repo_root,
            backup_path=backup,
            created_utc=_CREATED_UTC,
        )
    assert not (repo_root / PRIVATE_REVEAL_PATH).exists()
    assert not backup.exists()
