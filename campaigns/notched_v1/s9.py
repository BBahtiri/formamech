"""S9 preregistration ceremony and G9 gate for the notched campaign.

The ceremony is the DogBone ceremony with the notched proposal: secrets are
generated once, the private reveal is written first (a primary copy in the
Git-ignored ``.piginot-private/notched-v1/`` and an external backup), then the
public envelope, manifest, replacement policy, ``CP-ENVELOPE`` and the
live commitment-order evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from campaigns.dogbone_v1.ceremony import (
    S9Ceremony,
    _assert_no_reparse_components,
    _preflight_exact,
    _write_exact,
    _write_private_exact,
)
from campaigns.notched_v1.family import CAMPAIGN_ID, compiled_family
from campaigns.notched_v1.pilot import ARTIFACTS
from campaigns.notched_v1.proposal import (
    LOAD_APPLICATION_POLICY_ID,
    budget,
    design_spec,
    executable_envelope,
)
from piginot.contracts import canonical_json_bytes
from piginot.doe.allocation import ProductionRoleBudget, allocate_evidence_roles
from piginot.doe.checkpoint import EvidenceReference
from piginot.doe.design import generate_coupled_design
from piginot.doe.preregistration import LegacyCaseLedger, build_preregistration
from piginot.doe.provenance import verify_live_pre_artifact_order
from piginot.doe.replacement import OrderedReservePolicy
from piginot.doe.roles import EvidenceRole
from piginot.doe.secret_material import CampaignSecrets, generate_campaign_secrets
from piginot.git_environment import isolated_git_environment

CP_ENVELOPE_PATH = ARTIFACTS / "CP-ENVELOPE.json"
EXECUTABLE_ENVELOPE_PATH = ARTIFACTS / "executable-envelope.json"
PUBLIC_MANIFEST_PATH = ARTIFACTS / "public-case-manifest.json"
REPLACEMENT_POLICY_PATH = ARTIFACTS / "replacement-policy.json"
LIVE_ORDER_PATH = ARTIFACTS / "live-commitment-order.json"
PRIVATE_REVEAL_PATH = Path(f".piginot-private/{CAMPAIGN_ID}/reveal-v1.json")
GATES = ARTIFACTS / "gates"
PUBLIC_MANIFEST_ID = "notched-v1-public-cases"
REPLACEMENT_RULE_ID = "role-local-design-order-v1"
PRIVATE_REVEAL_SCHEMA = "piginot.s9.private-reveal-v1"

FAMILY_VALIDATION_EVIDENCE_HASH = (
    "1bb382419b32755276fcfa8da51cdb4b3c8657d30b49f4c6ab189d8cb51b4976"
)
FEM_PILOT_EVIDENCE_HASH = (
    "641e609b035738a574433c76b7dde7eddc422102a37d41868ee6f6c3a2b203c4"
)
_RESERVE_COUNTS = MappingProxyType(
    {EvidenceRole.PILOT: 2, EvidenceRole.TRAIN: 10, EvidenceRole.DEVELOPMENT: 4}
)


def approved_pilot_evidence() -> tuple[EvidenceReference, ...]:
    return (
        EvidenceReference(
            "notched-family-validation-20261005-v1",
            FAMILY_VALIDATION_EVIDENCE_HASH,
            EvidenceRole.PILOT,
        ),
        EvidenceReference(
            "notched-fem-pilot-20261005-v1",
            FEM_PILOT_EVIDENCE_HASH,
            EvidenceRole.PILOT,
        ),
    )


def reserve_policy(design: Any) -> OrderedReservePolicy:
    offset = 0
    reserves_by_role = {}
    for role, count in _RESERVE_COUNTS.items():
        reserves_by_role[role] = design.reserve_points[offset : offset + count]
        offset += count
    if offset != len(design.reserve_points):
        raise ValueError("reserve role counts do not match the design reserve count")
    return OrderedReservePolicy.preregister(
        rule_id=REPLACEMENT_RULE_ID, reserves_by_role=reserves_by_role
    )


def build_preregistration_closure(*, created_utc: str) -> S9Ceremony:
    """Run the production closure once with freshly generated secrets."""

    secrets = generate_campaign_secrets()
    if type(secrets) is not CampaignSecrets:
        raise TypeError("campaign secrets must come from generate_campaign_secrets()")
    envelope = executable_envelope()
    plan = budget()
    spec = design_spec(seed=secrets.doe_seed)
    design = generate_coupled_design(envelope, spec)
    allocation = allocate_evidence_roles(
        design.selected_points,
        ProductionRoleBudget(
            pilot=plan.pilot,
            train=plan.train,
            development=plan.development,
            holdout=plan.holdout,
        ),
        seed=secrets.allocation_seed,
    )
    replacement = reserve_policy(design)
    legacy = LegacyCaseLedger(())
    result = build_preregistration(
        compiled_family=compiled_family(),
        executable_envelope=envelope,
        pilot_evidence=approved_pilot_evidence(),
        spec=spec,
        design=design,
        role_allocation=allocation,
        legacy_ledger=legacy,
        replacement_policy=replacement,
        budget=plan,
        campaign_secrets=secrets,
        created_utc=created_utc,
        manifest_id=PUBLIC_MANIFEST_ID,
    )
    return S9Ceremony(
        campaign_secrets=secrets,
        executable_envelope=envelope,
        spec=spec,
        design=design,
        legacy_ledger=legacy,
        replacement_policy=replacement,
        result=result,
    )


def reveal_document(ceremony: S9Ceremony) -> Mapping[str, Any]:
    """The private reveal preimage, in the DogBone reveal schema."""

    assignments = ceremony.result.production_allocation.registry.assignments
    holdout = tuple(
        point.all_coordinates
        for point in ceremony.design.selected_points
        if assignments[point.case_id].role is EvidenceRole.HOLDOUT
    )
    secrets = ceremony.campaign_secrets
    binding = ceremony.result.binding
    return MappingProxyType(
        {
            "schema": PRIVATE_REVEAL_SCHEMA,
            "campaign_hash": ceremony.result.campaign_hash,
            "created_utc": ceremony.result.checkpoint.created_utc,
            "load_application_policy_id": LOAD_APPLICATION_POLICY_ID,
            "secret_generator_id": secrets.generator_id,
            "doe_seed_hex": secrets.doe_seed.hex(),
            "allocation_seed_hex": secrets.allocation_seed.hex(),
            "holdout_salt_hex": secrets.holdout_salt.hex(),
            "holdout_coordinates": tuple(sorted(holdout, key=canonical_json_bytes)),
            "doe_plan_preimage": MappingProxyType(
                {"design": ceremony.design, "spec": ceremony.spec}
            ),
            "role_ledger_preimage": MappingProxyType(
                {
                    "legacy_ledger": ceremony.legacy_ledger,
                    "production_allocation": ceremony.result.production_allocation,
                }
            ),
            "replacement_policy_preimage": ceremony.replacement_policy,
            "private_doe_plan_hash": binding.private_doe_plan_hash,
            "role_ledger_hash": binding.role_ledger_hash,
            "replacement_policy_hash": binding.replacement_policy_hash,
            "public_case_manifest_hash": ceremony.result.public_manifest.content_hash,
        }
    )


def _private_paths(repo_root: Path, backup_path: Path) -> tuple[Path, Path]:
    root = repo_root.resolve()
    if not backup_path.is_absolute():
        raise ValueError("private reveal backup must be an absolute path")
    _assert_no_reparse_components((root / PRIVATE_REVEAL_PATH).parent)
    _assert_no_reparse_components(backup_path.absolute().parent)
    primary = (root / PRIVATE_REVEAL_PATH).resolve()
    backup = backup_path.resolve()
    if not primary.is_relative_to(root):
        raise ValueError("private reveal primary must remain inside the repository")
    if backup == primary or backup.is_relative_to(root):
        raise ValueError("private reveal backup must be outside the repository")
    if backup.name != PRIVATE_REVEAL_PATH.name or backup.parent.name != CAMPAIGN_ID:
        raise ValueError(
            f"private reveal backup must use a dedicated {CAMPAIGN_ID} directory"
        )
    ignored = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "check-ignore",
            "--quiet",
            "--",
            PRIVATE_REVEAL_PATH.as_posix(),
        ],
        env=isolated_git_environment(),
        check=False,
        capture_output=True,
    )
    if ignored.returncode != 0:
        raise ValueError("the private reveal path must be ignored by Git")
    return primary, backup


def _public_payloads(ceremony: S9Ceremony, root: Path) -> dict[Path, bytes]:
    return {
        root / EXECUTABLE_ENVELOPE_PATH: ceremony.executable_envelope.to_bytes(),
        root / PUBLIC_MANIFEST_PATH: ceremony.result.public_manifest.to_bytes(),
        root / REPLACEMENT_POLICY_PATH: canonical_json_bytes(
            ceremony.replacement_policy
        ),
        root / CP_ENVELOPE_PATH: ceremony.result.checkpoint.to_bytes(),
    }


def execute_ceremony(
    *, repo_root: Path, backup_path: Path, created_utc: str
) -> S9Ceremony:
    """One-shot, private-first ceremony; the commitment is written last."""

    if FEM_PILOT_EVIDENCE_HASH == "PENDING":
        raise RuntimeError("the FEM pilot evidence hash is not pinned")
    root = repo_root.resolve()
    primary, backup = _private_paths(root, backup_path)
    targets = (primary, backup, *_public_payloads_targets(root), root / LIVE_ORDER_PATH)
    if any(path.exists() or path.is_symlink() for path in targets):
        raise FileExistsError("ceremony target already exists")
    ceremony = build_preregistration_closure(created_utc=created_utc)
    private = canonical_json_bytes(reveal_document(ceremony))
    created: list[Path] = []
    try:
        for path in (primary, backup):
            if _write_private_exact(path, private):
                created.append(path)
        payloads = _public_payloads(ceremony, root)
        _preflight_exact(payloads)
        for path, payload in payloads.items():
            if _write_exact(path, payload):
                created.append(path)
        live = verify_live_pre_artifact_order(
            repo_root=root, commitment_path=CP_ENVELOPE_PATH.as_posix(), artifacts=()
        )
        _write_exact(root / LIVE_ORDER_PATH, live.to_canonical_bytes())
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return ceremony


def _public_payloads_targets(root: Path) -> tuple[Path, ...]:
    return tuple(
        root / path
        for path in (
            EXECUTABLE_ENVELOPE_PATH,
            PUBLIC_MANIFEST_PATH,
            REPLACEMENT_POLICY_PATH,
            CP_ENVELOPE_PATH,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m campaigns.notched_v1.s9")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("ceremony")
    run.add_argument("--created-utc", required=True)
    run.add_argument("--private-backup", type=Path, required=True)
    gate = sub.add_parser("gate")
    gate.add_argument("--reveal-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if arguments.command == "ceremony":
        ceremony = execute_ceremony(
            repo_root=root,
            backup_path=arguments.private_backup,
            created_utc=arguments.created_utc,
        )
        print(f"campaign_hash: {ceremony.result.campaign_hash}")
        print("private reveal: primary and external backup written, operator-only")
        return 0
    from campaigns.notched_v1.s9_gate import write_gate

    document = write_gate(root, reveal_root=arguments.reveal_root)
    json.dump({"verdict": document["verdict"]}, sys.stdout)
    print()
    return 0 if document["verdict"] == "PASS" else 1


if __name__ == "__main__":
    os.umask(0o077)
    raise SystemExit(main())


__all__ = (
    "CP_ENVELOPE_PATH",
    "GATES",
    "LIVE_ORDER_PATH",
    "PRIVATE_REVEAL_PATH",
    "PUBLIC_MANIFEST_PATH",
    "approved_pilot_evidence",
    "build_preregistration_closure",
    "execute_ceremony",
    "reserve_policy",
    "reveal_document",
)
