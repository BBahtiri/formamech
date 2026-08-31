"""Production S9 ceremony for the first DogBone campaign."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from campaigns.dogbone_v1.campaign_proposal import (
    LOAD_APPLICATION_POLICY_ID,
    proposed_budget,
    proposed_design_spec,
    proposed_executable_envelope,
)
from campaigns.dogbone_v1.family import compiled_family
from piginot.contracts import canonical_json_bytes
from piginot.doe.allocation import ProductionRoleBudget, allocate_evidence_roles
from piginot.doe.checkpoint import EvidenceReference
from piginot.doe.design import (
    CoupledDesign,
    CoupledDesignSpec,
    generate_coupled_design,
)
from piginot.doe.envelope import ExecutableEnvelope
from piginot.doe.preregistration import (
    LegacyCaseLedger,
    PreregistrationResult,
    build_preregistration,
)
from piginot.doe.replacement import OrderedReservePolicy
from piginot.doe.roles import EvidenceRole
from piginot.doe.secret_material import (
    CampaignSecrets,
    generate_campaign_secrets,
)

PUBLIC_ARTIFACT_ROOT = Path("artifacts/campaigns/dogbone-v1")
CP_ENVELOPE_PATH = PUBLIC_ARTIFACT_ROOT / "CP-ENVELOPE.json"
EXECUTABLE_ENVELOPE_PATH = PUBLIC_ARTIFACT_ROOT / "executable-envelope.json"
PUBLIC_MANIFEST_PATH = PUBLIC_ARTIFACT_ROOT / "public-case-manifest.json"
REPLACEMENT_POLICY_PATH = PUBLIC_ARTIFACT_ROOT / "replacement-policy.json"
LIVE_ORDER_PATH = PUBLIC_ARTIFACT_ROOT / "live-commitment-order.json"
PRIVATE_REVEAL_PATH = Path(".piginot-private/dogbone-v1/reveal-v1.json")

PUBLIC_MANIFEST_ID = "dogbone-v1-public-cases"
REPLACEMENT_RULE_ID = "role-local-design-order-v1"
PRIVATE_REVEAL_SCHEMA = "piginot.s9.private-reveal-v1"

S1_GATE_EVIDENCE_HASH = (
    "cd0b01e8372d8f8756b373aab50ee77bdf1467755eec83a7ce33bf0ecfff9470"
)
FAMILY_VALIDATION_EVIDENCE_HASH = (
    "4dfac913a1fdbc5d752e83e5053e62fe45cada814b90638cdb89f05169aeb92d"
)
FEM_SOLVABILITY_EVIDENCE_HASH = (
    "e2aa1e0e2cd9d20b31043d9c6cb066c138c9b3cb00c7432111ddd38e6f156928"
)
STRAIN_64_STEP_EVIDENCE_HASH = (
    "2f6cfcb9ed56e072a25ed3b09ade2de9666ade5c7be15fcd33489eb67e4502a0"
)

_RESERVE_COUNTS = MappingProxyType(
    {
        EvidenceRole.PILOT: 4,
        EvidenceRole.TRAIN: 20,
        EvidenceRole.DEVELOPMENT: 8,
    }
)


@dataclass(frozen=True, slots=True, repr=False)
class S9Ceremony:
    """In-memory S9 closure with public outputs and private reveal material."""

    campaign_secrets: CampaignSecrets
    executable_envelope: ExecutableEnvelope
    spec: CoupledDesignSpec
    design: CoupledDesign
    legacy_ledger: LegacyCaseLedger
    replacement_policy: OrderedReservePolicy
    result: PreregistrationResult

    def __repr__(self) -> str:
        return (
            "S9Ceremony("
            f"campaign_hash={self.result.campaign_hash!r}, "
            f"created_utc={self.result.checkpoint.created_utc!r})"
        )


def approved_pilot_evidence() -> tuple[EvidenceReference, ...]:
    """Return the owner-ratified evidence set for the first S9 campaign."""

    return (
        EvidenceReference(
            "S1-G9-derived-axis-origin",
            S1_GATE_EVIDENCE_HASH,
            EvidenceRole.LEGACY,
        ),
        EvidenceReference(
            "dogbone-family-validation-20260828-v1",
            FAMILY_VALIDATION_EVIDENCE_HASH,
            EvidenceRole.PILOT,
        ),
        EvidenceReference(
            "dogbone-fem-solvability-20260829-v1",
            FEM_SOLVABILITY_EVIDENCE_HASH,
            EvidenceRole.PILOT,
        ),
        EvidenceReference(
            "dogbone-strain-envelope-20260830-v3",
            STRAIN_64_STEP_EVIDENCE_HASH,
            EvidenceRole.PILOT,
        ),
    )


def build_s9_reserve_policy(design: CoupledDesign) -> OrderedReservePolicy:
    """Assign the ratified reserve counts in committed design order."""

    offset = 0
    reserves_by_role = {}
    for role, count in _RESERVE_COUNTS.items():
        reserves_by_role[role] = design.reserve_points[offset : offset + count]
        offset += count
    if offset != len(design.reserve_points):
        raise ValueError("reserve role counts do not match the design reserve count")
    return OrderedReservePolicy.preregister(
        rule_id=REPLACEMENT_RULE_ID,
        reserves_by_role=reserves_by_role,
    )


def build_s9_preregistration(
    *,
    created_utc: str,
) -> S9Ceremony:
    """Run the approved production closure once with generated campaign secrets."""

    secrets = generate_campaign_secrets()
    if type(secrets) is not CampaignSecrets:
        raise TypeError("campaign secrets must come from generate_campaign_secrets()")

    family = compiled_family()
    envelope = proposed_executable_envelope()
    budget = proposed_budget()
    spec = proposed_design_spec(seed=secrets.doe_seed)
    design = generate_coupled_design(envelope, spec)
    role_allocation = allocate_evidence_roles(
        design.selected_points,
        ProductionRoleBudget(
            pilot=budget.pilot,
            train=budget.train,
            development=budget.development,
            holdout=budget.holdout,
        ),
        seed=secrets.allocation_seed,
    )
    replacement_policy = build_s9_reserve_policy(design)
    legacy_ledger = LegacyCaseLedger(())
    result = build_preregistration(
        compiled_family=family,
        executable_envelope=envelope,
        pilot_evidence=approved_pilot_evidence(),
        spec=spec,
        design=design,
        role_allocation=role_allocation,
        legacy_ledger=legacy_ledger,
        replacement_policy=replacement_policy,
        budget=budget,
        campaign_secrets=secrets,
        created_utc=created_utc,
        manifest_id=PUBLIC_MANIFEST_ID,
    )
    return S9Ceremony(
        campaign_secrets=secrets,
        executable_envelope=envelope,
        spec=spec,
        design=design,
        legacy_ledger=legacy_ledger,
        replacement_policy=replacement_policy,
        result=result,
    )


def build_s9_reveal_document(ceremony: S9Ceremony) -> Mapping[str, Any]:
    """Build the private reveal preimage; callers must keep it outside Git."""

    assignments = ceremony.result.production_allocation.registry.assignments
    holdout_coordinates = tuple(
        point.all_coordinates
        for point in ceremony.design.selected_points
        if assignments[point.case_id].role is EvidenceRole.HOLDOUT
    )
    document: dict[str, Any] = {
        "schema": PRIVATE_REVEAL_SCHEMA,
        "campaign_hash": ceremony.result.campaign_hash,
        "created_utc": ceremony.result.checkpoint.created_utc,
        "load_application_policy_id": LOAD_APPLICATION_POLICY_ID,
        "secret_generator_id": ceremony.campaign_secrets.generator_id,
        "doe_seed_hex": ceremony.campaign_secrets.doe_seed.hex(),
        "allocation_seed_hex": ceremony.campaign_secrets.allocation_seed.hex(),
        "holdout_salt_hex": ceremony.campaign_secrets.holdout_salt.hex(),
        "holdout_coordinates": tuple(
            sorted(holdout_coordinates, key=canonical_json_bytes)
        ),
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
        "private_doe_plan_hash": ceremony.result.binding.private_doe_plan_hash,
        "role_ledger_hash": ceremony.result.binding.role_ledger_hash,
        "replacement_policy_hash": ceremony.result.binding.replacement_policy_hash,
        "public_case_manifest_hash": ceremony.result.public_manifest.content_hash,
    }
    return MappingProxyType(document)


def _write_exact(path: Path, payload: bytes) -> bool:
    """Create and verify one immutable public file, returning whether it was new."""

    if path.exists() or path.is_symlink():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError("refusing to replace existing evidence")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise FileExistsError("refusing to replace existing evidence") from exc
    if path.read_bytes() != payload:
        path.unlink(missing_ok=True)
        raise OSError("written evidence failed exact read-back verification")
    return True


def _preflight_exact(payloads: Mapping[Path, bytes]) -> None:
    for path, payload in payloads.items():
        if (path.exists() or path.is_symlink()) and (
            not path.is_file() or path.read_bytes() != payload
        ):
            raise FileExistsError("refusing to replace existing evidence")


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _assert_no_reparse_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if _is_reparse_point(current):
            raise ValueError("ceremony paths must not traverse symlinks or junctions")


def _public_targets(repo_root: Path) -> tuple[Path, ...]:
    root = repo_root.resolve()
    relative_paths = (
        EXECUTABLE_ENVELOPE_PATH,
        PUBLIC_MANIFEST_PATH,
        REPLACEMENT_POLICY_PATH,
        CP_ENVELOPE_PATH,
    )
    targets: list[Path] = []
    for relative in relative_paths:
        target = root / relative
        _assert_no_reparse_components(target.parent)
        if not target.resolve().is_relative_to(root):
            raise ValueError(
                "public ceremony artifacts must remain inside the repository"
            )
        targets.append(target)
    return tuple(targets)


def _public_payloads(ceremony: S9Ceremony, repo_root: Path) -> Mapping[Path, bytes]:
    payload_values = (
        ceremony.executable_envelope.to_bytes(),
        ceremony.result.public_manifest.to_bytes(),
        canonical_json_bytes(ceremony.replacement_policy),
        ceremony.result.checkpoint.to_bytes(),
    )
    payloads = dict(zip(_public_targets(repo_root), payload_values, strict=True))
    return MappingProxyType(payloads)


def _resolved_private_paths(
    repo_root: Path,
    backup_path: Path,
) -> tuple[Path, Path]:
    root = repo_root.resolve()
    if not backup_path.is_absolute():
        raise ValueError("private reveal backup must be an absolute path")
    primary_lexical = root / PRIVATE_REVEAL_PATH
    backup_lexical = backup_path.absolute()
    _assert_no_reparse_components(primary_lexical.parent)
    _assert_no_reparse_components(backup_lexical.parent)
    primary = primary_lexical.resolve()
    backup = backup_path.resolve()
    if not primary.is_relative_to(root):
        raise ValueError("private reveal primary must remain inside the repository")
    if backup == primary or backup.is_relative_to(root):
        raise ValueError("private reveal backup must be outside the repository")
    if backup.name != PRIVATE_REVEAL_PATH.name or backup.parent.name != "dogbone-v1":
        raise ValueError(
            "private reveal backup must use a dedicated dogbone-v1 directory"
        )
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "check-ignore",
            "--quiet",
            "--",
            PRIVATE_REVEAL_PATH.as_posix(),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError("the private reveal path must be ignored by Git")
    return primary, backup


def _prepare_private_parent(path: Path) -> None:
    path.mkdir(mode=stat.S_IRWXU, parents=True, exist_ok=True)
    if not path.is_dir() or _is_reparse_point(path):
        raise ValueError("private reveal parent must be a real directory")
    _restrict_private_directory_to_operator(path)


def _write_private_exact(path: Path, payload: bytes) -> bool:
    if path.exists() or path.is_symlink():
        if not path.is_file():
            raise FileExistsError("refusing to replace existing evidence")
        _restrict_private_file_to_operator(path)
        if path.read_bytes() != payload:
            raise FileExistsError("refusing to replace existing evidence")
        return False

    _prepare_private_parent(path.parent)
    try:
        with path.open("xb+") as stream:
            stream.flush()
            os.fsync(stream.fileno())
            _restrict_private_file_to_operator(path)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
            if stream.read() != payload:
                raise OSError("private reveal failed exact read-back verification")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return True


def write_private_reveal(
    ceremony: S9Ceremony,
    *,
    repo_root: Path,
    backup_path: Path,
) -> tuple[Path, Path]:
    """Persist two restricted reveal copies before any public commitment."""

    payload = canonical_json_bytes(build_s9_reveal_document(ceremony))
    primary, backup = _resolved_private_paths(repo_root, backup_path)
    payloads = {primary: payload, backup: payload}
    _preflight_exact(payloads)
    created: list[Path] = []
    try:
        for path in payloads:
            if _write_private_exact(path, payload):
                created.append(path)
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return primary, backup


def write_public_preregistration_artifacts(
    ceremony: S9Ceremony,
    repo_root: Path = Path("."),
) -> None:
    """Write all public S9 preregistration artifacts except live order evidence."""

    payloads = _public_payloads(ceremony, repo_root)
    _preflight_exact(payloads)
    created: list[Path] = []
    try:
        for path, payload in payloads.items():
            if _write_exact(path, payload):
                created.append(path)
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise


def _assert_fresh_ceremony_targets(repo_root: Path, backup_path: Path) -> None:
    primary, backup = _resolved_private_paths(repo_root, backup_path)
    targets = (primary, backup, *_public_targets(repo_root))
    if any(path.exists() or path.is_symlink() for path in targets):
        raise FileExistsError("ceremony target already exists")


def execute_s9_ceremony(
    *,
    repo_root: Path,
    backup_path: Path,
    created_utc: str,
) -> S9Ceremony:
    """Run the one-shot private-first production ceremony."""

    _assert_fresh_ceremony_targets(repo_root, backup_path)
    ceremony = build_s9_preregistration(created_utc=created_utc)
    private_paths = write_private_reveal(
        ceremony,
        repo_root=repo_root,
        backup_path=backup_path,
    )
    try:
        write_public_preregistration_artifacts(ceremony, repo_root)
    except BaseException:
        for path in private_paths:
            path.unlink(missing_ok=True)
        raise
    return ceremony


def _windows_operator_sid() -> str:
    completed = subprocess.run(
        ["whoami", "/user", "/fo", "csv", "/nh"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise PermissionError("failed to resolve the ceremony operator SID")
    try:
        row = next(csv.reader((completed.stdout.strip(),)))
    except (csv.Error, StopIteration) as exc:
        raise PermissionError("failed to resolve the ceremony operator SID") from exc
    if len(row) < 2 or re.fullmatch(r"S-\d(?:-\d+)+", row[1]) is None:
        raise PermissionError("failed to resolve the ceremony operator SID")
    return row[1]


def _verify_windows_operator_only_acl(path: Path, sid: str) -> None:
    environment = os.environ.copy()
    environment["PIGINOT_S9_ACL_TARGET"] = str(path)
    script = (
        "Import-Module Microsoft.PowerShell.Security;"
        "$acl=Get-Acl -LiteralPath $env:PIGINOT_S9_ACL_TARGET;"
        "$entries=@($acl.Access|ForEach-Object {"
        "$translated=$_.IdentityReference.Translate("
        "[System.Security.Principal.SecurityIdentifier]).Value;"
        "[pscustomobject]@{sid=$translated;"
        "allow=($_.AccessControlType -eq "
        "[System.Security.AccessControl.AccessControlType]::Allow);"
        "inherited=$_.IsInherited;"
        "fullControl=(($_.FileSystemRights -band "
        "[System.Security.AccessControl.FileSystemRights]::FullControl) -eq "
        "[System.Security.AccessControl.FileSystemRights]::FullControl)}});"
        "[pscustomobject]@{protected=$acl.AreAccessRulesProtected;entries=$entries}|"
        "ConvertTo-Json -Depth 4 -Compress"
    )
    completed = subprocess.run(
        [
            _powershell_executable(),
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            script,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    try:
        record = json.loads(completed.stdout)
        entries = record["entries"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise PermissionError("failed to verify the private reveal ACL") from exc
    if isinstance(entries, dict):
        entries = [entries]
    expected = {
        "sid": sid,
        "allow": True,
        "inherited": False,
        "fullControl": True,
    }
    if (
        completed.returncode != 0
        or record.get("protected") is not True
        or entries != [expected]
    ):
        raise PermissionError("private reveal ACL is not operator-only")


def _powershell_executable() -> str:
    executable = shutil.which("pwsh") or shutil.which("powershell.exe")
    if executable is None:
        raise PermissionError("PowerShell is required to verify private reveal ACLs")
    return executable


def _restrict_private_path_to_operator(path: Path, *, directory: bool) -> None:
    if os.name != "nt":
        mode = stat.S_IRWXU if directory else stat.S_IRUSR | stat.S_IWUSR
        path.chmod(mode)
        if stat.S_IMODE(path.stat().st_mode) != mode:
            raise PermissionError("failed to verify private reveal permissions")
        return
    sid = _windows_operator_sid()
    grant = f"*{sid}:(OI)(CI)F" if directory else f"*{sid}:F"
    completed = subprocess.run(
        [
            "icacls",
            str(path),
            "/inheritance:r",
            "/remove:g",
            "*S-1-3-4",
            "*S-1-5-18",
            "*S-1-5-32-544",
            f"*{sid}",
            "/grant:r",
            grant,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise PermissionError("failed to restrict private reveal ACL")
    _verify_windows_operator_only_acl(path, sid)


def _restrict_private_directory_to_operator(path: Path) -> None:
    _restrict_private_path_to_operator(path, directory=True)


def _restrict_private_file_to_operator(path: Path) -> None:
    _restrict_private_path_to_operator(path, directory=False)


def main() -> int:
    """Create the private reveal and public S9 preregistration files."""

    created_utc = os.environ.get("PIGINOT_S9_CREATED_UTC")
    if created_utc is None:
        raise SystemExit("PIGINOT_S9_CREATED_UTC is required for the ceremony")
    backup = os.environ.get("PIGINOT_S9_PRIVATE_BACKUP")
    if backup is None:
        raise SystemExit("PIGINOT_S9_PRIVATE_BACKUP is required for the ceremony")
    repository_root = Path(__file__).resolve().parents[2]
    ceremony = execute_s9_ceremony(
        repo_root=repository_root,
        backup_path=Path(backup),
        created_utc=created_utc,
    )
    print(f"campaign_hash: {ceremony.result.campaign_hash}")
    print(f"CP-ENVELOPE: {CP_ENVELOPE_PATH.as_posix()}")
    print(f"executable_envelope: {EXECUTABLE_ENVELOPE_PATH.as_posix()}")
    print(f"public_manifest: {PUBLIC_MANIFEST_PATH.as_posix()}")
    print(f"replacement_policy: {REPLACEMENT_POLICY_PATH.as_posix()}")
    print(
        "private reveal: primary and external backup persisted with restricted access"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "CP_ENVELOPE_PATH",
    "EXECUTABLE_ENVELOPE_PATH",
    "FAMILY_VALIDATION_EVIDENCE_HASH",
    "FEM_SOLVABILITY_EVIDENCE_HASH",
    "LIVE_ORDER_PATH",
    "PRIVATE_REVEAL_PATH",
    "PUBLIC_MANIFEST_ID",
    "PUBLIC_MANIFEST_PATH",
    "REPLACEMENT_POLICY_PATH",
    "REPLACEMENT_RULE_ID",
    "S1_GATE_EVIDENCE_HASH",
    "S9Ceremony",
    "STRAIN_64_STEP_EVIDENCE_HASH",
    "approved_pilot_evidence",
    "build_s9_preregistration",
    "build_s9_reserve_policy",
    "build_s9_reveal_document",
    "execute_s9_ceremony",
    "write_private_reveal",
    "write_public_preregistration_artifacts",
)
