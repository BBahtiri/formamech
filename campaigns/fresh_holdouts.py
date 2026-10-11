"""Fresh sealed HOLDOUT sets H1 and H2 per family (S20, ADR 0101).

The ceremony runs once per family on the decision owner's machine, because
the new coordinates must stay sealed and the disjointness check needs the
family's private S9 reveal (the spent HOLDOUT coordinates).

1. The committed executable envelope is rebuilt and compared byte for byte.
2. Fresh campaign secrets come from ``generate_campaign_secrets()``.
3. The family's S9 design policy, with the fresh DOE seed, draws 128 cases
   plus 64 ordered reserves from the same envelope.
4. Every case within the policy's near-duplicate distance of any existing
   case (public manifest and spent HOLDOUT) is rejected; reserves replace
   rejected cases in design order.
5. The 128 cases are split 64/64 into H1 and H2, alternating within each
   stratum of the primitive axes in an order keyed by the allocation seed.
6. Each set gets a salted exact-64 ``HoldoutCommitment``.
7. The private reveal is written first (Git-ignored primary plus an external
   backup), then the public ``FRESH-HOLDOUTS.json``.

Usage (WSL, from a clean checkout of the S20 branch)::

    python -m campaigns.fresh_holdouts --campaign notched-v1 \\
        --old-reveal <path>/notched-v1/reveal-v1.json \\
        --backup <outside-repo>/notched-v1/fresh-holdouts-v1.json
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import hmac
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from campaigns.dogbone_v1.ceremony import (
    _assert_no_reparse_components,
    _write_exact,
    _write_private_exact,
)
from piginot.contracts import canonical_json_bytes, hash_spec
from piginot.doe.commitment import (
    HoldoutCommitment,
    compute_holdout_commitment,
    verify_holdout_reveal,
)
from piginot.doe.design import (
    CoupledDesign,
    CoupledDesignSpec,
    DesignPoint,
    generate_coupled_design,
    normalized_coordinates,
    weighted_distance,
)
from piginot.doe.secret_material import CampaignSecrets, generate_campaign_secrets
from piginot.git_environment import isolated_git_environment

REPOSITORY = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "s20-fresh-holdouts-v1"
PRIVATE_SCHEMA = "piginot.s20.fresh-holdouts-private-v1"
PUBLIC_SCHEMA = "piginot.s20.fresh-holdouts-v1"
SETS = ("H1", "H2")
SET_SIZE = 64
RESERVE_COUNT = 64
PRIVATE_NAME = "fresh-holdouts-v1.json"
PUBLIC_NAME = "FRESH-HOLDOUTS.json"


@dataclass(frozen=True, slots=True)
class FamilyAdapter:
    """Where one family keeps its envelope, design policy and records."""

    campaign_id: str
    envelope: Callable[[], Any]
    design_spec: Callable[..., CoupledDesignSpec]

    @property
    def artifacts(self) -> Path:
        return Path("artifacts/campaigns") / self.campaign_id

    @property
    def private_path(self) -> Path:
        return Path(".piginot-private") / self.campaign_id / PRIVATE_NAME


def _adapters() -> dict[str, FamilyAdapter]:
    from campaigns.dogbone_v1.campaign_proposal import (
        proposed_design_spec,
        proposed_executable_envelope,
    )
    from campaigns.notched_v1.proposal import design_spec, executable_envelope

    return {
        "dogbone-v1": FamilyAdapter(
            "dogbone-v1", proposed_executable_envelope, proposed_design_spec
        ),
        "notched-v1": FamilyAdapter("notched-v1", executable_envelope, design_spec),
    }


@dataclass(frozen=True, slots=True)
class FreshDraw:
    """The outcome of one family's draw: everything the reveal needs."""

    campaign_id: str
    created_utc: str
    secrets: CampaignSecrets
    spec: CoupledDesignSpec
    design: CoupledDesign
    sets: Mapping[str, tuple[tuple[str, DesignPoint], ...]]
    commitments: Mapping[str, HoldoutCommitment]
    rejected_near_duplicates: int
    existing_case_count: int
    envelope_hash: str


def _commitment(document: Mapping[str, Any]) -> HoldoutCommitment:
    return HoldoutCommitment(**{k: v for k, v in document.items() if k != "$type"})


def _require_reveal(
    commitment: HoldoutCommitment, salt: bytes, coordinates: Sequence[Any], label: str
) -> None:
    if not verify_holdout_reveal(
        commitment, salt=salt, holdout_coordinates=coordinates
    ):
        raise ValueError(f"{label} does not match its holdout commitment")


def set_campaign_hash(campaign_id: str, envelope_hash: str, set_name: str) -> str:
    """The ``campaign_hash`` a set's commitment binds."""

    payload = {
        "protocol_id": PROTOCOL_ID,
        "campaign_id": campaign_id,
        "executable_envelope_hash": envelope_hash,
        "set": set_name,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def existing_points(
    manifest: Mapping[str, Any], old_holdout: Sequence[Mapping[str, float]]
) -> tuple[DesignPoint, ...]:
    """Every case of the family so far, as design points for distance checks."""

    points = [
        DesignPoint(
            case_id=case["case_id"],
            leakage_group_id=case["leakage_group_id"],
            coordinates=case["primitive_coordinates"],
            derived_coordinates=case["derived_coordinates"],
            strata=case["strata"],
        )
        for case in manifest["cases"]
    ]
    derived = set(manifest["derived_axes"])
    for index, coordinates in enumerate(old_holdout):
        points.append(
            DesignPoint(
                case_id=f"spent-holdout-{index:03d}",
                leakage_group_id=f"spent-holdout-{index:03d}",
                coordinates={k: v for k, v in coordinates.items() if k not in derived},
                derived_coordinates={
                    k: v for k, v in coordinates.items() if k in derived
                },
                strata={"spent": 0},
            )
        )
    return tuple(points)


def _keyed_order(secret: bytes, point: DesignPoint) -> bytes:
    return hmac.new(
        secret, canonical_json_bytes(point.all_coordinates), "sha256"
    ).digest()


def split_sets(
    points: Sequence[DesignPoint], secret: bytes, primitive_axes: Sequence[str]
) -> dict[str, tuple[DesignPoint, ...]]:
    """Alternate H1/H2 within each primitive stratum, in keyed order."""

    strata: dict[tuple[int, ...], list[DesignPoint]] = {}
    for point in points:
        key = tuple(int(point.strata.get(axis, 0)) for axis in primitive_axes)
        strata.setdefault(key, []).append(point)
    sets: dict[str, list[DesignPoint]] = {name: [] for name in SETS}
    turn = 0
    for key in sorted(strata):
        for point in sorted(strata[key], key=lambda p: _keyed_order(secret, p)):
            sets[SETS[turn % 2]].append(point)
            turn += 1
    # odd strata can leave the sets one apart; move the excess in keyed order
    while len(sets["H1"]) != len(sets["H2"]):
        larger, smaller = (
            ("H1", "H2") if len(sets["H1"]) > len(sets["H2"]) else ("H2", "H1")
        )
        moved = max(sets[larger], key=lambda p: _keyed_order(secret, p))
        sets[larger].remove(moved)
        sets[smaller].append(moved)
    return {name: tuple(members) for name, members in sets.items()}


def draw(
    adapter: FamilyAdapter,
    *,
    secrets: CampaignSecrets,
    manifest: Mapping[str, Any],
    old_holdout: Sequence[Mapping[str, float]],
    committed_envelope: bytes,
    created_utc: str,
) -> FreshDraw:
    """Draw, filter, split and commit one family's H1 and H2."""

    envelope = adapter.envelope()
    if envelope.to_bytes() != committed_envelope:
        raise ValueError("the rebuilt envelope differs from the committed one")
    envelope_hash = hash_spec(envelope, "content")
    base = adapter.design_spec(seed=secrets.doe_seed)
    total = len(SETS) * SET_SIZE
    spec = dataclasses.replace(
        base,
        sample_count=total,
        reserve_count=RESERVE_COUNT,
        candidate_count=max(base.candidate_count, total + RESERVE_COUNT),
        guard_band_count=min(base.guard_band_count, total // 4),
    )
    design = generate_coupled_design(envelope, spec)
    existing = existing_points(manifest, old_holdout)
    known = [normalized_coordinates(envelope, spec.axes, p) for p in existing]
    tolerance = spec.leakage_policy.near_duplicate_tolerance
    accepted: list[DesignPoint] = []
    rejected = 0
    for point in (*design.selected_points, *design.reserve_points):
        if len(accepted) == total:
            break
        normal = normalized_coordinates(envelope, spec.axes, point)
        if (
            min(weighted_distance(normal, other, spec.axes) for other in known)
            <= tolerance
        ):
            rejected += 1
            continue
        accepted.append(point)
    if len(accepted) < total:
        raise RuntimeError(
            f"only {len(accepted)} of {total} fresh cases clear the near-duplicate rule"
        )
    split = split_sets(accepted, secrets.allocation_seed, manifest["primitive_axes"])
    sets = {
        name: tuple(
            (f"fresh-{name.lower()}-{index:03d}", point)
            for index, point in enumerate(members)
        )
        for name, members in split.items()
    }
    commitments = {
        name: compute_holdout_commitment(
            campaign_hash=set_campaign_hash(adapter.campaign_id, envelope_hash, name),
            campaign_secrets=secrets,
            holdout_coordinates=[point.all_coordinates for _, point in members],
            created_utc=created_utc,
        )
        for name, members in sets.items()
    }
    return FreshDraw(
        campaign_id=adapter.campaign_id,
        created_utc=created_utc,
        secrets=secrets,
        spec=spec,
        design=design,
        sets=sets,
        commitments=commitments,
        rejected_near_duplicates=rejected,
        existing_case_count=len(existing),
        envelope_hash=envelope_hash,
    )


def private_document(result: FreshDraw) -> dict[str, Any]:
    secrets = result.secrets
    return {
        "schema": PRIVATE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": result.campaign_id,
        "created_utc": result.created_utc,
        "secret_generator_id": secrets.generator_id,
        "doe_seed_hex": secrets.doe_seed.hex(),
        "allocation_seed_hex": secrets.allocation_seed.hex(),
        "holdout_salt_hex": secrets.holdout_salt.hex(),
        "design_spec_hash": hash_spec(result.spec, "content"),
        "design_hash": hash_spec(result.design, "content"),
        "sets": {
            name: [
                {"case_id": case_id, "coordinates": dict(point.all_coordinates)}
                for case_id, point in members
            ]
            for name, members in result.sets.items()
        },
    }


def public_document(result: FreshDraw) -> dict[str, Any]:
    return {
        "schema": PUBLIC_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "decision_record": "docs/adr/0101-s20-fresh-holdouts-and-post-mortem.md",
        "campaign_id": result.campaign_id,
        "created_utc": result.created_utc,
        "executable_envelope_hash": result.envelope_hash,
        "design_spec_hash": hash_spec(result.spec, "content"),
        "set_size": SET_SIZE,
        "sets": {
            name: {
                "case_ids": [case_id for case_id, _ in result.sets[name]],
                "commitment": json.loads(canonical_json_bytes(commitment)),
                "use": "S22-S23 certification of v0.1.0"
                if name == "H1"
                else "S25 model, sealed until its CP-FREEZE",
            }
            for name, commitment in result.commitments.items()
        },
        "near_duplicate_tolerance": result.spec.leakage_policy.near_duplicate_tolerance,
        "rejected_near_duplicates": result.rejected_near_duplicates,
        "existing_case_count": result.existing_case_count,
        "fem_labels": "none until each set's reveal",
    }


def verify_set_reveal(
    public: Mapping[str, Any], private: Mapping[str, Any], set_name: str
) -> tuple[Mapping[str, float], ...]:
    """Check a set's private coordinates against its public commitment."""

    commitment = _commitment(public["sets"][set_name]["commitment"])
    coordinates = tuple(entry["coordinates"] for entry in private["sets"][set_name])
    ids = [entry["case_id"] for entry in private["sets"][set_name]]
    if ids != public["sets"][set_name]["case_ids"]:
        raise ValueError(f"{set_name} case IDs differ from the public record")
    _require_reveal(
        commitment, bytes.fromhex(private["holdout_salt_hex"]), coordinates, set_name
    )
    return coordinates


def _private_targets(
    root: Path, adapter: FamilyAdapter, backup: Path
) -> tuple[Path, Path]:
    if not backup.is_absolute():
        raise ValueError("the private backup must be an absolute path")
    primary = (root / adapter.private_path).resolve()
    _assert_no_reparse_components(primary.parent)
    _assert_no_reparse_components(backup.absolute().parent)
    resolved = backup.resolve()
    if resolved.is_relative_to(root.resolve()):
        raise ValueError("the private backup must be outside the repository")
    if resolved.name != PRIVATE_NAME or resolved.parent.name != adapter.campaign_id:
        raise ValueError(
            f"the backup must be <dir>/{adapter.campaign_id}/{PRIVATE_NAME}"
        )
    ignored = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "check-ignore",
            "--quiet",
            "--",
            adapter.private_path.as_posix(),
        ],
        env=isolated_git_environment(),
        check=False,
        capture_output=True,
    )
    if ignored.returncode != 0:
        raise ValueError("the private path must be ignored by Git")
    return primary, resolved


def execute(
    *,
    root: Path,
    campaign_id: str,
    old_reveal: Path,
    backup: Path,
    created_utc: str | None = None,
    secrets: CampaignSecrets | None = None,
) -> FreshDraw:
    """One-shot ceremony: private reveal first, public record last."""

    adapter = _adapters()[campaign_id]
    primary, backup_path = _private_targets(root, adapter, backup)
    public_path = root / adapter.artifacts / PUBLIC_NAME
    if any(p.exists() or p.is_symlink() for p in (primary, backup_path, public_path)):
        raise FileExistsError("a fresh-holdout record already exists")
    cp_envelope = json.loads(
        (root / adapter.artifacts / "CP-ENVELOPE.json").read_text()
    )
    reveal = json.loads(old_reveal.read_text("utf-8"))
    if reveal.get("campaign_hash") != cp_envelope["campaign_hash"]:
        raise ValueError(
            "the old reveal does not belong to this campaign's CP-ENVELOPE"
        )
    _require_reveal(
        _commitment(cp_envelope["holdout_commitment"]),
        bytes.fromhex(reveal["holdout_salt_hex"]),
        reveal["holdout_coordinates"],
        "the old reveal",
    )
    result = draw(
        adapter,
        secrets=secrets if secrets is not None else generate_campaign_secrets(),
        manifest=json.loads(
            (root / adapter.artifacts / "public-case-manifest.json").read_text("utf-8")
        ),
        old_holdout=reveal["holdout_coordinates"],
        committed_envelope=(
            root / adapter.artifacts / "executable-envelope.json"
        ).read_bytes(),
        created_utc=created_utc
        or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    )
    private = canonical_json_bytes(private_document(result))
    written: list[Path] = []
    try:
        for path in (primary, backup_path):
            if _write_private_exact(path, private):
                written.append(path)
        public = (
            json.dumps(public_document(result), indent=2, sort_keys=True) + "\n"
        ).encode()
        if _write_exact(public_path, public):
            written.append(public_path)
    except BaseException:
        for path in reversed(written):
            path.unlink(missing_ok=True)
        raise
    return result


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw fresh sealed H1/H2 sets.")
    parser.add_argument("--campaign", required=True, choices=sorted(_adapters()))
    parser.add_argument("--old-reveal", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    options = parser.parse_args(arguments)
    result = execute(
        root=REPOSITORY,
        campaign_id=options.campaign,
        old_reveal=options.old_reveal,
        backup=options.backup,
    )
    for name, members in result.sets.items():
        print(
            f"{result.campaign_id} {name}: {len(members)} sealed cases, "
            f"commitment {result.commitments[name].commitment[:16]}..."
        )
    print(f"rejected near-duplicates: {result.rejected_near_duplicates}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
