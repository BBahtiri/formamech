"""S14 step 3 access paths: DEVELOPMENT opening and the one-time HOLDOUT reveal.

Neither path runs before the user has signed off ``S14-PROTOCOL.json``. Both
write a durable record before any label or coordinate is read:

- **DEVELOPMENT.** :func:`open_development` writes an exclusive, fsynced access
  record bound to the signed protocol, then loads the 64 DEVELOPMENT cases of
  the pinned S10 source. A second opening is refused.
- **HOLDOUT.** :func:`consume_and_reveal_holdout` writes the S14 consumption
  record (``piginot.verification.holdout``) and only then reads the private S9
  reveal. The revealed coordinates must open the public ``CP-ENVELOPE``
  holdout commitment. :func:`holdout_cases` turns them into role-neutral
  cases with S9 strata, and :func:`s10_request` builds the unchanged S10
  P2 request recipe for each.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from campaigns.dogbone_v1.campaign_proposal import proposed_design_spec
from campaigns.dogbone_v1.family import compiled_family, parameter_point
from campaigns.dogbone_v1.s10_requests import (
    _boundary_conditions,
    _material,
    _solver_config,
)
from campaigns.dogbone_v1.s14_candidate import S14Case
from campaigns.dogbone_v1.s14_envelope import dogbone_envelope
from piginot.contracts import ExecutionProfile, canonical_json_bytes
from piginot.doe import EvidenceRole, HoldoutCommitment, normalized_coordinates
from piginot.doe.design import DesignPoint, _stratum
from piginot.fem import FemRequest
from piginot.morphing import apply_morph
from piginot.verification.holdout import (
    ConsumptionRecord,
    consume_holdout,
    reveal_after_consumption,
)

_ROOT = Path(__file__).resolve().parents[2]
CP_ENVELOPE_PATH = _ROOT / "artifacts" / "campaigns" / "dogbone-v1" / "CP-ENVELOPE.json"
REVEAL_RELATIVE_PATH = Path(".piginot-private/dogbone-v1/reveal-v1.json")
REVEAL_SCHEMA = "piginot.s9.private-reveal-v1"
DEVELOPMENT_ACCESS_NAME = "S14-DEVELOPMENT-ACCESS.json"
HOLDOUT_CONSUMPTION_NAME = "S14-HOLDOUT-CONSUMED.json"
POPULATION_COUNT = 64
_GEOMETRY = ("L_total", "W_grip", "W_gauge", "R_fillet")


@dataclass(frozen=True, slots=True)
class DevelopmentAccess:
    """Why and under which signed protocol DEVELOPMENT was opened."""

    protocol_sha256: str
    signed_off_by: str
    signed_off_utc: str
    opened_utc: str
    case_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.protocol_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.protocol_sha256
        ):
            raise ValueError("protocol_sha256 must be a lowercase hex digest")
        for name in ("signed_off_by", "signed_off_utc", "opened_utc"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if len(self.case_ids) != POPULATION_COUNT or len(set(self.case_ids)) != len(
            self.case_ids
        ):
            raise ValueError("DEVELOPMENT holds exactly 64 distinct cases")

    def to_bytes(self) -> bytes:
        return (json.dumps(asdict(self), indent=1, sort_keys=True) + "\n").encode()


def _write_exclusive(path: Path, payload: bytes) -> None:
    if not path.parent.is_dir():
        raise FileNotFoundError("the record directory must exist")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"already recorded: {path}") from exc
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        directory = os.open(path.parent, flags)
    except OSError:  # pragma: no cover - platforms without directory handles
        return
    try:
        os.fsync(directory)
    except OSError:  # pragma: no cover
        pass
    finally:
        os.close(directory)


def open_development(
    source: object, record_path: Path, access: DevelopmentAccess
) -> Iterator[object]:
    """Record the access durably, then yield the 64 DEVELOPMENT S10 cases.

    ``source`` is the pinned ``SelectiveS10Source``; its DEVELOPMENT case list
    must equal the one named in ``access``.
    """

    expected = tuple(source._protocol.development_case_ids)  # type: ignore[attr-defined]
    if access.case_ids != expected:
        raise ValueError("access record names a different DEVELOPMENT population")
    _write_exclusive(record_path, access.to_bytes())

    def cases() -> Iterator[object]:
        for case_id in expected:
            yield source._load_case(case_id, EvidenceRole.DEVELOPMENT)  # type: ignore[attr-defined]

    return cases()


def holdout_commitment(path: Path = CP_ENVELOPE_PATH) -> HoldoutCommitment:
    """The public exact-64 holdout commitment of ``CP-ENVELOPE``."""

    record = json.loads(path.read_text(encoding="utf-8"))["holdout_commitment"]
    return HoldoutCommitment(**{k: v for k, v in record.items() if k != "$type"})


def parse_reveal(
    payload: bytes, commitment: HoldoutCommitment
) -> tuple[Mapping[str, float], ...]:
    """Coordinates of a reveal document that opens ``commitment``."""

    document = json.loads(payload)
    if document.get("schema") != REVEAL_SCHEMA:
        raise ValueError("not an S9 private reveal document")
    if document.get("campaign_hash") != commitment.campaign_hash:
        raise ValueError("reveal belongs to another campaign")
    salt = bytes.fromhex(document["holdout_salt_hex"])
    coordinates = tuple(document["holdout_coordinates"])
    if not commitment.verify_reveal(salt=salt, holdout_coordinates=coordinates):
        raise ValueError("revealed coordinates do not open the holdout commitment")
    return tuple(sorted(coordinates, key=canonical_json_bytes))


def consume_and_reveal_holdout(
    *,
    record_path: Path,
    record: ConsumptionRecord,
    reveal_path: Path,
    commitment: HoldoutCommitment | None = None,
) -> tuple[Mapping[str, float], ...]:
    """Consume the holdout once, then read and verify the private reveal."""

    consume_holdout(record_path, record)
    expected = holdout_commitment() if commitment is None else commitment
    return reveal_after_consumption(
        record_path,
        candidate_run_hash=record.candidate_run_hash,
        reader=lambda: parse_reveal(reveal_path.read_bytes(), expected),
    )


def holdout_cases(coordinates: Sequence[Mapping[str, float]]) -> tuple[S14Case, ...]:
    """Role-neutral HOLDOUT cases with the S9 strata of their coordinates."""

    envelope = dogbone_envelope()
    primitive = {bound.name for bound in envelope.primitive_bounds}
    derived = {bound.name for bound in envelope.derived_bounds}
    axes = proposed_design_spec(seed=b"public-axes-only").axes
    cases: list[S14Case] = []
    for index, values in enumerate(coordinates):
        if set(values) != primitive | derived:
            raise ValueError("holdout coordinates do not name every envelope axis")
        case_id = f"holdout-{index:02d}"
        point = DesignPoint(
            case_id=case_id,
            leakage_group_id=case_id,
            coordinates={name: float(values[name]) for name in primitive},
            derived_coordinates={name: float(values[name]) for name in derived},
            strata={"unassigned": 0},
        )
        normal = normalized_coordinates(envelope, axes, point)
        cases.append(
            S14Case(
                case_id=case_id,
                role=EvidenceRole.HOLDOUT,
                leakage_group_id=case_id,
                primitive_coordinates=point.coordinates,
                derived_coordinates=point.derived_coordinates,
                strata={
                    axis.name: _stratum(normal[axis.name], axis.strata) for axis in axes
                },
            )
        )
    return tuple(cases)


def s10_request(case: S14Case, profile: ExecutionProfile) -> FemRequest:
    """The unchanged S10 P2 request recipe for one case of any role."""

    family = compiled_family()
    theta = parameter_point(
        *(float(case.primitive_coordinates[name]) for name in _GEOMETRY)
    )
    morph = apply_morph(
        family.reference_mesh, family.morph_map, family.membership, theta
    )
    if not morph.accepted or morph.mesh is None:
        raise ValueError(f"{case.case_id} is outside the ratified family")
    return FemRequest(
        request_id=case.case_id,
        mesh=morph.mesh,
        material=_material(case),  # type: ignore[arg-type]
        boundary_conditions=_boundary_conditions(case),  # type: ignore[arg-type]
        loads=(),
        qois=(),
        config=_solver_config(),
        execution_profile=profile,
    )


__all__ = (
    "CP_ENVELOPE_PATH",
    "DEVELOPMENT_ACCESS_NAME",
    "HOLDOUT_CONSUMPTION_NAME",
    "REVEAL_RELATIVE_PATH",
    "DevelopmentAccess",
    "consume_and_reveal_holdout",
    "holdout_cases",
    "holdout_commitment",
    "open_development",
    "parse_reveal",
    "s10_request",
)
