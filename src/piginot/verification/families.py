"""The twelve verification families of the G14 reducer (ADR 0022).

Each family is the gate artifact of one upstream step (S2–S13). A family's
checks are the checks recorded in its artifact; its status is the worst of
them. A missing artifact is ``NOT_RUN``; an unreadable artifact, a wrong gate
ID, or a ``PASS`` verdict over a non-passing check is ``ERROR``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from piginot.verification.reducer import (
    Reduction,
    VerificationFamily,
    reduce_families,
)
from piginot.verification.status import CheckStatus, ClaimLevel

Reducer = Callable[..., Reduction]
_STATUSES = {status.value: status for status in CheckStatus}
_INJECTED = (CheckStatus.FAIL, CheckStatus.ERROR, CheckStatus.NOT_RUN)


@dataclass(frozen=True, slots=True)
class GateFamily:
    """One verification family and the gate artifact that evidences it."""

    family_id: str
    name: str
    gate_id: str
    artifact: str

    def __post_init__(self) -> None:
        for field in ("family_id", "name", "gate_id", "artifact"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be non-empty text")


REGISTERED_FAMILIES = (
    GateFamily("F01", "contracts-identity", "G2", "artifacts/gates/S2.json"),
    GateFamily("F02", "content-store", "G3", "artifacts/gates/S3.json"),
    GateFamily("F03", "physics-kernel", "G4", "artifacts/gates/S4.json"),
    GateFamily("F04", "geometry-tags", "G5", "artifacts/gates/S5.json"),
    GateFamily("F05", "morphing-membership", "G6", "artifacts/gates/S6.json"),
    GateFamily("F06", "primary-fem", "G7", "artifacts/gates/S7.json"),
    GateFamily("F07", "fem-verification", "G8", "artifacts/gates/S8.json"),
    GateFamily("F08", "doe-preregistration", "G9", "artifacts/gates/S9.json"),
    GateFamily("F09", "dataset-materialisation", "G10", "artifacts/gates/S10.json"),
    GateFamily(
        "F10", "operator-architecture", "G11", "artifacts/gates/S11-ENERGY.json"
    ),
    GateFamily("F11", "objectives-gradients", "G12", "artifacts/gates/S12.json"),
    GateFamily("F12", "training-selection", "G13", "artifacts/gates/S13.json"),
)


@dataclass(frozen=True, slots=True)
class FamilyEvidence:
    """The checks of one family as read from its artifact."""

    family: GateFamily
    check_ids: tuple[str, ...]
    statuses: Mapping[str, CheckStatus]
    problem: str | None = None


def _qualified(family: GateFamily, check_id: str) -> str:
    return f"{family.family_id}:{check_id}"


def read_family(family: GateFamily, root: Path) -> FamilyEvidence:
    """Read one family's checks; problems become a single synthetic check."""

    sentinel = _qualified(family, "artifact")
    path = root / family.artifact
    if not path.is_file():
        return FamilyEvidence(
            family, (sentinel,), {sentinel: CheckStatus.NOT_RUN}, "artifact missing"
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        checks = document["checks"]
        if not isinstance(checks, list) or not checks:
            raise ValueError("no checks recorded")
        statuses: dict[str, CheckStatus] = {}
        for check in checks:
            check_id = check.get("id") or check.get("check_id")
            if not isinstance(check_id, str) or not check_id:
                raise ValueError("a check has no identifier")
            qualified = _qualified(family, check_id)
            if qualified in statuses:
                raise ValueError(f"check {check_id} is recorded twice")
            statuses[qualified] = _STATUSES[check["status"]]
        if document.get("gate_id") != family.gate_id:
            raise ValueError("artifact belongs to another gate")
        all_pass = all(status is CheckStatus.PASS for status in statuses.values())
        if document.get("verdict") == "PASS" and not all_pass:
            raise ValueError("PASS verdict over a non-passing check")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return FamilyEvidence(
            family, (sentinel,), {sentinel: CheckStatus.ERROR}, str(exc)
        )
    return FamilyEvidence(family, tuple(statuses), statuses)


def reduce_registered(
    evidence: tuple[FamilyEvidence, ...],
    *,
    evidence_level: ClaimLevel,
    overrides: Mapping[str, CheckStatus] | None = None,
    removed: frozenset[str] = frozenset(),
    reducer: Reducer = reduce_families,
) -> Reduction:
    """Reduce the families, optionally with injected statuses or removed checks.

    ``reducer`` is the reduction under test; S16 substitutes mutants here.
    """

    families = tuple(
        VerificationFamily(item.family.family_id, True, item.check_ids)
        for item in evidence
    )
    statuses: dict[str, CheckStatus] = {}
    for item in evidence:
        statuses.update(item.statuses)
    statuses.update(overrides or {})
    for check in removed:
        statuses.pop(check, None)
    return reducer(families, statuses, evidence_level=evidence_level)


@dataclass(frozen=True, slots=True)
class InjectionResult:
    """Outcome of the G14.reducer injected-case test."""

    cases: int
    not_verified: int
    control_level: ClaimLevel

    @property
    def passed(self) -> bool:
        return (
            self.cases > 0
            and self.not_verified == self.cases
            and self.control_level is not ClaimLevel.NOT_VERIFIED
        )


def inject_failures(
    evidence: tuple[FamilyEvidence, ...],
    *,
    evidence_level: ClaimLevel,
    reducer: Reducer = reduce_families,
) -> InjectionResult:
    """Every family to FAIL, ERROR and NOT_RUN, and each with a check removed.

    The control is the all-``PASS`` version of the same families.
    """

    passing = tuple(
        FamilyEvidence(
            item.family,
            item.check_ids,
            {check: CheckStatus.PASS for check in item.check_ids},
        )
        for item in evidence
    )
    control = reduce_registered(passing, evidence_level=evidence_level, reducer=reducer)
    cases = 0
    not_verified = 0
    for item in passing:
        first = item.check_ids[0]
        for status in _INJECTED:
            cases += 1
            reduction = reduce_registered(
                passing,
                evidence_level=evidence_level,
                overrides={first: status},
                reducer=reducer,
            )
            not_verified += reduction.claim_level is ClaimLevel.NOT_VERIFIED
        cases += 1
        reduction = reduce_registered(
            passing,
            evidence_level=evidence_level,
            removed=frozenset({first}),
            reducer=reducer,
        )
        not_verified += reduction.claim_level is ClaimLevel.NOT_VERIFIED
    return InjectionResult(cases, not_verified, control.claim_level)


def read_registered(
    root: Path, families: tuple[GateFamily, ...] = REGISTERED_FAMILIES
) -> tuple[FamilyEvidence, ...]:
    """Read every registered family from the repository at ``root``."""

    return tuple(read_family(family, root) for family in families)


__all__ = (
    "REGISTERED_FAMILIES",
    "FamilyEvidence",
    "GateFamily",
    "InjectionResult",
    "inject_failures",
    "read_family",
    "read_registered",
    "reduce_registered",
)
