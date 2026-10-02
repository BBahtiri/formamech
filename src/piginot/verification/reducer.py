"""Gate reducer over declared verification families (``G14.reducer``).

Each family names the checks it owns and whether it is required. A family's
status is the worst status of its checks (``ERROR`` > ``FAIL`` > ``NOT_RUN`` >
``SKIPPED`` > ``PASS``); a declared check without a result is ``NOT_RUN``. Any
required family that is not ``PASS`` makes the reduction ``NOT_VERIFIED``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from piginot.verification.status import CheckStatus, ClaimLevel, worst_status


@dataclass(frozen=True, slots=True)
class VerificationFamily:
    """A named group of checks reduced together."""

    family_id: str
    required: bool
    check_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.family_id, str) or not self.family_id:
            raise ValueError("family_id must be non-empty text")
        if not isinstance(self.required, bool):
            raise TypeError("required must be boolean")
        if type(self.check_ids) is not tuple or not self.check_ids:
            raise ValueError("a family must own at least one check")
        if len(set(self.check_ids)) != len(self.check_ids):
            raise ValueError("a family must not repeat a check")


@dataclass(frozen=True, slots=True)
class FamilyOutcome:
    family_id: str
    required: bool
    status: CheckStatus
    missing_checks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Reduction:
    families: tuple[FamilyOutcome, ...]
    claim_level: ClaimLevel

    @property
    def verified(self) -> bool:
        return self.claim_level is not ClaimLevel.NOT_VERIFIED


def reduce_families(
    families: tuple[VerificationFamily, ...],
    statuses: Mapping[str, CheckStatus],
    *,
    evidence_level: ClaimLevel,
) -> Reduction:
    """Reduce check statuses family by family, fail-closed."""

    if type(families) is not tuple or not families:
        raise ValueError("at least one family is required")
    if any(type(family) is not VerificationFamily for family in families):
        raise TypeError("families must be VerificationFamily values")
    if len({family.family_id for family in families}) != len(families):
        raise ValueError("family ids must be unique")
    owned = [check for family in families for check in family.check_ids]
    if len(set(owned)) != len(owned):
        raise ValueError("a check must belong to exactly one family")
    unknown = set(statuses) - set(owned)
    if unknown:
        raise ValueError(f"results for undeclared checks: {sorted(unknown)}")
    if any(type(status) is not CheckStatus for status in statuses.values()):
        raise TypeError("statuses must be CheckStatus values")
    if type(evidence_level) is not ClaimLevel:
        raise TypeError("evidence_level must be a ClaimLevel")
    outcomes: list[FamilyOutcome] = []
    for family in families:
        missing = tuple(check for check in family.check_ids if check not in statuses)
        family_statuses = tuple(
            statuses.get(check, CheckStatus.NOT_RUN) for check in family.check_ids
        )
        outcomes.append(
            FamilyOutcome(
                family.family_id,
                family.required,
                worst_status(family_statuses),
                missing,
            )
        )
    verified = all(
        outcome.status is CheckStatus.PASS for outcome in outcomes if outcome.required
    )
    return Reduction(
        families=tuple(outcomes),
        claim_level=evidence_level if verified else ClaimLevel.NOT_VERIFIED,
    )


__all__ = ("FamilyOutcome", "Reduction", "VerificationFamily", "reduce_families")
