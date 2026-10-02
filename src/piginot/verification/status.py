"""Status vocabulary and claim levels shared by the S14 verification layer."""

from __future__ import annotations

from enum import StrEnum


class CheckStatus(StrEnum):
    """Outcome of one verification check.

    ``NOT_RUN`` is never a pass, and a ``SKIPPED`` required check is not a pass
    either; only ``PASS`` counts towards a claim.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    NOT_RUN = "NOT_RUN"
    SKIPPED = "SKIPPED"


class ClaimLevel(StrEnum):
    """Claim levels this code can emit, weakest first.

    ``HOLDOUT_VERIFIED`` is the ceiling: no stronger level exists here.
    """

    NOT_VERIFIED = "NOT_VERIFIED"
    DEVELOPMENT_VERIFIED = "DEVELOPMENT_VERIFIED"
    HOLDOUT_VERIFIED = "HOLDOUT_VERIFIED"


_SEVERITY = {
    CheckStatus.ERROR: 4,
    CheckStatus.FAIL: 3,
    CheckStatus.NOT_RUN: 2,
    CheckStatus.SKIPPED: 1,
    CheckStatus.PASS: 0,
}


def worst_status(statuses: tuple[CheckStatus, ...]) -> CheckStatus:
    """Combine statuses: ERROR > FAIL > NOT_RUN > SKIPPED > PASS; empty is NOT_RUN."""

    if not statuses:
        return CheckStatus.NOT_RUN
    if any(type(status) is not CheckStatus for status in statuses):
        raise TypeError("statuses must be CheckStatus values")
    return max(statuses, key=lambda status: _SEVERITY[status])


__all__ = ("CheckStatus", "ClaimLevel", "worst_status")
