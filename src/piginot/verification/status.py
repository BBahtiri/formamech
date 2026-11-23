"""Status vocabulary and claim levels shared by the S14 verification layer."""

from __future__ import annotations

from piginot.contracts.state_machine import CheckStatus, ClaimLevel

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
