"""Verification report with all six sections (``G14.report-sections``).

The report always emits ``required``, ``passed``, ``failed``, ``errored``,
``skipped`` and ``not_run``, even when a section is empty. The claim level is
``NOT_VERIFIED`` unless every required check passed; the strongest level the
code can emit is ``HOLDOUT_VERIFIED``. Generated output is scanned for
certification wording (``G14.no-certified``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from piginot.verification.status import CheckStatus, ClaimLevel

SECTIONS = ("required", "passed", "failed", "errored", "skipped", "not_run")
_FORBIDDEN = re.compile(r"certif", re.IGNORECASE)
_SECTION_OF = {
    CheckStatus.PASS: "passed",
    CheckStatus.FAIL: "failed",
    CheckStatus.ERROR: "errored",
    CheckStatus.SKIPPED: "skipped",
    CheckStatus.NOT_RUN: "not_run",
}


@dataclass(frozen=True, slots=True)
class CheckRecord:
    """One measured check with its decision rule."""

    check_id: str
    status: CheckStatus
    required: bool
    measured: object = None
    threshold: object = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.check_id, str) or not self.check_id:
            raise ValueError("check_id must be non-empty text")
        if type(self.status) is not CheckStatus:
            raise TypeError("status must be a CheckStatus")
        if not isinstance(self.required, bool):
            raise TypeError("required must be boolean")
        if self.status is not CheckStatus.PASS and self.status is not (
            CheckStatus.FAIL
        ):
            if not self.reason:
                raise ValueError(
                    f"{self.check_id}: non-measured statuses need a reason"
                )


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Sectioned report and the claim it supports."""

    checks: tuple[CheckRecord, ...]
    sections: Mapping[str, tuple[str, ...]]
    claim_level: ClaimLevel

    def to_document(self) -> dict[str, Any]:
        return {
            "claim_level": self.claim_level.value,
            "sections": {name: list(self.sections[name]) for name in SECTIONS},
            "checks": [
                {
                    "id": check.check_id,
                    "status": check.status.value,
                    "required": check.required,
                    "measured": check.measured,
                    "threshold": check.threshold,
                    "reason": check.reason,
                }
                for check in self.checks
            ],
        }


def build_report(
    checks: tuple[CheckRecord, ...],
    *,
    evidence_level: ClaimLevel,
) -> VerificationReport:
    """Sort checks into the six sections and derive the claim level."""

    if type(checks) is not tuple or any(type(c) is not CheckRecord for c in checks):
        raise TypeError("checks must be a tuple of CheckRecord")
    if type(evidence_level) is not ClaimLevel:
        raise TypeError("evidence_level must be a ClaimLevel")
    identifiers = [check.check_id for check in checks]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("check ids must be unique")
    sections: dict[str, list[str]] = {name: [] for name in SECTIONS}
    for check in checks:
        if check.required:
            sections["required"].append(check.check_id)
        sections[_SECTION_OF[check.status]].append(check.check_id)
    required = [check for check in checks if check.required]
    verified = bool(required) and all(
        check.status is CheckStatus.PASS for check in required
    )
    claim = evidence_level if verified else ClaimLevel.NOT_VERIFIED
    return VerificationReport(
        checks=checks,
        sections={name: tuple(values) for name, values in sections.items()},
        claim_level=claim,
    )


def forbidden_wording(text: str) -> tuple[str, ...]:
    """Return every occurrence of certification wording in generated text."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return tuple(match.group(0) for match in _FORBIDDEN.finditer(text))


def render_report(report: VerificationReport) -> str:
    """Serialise the report and refuse output that contains certification wording."""

    text = json.dumps(report.to_document(), indent=1, sort_keys=True, default=str)
    if forbidden_wording(text):
        raise ValueError("generated output must not contain certification wording")
    return text


__all__ = (
    "SECTIONS",
    "CheckRecord",
    "VerificationReport",
    "build_report",
    "forbidden_wording",
    "render_report",
)
