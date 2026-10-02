"""G14 check assembly for the DogBone campaign (``artifacts/gates/S14.json``).

Required checks decide the G14 verdict; any ``FAIL``, ``ERROR`` or ``NOT_RUN``
leaves the claim at ``NOT_VERIFIED``. The error budget is gated on
DEVELOPMENT. The HOLDOUT budget and the HOLDOUT calibration coverage are
reported checks: the HOLDOUT budget sets the claim level when every required
check passes (``HOLDOUT_VERIFIED`` if it passes, ``DEVELOPMENT_VERIFIED``
otherwise). A measurement that was not taken is ``NOT_RUN``, never ``PASS``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from piginot.verification import (
    CheckRecord,
    CheckStatus,
    ClaimLevel,
    VerificationReport,
    build_report,
)

REQUIRED_CHECKS = (
    "G14.budget",
    "G14.reducer",
    "G14.report-sections",
    "G14.refusal",
    "G14.reason-codes",
    "G14.confidence",
    "G14.latency",
    "G14.holdout-order",
    "G14.no-certified",
    "G14.coverage",
)
REPORTED_CHECKS = ("G14.holdout-budget", "G14.holdout-calibration")
REFUSAL_FIXTURE_MINIMUM = 200
FALSE_REFUSAL_MAXIMUM = 0.01
SPEARMAN_MINIMUM = 0.6
COVERAGE_MINIMUM = 85.0
SPEEDUP_TARGET = 100.0


@dataclass(frozen=True, slots=True)
class G14Measurements:
    """Everything G14 needs; ``None`` means not measured."""

    development_budget: CheckStatus | None = None
    holdout_budget: CheckStatus | None = None
    reducer: CheckStatus | None = None
    report_sections: bool | None = None
    fixture_count: int | None = None
    refused_with_expected_code: int | None = None
    false_refusal_rate: float | None = None
    reason_codes_stable: bool | None = None
    development_spearman: float | None = None
    latency: Mapping[str, float] | None = None
    holdout_order: bool | None = None
    no_certified: bool | None = None
    coverage_percent: float | None = None
    holdout_calibration_coverage: float | None = None
    notes: Mapping[str, str] = field(default_factory=dict)


def _not_run(check_id: str, required: bool, reason: str) -> CheckRecord:
    return CheckRecord(check_id, CheckStatus.NOT_RUN, required, reason=reason)


def _boolean(check_id: str, value: bool | None, threshold: str) -> CheckRecord:
    if value is None:
        return _not_run(check_id, True, "not measured")
    status = CheckStatus.PASS if value else CheckStatus.FAIL
    return CheckRecord(check_id, status, True, measured=value, threshold=threshold)


def _status(check_id: str, value: CheckStatus | None, required: bool) -> CheckRecord:
    if value is None:
        return _not_run(check_id, required, "not measured")
    if value in (CheckStatus.PASS, CheckStatus.FAIL):
        return CheckRecord(
            check_id, value, required, measured=value.value, threshold="every slice"
        )
    return CheckRecord(
        check_id, value, required, measured=value.value, reason="see slice table"
    )


def g14_checks(measurements: G14Measurements) -> tuple[CheckRecord, ...]:
    """One record per required and reported G14 check."""

    m = measurements
    checks: list[CheckRecord] = [_status("G14.budget", m.development_budget, True)]
    if m.reducer is None:
        checks.append(
            _not_run(
                "G14.reducer",
                True,
                "the twelve verification families of IMPLEMENTATION-v1.0-OSS.md "
                "are not available",
            )
        )
    else:
        checks.append(_status("G14.reducer", m.reducer, True))
    checks.append(
        _boolean("G14.report-sections", m.report_sections, "all six sections")
    )
    if None in (m.fixture_count, m.refused_with_expected_code, m.false_refusal_rate):
        checks.append(_not_run("G14.refusal", True, "not measured"))
    else:
        assert m.fixture_count is not None and m.refused_with_expected_code is not None
        assert m.false_refusal_rate is not None
        passed = (
            m.fixture_count >= REFUSAL_FIXTURE_MINIMUM
            and m.refused_with_expected_code == m.fixture_count
            and m.false_refusal_rate <= FALSE_REFUSAL_MAXIMUM
        )
        checks.append(
            CheckRecord(
                "G14.refusal",
                CheckStatus.PASS if passed else CheckStatus.FAIL,
                True,
                measured={
                    "fixtures": m.fixture_count,
                    "refused_with_expected_code": m.refused_with_expected_code,
                    "false_refusal_rate": m.false_refusal_rate,
                },
                threshold={
                    "fixtures": f">= {REFUSAL_FIXTURE_MINIMUM}",
                    "refused": "100%",
                    "false_refusal_rate": f"<= {FALSE_REFUSAL_MAXIMUM}",
                },
            )
        )
    checks.append(
        _boolean("G14.reason-codes", m.reason_codes_stable, "identical across runs")
    )
    if m.development_spearman is None:
        checks.append(_not_run("G14.confidence", True, "not measured"))
    else:
        checks.append(
            CheckRecord(
                "G14.confidence",
                CheckStatus.PASS
                if m.development_spearman >= SPEARMAN_MINIMUM
                else CheckStatus.FAIL,
                True,
                measured=m.development_spearman,
                threshold=f">= {SPEARMAN_MINIMUM}",
            )
        )
    if m.latency is None:
        checks.append(_not_run("G14.latency", True, "not measured"))
    else:
        keys = ("median_seconds", "p99_seconds", "speedup")
        complete = all(key in m.latency for key in keys)
        checks.append(
            CheckRecord(
                "G14.latency",
                CheckStatus.PASS if complete else CheckStatus.FAIL,
                True,
                measured=dict(m.latency),
                threshold=(
                    "median, p99 and speed-up measured on one thread; "
                    f"speed-up target {SPEEDUP_TARGET:g}x reported"
                ),
            )
        )
    checks.append(
        _boolean("G14.holdout-order", m.holdout_order, "record before reveal")
    )
    checks.append(
        _boolean("G14.no-certified", m.no_certified, "no forbidden claim wording")
    )
    if m.coverage_percent is None:
        checks.append(_not_run("G14.coverage", True, "not measured"))
    else:
        checks.append(
            CheckRecord(
                "G14.coverage",
                CheckStatus.PASS
                if m.coverage_percent >= COVERAGE_MINIMUM
                else CheckStatus.FAIL,
                True,
                measured=m.coverage_percent,
                threshold=f">= {COVERAGE_MINIMUM}",
            )
        )
    checks.append(_status("G14.holdout-budget", m.holdout_budget, False))
    if m.holdout_calibration_coverage is None:
        checks.append(
            _not_run("G14.holdout-calibration", False, "HOLDOUT not revealed")
        )
    else:
        checks.append(
            CheckRecord(
                "G14.holdout-calibration",
                CheckStatus.PASS,
                False,
                measured=m.holdout_calibration_coverage,
                threshold="reported (nominal 0.9)",
            )
        )
    identifiers = tuple(check.check_id for check in checks)
    if identifiers != (*REQUIRED_CHECKS, *REPORTED_CHECKS):
        raise AssertionError("G14 check list differs from the registered order")
    return tuple(checks)


def g14_report(measurements: G14Measurements) -> VerificationReport:
    """Sectioned report; the HOLDOUT budget sets the reachable claim level."""

    level = (
        ClaimLevel.HOLDOUT_VERIFIED
        if measurements.holdout_budget is CheckStatus.PASS
        else ClaimLevel.DEVELOPMENT_VERIFIED
    )
    return build_report(g14_checks(measurements), evidence_level=level)


__all__ = (
    "COVERAGE_MINIMUM",
    "FALSE_REFUSAL_MAXIMUM",
    "REFUSAL_FIXTURE_MINIMUM",
    "REPORTED_CHECKS",
    "REQUIRED_CHECKS",
    "SPEARMAN_MINIMUM",
    "SPEEDUP_TARGET",
    "G14Measurements",
    "g14_checks",
    "g14_report",
)
