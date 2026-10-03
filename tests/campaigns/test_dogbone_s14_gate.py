from __future__ import annotations

from dataclasses import replace

from campaigns.dogbone_v1.s14_gate import (
    REPORTED_CHECKS,
    REQUIRED_CHECKS,
    G14Measurements,
    g14_checks,
    g14_report,
)
from piginot.verification import (
    CheckStatus,
    ClaimLevel,
    forbidden_wording,
    render_report,
)

_PASSING = G14Measurements(
    development_budget=CheckStatus.PASS,
    holdout_budget=CheckStatus.PASS,
    reducer=CheckStatus.PASS,
    report_sections=True,
    fixture_count=240,
    refused_with_expected_code=240,
    false_refusal_rate=0.0,
    reason_codes_stable=True,
    development_spearman=0.7,
    latency={"median_seconds": 0.1, "p99_seconds": 0.2, "speedup": 900.0},
    holdout_order=True,
    no_certified=True,
    coverage_percent=95.0,
    holdout_calibration_coverage=0.88,
)


def test_all_checks_pass_and_reach_holdout_verified() -> None:
    report = g14_report(_PASSING)

    assert tuple(c.check_id for c in report.checks) == (
        *REQUIRED_CHECKS,
        *REPORTED_CHECKS,
    )
    assert report.claim_level is ClaimLevel.HOLDOUT_VERIFIED
    assert report.sections["failed"] == () and report.sections["not_run"] == ()
    assert forbidden_wording(render_report(report)) == ()


def test_holdout_budget_failure_caps_the_claim_at_development() -> None:
    report = g14_report(replace(_PASSING, holdout_budget=CheckStatus.FAIL))
    assert report.claim_level is ClaimLevel.DEVELOPMENT_VERIFIED
    assert "G14.holdout-budget" in report.sections["failed"]


def test_missing_or_failing_required_checks_give_not_verified() -> None:
    empty = g14_report(G14Measurements())
    assert empty.claim_level is ClaimLevel.NOT_VERIFIED
    assert set(empty.sections["not_run"]) == {*REQUIRED_CHECKS, *REPORTED_CHECKS}
    assert empty.sections["passed"] == ()

    for change in (
        {"development_budget": CheckStatus.FAIL},
        {"development_budget": CheckStatus.NOT_RUN},
        {"reducer": None},
        {"refused_with_expected_code": 239},
        {"fixture_count": 150, "refused_with_expected_code": 150},
        {"false_refusal_rate": 0.02},
        {"reason_codes_stable": False},
        {"development_spearman": 0.59},
        {"latency": {"median_seconds": 0.1}},
        {"holdout_order": False},
        {"no_certified": None},
        {"coverage_percent": 84.9},
    ):
        report = g14_report(replace(_PASSING, **change))  # type: ignore[arg-type]
        assert report.claim_level is ClaimLevel.NOT_VERIFIED, change


def test_reducer_is_not_run_without_the_family_list() -> None:
    checks = {c.check_id: c for c in g14_checks(replace(_PASSING, reducer=None))}
    assert checks["G14.reducer"].status is CheckStatus.NOT_RUN
    assert "ADR 0022" in (checks["G14.reducer"].reason or "")
    detail = {"injected_cases": 48, "injected_not_verified": 48}
    measured = {
        c.check_id: c for c in g14_checks(replace(_PASSING, reducer_detail=detail))
    }
    assert measured["G14.reducer"].measured == detail
    failed = g14_report(replace(_PASSING, reducer=CheckStatus.FAIL))
    assert failed.claim_level is ClaimLevel.NOT_VERIFIED
    assert "G14.reducer" in failed.sections["failed"]


def test_a_signed_waiver_passes_the_budget_and_keeps_the_computed_result() -> None:
    reason = "accepted by the decision owner (ADR 0021); computed FAIL on W_grip:2"
    waived = replace(
        _PASSING,
        development_budget=CheckStatus.FAIL,
        development_budget_waiver=reason,
    )
    checks = {c.check_id: c for c in g14_checks(waived)}
    assert checks["G14.budget"].status is CheckStatus.PASS
    assert checks["G14.budget"].reason == reason
    assert "FAIL" in str(checks["G14.budget"].measured)
    assert g14_report(waived).claim_level is ClaimLevel.HOLDOUT_VERIFIED
    unwaived = replace(_PASSING, development_budget=CheckStatus.FAIL)
    assert g14_report(unwaived).claim_level is ClaimLevel.NOT_VERIFIED
    not_run = replace(
        _PASSING,
        development_budget=CheckStatus.NOT_RUN,
        development_budget_waiver=reason,
    )
    assert g14_report(not_run).claim_level is ClaimLevel.NOT_VERIFIED
