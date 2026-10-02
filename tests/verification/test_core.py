from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from piginot.verification import (
    ALL_CASES_SLICE,
    BudgetTolerance,
    CaseBudgetEntry,
    CheckRecord,
    CheckStatus,
    ClaimLevel,
    ConsumptionRecord,
    HoldoutAlreadyConsumedError,
    VerificationFamily,
    build_report,
    consume_holdout,
    evaluate_budget,
    forbidden_wording,
    linear_quantile,
    read_consumption,
    reduce_families,
    render_report,
    reveal_after_consumption,
    slice_statistic,
    worst_status,
)

# ---------------------------------------------------------------- budget


def _entry(
    case_id: str,
    slices: tuple[str, ...],
    surrogate: float | None,
    discretisation: float | None = 0.0,
    projection: float | None = 0.0,
) -> CaseBudgetEntry:
    return CaseBudgetEntry(
        case_id,
        (ALL_CASES_SLICE, *slices),
        {
            "p": {
                "surrogate": surrogate,
                "discretisation": discretisation,
                "projection": projection,
            }
        },
    )


def test_budget_arithmetic_is_exact_on_dyadic_components() -> None:
    entries = (
        _entry("a", ("s1",), 0.0078125, 0.015625, 0.001953125),
        _entry("b", ("s1",), 0.03125, 0.0078125, 0.0),
    )
    results = evaluate_budget(
        entries, (BudgetTolerance("p", 0.05),), (ALL_CASES_SLICE, "s1")
    )

    assert entries[0].total("p") == 0.025390625
    assert entries[1].total("p") == 0.0390625
    assert [result.statistic for result in results] == [0.0390625, 0.0390625]
    assert all(result.status is CheckStatus.PASS for result in results)


def test_slice_statistic_uses_p90_from_ten_cases_and_the_maximum_below() -> None:
    nine = tuple(float(value) for value in range(9))
    ten = tuple(float(value) for value in range(10))

    assert slice_statistic(nine) == 8.0
    assert slice_statistic(ten) == pytest.approx(8.1)
    assert linear_quantile((0.0, 1.0), 0.5) == 0.5
    with pytest.raises(ValueError):
        slice_statistic(())
    with pytest.raises(ValueError):
        linear_quantile((1.0,), 1.5)


def test_budget_fails_closed_on_missing_components_and_empty_slices() -> None:
    entries = (
        _entry("a", ("s1",), 0.01),
        _entry("b", ("s1",), 0.2),
        _entry("c", (), None),
    )
    results = {
        result.slice_id: result
        for result in evaluate_budget(
            entries, (BudgetTolerance("p", 0.05),), (ALL_CASES_SLICE, "s1", "s2")
        )
    }

    assert results[ALL_CASES_SLICE].status is CheckStatus.NOT_RUN
    assert results["s1"].status is CheckStatus.FAIL
    assert results["s1"].statistic == 0.2
    assert results["s2"].status is CheckStatus.NOT_RUN
    assert results["s2"].case_count == 0
    assert _entry("d", (), 0.1).total("u") is None


@pytest.mark.parametrize(
    "build",
    [
        lambda: CaseBudgetEntry("a", ("s1",), {}),
        lambda: CaseBudgetEntry("a", (ALL_CASES_SLICE, ALL_CASES_SLICE), {}),
        lambda: CaseBudgetEntry("", (ALL_CASES_SLICE,), {}),
        lambda: CaseBudgetEntry(
            "a", (ALL_CASES_SLICE,), {"p": {"surrogate": 0.1, "projection": 0.0}}
        ),
        lambda: _entry("a", (), -0.1),
        lambda: _entry("a", (), float("nan")),
        lambda: BudgetTolerance("p", -1.0),
        lambda: BudgetTolerance(" p", 0.1),
    ],
)
def test_budget_inputs_are_validated(build: object) -> None:
    with pytest.raises(ValueError):
        build()  # type: ignore[operator]


def test_budget_rejects_malformed_calls() -> None:
    entry = _entry("a", (), 0.01)
    tolerance = BudgetTolerance("p", 0.05)
    with pytest.raises(TypeError):
        evaluate_budget([entry], (tolerance,), (ALL_CASES_SLICE,))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        evaluate_budget((entry,), (), (ALL_CASES_SLICE,))
    with pytest.raises(ValueError):
        evaluate_budget((entry,), (tolerance, tolerance), (ALL_CASES_SLICE,))
    with pytest.raises(ValueError):
        evaluate_budget((entry,), (tolerance,), ("s1",))
    with pytest.raises(ValueError):
        evaluate_budget((entry, entry), (tolerance,), (ALL_CASES_SLICE,))


# ---------------------------------------------------------------- report


def _check(check_id: str, status: CheckStatus, required: bool = True) -> CheckRecord:
    reason = None if status in (CheckStatus.PASS, CheckStatus.FAIL) else "because"
    return CheckRecord(check_id, status, required, reason=reason)


def test_a_fully_passing_report_still_emits_skipped_and_not_run() -> None:
    report = build_report(
        (_check("a", CheckStatus.PASS), _check("b", CheckStatus.PASS)),
        evidence_level=ClaimLevel.HOLDOUT_VERIFIED,
    )
    document = json.loads(render_report(report))

    assert list(document["sections"]) == sorted(
        ["required", "passed", "failed", "errored", "skipped", "not_run"]
    )
    assert document["sections"]["skipped"] == []
    assert document["sections"]["not_run"] == []
    assert document["claim_level"] == "HOLDOUT_VERIFIED"


@pytest.mark.parametrize(
    "status",
    [CheckStatus.FAIL, CheckStatus.ERROR, CheckStatus.NOT_RUN, CheckStatus.SKIPPED],
)
def test_any_required_non_pass_is_not_verified(status: CheckStatus) -> None:
    report = build_report(
        (_check("a", CheckStatus.PASS), _check("b", status)),
        evidence_level=ClaimLevel.DEVELOPMENT_VERIFIED,
    )

    assert report.claim_level is ClaimLevel.NOT_VERIFIED
    assert "b" in report.sections["required"]


def test_optional_checks_do_not_block_and_empty_required_does_not_verify() -> None:
    optional = build_report(
        (_check("a", CheckStatus.PASS), _check("b", CheckStatus.SKIPPED, False)),
        evidence_level=ClaimLevel.DEVELOPMENT_VERIFIED,
    )
    nothing = build_report(
        (_check("b", CheckStatus.SKIPPED, False),),
        evidence_level=ClaimLevel.DEVELOPMENT_VERIFIED,
    )

    assert optional.claim_level is ClaimLevel.DEVELOPMENT_VERIFIED
    assert optional.sections["skipped"] == ("b",)
    assert nothing.claim_level is ClaimLevel.NOT_VERIFIED


def test_certification_wording_is_detected_and_refused() -> None:
    assert forbidden_wording("This bundle is Certified.") == ("Certif",)
    assert forbidden_wording("verified within tolerance") == ()
    report = build_report(
        (CheckRecord("a", CheckStatus.NOT_RUN, True, reason="certified elsewhere"),),
        evidence_level=ClaimLevel.DEVELOPMENT_VERIFIED,
    )
    with pytest.raises(ValueError, match="certification"):
        render_report(report)
    with pytest.raises(TypeError):
        forbidden_wording(None)  # type: ignore[arg-type]


def test_report_inputs_are_validated() -> None:
    with pytest.raises(ValueError):
        CheckRecord("a", CheckStatus.NOT_RUN, True)
    with pytest.raises(ValueError):
        CheckRecord("", CheckStatus.PASS, True)
    with pytest.raises(TypeError):
        CheckRecord("a", "PASS", True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        CheckRecord("a", CheckStatus.PASS, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_report(
            (_check("a", CheckStatus.PASS), _check("a", CheckStatus.PASS)),
            evidence_level=ClaimLevel.HOLDOUT_VERIFIED,
        )
    with pytest.raises(TypeError):
        build_report([], evidence_level=ClaimLevel.HOLDOUT_VERIFIED)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_report((), evidence_level="HOLDOUT_VERIFIED")  # type: ignore[arg-type]


def test_no_claim_level_above_holdout_verified_exists() -> None:
    assert [level.value for level in ClaimLevel] == [
        "NOT_VERIFIED",
        "DEVELOPMENT_VERIFIED",
        "HOLDOUT_VERIFIED",
    ]


# ---------------------------------------------------------------- reducer

_FAMILIES = (
    VerificationFamily("f1", True, ("a", "b")),
    VerificationFamily("f2", True, ("c",)),
    VerificationFamily("f3", False, ("d",)),
)


def test_reducer_verifies_only_when_every_required_family_passes() -> None:
    passing = {name: CheckStatus.PASS for name in "abc"}
    reduction = reduce_families(
        _FAMILIES,
        {**passing, "d": CheckStatus.FAIL},
        evidence_level=ClaimLevel.DEVELOPMENT_VERIFIED,
    )

    assert reduction.verified
    assert reduction.families[2].status is CheckStatus.FAIL


@pytest.mark.parametrize(
    "status", [CheckStatus.FAIL, CheckStatus.ERROR, CheckStatus.NOT_RUN]
)
@pytest.mark.parametrize("check", ["a", "b", "c"])
def test_every_injected_required_non_pass_yields_not_verified(
    status: CheckStatus, check: str
) -> None:
    statuses = {name: CheckStatus.PASS for name in "abcd"}
    statuses[check] = status
    reduction = reduce_families(
        _FAMILIES, statuses, evidence_level=ClaimLevel.HOLDOUT_VERIFIED
    )

    assert reduction.claim_level is ClaimLevel.NOT_VERIFIED


def test_missing_results_are_not_run_and_status_order_is_fail_closed() -> None:
    reduction = reduce_families(
        _FAMILIES,
        {"a": CheckStatus.PASS, "c": CheckStatus.PASS},
        evidence_level=ClaimLevel.HOLDOUT_VERIFIED,
    )

    assert reduction.families[0].status is CheckStatus.NOT_RUN
    assert reduction.families[0].missing_checks == ("b",)
    assert not reduction.verified
    assert worst_status((CheckStatus.PASS, CheckStatus.ERROR)) is CheckStatus.ERROR
    assert worst_status(()) is CheckStatus.NOT_RUN
    with pytest.raises(TypeError):
        worst_status(("PASS",))  # type: ignore[arg-type]


def test_reducer_inputs_are_validated() -> None:
    with pytest.raises(ValueError):
        VerificationFamily("", True, ("a",))
    with pytest.raises(ValueError):
        VerificationFamily("f", True, ())
    with pytest.raises(ValueError):
        VerificationFamily("f", True, ("a", "a"))
    with pytest.raises(TypeError):
        VerificationFamily("f", 1, ("a",))  # type: ignore[arg-type]
    level = ClaimLevel.HOLDOUT_VERIFIED
    with pytest.raises(ValueError):
        reduce_families((), {}, evidence_level=level)
    with pytest.raises(TypeError):
        reduce_families((object(),), {}, evidence_level=level)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        reduce_families((_FAMILIES[0], _FAMILIES[0]), {}, evidence_level=level)
    with pytest.raises(ValueError):
        reduce_families(
            (_FAMILIES[0], VerificationFamily("g", True, ("a",))),
            {},
            evidence_level=level,
        )
    with pytest.raises(ValueError):
        reduce_families(_FAMILIES, {"z": CheckStatus.PASS}, evidence_level=level)
    with pytest.raises(TypeError):
        reduce_families(_FAMILIES, {"a": "PASS"}, evidence_level=level)  # type: ignore[dict-item]
    with pytest.raises(TypeError):
        reduce_families(_FAMILIES, {}, evidence_level="x")  # type: ignore[arg-type]


# ---------------------------------------------------------------- holdout

_RECORD = ConsumptionRecord(
    campaign_id="campaign",
    candidate_run_hash="1" * 64,
    freeze_record_sha256="2" * 64,
    consumed_utc="2026-10-02T12:00:00Z",
    purpose="S14 holdout evaluation",
)


def test_consumption_precedes_reveal_and_survives_a_crash(tmp_path: Path) -> None:
    record_path = tmp_path / "holdout-consumption.json"
    with pytest.raises(PermissionError, match="consumed before"):
        reveal_after_consumption(
            record_path, candidate_run_hash="1" * 64, reader=lambda: "secret"
        )
    consume_holdout(record_path, _RECORD)

    def crashing_reader() -> str:
        raise RuntimeError("crash mid-reveal")

    with pytest.raises(RuntimeError, match="crash"):
        reveal_after_consumption(
            record_path, candidate_run_hash="1" * 64, reader=crashing_reader
        )
    assert read_consumption(record_path) == _RECORD
    with pytest.raises(HoldoutAlreadyConsumedError):
        consume_holdout(record_path, _RECORD)
    assert (
        reveal_after_consumption(
            record_path, candidate_run_hash="1" * 64, reader=lambda: "secret"
        )
        == "secret"
    )
    with pytest.raises(PermissionError, match="different candidate"):
        reveal_after_consumption(
            record_path, candidate_run_hash="3" * 64, reader=lambda: "secret"
        )
    assert oct(os.stat(record_path).st_mode & 0o777) == "0o600"


def test_consumption_inputs_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ConsumptionRecord("c", "x", "2" * 64, "t", "p")
    with pytest.raises(ValueError):
        ConsumptionRecord("", "1" * 64, "2" * 64, "t", "p")
    with pytest.raises(TypeError):
        consume_holdout(str(tmp_path / "r.json"), _RECORD)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        consume_holdout(tmp_path / "r.json", object())  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        consume_holdout(tmp_path / "missing" / "r.json", _RECORD)
    (tmp_path / "bad.json").write_text("[]", "utf-8")
    with pytest.raises(ValueError):
        read_consumption(tmp_path / "bad.json")
