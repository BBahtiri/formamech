from __future__ import annotations

import pytest

from piginot.doe.gate import (
    DesignCoverageDiagnostics,
    GateStatus,
    ReplacementEvidence,
    S9GateMeasurements,
    evaluate_s9_gate,
)


def _measurements(**overrides: object) -> S9GateMeasurements:
    values: dict[str, object] = {
        "immutable_rewrite_raised": True,
        "one_role_trials": 1000,
        "one_role_violations": 0,
        "leakage_trials": 1000,
        "leakage_violations": 0,
        "holdout_visibility_violations": 0,
        "node_split_mutant_fired": True,
        "commitment_file_order_verified": True,
        "commitment_git_order_verified": True,
        "reveal_verified": True,
        "replacement": ReplacementEvidence(
            failures_injected=3,
            failures_recorded=3,
            deterministic_rule_matches=True,
        ),
        "code_coverage_percent": 92.25,
        "design_diagnostics": DesignCoverageDiagnostics(
            empirical_fill_distance=0.18,
            minimum_stratum_count=2,
            boundary_coverage=1.0,
            stratum_coverage=1.0,
            interaction_coverage=0.875,
            difficult_region_coverage=1.0,
            uncovered_region_ids=(),
            unresolved_region_ids=("interaction:width:ratio:3:3",),
        ),
    }
    values.update(overrides)
    return S9GateMeasurements(**values)  # type: ignore[arg-type]


def test_gate_uses_source_coverage_not_an_invented_doe_score() -> None:
    evaluation = evaluate_s9_gate(_measurements())

    assert evaluation.verdict is GateStatus.PASS
    coverage = evaluation.by_id("G9.coverage")
    assert coverage.measured == 92.25
    assert coverage.threshold == 90.0
    assert evaluation.design_diagnostics.interaction_coverage == 0.875


@pytest.mark.parametrize(
    ("overrides", "check_id", "status"),
    (
        ({"one_role_trials": 999}, "G9.one-role", GateStatus.NOT_RUN),
        ({"one_role_violations": 1}, "G9.one-role", GateStatus.FAIL),
        ({"leakage_trials": 999}, "G9.leakage", GateStatus.NOT_RUN),
        ({"holdout_visibility_violations": 1}, "G9.visibility", GateStatus.FAIL),
        ({"node_split_mutant_fired": False}, "G9.no-node-split", GateStatus.FAIL),
        (
            {"commitment_git_order_verified": False},
            "G9.commitment-order",
            GateStatus.FAIL,
        ),
        ({"reveal_verified": False}, "G9.commitment-order", GateStatus.FAIL),
        (
            {
                "replacement": ReplacementEvidence(
                    failures_injected=3,
                    failures_recorded=2,
                    deterministic_rule_matches=True,
                )
            },
            "G9.replacement",
            GateStatus.FAIL,
        ),
        ({"code_coverage_percent": 89.999}, "G9.coverage", GateStatus.FAIL),
        ({"code_coverage_percent": None}, "G9.coverage", GateStatus.NOT_RUN),
    ),
)
def test_gate_fails_closed_for_missing_or_bad_evidence(
    overrides: dict[str, object], check_id: str, status: GateStatus
) -> None:
    evaluation = evaluate_s9_gate(_measurements(**overrides))

    assert evaluation.verdict is GateStatus.FAIL
    assert evaluation.by_id(check_id).status is status


def test_replacement_check_is_not_vacuous_without_injected_failures() -> None:
    evidence = ReplacementEvidence(
        failures_injected=0,
        failures_recorded=0,
        deterministic_rule_matches=True,
    )

    evaluation = evaluate_s9_gate(_measurements(replacement=evidence))

    assert evaluation.by_id("G9.replacement").status is GateStatus.NOT_RUN


def test_gate_reports_all_missing_optional_evidence_as_not_run() -> None:
    evaluation = evaluate_s9_gate(
        _measurements(
            immutable_rewrite_raised=None,
            node_split_mutant_fired=None,
            commitment_file_order_verified=None,
            replacement=None,
        )
    )

    assert evaluation.verdict is GateStatus.FAIL
    assert evaluation.by_id("G9.immutable").status is GateStatus.NOT_RUN
    assert evaluation.by_id("G9.no-node-split").status is GateStatus.NOT_RUN
    assert evaluation.by_id("G9.commitment-order").status is GateStatus.NOT_RUN
    assert evaluation.by_id("G9.replacement").status is GateStatus.NOT_RUN
    assert "G9.immutable" in evaluation.verdict_reason


@pytest.mark.parametrize(
    ("overrides", "check_id"),
    (
        ({"immutable_rewrite_raised": False}, "G9.immutable"),
        ({"leakage_violations": 1}, "G9.leakage"),
        (
            {
                "replacement": ReplacementEvidence(
                    failures_injected=3,
                    failures_recorded=3,
                    deterministic_rule_matches=False,
                )
            },
            "G9.replacement",
        ),
    ),
)
def test_gate_rejects_additional_complete_but_failed_evidence(
    overrides: dict[str, object], check_id: str
) -> None:
    evaluation = evaluate_s9_gate(_measurements(**overrides))

    assert evaluation.by_id(check_id).status is GateStatus.FAIL


def test_code_coverage_threshold_is_inclusive() -> None:
    evaluation = evaluate_s9_gate(_measurements(code_coverage_percent=90))

    assert evaluation.verdict is GateStatus.PASS
    assert evaluation.by_id("G9.coverage").status is GateStatus.PASS


def test_gate_lookup_and_reducer_fail_closed_for_wrong_inputs() -> None:
    evaluation = evaluate_s9_gate(_measurements())

    with pytest.raises(KeyError, match="G9.unknown"):
        evaluation.by_id("G9.unknown")
    with pytest.raises(TypeError, match="S9GateMeasurements"):
        evaluate_s9_gate(object())  # type: ignore[arg-type]


def _diagnostics(**overrides: object) -> DesignCoverageDiagnostics:
    values: dict[str, object] = {
        "empirical_fill_distance": 0.18,
        "minimum_stratum_count": 2,
        "boundary_coverage": 1.0,
        "stratum_coverage": 1.0,
        "interaction_coverage": 0.875,
        "difficult_region_coverage": 1.0,
        "uncovered_region_ids": (),
        "unresolved_region_ids": ("z-region", "a-region"),
    }
    values.update(overrides)
    return DesignCoverageDiagnostics(**values)  # type: ignore[arg-type]


def test_design_diagnostics_canonicalize_identifier_order() -> None:
    diagnostics = _diagnostics()

    assert diagnostics.unresolved_region_ids == ("a-region", "z-region")


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"empirical_fill_distance": True}, "finite number"),
        ({"empirical_fill_distance": float("nan")}, "finite number"),
        ({"empirical_fill_distance": -0.01}, "non-negative"),
        ({"minimum_stratum_count": True}, "non-negative integer"),
        ({"minimum_stratum_count": 1.5}, "non-negative integer"),
        ({"minimum_stratum_count": -1}, "non-negative integer"),
        ({"boundary_coverage": True}, "finite number"),
        ({"stratum_coverage": float("inf")}, "finite number"),
        ({"interaction_coverage": -0.01}, r"\[0, 1\]"),
        ({"difficult_region_coverage": 1.01}, r"\[0, 1\]"),
        ({"uncovered_region_ids": []}, "must be a tuple"),
        ({"uncovered_region_ids": ("",)}, "non-empty text"),
        ({"uncovered_region_ids": (1,)}, "non-empty text"),
        ({"uncovered_region_ids": ("same", "same")}, "duplicates"),
    ),
)
def test_design_diagnostics_reject_malformed_measurements(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _diagnostics(**overrides)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"failures_injected": True}, "non-negative integer"),
        ({"failures_recorded": -1}, "non-negative integer"),
        ({"deterministic_rule_matches": 1}, "must be boolean"),
    ),
)
def test_replacement_evidence_rejects_invalid_values(
    kwargs: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "failures_injected": 1,
        "failures_recorded": 1,
        "deterministic_rule_matches": True,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        ReplacementEvidence(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"immutable_rewrite_raised": "yes"}, "boolean or None"),
        ({"one_role_trials": True}, "non-negative integer"),
        ({"leakage_trials": 1.5}, "non-negative integer"),
        ({"holdout_visibility_violations": -1}, "non-negative integer"),
        (
            {"one_role_trials": 1, "one_role_violations": 2},
            "cannot exceed",
        ),
        (
            {"leakage_trials": 1, "leakage_violations": 2},
            "cannot exceed",
        ),
        ({"replacement": object()}, "ReplacementEvidence or None"),
        ({"code_coverage_percent": True}, "finite number"),
        ({"code_coverage_percent": float("inf")}, "finite number"),
        ({"code_coverage_percent": -0.01}, r"\[0, 100\]"),
        ({"code_coverage_percent": 100.01}, r"\[0, 100\]"),
        ({"design_diagnostics": object()}, "DesignCoverageDiagnostics"),
    ),
)
def test_gate_measurements_reject_invalid_contracts(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _measurements(**overrides)
