"""P1/P2 and structured/unstructured discretization cross-checks."""

from __future__ import annotations

from piginot.fem.discretization_crosscheck import run_discretization_crosscheck
from piginot.fem.manufactured import MANUFACTURED_CASE_NAMES


def test_crosscheck_records_both_orders_and_mesh_families_without_overclaiming() -> (
    None
):
    report = run_discretization_crosscheck()

    assert report.claim_scope == "discretization_error_not_implementation_error"
    assert {case.case_name for case in report.cases} == set(MANUFACTURED_CASE_NAMES)
    for case in report.cases:
        assert set(case.displacement_l2_errors) == {
            "structured-p1",
            "structured-p2",
            "unstructured-p1",
            "unstructured-p2",
        }
        assert (
            case.displacement_l2_errors["structured-p2"]
            < case.displacement_l2_errors["structured-p1"]
        )
        assert (
            case.displacement_l2_errors["unstructured-p2"]
            < case.displacement_l2_errors["unstructured-p1"]
        )
        assert case.structured_unstructured_relative_delta >= 0.0
