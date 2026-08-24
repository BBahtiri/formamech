"""Integration tests for solved nonlinear manufactured FEM cases."""

from __future__ import annotations

import math

import pytest

from piginot.fem.manufactured import MANUFACTURED_CASE_NAMES, BoundaryKind
from piginot.fem.mms_runner import build_mms_problem, solve_mms_level
from piginot.fem.verification_meshes import VerificationMeshKind


@pytest.mark.parametrize("case_name", MANUFACTURED_CASE_NAMES)
def test_problem_builder_preserves_physical_boundary_family(case_name: str) -> None:
    problem = build_mms_problem(case_name, level=1, element_order=1)
    request = problem.request
    exact = problem.manufactured_solution
    dirichlet_tags = {
        condition.boundary_tag for condition in request.boundary_conditions
    }
    traction_tags = {
        load.boundary_tag for load in request.loads if load.kind == "traction"
    }

    assert dirichlet_tags == {
        tag
        for tag, kind in exact.boundary_types.items()
        if kind is BoundaryKind.DIRICHLET
    } or (
        dirichlet_tags == {"all"}
        and set(exact.boundary_types.values()) == {BoundaryKind.DIRICHLET}
    )
    assert traction_tags == {
        tag
        for tag, kind in exact.boundary_types.items()
        if kind is BoundaryKind.TRACTION
    }
    assert any(load.kind == "body_force" for load in request.loads)
    assert bool(request.rigid_mode_constraint) is bool(exact.rigid_mode_constraints)
    if case_name == "pure_traction_body_force":
        assert not request.boundary_conditions
        assert set(traction_tags) == {"left", "right", "top", "bottom"}
        assert request.rigid_mode_constraint == "zero_mean_translation_rotation"


@pytest.mark.parametrize("case_name", MANUFACTURED_CASE_NAMES)
def test_primary_backend_solves_each_mms_case_and_returns_true_norms(
    case_name: str,
) -> None:
    solved = solve_mms_level(case_name, level=1, element_order=1)

    assert solved.result_hash
    assert solved.dof_count > 0
    assert solved.h_max_mm > 0.0
    assert solved.norms.quadrature_point_count > 0
    assert math.isfinite(solved.norms.displacement_l2_error)
    assert math.isfinite(solved.norms.displacement_h1_seminorm_error)
    assert math.isfinite(solved.norms.first_piola_l2_error)
    assert solved.norms.displacement_l2_error > 0.0
    assert solved.norms.first_piola_l2_error > 0.0


def test_structured_and_unstructured_runs_are_distinct_discretizations() -> None:
    structured = solve_mms_level(
        "mixed_dirichlet_traction",
        level=1,
        element_order=1,
        mesh_kind=VerificationMeshKind.STRUCTURED,
    )
    unstructured = solve_mms_level(
        "mixed_dirichlet_traction",
        level=1,
        element_order=1,
        mesh_kind=VerificationMeshKind.UNSTRUCTURED,
    )

    assert structured.mesh_id != unstructured.mesh_id
    assert structured.result_hash != unstructured.result_hash
    assert structured.norms.displacement_l2_error != pytest.approx(
        unstructured.norms.displacement_l2_error,
        rel=1.0e-8,
        abs=1.0e-15,
    )
