"""Regression tests that distinguish a real FEM solve from the S7 affine shortcut."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import piginot.fem.backend as fem_backend
from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    CanonicalQuantity,
    ExecutionProfile,
    LoadSpec,
    MaterialSpec,
    QoISpec,
    Unit,
    canonical_json_bytes,
    hash_spec,
)
from piginot.fem import (
    ConstantBodyForce,
    FemBackend,
    FemFailureCode,
    FemRequest,
    FemSolverConfig,
    ScikitFemBackend,
)
from piginot.store import ObjectStore


def _quantity(value: float, unit: Unit) -> CanonicalQuantity:
    return CanonicalQuantity(value, unit)


def _unit_square() -> CanonicalMesh:
    return CanonicalMesh(
        mesh_id="traction-square",
        nodes=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        cells=((0, 1, 2), (0, 2, 3)),
        cell_type="triangle",
        boundary_facets=((0, 1), (1, 2), (2, 3), (3, 0)),
        semantic_tags={
            "bottom": (0,),
            "right": (1,),
            "top": (2,),
            "left": (3,),
        },
        region_ids=(1, 1),
        unit=Unit.MM,
        orientation="ccw",
        topology_signature="traction-square-v1",
        quality_metrics={"min_scaled_jacobian": 1.0},
        source_provenance={"fixture": "s7-real-solver"},
    )


def _refined_unit_square() -> CanonicalMesh:
    return CanonicalMesh(
        mesh_id="refined-traction-square",
        nodes=(
            (0.0, 0.0),
            (0.5, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (0.5, 1.0),
            (1.0, 1.0),
        ),
        cells=((0, 1, 4), (0, 4, 3), (1, 2, 5), (1, 5, 4)),
        cell_type="triangle",
        boundary_facets=((0, 1), (1, 2), (2, 5), (5, 4), (4, 3), (3, 0)),
        semantic_tags={
            "bottom": (0, 1),
            "right": (2,),
            "top": (3, 4),
            "top_right": (3,),
            "left": (5,),
        },
        region_ids=(1, 1, 1, 1),
        unit=Unit.MM,
        orientation="ccw",
        topology_signature="refined-traction-square-v1",
        quality_metrics={"min_scaled_jacobian": 0.5},
        source_provenance={"fixture": "s7-recovery"},
    )


def _profile() -> ExecutionProfile:
    return ExecutionProfile(
        python_version="3.12.0",
        library_versions={
            "numpy": "2.2.6",
            "scikit-fem": "12.0.2",
            "scipy": "1.18.0",
            "torch": "2.13.0",
        },
        blas_threads=1,
        device="cpu",
        dtype="float64",
        rng_algorithm="none",
        seed=0,
    )


def _traction_request(*, element_order: int = 1) -> FemRequest:
    traction = 17.492914739624112
    return FemRequest(
        request_id="traction-driven-uniaxial",
        mesh=_unit_square(),
        material=MaterialSpec(
            "rubber",
            "compressible_neo_hookean",
            {
                "elastic_modulus": _quantity(100.0, Unit.MPA),
                "poisson_ratio": _quantity(0.3, Unit.ONE),
            },
        ),
        boundary_conditions=(
            BoundaryConditionSpec(
                "left_x", "left", "dirichlet", ("x",), (_quantity(0.0, Unit.MM),)
            ),
            BoundaryConditionSpec(
                "bottom_y",
                "bottom",
                "symmetry",
                ("y",),
                (_quantity(0.0, Unit.MM),),
            ),
        ),
        loads=(
            LoadSpec(
                "right_pull",
                "traction",
                "right",
                ("x",),
                (_quantity(traction, Unit.MPA),),
            ),
        ),
        qois=(),
        config=FemSolverConfig(
            element_order=element_order,
            solver_tolerance=1.0e-11,
            max_newton_iterations=20,
            load_steps=4,
        ),
        execution_profile=_profile(),
    )


def test_nonzero_traction_is_solved_by_newton_load_continuation() -> None:
    attempt = ScikitFemBackend().solve_many((_traction_request(),))[0]

    assert attempt.success, attempt.message
    assert attempt.result is not None
    expected = (
        (0.0, 0.0),
        (0.2, 0.0),
        (0.2, -0.05436737570887096),
        (0.0, -0.05436737570887096),
    )
    for actual, target in zip(attempt.result.displacement, expected, strict=True):
        assert math.dist(actual, target) < 1.0e-8

    assert len(attempt.result.convergence_history) >= 4
    assert attempt.result.convergence_history[-1].load_step == 4
    assert attempt.result.convergence_history[-1].residual_norm < 1.0e-8
    assert attempt.result.total_strain_energy > 0.0


def test_p2_returns_all_dofs_quadrature_fields_and_recovered_stress() -> None:
    attempt = ScikitFemBackend().solve_many((_traction_request(element_order=2),))[0]

    assert attempt.success, attempt.message
    assert attempt.result is not None
    result = attempt.result
    assert len(result.displacement) == 9
    assert len(result.displacement_dof_coordinates) == 9
    for (x, y), displacement in zip(
        result.displacement_dof_coordinates,
        result.displacement,
        strict=True,
    ):
        expected = (0.2 * x, -0.05436737570887096 * y)
        assert math.dist(displacement, expected) < 1.0e-8

    assert len(result.recovered_first_piola_stress) == 9
    assert len(result.quadrature_points) == len(result.first_piola_stress) == 2
    assert len(result.quadrature_weights) == 2
    for cell_points, cell_weights, cell_stress, cell_gradient in zip(
        result.quadrature_points,
        result.quadrature_weights,
        result.first_piola_stress,
        result.deformation_gradient,
        strict=True,
    ):
        assert len(cell_points) == len(cell_weights) == len(cell_stress)
        assert len(cell_gradient) == len(cell_stress)
        assert all(weight > 0.0 for weight in cell_weights)
        for stress, gradient in zip(cell_stress, cell_gradient, strict=True):
            assert len(stress) == len(gradient) == 3
            assert abs(stress[0][0] - 17.492914739624112) < 1.0e-8
            assert abs(stress[1][1]) < 1.0e-8
            assert abs(stress[2][2]) < 1.0e-8
            assert abs(gradient[0][0] - 1.2) < 1.0e-8
            assert abs(gradient[1][1] - 0.945632624291129) < 1.0e-8
            assert abs(gradient[2][2] - 0.945632624291129) < 1.0e-8


def test_registered_body_force_enters_equilibrium_and_recipe_identity() -> None:
    provider = ConstantBodyForce((0.0, -0.1))
    request = FemRequest(
        request_id="body-force-square",
        mesh=_unit_square(),
        material=_traction_request().material,
        boundary_conditions=(
            BoundaryConditionSpec(
                "left_clamp",
                "left",
                "dirichlet",
                ("x", "y"),
                (_quantity(0.0, Unit.MM), _quantity(0.0, Unit.MM)),
            ),
        ),
        loads=(LoadSpec("gravity", "body_force"),),
        qois=(),
        config=FemSolverConfig(
            solver_tolerance=1.0e-11,
            max_newton_iterations=20,
            load_steps=2,
        ),
        execution_profile=_profile(),
        body_force_provider_ids={"gravity": provider.provider_id},
    )

    attempt = ScikitFemBackend(body_force_providers=(provider,)).solve_many((request,))[
        0
    ]

    assert attempt.success, attempt.message
    assert attempt.result is not None
    assert min(value[1] for value in attempt.result.displacement) < 0.0
    assert abs(attempt.result.reactions["left_clamp"][1] - 0.1) < 1.0e-8

    unknown = ScikitFemBackend().solve_many((request,))[0]
    assert not unknown.success
    assert unknown.failure_code is FemFailureCode.UNSUPPORTED_REQUEST


def test_body_force_provider_failure_is_a_visible_failed_attempt() -> None:
    class FailingBodyForce:
        provider_id = "failing-body-force-v1"

        def evaluate(self, _reference_points: np.ndarray) -> np.ndarray:
            raise RuntimeError("provider unavailable")

    provider = FailingBodyForce()
    request = replace(
        _traction_request(),
        boundary_conditions=(
            BoundaryConditionSpec(
                "left_clamp",
                "left",
                "dirichlet",
                ("x", "y"),
                (_quantity(0.0, Unit.MM), _quantity(0.0, Unit.MM)),
            ),
        ),
        loads=(LoadSpec("failing_body", "body_force"),),
        body_force_provider_ids={"failing_body": provider.provider_id},
    )

    attempt = ScikitFemBackend(body_force_providers=(provider,)).solve_many((request,))[
        0
    ]

    assert not attempt.success
    assert attempt.failure_code is FemFailureCode.UNSUPPORTED_REQUEST


def test_scikit_fem_backend_implements_public_protocol() -> None:
    assert isinstance(ScikitFemBackend(), FemBackend)


@pytest.mark.parametrize("reduction", ("pointwise_peak", "maximum"))
def test_peak_stress_at_mixed_boundary_junction_is_rejected(reduction: str) -> None:
    request = _traction_request().with_qois(
        (
            QoISpec(
                "top_peak",
                "first_piola_stress",
                reduction,
                "top",
                Unit.MPA,
            ),
        )
    )

    attempt = ScikitFemBackend().solve_many((request,))[0]

    assert not attempt.success
    assert attempt.failure_code is FemFailureCode.QOI_REGULARITY_REJECTED
    assert "mixed-BC junction" in attempt.message


@pytest.mark.parametrize("reduction", ("pointwise_peak", "maximum"))
def test_safe_peak_stress_requires_p2_or_declared_graded_mesh(
    reduction: str,
) -> None:
    baseline = _traction_request()
    request = replace(
        baseline,
        boundary_conditions=(),
        loads=(),
        qois=(
            QoISpec(
                "top_peak",
                "first_piola_stress",
                reduction,
                "top",
                Unit.MPA,
            ),
        ),
    )

    attempt = ScikitFemBackend().solve_many((request,))[0]

    assert not attempt.success
    assert attempt.failure_code is FemFailureCode.QOI_RESOLUTION_REJECTED
    assert "P2 or a declared graded mesh" in attempt.message


def test_successful_result_is_associated_with_recipe_in_s3_store(
    tmp_path: Path,
) -> None:
    store = ObjectStore.create(tmp_path / "fem-store")
    request = _traction_request()

    attempt = ScikitFemBackend(store=store).solve_many((request,))[0]

    assert attempt.success
    assert attempt.result is not None
    assert attempt.result_hash == hash_spec(attempt.result, "content")
    assert attempt.store_content_hash is not None
    assert store.resolve_recipe(request.recipe_hash) == attempt.store_content_hash
    assert store.get(attempt.store_content_hash) == canonical_json_bytes(
        {"domain": "content", "payload": attempt.result}
    )

    retry = ScikitFemBackend(store=store).solve_many((request,))[0]
    assert retry.store_content_hash == attempt.store_content_hash


def test_load_history_reactions_and_dead_load_work_are_physical() -> None:
    request = _traction_request()
    attempt = ScikitFemBackend().solve_many((request,))[0]

    assert attempt.success
    assert attempt.result is not None
    result = attempt.result
    assert {entry.load_step for entry in result.convergence_history} == {1, 2, 3, 4}
    assert any(entry.increment_norm > 0.0 for entry in result.convergence_history)
    for load_step in range(1, 5):
        entries = [
            entry
            for entry in result.convergence_history
            if entry.load_step == load_step
        ]
        assert [entry.iteration for entry in entries] == list(
            range(1, len(entries) + 1)
        )
        assert entries[-1].residual_norm < 1.0e-8

    traction = 17.492914739624112
    assert set(result.reactions) == {"left_x", "bottom_y"}
    assert abs(result.reactions["left_x"][0] + traction) < 1.0e-8
    assert abs(result.reactions["bottom_y"][1]) < 1.0e-8
    assert abs(result.external_work - traction * 0.2) < 1.0e-8

    one_step = replace(
        request,
        config=replace(request.config, load_steps=1),
    )
    one_step_result = ScikitFemBackend().solve_many((one_step,))[0]
    assert one_step_result.success
    assert one_step_result.result is not None
    for left, right in zip(
        one_step_result.result.displacement,
        result.displacement,
        strict=True,
    ):
        assert math.dist(left, right) < 1.0e-8


def test_implicit_symmetry_and_explicit_free_boundary_are_bound() -> None:
    baseline = _traction_request()
    request = replace(
        baseline,
        boundary_conditions=(
            BoundaryConditionSpec(
                "left_x",
                "left",
                "dirichlet",
                ("x",),
                (_quantity(0.0, Unit.MM),),
            ),
            BoundaryConditionSpec("bottom_symmetry", "bottom", "symmetry"),
            BoundaryConditionSpec("top_free", "top", "free"),
        ),
    )

    attempt = ScikitFemBackend().solve_many((request,))[0]

    assert attempt.success, attempt.message
    assert attempt.result is not None
    assert set(attempt.result.reactions) == {"left_x", "bottom_symmetry"}
    assert (
        math.dist(attempt.result.displacement[2], (0.2, -0.05436737570887096)) < 1.0e-8
    )


def test_invalid_traction_component_is_a_visible_failed_attempt() -> None:
    baseline = _traction_request()
    request = replace(
        baseline,
        loads=(
            LoadSpec(
                "bad_traction",
                "traction",
                "right",
                ("z",),
                (_quantity(1.0, Unit.MPA),),
            ),
        ),
    )

    attempt = ScikitFemBackend().solve_many((request,))[0]

    assert not attempt.success
    assert attempt.failure_code is FemFailureCode.UNSUPPORTED_REQUEST


@pytest.mark.parametrize("reduction", ("pointwise_peak", "maximum"))
def test_safe_stress_maximum_uses_recovered_p2_boundary_field(
    monkeypatch: pytest.MonkeyPatch,
    reduction: str,
) -> None:
    request = replace(
        _traction_request(element_order=2),
        mesh=_refined_unit_square(),
        qois=(
            QoISpec(
                "safe_peak",
                "first_piola_stress",
                reduction,
                "top_right",
                Unit.MPA,
            ),
        ),
    )

    def distinctive_recovery(scalar_basis: Any, _stress: object) -> np.ndarray:
        recovered = np.zeros((scalar_basis.N, 3, 3), dtype=np.float64)
        recovered[:, 0, 0] = 123.0
        return recovered

    monkeypatch.setattr(fem_backend, "recover_stress", distinctive_recovery)
    attempt = ScikitFemBackend().solve_many((request,))[0]

    assert attempt.success, attempt.message
    assert attempt.result is not None
    assert attempt.result.qois["safe_peak"] == 123.0


@pytest.mark.parametrize(
    ("field", "reduction", "unit"),
    (
        ("cauchy_stress", "pointwise_peak", Unit.MPA),
        ("first_piola_stress", "average", Unit.MPA),
        ("first_piola_stress", "pointwise_peak", Unit.MM),
    ),
)
def test_unsupported_qoi_is_a_visible_failed_attempt(
    field: str,
    reduction: str,
    unit: Unit,
) -> None:
    request = replace(
        _traction_request(element_order=2),
        mesh=_refined_unit_square(),
        qois=(QoISpec("unsupported", field, reduction, "top_right", unit),),
    )

    attempt = ScikitFemBackend().solve_many((request,))[0]

    assert not attempt.success
    assert attempt.failure_code is FemFailureCode.UNSUPPORTED_REQUEST


def test_empty_qoi_region_is_visible_failure_without_dropping_batch() -> None:
    valid = _traction_request(element_order=2)
    empty_region = replace(
        valid,
        mesh=replace(
            valid.mesh,
            semantic_tags={**dict(valid.mesh.semantic_tags), "empty_region": ()},
        ),
        qois=(
            QoISpec(
                "empty_peak",
                "first_piola_stress",
                "pointwise_peak",
                "empty_region",
                Unit.MPA,
            ),
        ),
    )

    attempts = ScikitFemBackend().solve_many((empty_region, valid))

    assert len(attempts) == 2
    assert not attempts[0].success
    assert attempts[0].failure_code is FemFailureCode.UNSUPPORTED_REQUEST
    assert attempts[1].success
