"""Regression tests that distinguish a real FEM solve from the S7 affine shortcut."""

from __future__ import annotations

import math

from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    CanonicalQuantity,
    ExecutionProfile,
    LoadSpec,
    MaterialSpec,
    Unit,
)
from piginot.fem import FemRequest, FemSolverConfig, ScikitFemBackend


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


def _traction_request() -> FemRequest:
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
            element_order=1,
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
