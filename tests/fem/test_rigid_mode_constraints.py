"""Global rigid-mode constraints for physically pure-traction FEM cases."""

from __future__ import annotations

from dataclasses import replace

import pytest

from piginot.contracts import (
    CanonicalMesh,
    CanonicalQuantity,
    ExecutionProfile,
    MaterialSpec,
    Unit,
)
from piginot.fem import FemFailureCode, FemRequest, FemSolverConfig, ScikitFemBackend

_ZERO_MEAN_RIGID_MODES = "zero_mean_translation_rotation"


def _mesh() -> CanonicalMesh:
    return CanonicalMesh(
        mesh_id="rigid-mode-square",
        nodes=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        cells=((0, 1, 2), (0, 2, 3)),
        cell_type="triangle",
        boundary_facets=((0, 1), (1, 2), (2, 3), (3, 0)),
        semantic_tags={"bottom": (0,), "right": (1,), "top": (2,), "left": (3,)},
        region_ids=(1, 1),
        unit=Unit.MM,
        orientation="ccw",
        topology_signature="rigid-mode-square-v1",
        quality_metrics={"min_scaled_jacobian": 1.0},
        source_provenance={"fixture": "s8-rigid-mode-constraint"},
    )


def _request(strategy: str | None = _ZERO_MEAN_RIGID_MODES) -> FemRequest:
    return FemRequest(
        request_id="pure-traction-rigid-mode-contract",
        mesh=_mesh(),
        material=MaterialSpec(
            "rigid_mode_material",
            "compressible_neo_hookean",
            {
                "elastic_modulus": CanonicalQuantity(100.0, Unit.MPA),
                "poisson_ratio": CanonicalQuantity(0.3, Unit.ONE),
            },
        ),
        boundary_conditions=(),
        loads=(),
        qois=(),
        config=FemSolverConfig(element_order=2),
        execution_profile=ExecutionProfile(
            python_version="3.12.0",
            library_versions={"scikit-fem": "12.0.2"},
            blas_threads=1,
            device="cpu",
            dtype="float64",
            rng_algorithm="none",
            seed=0,
        ),
        rigid_mode_constraint=strategy,
    )


def test_integral_constraints_remove_only_the_three_rigid_null_modes() -> None:
    attempt = ScikitFemBackend().solve_many((_request(),))[0]

    assert attempt.success, attempt.message
    assert attempt.result is not None
    assert not attempt.result.reactions
    assert len(attempt.result.rigid_mode_multipliers) == 3
    assert max(abs(value) for value in attempt.result.rigid_mode_multipliers) < 1.0e-14
    assert all(
        displacement == (0.0, 0.0) for displacement in attempt.result.displacement
    )


def test_integral_constraint_strategy_is_part_of_the_fem_recipe() -> None:
    constrained = _request()
    unconstrained = replace(constrained, rigid_mode_constraint=None)

    assert constrained.recipe_hash != unconstrained.recipe_hash


def test_nonfinite_rigid_mode_multiplier_is_rejected() -> None:
    attempt = ScikitFemBackend().solve_many((_request(),))[0]

    assert attempt.result is not None
    with pytest.raises(ValueError, match="rigid_mode_multipliers"):
        replace(attempt.result, rigid_mode_multipliers=(float("nan"),))


def test_missing_constraint_is_a_visible_failure_without_dropping_batch() -> None:
    attempts = ScikitFemBackend().solve_many((_request(None), _request()))

    assert len(attempts) == 2
    assert attempts[0].failure_code is FemFailureCode.CONSTRAINT_RANK_LOSS
    assert attempts[1].success


def test_unknown_rigid_mode_strategy_is_rejected_at_the_contract_boundary() -> None:
    with pytest.raises(ValueError, match="rigid_mode_constraint"):
        _request("point-pins")
