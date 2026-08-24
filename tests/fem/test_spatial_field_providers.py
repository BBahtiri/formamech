"""S8 tests for recipe-addressable spatial boundary fields."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    CanonicalQuantity,
    ExecutionProfile,
    LoadSpec,
    MaterialSpec,
    Unit,
)
from piginot.fem import FemFailureCode, FemRequest, FemSolverConfig, ScikitFemBackend
from piginot.fem.field_providers import immutable_provider_registry


def _quantity(value: float, unit: Unit) -> CanonicalQuantity:
    return CanonicalQuantity(value, unit)


def _unit_square() -> CanonicalMesh:
    return CanonicalMesh(
        mesh_id="spatial-field-square",
        nodes=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        cells=((0, 1, 2), (0, 2, 3)),
        cell_type="triangle",
        boundary_facets=((0, 1), (1, 2), (2, 3), (3, 0)),
        semantic_tags={
            "bottom": (0,),
            "right": (1,),
            "top": (2,),
            "left": (3,),
            "all": (0, 1, 2, 3),
        },
        region_ids=(1, 1),
        unit=Unit.MM,
        orientation="ccw",
        topology_signature="spatial-field-square-v1",
        quality_metrics={"min_scaled_jacobian": 1.0},
        source_provenance={"fixture": "s8-spatial-fields"},
    )


def _material() -> MaterialSpec:
    return MaterialSpec(
        "rubber",
        "compressible_neo_hookean",
        {
            "elastic_modulus": _quantity(100.0, Unit.MPA),
            "poisson_ratio": _quantity(0.3, Unit.ONE),
        },
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


@dataclass(frozen=True, slots=True)
class AffineDirichletField:
    provider_id: str = "affine-dirichlet-v1"

    def evaluate(self, reference_points: np.ndarray) -> np.ndarray:
        x, y = reference_points
        return np.stack((0.01 * x + 0.002 * y, -0.003 * x + 0.004 * y))


@dataclass(frozen=True, slots=True)
class NormalAwareTractionField:
    provider_id: str = "normal-aware-traction-v1"

    def evaluate(
        self,
        reference_points: np.ndarray,
        outward_reference_normals: np.ndarray,
    ) -> np.ndarray:
        magnitude = 0.1 * (1.0 + reference_points[1])
        return outward_reference_normals * magnitude[None, ...]


def _affine_request(provider_id: str = "affine-dirichlet-v1") -> FemRequest:
    return FemRequest(
        request_id="spatial-dirichlet",
        mesh=_unit_square(),
        material=_material(),
        boundary_conditions=(
            BoundaryConditionSpec("prescribed", "all", "dirichlet", ("x", "y")),
        ),
        loads=(),
        qois=(),
        config=FemSolverConfig(
            element_order=2,
            solver_tolerance=1.0e-11,
            max_newton_iterations=16,
            load_steps=2,
        ),
        execution_profile=_profile(),
        boundary_field_provider_ids={"prescribed": provider_id},
    )


def _traction_request(provider_id: str = "normal-aware-traction-v1") -> FemRequest:
    return FemRequest(
        request_id="spatial-traction",
        mesh=_unit_square(),
        material=_material(),
        boundary_conditions=(
            BoundaryConditionSpec(
                "left_clamp",
                "left",
                "dirichlet",
                ("x", "y"),
                (_quantity(0.0, Unit.MM), _quantity(0.0, Unit.MM)),
            ),
        ),
        loads=(LoadSpec("edge_field", "traction", "right", ("x", "y")),),
        qois=(),
        config=FemSolverConfig(
            element_order=2,
            solver_tolerance=1.0e-11,
            max_newton_iterations=20,
            load_steps=2,
        ),
        execution_profile=_profile(),
        traction_field_provider_ids={"edge_field": provider_id},
    )


def test_spatial_provider_ids_are_immutable_recipe_inputs() -> None:
    baseline = _affine_request()
    changed_boundary = replace(
        baseline,
        boundary_field_provider_ids={"prescribed": "affine-dirichlet-v2"},
    )
    changed_traction = replace(
        baseline,
        traction_field_provider_ids={"unused": "traction-v1"},
    )

    assert baseline.recipe_hash != changed_boundary.recipe_hash
    assert baseline.recipe_hash != changed_traction.recipe_hash
    with pytest.raises(TypeError):
        baseline.boundary_field_provider_ids["prescribed"] = "mutated"  # type: ignore[index]


def test_provider_registry_is_sorted_immutable_and_rejects_duplicate_ids() -> None:
    late = AffineDirichletField("z-provider")
    early = AffineDirichletField("a-provider")

    registry = immutable_provider_registry((late, early), label="Dirichlet-field")

    assert tuple(registry) == ("a-provider", "z-provider")
    with pytest.raises(TypeError):
        registry["new"] = early  # type: ignore[index]
    with pytest.raises(ValueError, match="unique"):
        immutable_provider_registry((early, early), label="Dirichlet-field")
    with pytest.raises(ValueError, match="non-empty strings"):
        immutable_provider_registry(
            (AffineDirichletField(" "),), label="Dirichlet-field"
        )


def test_spatial_dirichlet_field_is_applied_at_p2_boundary_dofs() -> None:
    provider = AffineDirichletField()

    attempt = ScikitFemBackend(dirichlet_field_providers=(provider,)).solve_many(
        (_affine_request(),)
    )[0]

    assert attempt.success, attempt.message
    assert attempt.result is not None
    for point, displacement in zip(
        attempt.result.displacement_dof_coordinates,
        attempt.result.displacement,
        strict=True,
    ):
        expected = provider.evaluate(np.asarray(point, dtype=np.float64)[:, None])
        assert np.allclose(displacement, expected[:, 0], rtol=0.0, atol=1.0e-10)


def test_spatial_nominal_traction_receives_reference_normals() -> None:
    provider = NormalAwareTractionField()

    attempt = ScikitFemBackend(traction_field_providers=(provider,)).solve_many(
        (_traction_request(),)
    )[0]

    assert attempt.success, attempt.message
    assert attempt.result is not None
    assert attempt.result.reactions["left_clamp"][0] == pytest.approx(-0.15, abs=1.0e-9)
    assert attempt.result.reactions["left_clamp"][1] == pytest.approx(0.0, abs=1.0e-9)


@dataclass(frozen=True, slots=True)
class MalformedDirichletField:
    provider_id: str = "malformed-dirichlet-v1"

    def evaluate(self, reference_points: np.ndarray) -> np.ndarray:
        return np.zeros((1, reference_points.shape[-1]), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class MalformedTractionField:
    provider_id: str = "malformed-traction-v1"

    def evaluate(
        self,
        reference_points: np.ndarray,
        outward_reference_normals: np.ndarray,
    ) -> np.ndarray:
        del outward_reference_normals
        return np.zeros(reference_points.shape[1:], dtype=np.float64)


def test_unknown_and_malformed_fields_are_visible_without_dropping_batch() -> None:
    valid_provider = AffineDirichletField()
    malformed_dirichlet = MalformedDirichletField()
    malformed_traction = MalformedTractionField()
    requests = (
        _affine_request("unknown-dirichlet-v1"),
        _affine_request(malformed_dirichlet.provider_id),
        _traction_request("unknown-traction-v1"),
        _traction_request(malformed_traction.provider_id),
        _affine_request(),
    )
    backend = ScikitFemBackend(
        dirichlet_field_providers=(valid_provider, malformed_dirichlet),
        traction_field_providers=(malformed_traction,),
    )

    attempts = backend.solve_many(requests)

    assert len(attempts) == len(requests)
    assert tuple(attempt.request.request_id for attempt in attempts) == tuple(
        request.request_id for request in requests
    )
    assert all(
        attempt.failure_code is FemFailureCode.UNSUPPORTED_REQUEST
        for attempt in attempts[:4]
    )
    assert "not registered" in attempts[0].message
    assert "invalid canonical field" in attempts[1].message
    assert "not registered" in attempts[2].message
    assert "invalid canonical field" in attempts[3].message
    assert attempts[4].success


def test_provider_references_cannot_ambiguously_override_constant_fields() -> None:
    provider = AffineDirichletField()
    ambiguous_dirichlet = replace(
        _affine_request(),
        boundary_conditions=(
            BoundaryConditionSpec(
                "prescribed",
                "all",
                "dirichlet",
                ("x", "y"),
                (_quantity(0.0, Unit.MM), _quantity(0.0, Unit.MM)),
            ),
        ),
    )
    traction_provider = NormalAwareTractionField()
    ambiguous_traction = replace(
        _traction_request(),
        loads=(
            LoadSpec(
                "edge_field",
                "traction",
                "right",
                ("x", "y"),
                (_quantity(0.0, Unit.MPA), _quantity(0.0, Unit.MPA)),
            ),
        ),
    )
    backend = ScikitFemBackend(
        dirichlet_field_providers=(provider,),
        traction_field_providers=(traction_provider,),
    )

    attempts = backend.solve_many((ambiguous_dirichlet, ambiguous_traction))

    assert all(not attempt.success for attempt in attempts)
    assert all(
        attempt.failure_code is FemFailureCode.UNSUPPORTED_REQUEST
        for attempt in attempts
    )
    assert all("cannot also carry constant values" in item.message for item in attempts)


def test_spatial_provider_references_reject_nonmatching_condition_and_load_kinds() -> (
    None
):
    provider = AffineDirichletField()
    free_boundary = replace(
        _affine_request(),
        boundary_conditions=(BoundaryConditionSpec("prescribed", "all", "free"),),
    )
    traction_provider = NormalAwareTractionField()
    body_load = replace(
        _traction_request(),
        loads=(LoadSpec("edge_field", "body_force"),),
    )
    backend = ScikitFemBackend(
        dirichlet_field_providers=(provider,),
        traction_field_providers=(traction_provider,),
    )

    free_attempt, body_attempt = backend.solve_many((free_boundary, body_load))

    assert free_attempt.failure_code is FemFailureCode.UNSUPPORTED_REQUEST
    assert "require a Dirichlet condition" in free_attempt.message
    assert body_attempt.failure_code is FemFailureCode.UNSUPPORTED_REQUEST
    assert "require a traction load" in body_attempt.message
