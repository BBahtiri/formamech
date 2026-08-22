from __future__ import annotations

import math

import torch

from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    CanonicalQuantity,
    ExecutionProfile,
    LoadSpec,
    MaterialSpec,
    QoISpec,
    Unit,
)
from piginot.contracts.hashing import hash_spec
from piginot.fem import (
    FemFailureCode,
    FemRequest,
    FemSolverConfig,
    ScikitFemBackend,
)
from piginot.physics import first_piola, plane_stress_closure


def _quantity(value: float, unit: Unit) -> CanonicalQuantity:
    return CanonicalQuantity(value, unit)


def _material() -> MaterialSpec:
    return MaterialSpec(
        "rubber",
        "compressible_neo_hookean",
        {
            "elastic_modulus": _quantity(100.0, Unit.MPA),
            "poisson_ratio": _quantity(0.3, Unit.ONE),
        },
    )


def _profile(
    *, blas_threads: int = 1, skfem_version: str = "12.0.2"
) -> ExecutionProfile:
    return ExecutionProfile(
        python_version="3.12.0",
        library_versions={
            "numpy": "2.2.6",
            "scikit-fem": skfem_version,
            "torch": "2.13.0",
        },
        blas_threads=blas_threads,
        device="cpu",
        dtype="float64",
        rng_algorithm="none",
        seed=0,
    )


def _unit_square_mesh(mesh_id: str = "square") -> CanonicalMesh:
    return CanonicalMesh(
        mesh_id=mesh_id,
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
        topology_signature="square-v1",
        quality_metrics={"min_scaled_jacobian": 1.0},
        source_provenance={"fixture": "s7"},
    )


def _regularity_mesh() -> CanonicalMesh:
    return CanonicalMesh(
        mesh_id="l-shape",
        nodes=(
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.5),
            (0.5, 0.5),
            (0.5, 1.0),
            (0.0, 1.0),
        ),
        cells=((0, 1, 3), (1, 2, 3), (0, 3, 5), (3, 4, 5)),
        cell_type="triangle",
        boundary_facets=((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)),
        semantic_tags={"all": (0, 1, 2, 3, 4, 5), "corner": (2, 3)},
        region_ids=(1, 1, 1, 1),
        unit=Unit.MM,
        orientation="ccw",
        topology_signature="l-shape-reentrant-v1",
        quality_metrics={"min_scaled_jacobian": 0.5},
        source_provenance={"fixture": "s7"},
    )


def _extension_request(
    *,
    mesh: CanonicalMesh | None = None,
    element_order: int = 1,
    solver_tolerance: float = 1e-10,
    profile: ExecutionProfile | None = None,
    displacement: float = 0.2,
    max_iterations: int = 8,
    closure_iterations: int = 32,
) -> FemRequest:
    bcs = (
        BoundaryConditionSpec(
            "left_x",
            "left",
            "dirichlet",
            ("x",),
            (_quantity(0.0, Unit.MM),),
        ),
        BoundaryConditionSpec(
            "right_x",
            "right",
            "dirichlet",
            ("x",),
            (_quantity(displacement, Unit.MM),),
        ),
        BoundaryConditionSpec(
            "bottom_y",
            "bottom",
            "dirichlet",
            ("y",),
            (_quantity(0.0, Unit.MM),),
        ),
        BoundaryConditionSpec(
            "top_y",
            "top",
            "dirichlet",
            ("y",),
            (_quantity(0.0, Unit.MM),),
        ),
    )
    return FemRequest(
        request_id="uniaxial-square",
        mesh=_unit_square_mesh() if mesh is None else mesh,
        material=_material(),
        boundary_conditions=bcs,
        loads=(),
        qois=(),
        config=FemSolverConfig(
            element_order=element_order,
            solver_tolerance=solver_tolerance,
            max_newton_iterations=max_iterations,
            plane_stress_max_iterations=closure_iterations,
        ),
        execution_profile=_profile() if profile is None else profile,
    )


def _traction_request(
    *,
    element_order: int = 1,
    solver_tolerance: float = 1e-10,
    max_iterations: int = 12,
    traction: float = 17.492914739624112,
) -> FemRequest:
    bcs = (
        BoundaryConditionSpec(
            "left_x",
            "left",
            "dirichlet",
            ("x",),
            (_quantity(0.0, Unit.MM),),
        ),
        BoundaryConditionSpec(
            "bottom_y",
            "bottom",
            "dirichlet",
            ("y",),
            (_quantity(0.0, Unit.MM),),
        ),
    )
    loads = (
        LoadSpec(
            "right_pull",
            "traction",
            "right",
            ("x",),
            (_quantity(traction, Unit.MPA),),
        ),
    )
    return FemRequest(
        request_id="traction-square",
        mesh=_unit_square_mesh(),
        material=_material(),
        boundary_conditions=bcs,
        loads=loads,
        qois=(),
        config=FemSolverConfig(
            element_order=element_order,
            solver_tolerance=solver_tolerance,
            max_newton_iterations=max_iterations,
            load_steps=4,
        ),
        execution_profile=_profile(),
    )


def test_g7_analytic_unit_square_matches_closed_form_to_1e_minus_8() -> None:
    result = ScikitFemBackend().solve_many((_traction_request(),))

    assert len(result) == 1
    attempt = result[0]
    assert attempt.success
    assert attempt.failure_code is None

    solved = attempt.result
    assert solved is not None
    expected_displacement = (
        (0.0, 0.0),
        (0.2, 0.0),
        (0.2, -0.05436737570887096),
        (0.0, -0.05436737570887096),
    )
    for actual, expected_value in zip(
        solved.displacement, expected_displacement, strict=True
    ):
        assert math.dist(actual, expected_value) < 1.0e-8

    computed = torch.tensor(solved.first_piola_stress, dtype=torch.float64)
    deformation = torch.tensor(
        [[1.2, 0.0], [0.0, 0.945632624291129]],
        dtype=torch.float64,
    )
    f33 = plane_stress_closure(deformation, _material())
    full_deformation = torch.eye(3, dtype=torch.float64)
    full_deformation[:2, :2] = deformation
    full_deformation[2, 2] = f33
    expected = first_piola(full_deformation, _material())

    assert torch.max(torch.abs(computed - expected)).item() < 1e-8
    assert attempt.result_hash == hash_spec(attempt.result, "content")


def test_g7_p2_unit_square_matches_closed_form_to_1e_minus_8() -> None:
    result = ScikitFemBackend().solve_many((_traction_request(element_order=2),))

    attempt = result[0]
    assert attempt.success
    assert attempt.result is not None
    computed = torch.tensor(attempt.result.first_piola_stress, dtype=torch.float64)
    deformation = torch.tensor(
        [[1.2, 0.0], [0.0, 0.945632624291129]], dtype=torch.float64
    )
    f33 = plane_stress_closure(deformation, _material())
    full_deformation = torch.eye(3, dtype=torch.float64)
    full_deformation[:2, :2] = deformation
    full_deformation[2, 2] = f33
    expected = first_piola(full_deformation, _material())

    assert torch.max(torch.abs(computed - expected)).item() < 1e-8


def test_nonzero_dead_traction_enters_equilibrium_and_reactions() -> None:
    attempt = ScikitFemBackend().solve_many((_traction_request(traction=1.0),))[0]

    assert attempt.success
    assert attempt.result is not None
    right_node_displacements = [
        attempt.result.displacement[index][0]
        for index, point in enumerate(attempt.request.mesh.nodes)
        if math.isclose(point[0], 1.0)
    ]
    assert min(right_node_displacements) > 0.0
    assert attempt.result.external_work > 0.0
    assert attempt.result.reactions["left_x"][0] < 0.0


def test_g7_outcomes_records_five_distinct_failure_codes() -> None:
    valid = _extension_request()
    attempts = (
        _extension_request(displacement=-1.2),
        _extension_request(closure_iterations=0),
        FemRequest(
            request_id="rank-loss",
            mesh=_unit_square_mesh(),
            material=_material(),
            boundary_conditions=(),
            loads=(),
            qois=(),
            config=valid.config,
            execution_profile=valid.execution_profile,
        ),
        _traction_request(max_iterations=1, solver_tolerance=1e-20),
        _extension_request(mesh=_unit_square_mesh("too-large")).with_config(
            valid.config.with_resource_limit(max_dofs=2)
        ),
    )

    result = ScikitFemBackend().solve_many(attempts)

    assert {attempt.failure_code for attempt in result} == {
        FemFailureCode.NEGATIVE_JACOBIAN,
        FemFailureCode.CLOSURE_FAILURE,
        FemFailureCode.CONSTRAINT_RANK_LOSS,
        FemFailureCode.NEWTON_NON_CONVERGENCE,
        FemFailureCode.RESOURCE_EXHAUSTION,
    }


def test_g7_no_drop_failed_cases_are_visible_in_result_set() -> None:
    requests = (
        _extension_request(),
        _extension_request(displacement=-1.2),
        _extension_request(max_iterations=0),
    )

    result = ScikitFemBackend().solve_many(requests)

    assert tuple(attempt.request.request_id for attempt in result) == tuple(
        request.request_id for request in requests
    )
    assert len(result) == 3
    assert sum(1 for attempt in result if not attempt.success) == 2
    assert result[2].failure_code is FemFailureCode.NEWTON_NON_CONVERGENCE


def test_g7_recipe_changes_with_mesh_tolerance_order_threads_and_library() -> None:
    baseline = _extension_request()
    variants = (
        _extension_request(mesh=_unit_square_mesh("square-other")),
        _extension_request(solver_tolerance=1e-9),
        _extension_request(element_order=2),
        _extension_request(profile=_profile(blas_threads=2)),
        _extension_request(profile=_profile(skfem_version="12.0.1")),
    )

    baseline_hash = baseline.recipe_hash

    assert all(variant.recipe_hash != baseline_hash for variant in variants)
    assert len({variant.recipe_hash for variant in variants}) == len(variants)


def test_g7_retry_reuses_byte_identical_recipe_only() -> None:
    request = _extension_request()
    retry = _extension_request()
    changed = _extension_request(displacement=0.25)

    assert retry.recipe_hash == request.recipe_hash
    assert changed.recipe_hash != request.recipe_hash
    assert (
        ScikitFemBackend().solve_many((request, retry))[0].recipe_hash
        == request.recipe_hash
    )


def test_g7_regularity_rejects_reentrant_corner_peak_stress_qoi() -> None:
    request = _extension_request(mesh=_regularity_mesh()).with_qois(
        (
            QoISpec(
                "bad_peak",
                "first_piola_stress",
                "pointwise_peak",
                "corner",
                Unit.MPA,
            ),
        )
    )

    attempt = ScikitFemBackend().solve_many((request,))[0]

    assert not attempt.success
    assert attempt.failure_code is FemFailureCode.QOI_REGULARITY_REJECTED
    assert "pointwise peak stress" in attempt.message


def test_body_force_density_cannot_masquerade_as_mpa() -> None:
    request = _extension_request().with_loads(
        (
            LoadSpec(
                "gravity",
                "body_force",
                None,
                ("y",),
                (_quantity(0.0, Unit.MPA),),
            ),
        )
    )

    attempt = ScikitFemBackend().solve_many((request,))[0]

    assert not attempt.success
    assert attempt.failure_code is FemFailureCode.UNSUPPORTED_REQUEST
