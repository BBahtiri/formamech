from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest
import torch

from campaigns.dogbone_v1.s11_contingency import (
    DERIVED_PARAMETER_NAMES,
    RAW_PARAMETER_NAMES,
)
from campaigns.dogbone_v1.s11_contingency_data import (
    BOUNDARY_POINT_COUNT,
    BOUNDARY_TAG_NAMES,
    STRESS_REFERENCE_MPA,
    OperatorCaseTensors,
    fit_case_normalization,
    normalize_case_values,
    prepare_operator_case,
)
from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    CanonicalQuantity,
    ExecutionProfile,
    MaterialSpec,
    Unit,
)
from piginot.doe import EvidenceRole, PublishedCase, PublishedCaseStatus
from piginot.fem import FemRequest, FemResult, FemSolverConfig


def _mesh() -> CanonicalMesh:
    nodes = (
        (0.0, 0.0),
        (2.0, 0.0),
        (3.0, 0.5),
        (3.0, 1.5),
        (2.0, 2.0),
        (1.0, 2.0),
        (0.0, 2.0),
    )
    return CanonicalMesh(
        mesh_id="synthetic-dogbone-v1",
        nodes=nodes,
        cells=((0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 5, 6)),
        cell_type="triangle",
        boundary_facets=((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0)),
        semantic_tags={
            "bottom": (0,),
            "gauge_top": (4, 5),
            "left_symmetry": (6,),
            "right_arc": (1, 3),
            "right_grip": (2,),
        },
        region_ids=(1, 1, 1, 1, 1),
        unit=Unit.MM,
        orientation="ccw",
        topology_signature="synthetic-dogbone-v1",
        quality_metrics={"minimum_angle_degrees": 10.0},
        source_provenance={"fixture": "s11-contingency-data"},
    )


def _profile() -> ExecutionProfile:
    return ExecutionProfile(
        python_version="3.13.0",
        library_versions={"piginot": "0.1.0", "torch": torch.__version__},
        blas_threads=1,
        device="cpu",
        dtype="float64",
        rng_algorithm="none",
        seed=0,
    )


def _published_case(
    case_id: str,
    *,
    offset: float = 0.0,
    role: EvidenceRole = EvidenceRole.TRAIN,
    status: PublishedCaseStatus = PublishedCaseStatus.ACTIVE,
) -> PublishedCase:
    return PublishedCase(
        case_id=case_id,
        leakage_group_id=f"group-{case_id}",
        role=role,
        primitive_coordinates={
            "L_total": 54.0 + offset,
            "W_grip": 20.0 + offset,
            "W_gauge": 10.0 + offset,
            "R_fillet": 12.0 + offset,
            "elastic_modulus": 760.0 + offset,
            "poisson_ratio": 0.23 + 0.01 * offset,
            "nominal_axial_strain": 0.10 + 0.01 * offset,
        },
        derived_coordinates={
            "arc_feasibility_ratio": 2.4 + offset,
            "fillet_to_gauge_ratio": 1.2 + offset,
            "grip_to_gauge_ratio": 2.0 + offset,
            "length_to_gauge_ratio": 5.4 + offset,
        },
        strata={name: 0 for name in DERIVED_PARAMETER_NAMES},
        status=status,
    )


def _request(case: PublishedCase) -> FemRequest:
    primitives = case.primitive_coordinates
    displacement = primitives["L_total"] * primitives["nominal_axial_strain"] / 2.0
    return FemRequest(
        request_id=case.case_id,
        mesh=_mesh(),
        material=MaterialSpec(
            "synthetic-material",
            "finite_strain_plane_stress_neo_hookean",
            {
                "elastic_modulus": CanonicalQuantity(
                    primitives["elastic_modulus"], Unit.MPA
                ),
                "poisson_ratio": CanonicalQuantity(
                    primitives["poisson_ratio"], Unit.ONE
                ),
            },
        ),
        boundary_conditions=(
            BoundaryConditionSpec(
                "right_grip_displacement",
                "right_grip",
                "dirichlet",
                ("x",),
                (CanonicalQuantity(displacement, Unit.MM),),
            ),
        ),
        loads=(),
        qois=(),
        config=FemSolverConfig(element_order=2),
        execution_profile=_profile(),
    )


def _result(request: FemRequest) -> FemResult:
    mesh = request.mesh
    extra_coordinates = ((1.0, 0.0), (2.5, 0.25))
    displacement = tuple(
        (0.1 * index, -0.2 * index) for index in range(len(mesh.nodes) + 2)
    )
    recovered = tuple(
        (
            (float(index + 1), float(index + 2), 1000.0 + index),
            (float(index + 3), float(index + 4), 2000.0 + index),
            (3000.0 + index, 4000.0 + index, 5000.0 + index),
        )
        for index in range(len(mesh.nodes) + 2)
    )
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    stress = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0))
    return FemResult(
        request_id=request.request_id,
        mesh_id=mesh.mesh_id,
        element_order=2,
        displacement_dof_coordinates=(*mesh.nodes, *extra_coordinates),
        displacement=displacement,
        quadrature_points=(),
        quadrature_weights=(),
        first_piola_stress=((stress,),) * len(mesh.cells),
        deformation_gradient=((identity,),) * len(mesh.cells),
        recovered_first_piola_stress=recovered,
        reactions={},
        rigid_mode_multipliers=(),
        total_strain_energy=1.0,
        external_work=2.0,
        total_potential_energy=-1.0,
        convergence_history=(),
        qois={},
    )


def _prepared(
    case_id: str,
    *,
    offset: float = 0.0,
    role: EvidenceRole = EvidenceRole.TRAIN,
) -> OperatorCaseTensors:
    case = _published_case(case_id, offset=offset, role=role)
    request = _request(case)
    return prepare_operator_case(case, request, _result(request))


def test_prepare_operator_case_extracts_vertex_fields_and_parameter_order() -> None:
    case = _published_case("train-a")
    request = _request(case)
    result = _result(request)

    prepared = prepare_operator_case(case, request, result)

    assert prepared.case_id == "train-a"
    assert prepared.role is EvidenceRole.TRAIN
    assert prepared.status is PublishedCaseStatus.ACTIVE
    assert prepared.parameter_names == (*RAW_PARAMETER_NAMES, *DERIVED_PARAMETER_NAMES)
    torch.testing.assert_close(
        prepared.raw_parameters,
        torch.tensor(
            [54.0, 20.0, 10.0, 12.0, 760.0, 0.23, 0.10],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        prepared.derived_parameters,
        torch.tensor([2.4, 1.2, 2.0, 5.4], dtype=torch.float64),
    )
    assert prepared.query_xy.shape == (len(request.mesh.nodes), 2)
    assert prepared.target_u.shape == (len(request.mesh.nodes), 2)
    assert prepared.target_P.shape == (len(request.mesh.nodes), 2, 2)
    torch.testing.assert_close(
        prepared.target_u[-1], torch.tensor([0.6, -1.2], dtype=torch.float64)
    )
    torch.testing.assert_close(
        prepared.target_P[0],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64),
    )
    assert prepared.target_u.shape[0] < len(result.displacement)
    assert prepared.target_P.shape[0] < len(result.recovered_first_piola_stress)
    assert prepared.qoi_indices.tolist() == [1, 4]
    assert all(
        tensor.device.type == "cpu"
        and tensor.dtype is torch.float64
        and not tensor.requires_grad
        for tensor in (
            prepared.boundary_xy,
            prepared.boundary_tags,
            prepared.boundary_normals,
            prepared.boundary_weights,
            prepared.query_xy,
            prepared.target_u,
            prepared.target_P,
            prepared.raw_parameters,
            prepared.derived_parameters,
        )
    )
    assert prepared.boundary_mask.dtype is torch.bool
    assert prepared.qoi_indices.dtype is torch.int64


def test_boundary_cloud_is_deterministic_measure_weighted_and_outward() -> None:
    prepared = _prepared("train-a")
    replay = _prepared("train-a")
    mesh = _mesh()

    assert prepared.boundary_xy.shape == (BOUNDARY_POINT_COUNT, 2)
    assert prepared.boundary_tags.shape == (
        BOUNDARY_POINT_COUNT,
        len(BOUNDARY_TAG_NAMES),
    )
    assert prepared.boundary_normals.shape == (BOUNDARY_POINT_COUNT, 2)
    assert prepared.boundary_weights.shape == (BOUNDARY_POINT_COUNT,)
    assert bool(prepared.boundary_mask.all().item())
    for name in (
        "boundary_xy",
        "boundary_tags",
        "boundary_normals",
        "boundary_weights",
    ):
        torch.testing.assert_close(
            getattr(prepared, name), getattr(replay, name), rtol=0.0, atol=0.0
        )

    nodes = torch.tensor(mesh.nodes, dtype=torch.float64)
    facets = torch.tensor(mesh.boundary_facets, dtype=torch.long)
    starts = nodes[facets[:, 0]]
    ends = nodes[facets[:, 1]]
    lengths = torch.linalg.vector_norm(ends - starts, dim=1)
    perimeter = lengths.sum()
    torch.testing.assert_close(
        prepared.boundary_weights,
        torch.full(
            (BOUNDARY_POINT_COUNT,),
            perimeter / BOUNDARY_POINT_COUNT,
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(prepared.boundary_weights.sum(), perimeter)
    torch.testing.assert_close(
        prepared.boundary_tags.sum(dim=1),
        torch.ones(BOUNDARY_POINT_COUNT, dtype=torch.float64),
    )

    locations = (
        (torch.arange(BOUNDARY_POINT_COUNT, dtype=torch.float64) + 0.5)
        * perimeter
        / BOUNDARY_POINT_COUNT
    )
    facet_ids = torch.searchsorted(torch.cumsum(lengths, dim=0), locations)
    for sample_id, facet_id_tensor in enumerate(facet_ids):
        facet_id = int(facet_id_tensor.item())
        owning_tags = tuple(
            tag_name
            for tag_name in BOUNDARY_TAG_NAMES
            if facet_id in mesh.semantic_tags[tag_name]
        )
        assert len(owning_tags) == 1
        assert int(prepared.boundary_tags[sample_id].argmax().item()) == (
            BOUNDARY_TAG_NAMES.index(owning_tags[0])
        )
        a, b = mesh.boundary_facets[facet_id]
        adjacent = tuple(cell for cell in mesh.cells if a in cell and b in cell)
        assert len(adjacent) == 1
        cell_centroid = nodes[list(adjacent[0])].mean(dim=0)
        away = prepared.boundary_xy[sample_id] - cell_centroid
        assert float(torch.dot(prepared.boundary_normals[sample_id], away)) > 0.0


def test_prepare_rejects_non_p2_or_unaligned_vertex_results() -> None:
    case = _published_case("train-a")
    request = _request(case)
    result = _result(request)

    with pytest.raises(ValueError, match="P2"):
        prepare_operator_case(case, replace(request, config=FemSolverConfig()), result)
    with pytest.raises(ValueError, match="vertex coordinates"):
        prepare_operator_case(
            case,
            request,
            replace(
                result,
                displacement_dof_coordinates=(
                    (99.0, 99.0),
                    *result.displacement_dof_coordinates[1:],
                ),
            ),
        )
    with pytest.raises(ValueError, match="recovered stress"):
        prepare_operator_case(
            case,
            request,
            replace(result, recovered_first_piola_stress=()),
        )


def test_raw_case_contract_rejects_invalid_tensor_boundaries() -> None:
    case = _prepared("train-a")
    nan_parameters = case.raw_parameters.clone()
    nan_parameters[0] = torch.nan
    bad_tags = case.boundary_tags.clone()
    bad_tags[0] = 0.0
    bad_normals = 2.0 * case.boundary_normals
    invalid = (
        ({"case_id": ""}, "case_id"),
        ({"leakage_group_id": ""}, "leakage_group_id"),
        ({"status": PublishedCaseStatus.RESERVE}, "ACTIVE"),
        ({"raw_parameters": case.raw_parameters.float()}, "float64"),
        ({"raw_parameters": case.raw_parameters[:-1]}, "shape"),
        ({"raw_parameters": nan_parameters}, "finite"),
        ({"boundary_weights": torch.zeros_like(case.boundary_weights)}, "positive"),
        ({"boundary_mask": torch.zeros_like(case.boundary_mask)}, "expose"),
        ({"boundary_tags": bad_tags}, "one-hot"),
        ({"boundary_normals": bad_normals}, "unit"),
        ({"query_xy": case.query_xy[:, :1]}, "shape"),
        ({"target_u": case.target_u[:-1]}, "shape"),
        ({"target_P": case.target_P[:, :1]}, "shape"),
        ({"qoi_indices": case.qoi_indices.float()}, "int64"),
        ({"qoi_indices": torch.tensor([1, 1])}, "sorted unique"),
        ({"qoi_indices": torch.tensor([99])}, "sorted unique"),
    )
    for change, message in invalid:
        with pytest.raises((TypeError, ValueError), match=message):
            replace(case, **cast(Any, change))


def test_prepare_rejects_identity_shape_and_semantic_drift() -> None:
    case = _published_case("train-a")
    request = _request(case)
    result = _result(request)
    wrong_tags = replace(
        request.mesh,
        semantic_tags={
            key: value
            for key, value in request.mesh.semantic_tags.items()
            if key != "gauge_top"
        },
    )

    invalid = (
        (case, request, replace(result, request_id="other"), "identities"),
        (case, request, replace(result, mesh_id="other"), "mesh identity"),
        (
            case,
            request,
            replace(result, displacement=result.displacement[:1]),
            "displacement omitted",
        ),
        (
            case,
            request,
            replace(result, displacement_dof_coordinates=()),
            "coordinates omitted",
        ),
        (
            case,
            request,
            replace(
                result,
                displacement=cast(
                    Any,
                    ((1.0,),) * len(request.mesh.nodes),
                ),
            ),
            "displacement must have shape",
        ),
        (
            case,
            replace(request, mesh=wrong_tags),
            result,
            "semantic tags",
        ),
    )
    for invalid_case, invalid_request, invalid_result, message in invalid:
        with pytest.raises(ValueError, match=message):
            prepare_operator_case(invalid_case, invalid_request, invalid_result)


def test_fit_normalization_uses_exact_active_train_population() -> None:
    first = _prepared("train-a", offset=0.0)
    second = _prepared("train-b", offset=2.0)

    statistics = fit_case_normalization(
        (second, first),
        fit_case_ids=("train-a", "train-b"),
        training_snapshot_hash="a" * 64,
    )

    assert statistics.parameter_names == (
        *RAW_PARAMETER_NAMES,
        *DERIVED_PARAMETER_NAMES,
    )
    assert statistics.fit_case_ids == ("train-a", "train-b")
    assert statistics.training_snapshot_hash == "a" * 64
    normalized_first = normalize_case_values(first, statistics)
    normalized_second = normalize_case_values(second, statistics)
    torch.testing.assert_close(
        normalized_first.normalized_parameters,
        -torch.ones(len(statistics.parameter_names), dtype=torch.float64),
    )
    torch.testing.assert_close(
        normalized_second.normalized_parameters,
        torch.ones(len(statistics.parameter_names), dtype=torch.float64),
    )

    before_u = first.target_u.clone()
    before_p = first.target_P.clone()
    displacement_scale = 54.0 * 0.10 / 2.0
    torch.testing.assert_close(
        normalized_first.target_u, first.target_u / displacement_scale
    )
    torch.testing.assert_close(
        normalized_first.target_P, first.target_P / STRESS_REFERENCE_MPA
    )
    torch.testing.assert_close(first.target_u, before_u, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.target_P, before_p, rtol=0.0, atol=0.0)
    assert normalized_first.target_u.data_ptr() != first.target_u.data_ptr()
    assert normalized_first.target_P.data_ptr() != first.target_P.data_ptr()
    assert normalized_first.normalization_hash == statistics.content_hash


def test_normalization_statistics_fail_closed_and_handle_constant_axes() -> None:
    case = _prepared("train-a")
    constant = fit_case_normalization(
        (case,),
        fit_case_ids=("train-a",),
        training_snapshot_hash="c" * 64,
    )
    assert constant.standard_deviation == (1.0,) * len(constant.parameter_names)
    invalid: tuple[tuple[dict[str, object], str], ...] = (
        ({"parameter_names": ("wrong",)}, "protocol order"),
        ({"mean": [1.0]}, "tuple"),
        ({"mean": (float("nan"),) * 11}, "finite"),
        ({"mean": (1.0,)}, "width"),
        ({"standard_deviation": (0.0,) * 11}, "positive"),
        ({"fit_case_ids": ("train-b", "train-a")}, "sorted unique"),
        ({"training_snapshot_hash": "BAD"}, "digest"),
        ({"stress_reference_mpa": 1.0}, "760"),
        ({"schema_version": "2.0"}, "schema"),
    )
    for change, message in invalid:
        with pytest.raises(ValueError, match=message):
            replace(constant, **cast(Any, change))


def test_normalization_rejects_role_leakage_or_fit_population_drift() -> None:
    train = _prepared("train-a")
    development = _prepared("development-a", role=EvidenceRole.DEVELOPMENT)

    with pytest.raises(ValueError, match="ACTIVE TRAIN"):
        fit_case_normalization(
            (train, development),
            fit_case_ids=("development-a", "train-a"),
            training_snapshot_hash="a" * 64,
        )
    with pytest.raises(ValueError, match="exact fit_case_ids"):
        fit_case_normalization(
            (train,),
            fit_case_ids=("train-a", "train-b"),
            training_snapshot_hash="a" * 64,
        )
    with pytest.raises(ValueError, match="non-empty tuple"):
        fit_case_normalization(
            (),
            fit_case_ids=("train-a",),
            training_snapshot_hash="a" * 64,
        )
    with pytest.raises(ValueError, match="sorted unique"):
        fit_case_normalization(
            (train,),
            fit_case_ids=("train-b", "train-a"),
            training_snapshot_hash="a" * 64,
        )


def test_frozen_train_statistics_normalize_development_without_role_relabeling() -> (
    None
):
    first = _prepared("train-a", offset=0.0)
    second = _prepared("train-b", offset=2.0)
    development = _prepared(
        "development-a",
        offset=1.0,
        role=EvidenceRole.DEVELOPMENT,
    )
    statistics = fit_case_normalization(
        (first, second),
        fit_case_ids=("train-a", "train-b"),
        training_snapshot_hash="d" * 64,
    )

    normalized = normalize_case_values(development, statistics)

    assert normalized.role is EvidenceRole.DEVELOPMENT
    assert normalized.normalization_hash == statistics.content_hash
    torch.testing.assert_close(
        normalized.normalized_parameters,
        torch.zeros(len(statistics.parameter_names), dtype=torch.float64),
    )


def test_normalized_values_select_candidate_parameters_without_reordering() -> None:
    first = _prepared("train-a", offset=0.0)
    second = _prepared("train-b", offset=2.0)
    statistics = fit_case_normalization(
        (first, second),
        fit_case_ids=("train-a", "train-b"),
        training_snapshot_hash="b" * 64,
    )
    normalized = normalize_case_values(first, statistics)

    torch.testing.assert_close(
        normalized.parameters_for(RAW_PARAMETER_NAMES),
        normalized.normalized_parameters[: len(RAW_PARAMETER_NAMES)],
    )
    torch.testing.assert_close(
        normalized.parameters_for((*RAW_PARAMETER_NAMES, *DERIVED_PARAMETER_NAMES)),
        normalized.normalized_parameters,
    )
    with pytest.raises(ValueError, match="registered"):
        normalized.parameters_for(("nu", "E"))
