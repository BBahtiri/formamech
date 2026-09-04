from __future__ import annotations

from typing import TypedDict

from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    CanonicalQuantity,
    ExecutionProfile,
    LoadSpec,
    MaterialSpec,
    QoISpec,
    Unit,
    hash_spec,
)
from piginot.datasets.snapshots import (
    DatasetCaseRecord,
    RoleDatasetSnapshot,
    SampleTrace,
)
from piginot.doe import EvidenceRole
from piginot.fem import FemFailureCode, FemRequest, FemSolverConfig

CAMPAIGN_HASH = "a" * 64
MANIFEST_HASH = "b" * 64
CASE_ID = "case-001"


class CollocationCounts(TypedDict):
    interior_count: int
    boundary_count: int
    interface_count: int


def snapshot(
    *,
    mesh_id: str = "two-region-square",
    mesh_content_hash: str | None = None,
    request: FemRequest | None = None,
    case_id: str = CASE_ID,
    role: EvidenceRole = EvidenceRole.TRAIN,
    success: bool = True,
) -> RoleDatasetSnapshot:
    if mesh_content_hash is None:
        mesh_content_hash = hash_spec(mesh(), "content")
    record = DatasetCaseRecord(
        case_id=case_id,
        leakage_group_id="group-001",
        role=role,
        primitive_coordinates={"width": 1.0},
        derived_coordinates={"net_section_ratio": 0.5},
        strata={"width": 0},
        request_recipe_hash=request.recipe_hash if request is not None else "c" * 64,
        request_content_hash=(
            hash_spec(request, "content") if request is not None else "f" * 64
        ),
        mesh_id=mesh_id,
        mesh_content_hash=mesh_content_hash,
        success=success,
        sample_trace=SampleTrace(fem_result_hash="d" * 64) if success else None,
        store_content_hash="e" * 64 if success else None,
        failure_code=None if success else FemFailureCode.NEWTON_NON_CONVERGENCE,
        message=("succeeded" if success else "failed:NEWTON_NON_CONVERGENCE"),
    )
    return RoleDatasetSnapshot(
        campaign_hash=CAMPAIGN_HASH,
        public_case_manifest_hash=MANIFEST_HASH,
        request_plan_hash="1" * 64,
        role=role,
        cases=(record,),
    )


def mesh(*, regions: tuple[int, int] = (0, 1)) -> CanonicalMesh:
    return CanonicalMesh(
        mesh_id="two-region-square",
        nodes=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        cells=((0, 1, 2), (0, 2, 3)),
        cell_type="triangle",
        boundary_facets=((0, 1), (1, 2), (2, 3), (3, 0)),
        semantic_tags={
            "bottom": (0,),
            "left": (3,),
            "right": (1,),
            "top": (2,),
        },
        region_ids=regions,
        unit=Unit.MM,
        orientation="counterclockwise",
        topology_signature="square-v1",
        quality_metrics={"minimum_scaled_jacobian": 1.0},
        source_provenance={"fixture": "generic-two-region-square"},
    )


def boundary_conditions() -> tuple[BoundaryConditionSpec, ...]:
    return (
        BoundaryConditionSpec("left_fixed", "left", "dirichlet", ("x",)),
        BoundaryConditionSpec("right_y", "right", "symmetry", ("y",)),
        BoundaryConditionSpec("bottom_free", "bottom", "free"),
    )


def loads() -> tuple[LoadSpec, ...]:
    return (
        LoadSpec("right_pull", "traction", "right", ("x",)),
        LoadSpec("top_pull", "traction", "top", ("y",)),
        LoadSpec("gravity", "body_force"),
    )


def request_with_qoi() -> FemRequest:
    return FemRequest(
        request_id=CASE_ID,
        mesh=mesh(),
        material=MaterialSpec(
            name="generic",
            model="compressible_neo_hookean_plane_stress",
            properties={
                "elastic_modulus": CanonicalQuantity(100.0, Unit.MPA),
                "poisson_ratio": CanonicalQuantity(0.3, Unit.ONE),
            },
        ),
        boundary_conditions=boundary_conditions(),
        loads=loads(),
        qois=(
            QoISpec(
                name="peak_left",
                field="first_piola_stress",
                reduction="maximum",
                region_tag="left",
                unit=Unit.MPA,
            ),
        ),
        config=FemSolverConfig(element_order=2),
        execution_profile=ExecutionProfile(
            python_version="3.13.0",
            library_versions={"numpy": "2.2.6"},
            blas_threads=1,
            device="cpu",
            dtype="float64",
            rng_algorithm="none",
            seed=0,
        ),
    )
