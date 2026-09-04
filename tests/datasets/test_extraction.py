from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    CanonicalQuantity,
    Dimension,
    ExecutionProfile,
    MaterialSpec,
    QoISpec,
    Unit,
    canonical_json_bytes,
    hash_spec,
)
from piginot.datasets.extraction import (
    ExtractedArtifact,
    ExtractedFemArtifacts,
    FemExtractionError,
    FieldLocation,
    extract_fem_artifacts,
)
from piginot.datasets.request_plan import (
    PublicFemRequestPlan,
    PublicFemRequestPlanEntry,
)
from piginot.datasets.snapshots import (
    DatasetCaseRecord,
    RoleDatasetSnapshot,
    SampleTrace,
)
from piginot.datasets.units import AreaUnit
from piginot.doe import EvidenceRole
from piginot.fem import FemAttempt, FemRequest, FemResult, FemSolverConfig
from piginot.store import ObjectStore
from piginot.store.store import content_hash as store_content_hash


def _request() -> FemRequest:
    mesh = CanonicalMesh(
        mesh_id="one-triangle",
        nodes=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        cells=((0, 1, 2),),
        cell_type="triangle",
        boundary_facets=((0, 1), (1, 2), (2, 0)),
        semantic_tags={"edge": (0, 1, 2)},
        region_ids=(1,),
        unit=Unit.MM,
        orientation="ccw",
        topology_signature="one-triangle",
        quality_metrics={"min_scaled_jacobian": 0.5},
        source_provenance={"fixture": "s10-extraction"},
    )
    return FemRequest(
        request_id="case-1",
        mesh=mesh,
        material=MaterialSpec(
            "rubber",
            "compressible_neo_hookean_plane_stress",
            {
                "elastic_modulus": CanonicalQuantity(100.0, Unit.MPA),
                "poisson_ratio": CanonicalQuantity(0.3, Unit.ONE),
            },
        ),
        boundary_conditions=(
            BoundaryConditionSpec(
                "fixed",
                "edge",
                "dirichlet",
                ("x",),
                (CanonicalQuantity(0.0, Unit.MM),),
            ),
        ),
        loads=(),
        qois=(QoISpec("peak", "first_piola", "max", "edge", Unit.MPA),),
        config=FemSolverConfig(),
        execution_profile=ExecutionProfile(
            python_version="3.12.0",
            library_versions={"numpy": "2.2.6"},
            blas_threads=1,
            device="cpu",
            dtype="float64",
            rng_algorithm="none",
            seed=0,
        ),
    )


def _attempt() -> FemAttempt:
    request = _request()
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    stress = ((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0))
    result = FemResult(
        request_id=request.request_id,
        mesh_id=request.mesh.mesh_id,
        element_order=1,
        displacement_dof_coordinates=request.mesh.nodes,
        displacement=((0.0, 0.0), (0.1, 0.0), (0.0, 0.1)),
        quadrature_points=(((1.0 / 3.0, 1.0 / 3.0),),),
        quadrature_weights=((0.5,),),
        first_piola_stress=((stress,),),
        deformation_gradient=((identity,),),
        recovered_first_piola_stress=(stress, stress, stress),
        reactions={"fixed": (-2.0, 0.0)},
        rigid_mode_multipliers=(),
        total_strain_energy=0.5,
        external_work=1.0,
        total_potential_energy=-0.5,
        convergence_history=(),
        qois={"peak": 2.0},
    )
    return FemAttempt(
        request=request,
        recipe_hash=request.recipe_hash,
        success=True,
        result=result,
        result_hash=hash_spec(result, "content"),
        store_content_hash=store_content_hash(
            canonical_json_bytes({"domain": "content", "payload": result})
        ),
        failure_code=None,
        message="succeeded",
    )


def _context(
    tmp_path: Path,
) -> tuple[RoleDatasetSnapshot, PublicFemRequestPlan, FemAttempt, ObjectStore]:
    attempt = _attempt()
    request = attempt.request
    assert attempt.result is not None
    store = ObjectStore.create(tmp_path)
    persisted = store.put(
        canonical_json_bytes({"domain": "content", "payload": attempt.result}),
        recipe_hash=attempt.recipe_hash,
    )
    assert persisted == attempt.store_content_hash
    plan = PublicFemRequestPlan(
        campaign_hash="a" * 64,
        public_case_manifest_hash="b" * 64,
        execution_profile_hash=hash_spec(request.execution_profile, "content"),
        entries=(
            PublicFemRequestPlanEntry(
                case_id=request.request_id,
                leakage_group_id="group-1",
                role=EvidenceRole.TRAIN,
                request_recipe_hash=request.recipe_hash,
                request_content_hash=hash_spec(request, "content"),
                mesh_id=request.mesh.mesh_id,
                mesh_content_hash=hash_spec(request.mesh, "content"),
            ),
        ),
    )
    snapshot = RoleDatasetSnapshot(
        campaign_hash=plan.campaign_hash,
        public_case_manifest_hash=plan.public_case_manifest_hash,
        request_plan_hash=plan.content_hash,
        role=EvidenceRole.TRAIN,
        cases=(
            DatasetCaseRecord(
                case_id=request.request_id,
                leakage_group_id="group-1",
                role=EvidenceRole.TRAIN,
                primitive_coordinates={"width": 1.0},
                derived_coordinates={"net_section_ratio": 0.5},
                strata={"width": 0},
                request_recipe_hash=request.recipe_hash,
                request_content_hash=hash_spec(request, "content"),
                mesh_id=request.mesh.mesh_id,
                mesh_content_hash=hash_spec(request.mesh, "content"),
                success=True,
                sample_trace=SampleTrace(fem_result_hash=attempt.result_hash),
                store_content_hash=attempt.store_content_hash,
                failure_code=None,
                message="succeeded",
            ),
        ),
    )
    return snapshot, plan, attempt, store


def _extract(tmp_path: Path) -> ExtractedFemArtifacts:
    snapshot, plan, attempt, store = _context(tmp_path)
    return extract_fem_artifacts(snapshot, plan, attempt, store)


def test_extracts_all_required_fields_with_explicit_units_and_shared_trace(
    tmp_path: Path,
) -> None:
    snapshot, plan, attempt, store = _context(tmp_path)

    artifacts = extract_fem_artifacts(snapshot, plan, attempt, store)
    by_name = {artifact.name: artifact for artifact in artifacts.fields}

    assert set(by_name) == {
        "displacement",
        "first_piola_stress",
        "deformation_gradient",
        "recovered_first_piola_stress",
    }
    assert (by_name["displacement"].unit, by_name["displacement"].dimension) == (
        Unit.MM,
        Dimension.LENGTH,
    )
    assert by_name["displacement"].location == "DISPLACEMENT_DOF"
    assert by_name["first_piola_stress"].unit is Unit.MPA
    assert by_name["first_piola_stress"].location == "QUADRATURE"
    assert by_name["first_piola_stress"].coordinate_unit is Unit.MM
    assert (
        by_name["first_piola_stress"].quadrature_weight_unit
        is AreaUnit.SQUARE_MILLIMETRE
    )
    assert by_name["displacement"].coordinate_unit is Unit.MM
    assert by_name["displacement"].quadrature_weight_unit is None
    assert by_name["deformation_gradient"].unit is Unit.ONE
    assert by_name["recovered_first_piola_stress"].location == "RECOVERED_DOF"
    assert all(
        artifact.fem_result_hash == attempt.result_hash for artifact in artifacts.fields
    )
    assert artifacts.snapshot_content_hash == snapshot.content_hash
    assert artifacts.request_plan_hash == plan.content_hash
    assert artifacts.role is EvidenceRole.TRAIN
    assert artifacts.leakage_group_id == "group-1"


def test_extracts_reactions_energies_and_request_declared_qoi_units(
    tmp_path: Path,
) -> None:
    snapshot, plan, attempt, store = _context(tmp_path)

    artifacts = extract_fem_artifacts(snapshot, plan, attempt, store)

    assert artifacts.reactions[0].name == "reaction:fixed"
    assert artifacts.reactions[0].unit is Unit.N
    assert artifacts.reactions[0].dimension is Dimension.FORCE
    assert {item.name for item in artifacts.energies} == {
        "external_work",
        "total_potential_energy",
        "total_strain_energy",
    }
    assert all(item.unit is Unit.N_MM for item in artifacts.energies)
    assert all(item.dimension is Dimension.ENERGY for item in artifacts.energies)
    assert artifacts.qois[0].name == "qoi:peak"
    assert artifacts.qois[0].unit is Unit.MPA
    assert artifacts.qois[0].dimension is Dimension.STRESS
    assert all(
        item.fem_result_hash == attempt.result_hash
        for item in (*artifacts.reactions, *artifacts.energies, *artifacts.qois)
    )


def test_extraction_rejects_failed_or_identity_corrupted_attempts(
    tmp_path: Path,
) -> None:
    snapshot, plan, attempt, store = _context(tmp_path)

    with pytest.raises(FemExtractionError, match="successful"):
        extract_fem_artifacts(
            snapshot,
            plan,
            replace(
                attempt,
                success=False,
                result=None,
                result_hash=None,
                failure_code=None,
            ),
            store,
        )
    with pytest.raises(FemExtractionError, match="result_hash"):
        extract_fem_artifacts(
            snapshot, plan, replace(attempt, result_hash="0" * 64), store
        )


def test_extraction_rejects_snapshot_plan_or_store_bypass(tmp_path: Path) -> None:
    snapshot, plan, attempt, store = _context(tmp_path / "trusted")
    other_store = ObjectStore.create(tmp_path / "empty")

    with pytest.raises(FemExtractionError, match="request plan"):
        extract_fem_artifacts(
            replace(snapshot, request_plan_hash="f" * 64), plan, attempt, store
        )
    with pytest.raises(FemExtractionError, match="ObjectStore"):
        extract_fem_artifacts(snapshot, plan, attempt, other_store)
    with pytest.raises(TypeError, match="RoleDatasetSnapshot"):
        extract_fem_artifacts(object(), plan, attempt, store)  # type: ignore[arg-type]


def test_extracted_artifact_constructor_rejects_bad_numeric_or_unit_metadata(
    tmp_path: Path,
) -> None:
    artifact = _extract(tmp_path).fields[0]

    with pytest.raises(FemExtractionError, match="digest"):
        replace(artifact, fem_result_hash="g" * 64)
    with pytest.raises(FemExtractionError, match="finite"):
        replace(artifact, values=(float("nan"),))
    with pytest.raises(FemExtractionError, match="coordinate_unit"):
        replace(artifact, coordinate_unit=None)
    with pytest.raises(FemExtractionError, match="quadrature_weight_unit"):
        replace(artifact, quadrature_weight_unit=None)


def test_extracted_artifact_rejects_incoherent_support_layouts(
    tmp_path: Path,
) -> None:
    artifacts = _extract(tmp_path)
    quadrature = artifacts.fields[0]
    displacement = next(
        artifact for artifact in artifacts.fields if artifact.name == "displacement"
    )

    with pytest.raises(FemExtractionError, match="quadrature layout"):
        replace(
            quadrature,
            coordinates=(*quadrature.coordinates, quadrature.coordinates[0]),
        )
    with pytest.raises(FemExtractionError, match="quadrature layout"):
        replace(quadrature, values=(1.0,))
    with pytest.raises(FemExtractionError, match="QUADRATURE"):
        replace(quadrature, location=FieldLocation.GLOBAL)
    with pytest.raises(FemExtractionError, match="DOF layout"):
        replace(displacement, values=displacement.values[:-1])


def test_extracted_bundle_rejects_cross_case_artifact_injection(
    tmp_path: Path,
) -> None:
    bundle = _extract(tmp_path)
    foreign = replace(bundle.fields[0], case_id="other-case")

    with pytest.raises(FemExtractionError, match="case and result provenance"):
        replace(bundle, fields=(foreign, *bundle.fields[1:]))


def test_extracted_artifact_requires_exact_runtime_type_in_bundle(
    tmp_path: Path,
) -> None:
    bundle = _extract(tmp_path)

    with pytest.raises(FemExtractionError, match="ExtractedArtifact"):
        replace(bundle, fields=(object(),))  # type: ignore[arg-type]

    assert isinstance(bundle.fields[0], ExtractedArtifact)
