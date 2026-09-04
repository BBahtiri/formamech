from __future__ import annotations

from pathlib import Path

from campaigns.dogbone_v1.s10_requests import (
    DOGBONE_S10_LOAD_POLICY_ID,
    build_dogbone_s10_public_requests,
    build_dogbone_s10_request_plan,
)
from piginot.contracts import ExecutionProfile, Unit, hash_spec
from piginot.doe import (
    CPEnvelope,
    EvidenceRole,
    PublishedCaseManifest,
    PublishedCaseStatus,
)


def test_s10_public_source_is_exactly_the_cp_envelope_bound_manifest() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    campaign_root = repository_root / "artifacts" / "campaigns" / "dogbone-v1"
    checkpoint = CPEnvelope.from_bytes(
        (campaign_root / "CP-ENVELOPE.json").read_bytes()
    )
    manifest = PublishedCaseManifest.from_bytes(
        (campaign_root / "public-case-manifest.json").read_bytes()
    )

    assert checkpoint.campaign_hash == (
        "8f58fe30803e2bd147f863ac000e1e89a19092c8b347df6b4d74f38487d96d46"
    )
    assert checkpoint.public_case_manifest_hash == (
        "ce95c03f5b22308a7aa9bf627966a6b21300a4e7b0272d3affa7dd503af564a9"
    )
    assert manifest.content_hash == checkpoint.public_case_manifest_hash
    assert len(manifest.cases) == 376
    assert (
        sum(case.status is PublishedCaseStatus.ACTIVE for case in manifest.cases) == 344
    )
    assert (
        sum(case.status is PublishedCaseStatus.RESERVE for case in manifest.cases) == 32
    )
    assert all(case.role is not EvidenceRole.HOLDOUT for case in manifest.cases)

    role_status_counts = {
        (role, status): sum(
            case.role is role and case.status is status for case in manifest.cases
        )
        for role in (
            EvidenceRole.PILOT,
            EvidenceRole.TRAIN,
            EvidenceRole.DEVELOPMENT,
        )
        for status in (PublishedCaseStatus.ACTIVE, PublishedCaseStatus.RESERVE)
    }
    assert role_status_counts == {
        (EvidenceRole.PILOT, PublishedCaseStatus.ACTIVE): 24,
        (EvidenceRole.PILOT, PublishedCaseStatus.RESERVE): 4,
        (EvidenceRole.TRAIN, PublishedCaseStatus.ACTIVE): 256,
        (EvidenceRole.TRAIN, PublishedCaseStatus.RESERVE): 20,
        (EvidenceRole.DEVELOPMENT, PublishedCaseStatus.ACTIVE): 64,
        (EvidenceRole.DEVELOPMENT, PublishedCaseStatus.RESERVE): 8,
    }


def _profile() -> ExecutionProfile:
    return ExecutionProfile(
        python_version="3.12.0",
        library_versions={"numpy": "2.2.6", "scikit-fem": "11.0.0"},
        blas_threads=1,
        device="cpu",
        dtype="float64",
        rng_algorithm="none",
        seed=0,
    )


def test_s10_materializes_exact_public_active_requests_from_ratified_sources() -> None:
    requests = build_dogbone_s10_public_requests(execution_profile=_profile())
    repository_root = Path(__file__).resolve().parents[2]
    manifest = PublishedCaseManifest.from_bytes(
        (
            repository_root
            / "artifacts"
            / "campaigns"
            / "dogbone-v1"
            / "public-case-manifest.json"
        ).read_bytes()
    )
    active = tuple(
        sorted(
            (
                case
                for case in manifest.cases
                if case.status is PublishedCaseStatus.ACTIVE
            ),
            key=lambda case: case.case_id,
        )
    )

    assert len(requests) == 344
    assert tuple(request.request_id for request in requests) == tuple(
        sorted(request.request_id for request in requests)
    )
    assert all("reserve" not in request.request_id.casefold() for request in requests)
    assert len({request.mesh.mesh_id for request in requests}) == len(requests)

    first = requests[0]
    length = active[0].primitive_coordinates["L_total"]
    strain = active[0].primitive_coordinates["nominal_axial_strain"]
    right = next(
        condition
        for condition in first.boundary_conditions
        if condition.name == "right_grip_displacement"
    )

    assert first.material.name == "dogbone-pinned-reference-material-v1"
    assert first.material.model == "finite_strain_plane_stress_neo_hookean"
    assert first.material.properties["elastic_modulus"].value == 760.0
    assert first.material.properties["poisson_ratio"].value == 0.23
    assert first.config.element_order == 2
    assert first.config.load_steps == 64
    assert first.config.solver_tolerance == 1.0e-10
    assert first.config.max_newton_iterations == 20
    assert first.config.plane_stress_max_iterations == 32
    assert first.config.max_dofs == 100_000
    assert right.values[0].unit is Unit.MM
    assert right.values[0].value == strain * length / 2.0
    assert DOGBONE_S10_LOAD_POLICY_ID == "nominal-axial-strain-times-half-length-v1"


def test_s10_request_plan_binds_the_materialized_requests_to_cp_envelope() -> None:
    requests = build_dogbone_s10_public_requests(execution_profile=_profile())
    plan = build_dogbone_s10_request_plan(execution_profile=_profile())

    assert len(plan.entries) == len(requests)
    assert plan.campaign_hash == (
        "8f58fe30803e2bd147f863ac000e1e89a19092c8b347df6b4d74f38487d96d46"
    )
    assert plan.public_case_manifest_hash == (
        "ce95c03f5b22308a7aa9bf627966a6b21300a4e7b0272d3affa7dd503af564a9"
    )
    assert plan.execution_profile_hash == hash_spec(_profile(), "content")
    assert tuple(entry.case_id for entry in plan.entries) == tuple(
        request.request_id for request in requests
    )
