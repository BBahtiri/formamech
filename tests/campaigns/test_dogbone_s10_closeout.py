from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pytest

import campaigns.dogbone_v1.s10_closeout as closeout
from campaigns.dogbone_v1.s10_closeout import (
    S10DatasetPolicy,
    S10MutationEvidence,
    gate_measurements_from_report,
    materialize_s10_dataset,
    write_s10_closeout_report,
)
from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    CanonicalQuantity,
    ExecutionProfile,
    MaterialSpec,
    Unit,
    canonical_json_bytes,
    hash_spec,
)
from piginot.datasets import (
    CollocationKind,
    CrossMeshMethod,
    GateStatus,
    SemanticCollocationQuota,
    build_public_fem_request_plan,
    evaluate_s10_gate,
)
from piginot.datasets.request_plan import PublicFemRequestPlan
from piginot.doe.checkpoint import (
    CampaignBinding,
    CampaignBudget,
    CPEnvelope,
    EvidenceReference,
    compute_campaign_hash,
)
from piginot.doe.commitment import HoldoutCommitment
from piginot.doe.publication import (
    PublishedCase,
    PublishedCaseManifest,
    PublishedCaseStatus,
)
from piginot.doe.roles import EvidenceRole
from piginot.doe.secret_material import CAMPAIGN_SECRET_GENERATOR_ID
from piginot.fem import (
    FemAttempt,
    FemFailureCode,
    FemRequest,
    FemResult,
    FemSolverConfig,
)
from piginot.store import ObjectStore

_RUN_ID = "fixture-s10-complete-run"
_FEM_SOURCE_REVISION = "a" * 40
_MATERIALIZER_SOURCE_REVISION = "b" * 40


@dataclass(frozen=True)
class _Fixture:
    checkpoint: CPEnvelope
    manifest: PublishedCaseManifest
    request_plan: PublicFemRequestPlan
    attempts: dict[str, FemAttempt]
    store: ObjectStore


def _profile() -> ExecutionProfile:
    return ExecutionProfile(
        python_version="3.12.0",
        library_versions={"numpy": "2.2.6"},
        blas_threads=1,
        device="cpu",
        dtype="float64",
        rng_algorithm="none",
        seed=0,
    )


def _mesh(case_id: str) -> CanonicalMesh:
    return CanonicalMesh(
        mesh_id=f"two-triangle-square-{case_id}",
        nodes=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        cells=((0, 1, 2), (0, 2, 3)),
        cell_type="triangle",
        boundary_facets=((0, 1), (1, 2), (2, 3), (3, 0)),
        semantic_tags={"side": (0, 1, 2, 3)},
        region_ids=(0, 0),
        unit=Unit.MM,
        orientation="ccw",
        topology_signature="two-triangle-square-v1",
        quality_metrics={"min_scaled_jacobian": 1.0},
        source_provenance={"fixture": "s10-closeout"},
    )


def _request(case_id: str) -> FemRequest:
    return FemRequest(
        request_id=case_id,
        mesh=_mesh(case_id),
        material=MaterialSpec(
            name="fixture-rubber",
            model="compressible_neo_hookean_plane_stress",
            properties={
                "elastic_modulus": CanonicalQuantity(100.0, Unit.MPA),
                "poisson_ratio": CanonicalQuantity(0.3, Unit.ONE),
            },
        ),
        boundary_conditions=(BoundaryConditionSpec("side-free", "side", "free"),),
        loads=(),
        qois=(),
        config=FemSolverConfig(element_order=1),
        execution_profile=_profile(),
    )


def _result(request: FemRequest, scale: float) -> FemResult:
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    stress = (
        (2.0 * scale, 0.1 * scale, 0.0),
        (0.1 * scale, 1.0 * scale, 0.0),
        (0.0, 0.0, 0.0),
    )
    return FemResult(
        request_id=request.request_id,
        mesh_id=request.mesh.mesh_id,
        element_order=1,
        displacement_dof_coordinates=request.mesh.nodes,
        displacement=((0.0, 0.0), (0.01, 0.0), (0.01, 0.01), (0.0, 0.01)),
        quadrature_points=(
            (((2.0 / 3.0), (1.0 / 3.0)),),
            (((1.0 / 3.0), (2.0 / 3.0)),),
        ),
        quadrature_weights=((0.5,), (0.5,)),
        first_piola_stress=((stress,), (stress,)),
        deformation_gradient=((identity,), (identity,)),
        recovered_first_piola_stress=(stress, stress, stress, stress),
        reactions={"side": (-scale, 0.0)},
        rigid_mode_multipliers=(),
        total_strain_energy=0.5 * scale,
        external_work=scale,
        total_potential_energy=-0.5 * scale,
        convergence_history=(),
        qois={},
    )


def _success(request: FemRequest, scale: float, store: ObjectStore) -> FemAttempt:
    result = _result(request, scale)
    store_hash = store.put(
        canonical_json_bytes({"domain": "content", "payload": result}),
        recipe_hash=request.recipe_hash,
    )
    return FemAttempt(
        request=request,
        recipe_hash=request.recipe_hash,
        success=True,
        result=result,
        result_hash=hash_spec(result, "content"),
        store_content_hash=store_hash,
        failure_code=None,
        message="succeeded",
    )


def _failure(request: FemRequest) -> FemAttempt:
    return FemAttempt(
        request=request,
        recipe_hash=request.recipe_hash,
        success=False,
        result=None,
        result_hash=None,
        store_content_hash=None,
        failure_code=FemFailureCode.RESOURCE_EXHAUSTION,
        message="backend detail that must be sanitized",
    )


def _case(case_id: str, role: EvidenceRole, stratum: int) -> PublishedCase:
    return PublishedCase(
        case_id=case_id,
        leakage_group_id=f"group-{case_id}",
        role=role,
        primitive_coordinates={"width": 1.0 + stratum},
        derived_coordinates={"net_section_ratio": 0.5 + 0.1 * stratum},
        strata={"width": stratum},
        status=PublishedCaseStatus.ACTIVE,
    )


def _manifest() -> PublishedCaseManifest:
    return PublishedCaseManifest(
        manifest_id="s10-closeout-public-cases",
        primitive_axes=("width",),
        derived_axes=("net_section_ratio",),
        stratum_axes=("width",),
        cases=(
            _case("pilot-1", EvidenceRole.PILOT, 0),
            _case("train-1", EvidenceRole.TRAIN, 1),
            _case("development-1", EvidenceRole.DEVELOPMENT, 2),
        ),
    )


def _checkpoint(manifest: PublishedCaseManifest) -> CPEnvelope:
    budget = CampaignBudget(
        pilot=1,
        train=1,
        development=1,
        holdout=64,
        legacy=0,
        reserve=0,
    )
    binding = CampaignBinding(
        compiled_family_hash="1" * 64,
        pilot_evidence=(
            EvidenceReference(
                evidence_id="s1-closeout",
                content_hash="2" * 64,
                role=EvidenceRole.LEGACY,
            ),
        ),
        executable_envelope_hash="3" * 64,
        private_doe_plan_hash="4" * 64,
        role_ledger_hash="5" * 64,
        replacement_policy_hash="6" * 64,
        public_case_manifest_hash=manifest.content_hash,
        doe_seed_commitment="7" * 64,
        secret_generator_id=CAMPAIGN_SECRET_GENERATOR_ID,
        budget=budget,
        derived_axes=("net_section_ratio",),
    )
    campaign_hash = compute_campaign_hash(binding)
    return CPEnvelope(
        campaign_hash=campaign_hash,
        compiled_family_hash=binding.compiled_family_hash,
        pilot_evidence=binding.pilot_evidence,
        executable_envelope_hash=binding.executable_envelope_hash,
        private_doe_plan_hash=binding.private_doe_plan_hash,
        role_ledger_hash=binding.role_ledger_hash,
        replacement_policy_hash=binding.replacement_policy_hash,
        public_case_manifest_hash=binding.public_case_manifest_hash,
        doe_seed_commitment=binding.doe_seed_commitment,
        secret_generator_id=binding.secret_generator_id,
        holdout_commitment=HoldoutCommitment(
            scheme="blake2b-256",
            campaign_hash=campaign_hash,
            n_holdout_points=64,
            commitment="8" * 64,
            created_utc="2026-09-01T12:00:00Z",
        ),
        budget=budget,
        derived_axes=binding.derived_axes,
        created_utc="2026-09-01T12:01:00Z",
    )


def _fixture(tmp_path: Path) -> _Fixture:
    manifest = _manifest()
    checkpoint = _checkpoint(manifest)
    requests = {
        case.case_id: _request(case.case_id)
        for case in manifest.cases
        if case.status is PublishedCaseStatus.ACTIVE
    }
    request_plan = build_public_fem_request_plan(checkpoint, manifest, requests)
    store = ObjectStore.create(tmp_path / "object-store")
    attempts = {
        case_id: _success(request, float(index), store)
        for index, (case_id, request) in enumerate(requests.items(), start=1)
    }
    return _Fixture(checkpoint, manifest, request_plan, attempts, store)


def _policy() -> S10DatasetPolicy:
    return S10DatasetPolicy(
        query_points=((0.75, 0.25),),
        interior_count=2,
        boundary_count=1,
        interface_count=0,
        qoi_quotas=(
            SemanticCollocationQuota(
                quota_id="stress-critical-side",
                kind=CollocationKind.BOUNDARY,
                count=1,
                semantic_tag="side",
            ),
        ),
        reference_scale=CanonicalQuantity(100.0, Unit.MPA),
    )


def _materialize(fixture: _Fixture):  # type: ignore[no-untyped-def]
    return materialize_s10_dataset(
        checkpoint=fixture.checkpoint,
        manifest=fixture.manifest,
        request_plan=fixture.request_plan,
        attempts=fixture.attempts,
        store=fixture.store,
        policy=_policy(),
        run_id=_RUN_ID,
        fem_source_revision=_FEM_SOURCE_REVISION,
        materializer_source_revision=_MATERIALIZER_SOURCE_REVISION,
    )


@pytest.mark.parametrize(
    ("changes", "exception", "match"),
    (
        ({"query_points": ()}, ValueError, "query_points"),
        ({"query_points": ((float("nan"), 0.0),)}, ValueError, "query_points"),
        ({"query_points": ((0.5, 0.5), (0.5, 0.5))}, ValueError, "unique"),
        ({"interior_count": -1}, ValueError, "non-negative"),
        (
            {
                "interior_count": 0,
                "boundary_count": 0,
                "interface_count": 0,
                "qoi_quotas": (),
            },
            ValueError,
            "at least one",
        ),
        ({"qoi_quotas": cast(Any, (object(),))}, TypeError, "qoi_quotas"),
        (
            {"reference_scale": CanonicalQuantity(1.0, Unit.MM)},
            ValueError,
            "MPa stress",
        ),
        ({"field_name": "cauchy_stress"}, ValueError, "field_name"),
        ({"component_shape": (2, 2)}, ValueError, "3x3"),
        ({"schema_version": "2.0"}, ValueError, "schema"),
    ),
)
def test_dataset_policy_rejects_ambiguous_or_unphysical_inputs(
    changes: dict[str, object],
    exception: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exception, match=match):
        replace(_policy(), **changes)


def test_mutation_evidence_validates_test_results() -> None:
    with pytest.raises(TypeError, match="boolean"):
        S10MutationEvidence(
            drop_projection_error_mutant_fired=cast(Any, 1),
            stress_node_pairing_mutant_fired=True,
            normalize_on_all_splits_mutant_fired=True,
            code_coverage_percent=90.0,
        )
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        S10MutationEvidence(True, True, True, 101.0)


def test_materializer_builds_deterministic_content_addressed_evidence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    report = _materialize(fixture)
    repeated = materialize_s10_dataset(
        checkpoint=fixture.checkpoint,
        manifest=fixture.manifest,
        request_plan=fixture.request_plan,
        attempts=dict(reversed(tuple(fixture.attempts.items()))),
        store=fixture.store,
        policy=_policy(),
        run_id=_RUN_ID,
        fem_source_revision=_FEM_SOURCE_REVISION,
        materializer_source_revision=_MATERIALIZER_SOURCE_REVISION,
    )

    assert repeated == report
    assert repeated.to_bytes() == report.to_bytes()
    assert report.run_id == _RUN_ID
    assert report.fem_source_revision == _FEM_SOURCE_REVISION
    assert report.materializer_source_revision == _MATERIALIZER_SOURCE_REVISION
    assert report.campaign_hash == fixture.checkpoint.campaign_hash
    assert report.public_case_manifest_hash == fixture.manifest.content_hash
    assert report.request_plan_hash == fixture.request_plan.content_hash
    assert report.to_bytes() == canonical_json_bytes(report)
    assert report.content_hash == hash_spec(report, "content")

    snapshots = {evidence.role: evidence for evidence in report.role_snapshots}
    assert set(snapshots) == {
        EvidenceRole.PILOT,
        EvidenceRole.TRAIN,
        EvidenceRole.DEVELOPMENT,
    }
    assert all(evidence.case_count == 1 for evidence in snapshots.values())
    assert all(evidence.successful_case_count == 1 for evidence in snapshots.values())
    assert all(evidence.failed_case_count == 0 for evidence in snapshots.values())
    assert len({evidence.snapshot_content_hash for evidence in snapshots.values()}) == 3
    assert all(
        fixture.store.get(evidence.store_content_hash)
        for evidence in snapshots.values()
    )

    assert report.successful_case_count == 3
    assert report.source_verified_extraction_count == 3
    assert report.projection_field_count == 3
    assert report.projection_error_artifact_count == 3
    assert report.source_verified_projection_count == 3
    assert all(fixture.store.get(digest) for digest in report.store_content_hashes)


def test_materializer_uses_physical_integrals_train_only_stats_and_replay(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    report = _materialize(fixture)

    assert len(report.cross_mesh_comparisons) == 2
    assert all(
        evidence.method is CrossMeshMethod.PHYSICAL_DOMAIN_INTEGRAL
        for evidence in report.cross_mesh_comparisons
    )
    assert report.physical_integral_comparison_count == 2
    assert report.source_verified_integral_comparison_count == 2

    train_snapshot = next(
        evidence
        for evidence in report.role_snapshots
        if evidence.role is EvidenceRole.TRAIN
    )
    assert report.normalization.training_case_ids == ("train-1",)
    assert report.normalization.training_snapshot_hash == (
        train_snapshot.snapshot_content_hash
    )
    assert report.normalization.statistic_count == 9
    assert report.normalization.non_train_source_count == 0
    assert report.normalization.non_active_source_count == 0
    assert report.normalization.snapshot_binding_verified is True
    assert report.normalization.round_trip_verified_count == 1

    assert report.collocation_point_count == 12
    assert report.collocation_replay_count == 3
    assert report.byte_identical_replay_count == 3
    assert report.collocation_request_binding_verified is True


def test_zero_natural_failure_run_has_an_explicit_nonproduction_canary(
    tmp_path: Path,
) -> None:
    report = _materialize(_fixture(tmp_path))

    assert report.actual_failed_case_ids == ()
    canary = report.failure_visibility_canary
    assert canary is not None
    assert canary.is_production_failure is False
    assert canary.failure_code is FemFailureCode.NEWTON_NON_CONVERGENCE
    assert canary.represented_failure_count == 1
    assert canary.message == "failed:NEWTON_NON_CONVERGENCE"
    assert canary.snapshot_content_hash
    assert canary.store_content_hash in report.store_content_hashes


def test_natural_failure_remains_visible_without_a_canary(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    attempts = dict(fixture.attempts)
    attempts["development-1"] = _failure(attempts["development-1"].request)

    report = materialize_s10_dataset(
        checkpoint=fixture.checkpoint,
        manifest=fixture.manifest,
        request_plan=fixture.request_plan,
        attempts=attempts,
        store=fixture.store,
        policy=_policy(),
        run_id=_RUN_ID,
        fem_source_revision=_FEM_SOURCE_REVISION,
        materializer_source_revision=_MATERIALIZER_SOURCE_REVISION,
    )
    measurements = gate_measurements_from_report(
        report,
        mutation_evidence=S10MutationEvidence(True, True, True, 90.0),
    )

    assert report.actual_failed_case_ids == ("development-1",)
    assert report.failed_cases[0].message == "failed:RESOURCE_EXHAUSTION"
    assert report.failure_visibility_canary is None
    assert measurements.failed_fem_case_count == 1
    assert measurements.represented_failed_fem_case_count == 1
    assert measurements.failure_visibility_canary_count == 0
    assert evaluate_s10_gate(measurements).verdict is GateStatus.PASS


def test_gate_measurements_reconcile_to_a_complete_passing_fixture(
    tmp_path: Path,
) -> None:
    report = _materialize(_fixture(tmp_path))
    measurements = gate_measurements_from_report(
        report,
        mutation_evidence=S10MutationEvidence(
            drop_projection_error_mutant_fired=True,
            stress_node_pairing_mutant_fired=True,
            normalize_on_all_splits_mutant_fired=True,
            code_coverage_percent=90.0,
        ),
    )

    assert measurements.sample_count == 15
    assert measurements.fem_result_trace_count == 3
    assert measurements.sampler_stream_trace_count == 12
    assert measurements.ambiguous_sample_trace_count == 0
    assert measurements.projection_field_count == 3
    assert measurements.projection_error_artifact_count == 3
    assert measurements.source_verified_projection_count == 3
    assert measurements.physical_integral_comparison_count == 2
    assert measurements.source_verified_integral_comparison_count == 2
    assert measurements.normalization_statistic_count == 9
    assert measurements.failed_fem_case_count == 0
    assert measurements.represented_failed_fem_case_count == 0
    assert measurements.failure_visibility_canary_count == 1
    assert measurements.represented_failure_visibility_canary_count == 1
    assert measurements.collocation_replay_count == 3
    assert measurements.byte_identical_replay_count == 3
    assert evaluate_s10_gate(measurements).verdict is GateStatus.PASS


def test_closeout_report_write_is_create_once_and_exact(tmp_path: Path) -> None:
    report = _materialize(_fixture(tmp_path / "fixture"))
    target = tmp_path / "evidence" / "s10-closeout.json"

    write_s10_closeout_report(report, target)
    write_s10_closeout_report(report, target)

    assert target.read_bytes() == report.to_bytes()
    target.write_bytes(report.to_bytes() + b"\n")
    with pytest.raises(ValueError, match="different"):
        write_s10_closeout_report(report, target)


def test_closeout_loads_exact_persisted_execution_profile(tmp_path: Path) -> None:
    profile = _profile()
    path = tmp_path / "execution-profile.json"
    path.write_bytes(canonical_json_bytes(profile))

    assert closeout._load_run_execution_profile(tmp_path) == profile

    path.write_bytes(canonical_json_bytes(profile) + b"\n")
    with pytest.raises(ValueError, match="canonical"):
        closeout._load_run_execution_profile(tmp_path)


def test_materializer_fails_closed_on_incomplete_or_mismatched_attempts(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    incomplete = dict(fixture.attempts)
    incomplete.pop("pilot-1")

    with pytest.raises(ValueError, match="attempt case IDs|active public cases"):
        materialize_s10_dataset(
            checkpoint=fixture.checkpoint,
            manifest=fixture.manifest,
            request_plan=fixture.request_plan,
            attempts=incomplete,
            store=fixture.store,
            policy=_policy(),
            run_id=_RUN_ID,
            fem_source_revision=_FEM_SOURCE_REVISION,
            materializer_source_revision=_MATERIALIZER_SOURCE_REVISION,
        )

    mismatched = dict(fixture.attempts)
    other_request = _request("other-case")
    mismatched["pilot-1"] = _success(other_request, 1.0, fixture.store)
    with pytest.raises(ValueError, match="request identity|request.*plan"):
        materialize_s10_dataset(
            checkpoint=fixture.checkpoint,
            manifest=fixture.manifest,
            request_plan=fixture.request_plan,
            attempts=mismatched,
            store=fixture.store,
            policy=_policy(),
            run_id=_RUN_ID,
            fem_source_revision=_FEM_SOURCE_REVISION,
            materializer_source_revision=_MATERIALIZER_SOURCE_REVISION,
        )
