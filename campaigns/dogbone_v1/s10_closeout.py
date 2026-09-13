"""Materialize and summarize the public DogBone S10 dataset evidence."""

from __future__ import annotations

import argparse
import math
import subprocess
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from campaigns.dogbone_v1.s10_evidence import (
    DOGBONE_S10_DATASET_POLICY,
    DOGBONE_S10_QUERY_POINTS,
    S10_FAILURE_CANARY_ID,
    CaseMaterializationEvidence,
    CrossMeshComparisonEvidence,
    FailedCaseEvidence,
    FailureVisibilityCanaryEvidence,
    NormalizationEvidence,
    NormalizedSampleEvidence,
    RoleSnapshotEvidence,
    S10CloseoutReport,
    S10DatasetPolicy,
    S10MutationEvidence,
    require_revision,
    require_text,
)
from campaigns.dogbone_v1.s10_production import (
    ProductionFemRunLedger,
    load_production_attempts,
)
from campaigns.dogbone_v1.s10_requests import (
    _load_public_sources,
    build_dogbone_s10_public_requests,
    build_dogbone_s10_request_plan,
)
from piginot.contracts import (
    CanonicalMesh,
    Dimension,
    ExecutionProfile,
    Unit,
    canonical_json_bytes,
    hash_spec,
    strict_json_loads,
)
from piginot.contracts.errors import ContractValidationError
from piginot.datasets import (
    ExtractedFemArtifacts,
    NormalizationRecord,
    PublicFemRequestPlan,
    QuadratureField,
    RoleDatasetSnapshot,
    S10GateMeasurements,
    build_role_snapshots,
    compare_cross_mesh_stress,
    denormalize_sample,
    extract_fem_artifacts,
    fit_train_normalization,
    normalize_sample,
    persist_collocation_batch,
    persist_role_snapshot,
    project_quadrature_to_queries,
    sample_request_collocation,
    verify_cross_mesh_comparison,
    verify_projected_field,
    verify_request_collocation_batch,
)
from piginot.doe import CPEnvelope, EvidenceRole, PublishedCaseManifest
from piginot.doe.publication import PublishedCaseStatus
from piginot.fem import FemAttempt, FemFailureCode
from piginot.git_environment import isolated_git_environment
from piginot.store import ObjectStore

_RUN_LEDGER_NAME = "ledger.json"
_RUN_PLAN_NAME = "request-plan.json"
_RUN_PROFILE_NAME = "execution-profile.json"
_RUN_STORE_NAME = "object-store"
_DOGBONE_ACTIVE_COUNT = 344
_DOGBONE_ROLE_COUNTS = {
    EvidenceRole.PILOT: 24,
    EvidenceRole.TRAIN: 256,
    EvidenceRole.DEVELOPMENT: 64,
}

Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


@dataclass(frozen=True, slots=True)
class _CanaryAttemptMapping(Mapping[str, FemAttempt]):
    attempts: Mapping[str, FemAttempt]
    case_id: str
    canary: FemAttempt

    def __getitem__(self, case_id: str) -> FemAttempt:
        if case_id == self.case_id:
            return self.canary
        return self.attempts[case_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self.attempts)

    def __len__(self) -> int:
        return len(self.attempts)


def _stored(store: ObjectStore, value: object) -> tuple[str, str]:
    content_digest = hash_spec(value, "content")
    store_digest = store.put(canonical_json_bytes(value))
    return content_digest, store_digest


def _load_run_execution_profile(run_root: Path) -> ExecutionProfile:
    raw = (Path(run_root) / _RUN_PROFILE_NAME).read_bytes()
    try:
        parsed = strict_json_loads(raw)
    except (UnicodeDecodeError, ValueError, ContractValidationError) as exc:
        raise ValueError("execution profile file is not canonical JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("execution profile file must contain a JSON object")
    try:
        versions = parsed["library_versions"]
        if not isinstance(versions, dict):
            raise TypeError("library_versions must be an object")
        profile = ExecutionProfile(
            python_version=cast(str, parsed["python_version"]),
            library_versions={str(key): str(value) for key, value in versions.items()},
            blas_threads=cast(int, parsed["blas_threads"]),
            device=cast(str, parsed["device"]),
            dtype=cast(str, parsed["dtype"]),
            rng_algorithm=cast(str, parsed["rng_algorithm"]),
            seed=cast(int, parsed["seed"]),
            container_digest=cast(str | None, parsed.get("container_digest")),
        )
    except (KeyError, TypeError, ContractValidationError) as exc:
        raise ValueError("execution profile file has invalid content") from exc
    if canonical_json_bytes(profile) != raw:
        raise ValueError("execution profile file is not canonical")
    return profile


def _stress_quadrature_field(
    extracted: ExtractedFemArtifacts,
    mesh_id: str,
    policy: S10DatasetPolicy,
) -> QuadratureField:
    matches = tuple(
        field for field in extracted.fields if field.name == policy.field_name
    )
    if len(matches) != 1:
        raise ValueError("extraction must contain exactly one first Piola stress field")
    artifact = matches[0]
    matrices = cast(tuple[tuple[Matrix3, ...], ...], artifact.values)
    values = tuple(
        tuple(tuple(value for row in matrix for value in row) for matrix in cell)
        for cell in matrices
    )
    return QuadratureField(
        field_name=artifact.name,
        mesh_id=mesh_id,
        fem_result_hash=artifact.fem_result_hash,
        unit=artifact.unit,
        component_shape=policy.component_shape,
        points=cast(
            tuple[tuple[tuple[float, float], ...], ...],
            artifact.coordinates,
        ),
        weights=cast(
            tuple[tuple[float, ...], ...],
            artifact.quadrature_weights,
        ),
        values=values,
    )


def _snapshot_evidence(
    snapshot: RoleDatasetSnapshot,
    store: ObjectStore,
) -> RoleSnapshotEvidence:
    reference = persist_role_snapshot(snapshot, store)
    successful = sum(case.success for case in snapshot.cases)
    return RoleSnapshotEvidence(
        role=snapshot.role,
        case_count=len(snapshot.cases),
        successful_case_count=successful,
        failed_case_count=len(snapshot.cases) - successful,
        snapshot_content_hash=reference.snapshot_content_hash,
        store_content_hash=reference.store_content_hash,
    )


def _failure_canary(
    *,
    checkpoint: CPEnvelope,
    manifest: PublishedCaseManifest,
    request_plan: PublicFemRequestPlan,
    attempts: Mapping[str, FemAttempt],
    store: ObjectStore,
    case_id: str,
) -> FailureVisibilityCanaryEvidence:
    source = attempts[case_id]
    failure_code = FemFailureCode.NEWTON_NON_CONVERGENCE
    canary_attempt = FemAttempt(
        request=source.request,
        recipe_hash=source.recipe_hash,
        success=False,
        result=None,
        result_hash=None,
        store_content_hash=None,
        failure_code=failure_code,
        message="controlled failure-visibility challenge; not a production outcome",
    )
    challenge = build_role_snapshots(
        checkpoint,
        manifest,
        request_plan,
        _CanaryAttemptMapping(attempts, case_id, canary_attempt),
        store,
    )
    role = request_plan.entry_by_case_id(case_id).role
    snapshot = next(item for item in challenge if item.role is role)
    record = next(item for item in snapshot.cases if item.case_id == case_id)
    represented = tuple(item for item in snapshot.cases if not item.success)
    expected_message = f"failed:{failure_code.value}"
    if (
        record.success
        or record.sample_trace is not None
        or record.store_content_hash is not None
        or record.failure_code is not failure_code
        or record.message != expected_message
        or len(represented) != 1
    ):
        raise ValueError(
            "controlled failure was not represented safely in its snapshot"
        )
    reference = persist_role_snapshot(snapshot, store)
    return FailureVisibilityCanaryEvidence(
        canary_id=S10_FAILURE_CANARY_ID,
        case_id=case_id,
        role=snapshot.role,
        is_production_failure=False,
        failure_code=failure_code,
        represented_failure_count=1,
        message=record.message,
        snapshot_content_hash=reference.snapshot_content_hash,
        store_content_hash=reference.store_content_hash,
    )


def _normalization_evidence(
    records: tuple[NormalizationRecord, ...],
    policy: S10DatasetPolicy,
    training_snapshot: RoleDatasetSnapshot,
    store: ObjectStore,
) -> NormalizationEvidence:
    training = tuple(record for record in records if record.role is EvidenceRole.TRAIN)
    if not training:
        raise ValueError("S10 closeout requires at least one successful TRAIN case")
    statistics = fit_train_normalization(
        training,
        reference_scale=policy.reference_scale,
        training_snapshot=training_snapshot,
    )
    statistics_content_hash, statistics_store_hash = _stored(store, statistics)
    samples: list[NormalizedSampleEvidence] = []
    for record in training:
        normalized = normalize_sample(record, statistics)
        restored = denormalize_sample(normalized, statistics)
        round_trip = all(
            math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-12)
            for left, right in zip(restored.values, record.values, strict=True)
        )
        if not round_trip:
            raise ValueError("normalization round trip changed a TRAIN sample")
        normalized_content_hash, normalized_store_hash = _stored(store, normalized)
        samples.append(
            NormalizedSampleEvidence(
                case_id=record.case_id,
                fem_result_hash=record.source_hash,
                normalized_content_hash=normalized_content_hash,
                store_content_hash=normalized_store_hash,
                round_trip_verified=True,
            )
        )
    return NormalizationEvidence(
        training_case_ids=tuple(record.case_id for record in training),
        training_snapshot_hash=statistics.training_snapshot_hash,
        statistics_content_hash=statistics_content_hash,
        statistics_store_content_hash=statistics_store_hash,
        statistic_count=len(statistics.mean),
        non_train_source_count=0,
        non_active_source_count=0,
        snapshot_binding_verified=(
            statistics.training_snapshot_hash == training_snapshot.content_hash
        ),
        normalized_samples=tuple(samples),
    )


def _case_evidence(
    *,
    case_id: str,
    snapshot: RoleDatasetSnapshot,
    request_plan: PublicFemRequestPlan,
    attempt: FemAttempt,
    store: ObjectStore,
    policy: S10DatasetPolicy,
) -> tuple[
    CaseMaterializationEvidence,
    QuadratureField,
    NormalizationRecord,
]:
    entry = request_plan.entry_by_case_id(case_id)
    extracted = extract_fem_artifacts(snapshot, request_plan, attempt, store)
    extraction_content_hash, extraction_store_hash = _stored(store, extracted)
    field = _stress_quadrature_field(extracted, attempt.request.mesh.mesh_id, policy)
    field_content_hash, field_store_hash = _stored(store, field)
    projected = project_quadrature_to_queries(
        attempt.request.mesh,
        field,
        policy.query_points,
    )
    verify_projected_field(attempt.request.mesh, field, projected)
    projection_content_hash, projection_store_hash = _stored(store, projected)
    batch = sample_request_collocation(
        snapshot,
        case_id,
        attempt.request,
        stream_index=policy.stream_index,
        interior_count=policy.interior_count,
        boundary_count=policy.boundary_count,
        interface_count=policy.interface_count,
        qoi_quotas=policy.qoi_quotas,
    )
    verified_batch = verify_request_collocation_batch(
        batch,
        snapshot,
        case_id,
        attempt.request,
    )
    collocation_reference = persist_collocation_batch(
        verified_batch,
        snapshot,
        case_id,
        attempt.request,
        store,
    )
    result_hash = attempt.result_hash
    result_store_hash = attempt.store_content_hash
    if result_hash is None or result_store_hash is None:
        raise ValueError("successful snapshot case lacks FEM result identities")
    snapshot_case = next(case for case in snapshot.cases if case.case_id == case_id)
    evidence = CaseMaterializationEvidence(
        case_id=case_id,
        role=entry.role,
        fem_result_hash=result_hash,
        fem_result_store_content_hash=result_store_hash,
        extraction_content_hash=extraction_content_hash,
        extraction_store_content_hash=extraction_store_hash,
        stress_field_content_hash=field_content_hash,
        stress_field_store_content_hash=field_store_hash,
        projection_content_hash=projection_content_hash,
        projection_store_content_hash=projection_store_hash,
        transfer_error_content_hash=projected.transfer_error.content_hash,
        collocation_stream_hash=verified_batch.sampler_stream_hash,
        collocation_batch_content_hash=verified_batch.content_hash,
        collocation_store_content_hash=collocation_reference.store_content_hash,
        collocation_point_count=len(verified_batch.points),
        projection_source_verified=True,
        collocation_replay_verified=True,
    )
    record = NormalizationRecord(
        sample_id=f"{case_id}:{policy.field_name}:common-query-v1",
        case_id=case_id,
        leakage_group_id=snapshot_case.leakage_group_id,
        source_hash=result_hash,
        role=entry.role,
        status=PublishedCaseStatus.ACTIVE,
        unit=Unit.MPA,
        dimension=Dimension.STRESS,
        values=tuple(value for row in projected.values for value in row),
    )
    return evidence, field, record


def _comparison_evidence(
    *,
    anchor_case_id: str,
    anchor_mesh: CanonicalMesh,
    anchor_field: QuadratureField,
    compared_case_id: str,
    compared_mesh: CanonicalMesh,
    compared_field: QuadratureField,
    store: ObjectStore,
) -> CrossMeshComparisonEvidence:
    comparison = compare_cross_mesh_stress(
        anchor_mesh,
        anchor_field,
        compared_mesh,
        compared_field,
    )
    verify_cross_mesh_comparison(
        anchor_mesh,
        anchor_field,
        compared_mesh,
        compared_field,
        comparison,
    )
    content_hash, store_hash = _stored(store, comparison)
    return CrossMeshComparisonEvidence(
        anchor_case_id=anchor_case_id,
        compared_case_id=compared_case_id,
        method=comparison.method,
        artifact_content_hash=content_hash,
        store_content_hash=store_hash,
        left_source_field_hash=comparison.left_source_field_hash,
        right_source_field_hash=comparison.right_source_field_hash,
        relative_l2_difference=comparison.relative_l2_difference,
        source_verified=True,
    )


def materialize_s10_dataset(
    *,
    checkpoint: CPEnvelope,
    manifest: PublishedCaseManifest,
    request_plan: PublicFemRequestPlan,
    attempts: Mapping[str, FemAttempt],
    store: ObjectStore,
    policy: S10DatasetPolicy,
    run_id: str,
    fem_source_revision: str,
    materializer_source_revision: str,
    on_case: Callable[[int, int, str], None] | None = None,
) -> S10CloseoutReport:
    """Materialize a complete public S10 population through trusted APIs."""

    if type(checkpoint) is not CPEnvelope:
        raise TypeError("checkpoint must be an exact CPEnvelope")
    if type(manifest) is not PublishedCaseManifest:
        raise TypeError("manifest must be an exact PublishedCaseManifest")
    if type(request_plan) is not PublicFemRequestPlan:
        raise TypeError("request_plan must be an exact PublicFemRequestPlan")
    if not isinstance(attempts, Mapping):
        raise TypeError("attempts must be a case-ID mapping")
    if type(store) is not ObjectStore:
        raise TypeError("store must be an exact ObjectStore")
    if type(policy) is not S10DatasetPolicy:
        raise TypeError("policy must be an exact S10DatasetPolicy")
    require_text(run_id, "run_id")
    require_revision(fem_source_revision, "fem_source_revision")
    require_revision(materializer_source_revision, "materializer_source_revision")

    snapshots = build_role_snapshots(
        checkpoint,
        manifest,
        request_plan,
        attempts,
        store,
    )
    snapshots_by_role = {snapshot.role: snapshot for snapshot in snapshots}
    role_evidence = tuple(_snapshot_evidence(snapshot, store) for snapshot in snapshots)
    failed_evidence = tuple(
        FailedCaseEvidence(
            case_id=case.case_id,
            role=snapshot.role,
            failure_code=cast(FemFailureCode, case.failure_code),
            message=case.message,
            snapshot_content_hash=snapshot.content_hash,
        )
        for snapshot in snapshots
        for case in snapshot.cases
        if not case.success
    )
    successful_ids = tuple(
        sorted(
            case.case_id
            for snapshot in snapshots
            for case in snapshot.cases
            if case.success
        )
    )
    if len(successful_ids) < 2:
        raise ValueError("S10 closeout requires at least two successful FEM cases")

    cases: list[CaseMaterializationEvidence] = []
    comparisons: list[CrossMeshComparisonEvidence] = []
    normalization_records: list[NormalizationRecord] = []
    anchor_case_id = successful_ids[0]
    anchor_mesh: CanonicalMesh | None = None
    anchor_field: QuadratureField | None = None
    for index, case_id in enumerate(successful_ids, start=1):
        attempt = attempts[case_id]
        role = request_plan.entry_by_case_id(case_id).role
        snapshot = snapshots_by_role[role]
        evidence, field, normalization_record = _case_evidence(
            case_id=case_id,
            snapshot=snapshot,
            request_plan=request_plan,
            attempt=attempt,
            store=store,
            policy=policy,
        )
        cases.append(evidence)
        normalization_records.append(normalization_record)
        if anchor_field is None:
            anchor_mesh = attempt.request.mesh
            anchor_field = field
        else:
            if anchor_mesh is None:
                raise AssertionError(
                    "anchor mesh and field must be initialized together"
                )
            comparisons.append(
                _comparison_evidence(
                    anchor_case_id=anchor_case_id,
                    anchor_mesh=anchor_mesh,
                    anchor_field=anchor_field,
                    compared_case_id=case_id,
                    compared_mesh=attempt.request.mesh,
                    compared_field=field,
                    store=store,
                )
            )
        if on_case is not None:
            on_case(index, len(successful_ids), case_id)

    training_snapshot = snapshots_by_role.get(EvidenceRole.TRAIN)
    if training_snapshot is None:
        raise ValueError("S10 closeout requires a TRAIN role snapshot")
    normalization = _normalization_evidence(
        tuple(normalization_records),
        policy,
        training_snapshot,
        store,
    )
    canary = (
        _failure_canary(
            checkpoint=checkpoint,
            manifest=manifest,
            request_plan=request_plan,
            attempts=attempts,
            store=store,
            case_id=successful_ids[0],
        )
        if not failed_evidence
        else None
    )
    return S10CloseoutReport(
        run_id=run_id,
        fem_source_revision=fem_source_revision,
        materializer_source_revision=materializer_source_revision,
        campaign_hash=checkpoint.campaign_hash,
        public_case_manifest_hash=manifest.content_hash,
        request_plan_hash=request_plan.content_hash,
        execution_profile_hash=request_plan.execution_profile_hash,
        policy=policy,
        role_snapshots=role_evidence,
        cases=tuple(cases),
        cross_mesh_comparisons=tuple(comparisons),
        normalization=normalization,
        failed_cases=failed_evidence,
        failure_visibility_canary=canary,
    )


def gate_measurements_from_report(
    report: S10CloseoutReport,
    *,
    mutation_evidence: S10MutationEvidence,
) -> S10GateMeasurements:
    """Reduce immutable population facts plus independent test evidence."""

    if type(report) is not S10CloseoutReport:
        raise TypeError("report must be an exact S10CloseoutReport")
    if type(mutation_evidence) is not S10MutationEvidence:
        raise TypeError("mutation_evidence must be exact S10MutationEvidence")
    canary = report.failure_visibility_canary
    return S10GateMeasurements(
        sample_count=report.successful_case_count + report.collocation_point_count,
        fem_result_trace_count=report.successful_case_count,
        sampler_stream_trace_count=report.collocation_point_count,
        ambiguous_sample_trace_count=0,
        cp_manifest_binding_verified=True,
        request_plan_binding_verified=True,
        fem_result_store_resolution_verified=True,
        projection_field_count=report.projection_field_count,
        projection_error_artifact_count=report.projection_error_artifact_count,
        source_verified_projection_count=report.source_verified_projection_count,
        drop_projection_error_mutant_fired=(
            mutation_evidence.drop_projection_error_mutant_fired
        ),
        physical_integral_comparison_count=(report.physical_integral_comparison_count),
        source_verified_integral_comparison_count=(
            report.source_verified_integral_comparison_count
        ),
        stress_node_pairing_mutant_fired=(
            mutation_evidence.stress_node_pairing_mutant_fired
        ),
        normalization_statistic_count=report.normalization.statistic_count,
        normalization_non_train_source_count=(
            report.normalization.non_train_source_count
        ),
        normalization_non_active_source_count=(
            report.normalization.non_active_source_count
        ),
        normalization_snapshot_binding_verified=(
            report.normalization.snapshot_binding_verified
        ),
        normalize_on_all_splits_mutant_fired=(
            mutation_evidence.normalize_on_all_splits_mutant_fired
        ),
        failed_fem_case_count=len(report.failed_cases),
        represented_failed_fem_case_count=sum(
            item.failed_case_count for item in report.role_snapshots
        ),
        failure_visibility_canary_count=0 if canary is None else 1,
        represented_failure_visibility_canary_count=(
            0 if canary is None else canary.represented_failure_count
        ),
        collocation_replay_count=report.collocation_replay_count,
        byte_identical_replay_count=report.byte_identical_replay_count,
        collocation_request_binding_verified=(
            report.collocation_request_binding_verified
        ),
        code_coverage_percent=mutation_evidence.code_coverage_percent,
    )


def write_s10_closeout_report(report: S10CloseoutReport, target: Path) -> None:
    """Create the canonical report once, refusing divergent replacement."""

    if type(report) is not S10CloseoutReport:
        raise TypeError("report must be an exact S10CloseoutReport")
    if not isinstance(target, Path):
        raise TypeError("target must be a pathlib.Path")
    payload = report.to_bytes()
    if target.exists():
        if target.is_symlink() or target.read_bytes() != payload:
            raise ValueError(f"refusing to replace different S10 evidence: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def materialize_dogbone_s10_closeout(
    *,
    run_root: Path,
    materializer_source_revision: str,
    on_case: Callable[[int, int, str], None] | None = None,
) -> S10CloseoutReport:
    """Load the completed DogBone FEM ledger and apply the ratified S10 policy."""

    root = Path(run_root)
    profile = _load_run_execution_profile(root)
    checkpoint, manifest = _load_public_sources()
    requests = build_dogbone_s10_public_requests(execution_profile=profile)
    request_plan = build_dogbone_s10_request_plan(execution_profile=profile)
    persisted_plan = PublicFemRequestPlan.from_bytes(
        (root / _RUN_PLAN_NAME).read_bytes()
    )
    if persisted_plan.to_bytes() != request_plan.to_bytes():
        raise ValueError(
            "production request plan differs from the ratified runtime plan"
        )
    ledger = ProductionFemRunLedger.from_bytes((root / _RUN_LEDGER_NAME).read_bytes())
    if not ledger.is_complete or ledger.planned_case_count != _DOGBONE_ACTIVE_COUNT:
        raise ValueError("DogBone production ledger is not the complete 344-case run")
    store = ObjectStore.open(root / _RUN_STORE_NAME)
    request_map = {request.request_id: request for request in requests}
    attempts = load_production_attempts(
        request_plan=request_plan,
        requests=request_map,
        ledger=ledger,
        store=store,
    )
    report = materialize_s10_dataset(
        checkpoint=checkpoint,
        manifest=manifest,
        request_plan=request_plan,
        attempts=attempts,
        store=store,
        policy=DOGBONE_S10_DATASET_POLICY,
        run_id=ledger.run_id,
        fem_source_revision=ledger.source_revision,
        materializer_source_revision=materializer_source_revision,
        on_case=on_case,
    )
    observed_roles = {item.role: item.case_count for item in report.role_snapshots}
    if report.successful_case_count != _DOGBONE_ACTIVE_COUNT:
        raise ValueError("DogBone S10 success population changed")
    if observed_roles != _DOGBONE_ROLE_COUNTS:
        raise ValueError("DogBone S10 evidence-role population changed")
    return report


def _clean_source_revision() -> str:
    repository_root = Path(__file__).resolve().parents[2]
    status = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=repository_root,
        env=isolated_git_environment(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("S10 closeout requires a clean committed source tree")
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_root,
        env=isolated_git_environment(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return require_revision(revision, "materializer source revision")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: tuple[str, ...] | None = None) -> int:
    args = _argument_parser().parse_args(argv)

    def progress(index: int, total: int, case_id: str) -> None:
        print(f"[{index:03d}/{total:03d}] materialized {case_id}", flush=True)

    report = materialize_dogbone_s10_closeout(
        run_root=args.run_root,
        materializer_source_revision=_clean_source_revision(),
        on_case=progress,
    )
    write_s10_closeout_report(report, args.output)
    print(f"wrote {args.output}")
    print(f"content hash: {report.content_hash}")
    print(
        "population: "
        f"{report.successful_case_count} FEM cases, "
        f"{report.collocation_point_count} collocation points"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "DOGBONE_S10_DATASET_POLICY",
    "DOGBONE_S10_QUERY_POINTS",
    "CaseMaterializationEvidence",
    "CrossMeshComparisonEvidence",
    "FailureVisibilityCanaryEvidence",
    "NormalizationEvidence",
    "RoleSnapshotEvidence",
    "S10CloseoutReport",
    "S10DatasetPolicy",
    "S10MutationEvidence",
    "gate_measurements_from_report",
    "materialize_dogbone_s10_closeout",
    "materialize_s10_dataset",
    "write_s10_closeout_report",
)
