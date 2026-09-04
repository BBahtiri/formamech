from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

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
from piginot.datasets.request_plan import (
    PublicFemRequestPlan,
    build_public_fem_request_plan,
)
from piginot.datasets.snapshots import (
    DatasetSnapshotValidationError,
    RoleDatasetSnapshot,
    build_role_snapshots,
    persist_role_snapshot,
)
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
from piginot.store.store import content_hash as store_content_hash


def _mesh() -> CanonicalMesh:
    return CanonicalMesh(
        mesh_id="triangle-v1",
        nodes=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        cells=((0, 1, 2),),
        cell_type="triangle",
        boundary_facets=((0, 1), (1, 2), (2, 0)),
        semantic_tags={"bottom": (0,), "side": (1, 2)},
        region_ids=(1,),
        unit=Unit.MM,
        orientation="ccw",
        topology_signature="triangle-v1",
        quality_metrics={"min_scaled_jacobian": 0.5},
        source_provenance={"fixture": "s10-snapshot"},
    )


def _request(case_id: str) -> FemRequest:
    return FemRequest(
        request_id=case_id,
        mesh=_mesh(),
        material=MaterialSpec(
            name="rubber",
            model="compressible_neo_hookean_plane_stress",
            properties={
                "elastic_modulus": CanonicalQuantity(100.0, Unit.MPA),
                "poisson_ratio": CanonicalQuantity(0.3, Unit.ONE),
            },
        ),
        boundary_conditions=(
            BoundaryConditionSpec(
                "fixed",
                "bottom",
                "dirichlet",
                ("x", "y"),
                (CanonicalQuantity(0.0, Unit.MM), CanonicalQuantity(0.0, Unit.MM)),
            ),
        ),
        loads=(),
        qois=(),
        config=FemSolverConfig(element_order=1),
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


def _result(request: FemRequest, **overrides: object) -> FemResult:
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    stress = ((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0))
    values: dict[str, object] = {
        "request_id": request.request_id,
        "mesh_id": request.mesh.mesh_id,
        "element_order": 1,
        "displacement_dof_coordinates": request.mesh.nodes,
        "displacement": ((0.0, 0.0), (0.1, 0.0), (0.0, 0.1)),
        "quadrature_points": (((1.0 / 3.0, 1.0 / 3.0),),),
        "quadrature_weights": ((0.5,),),
        "first_piola_stress": ((stress,),),
        "deformation_gradient": ((identity,),),
        "recovered_first_piola_stress": (stress, stress, stress),
        "reactions": {"bottom": (-1.0, 0.0)},
        "rigid_mode_multipliers": (),
        "total_strain_energy": 0.5,
        "external_work": 1.0,
        "total_potential_energy": -0.5,
        "convergence_history": (),
        "qois": {},
    }
    values.update(overrides)
    return FemResult(**values)  # type: ignore[arg-type]


def _success(case_id: str) -> FemAttempt:
    request = _request(case_id)
    result = _result(request)
    payload = canonical_json_bytes({"domain": "content", "payload": result})
    return FemAttempt(
        request=request,
        recipe_hash=request.recipe_hash,
        success=True,
        result=result,
        result_hash=hash_spec(result, "content"),
        store_content_hash=store_content_hash(payload),
        failure_code=None,
        message="succeeded",
    )


def _failure(case_id: str) -> FemAttempt:
    request = _request(case_id)
    return FemAttempt(
        request=request,
        recipe_hash=request.recipe_hash,
        success=False,
        result=None,
        result_hash=None,
        store_content_hash=None,
        failure_code=FemFailureCode.NEWTON_NON_CONVERGENCE,
        message="Newton iteration did not converge",
    )


def _case(
    case_id: str,
    role: EvidenceRole,
    *,
    status: PublishedCaseStatus = PublishedCaseStatus.ACTIVE,
) -> PublishedCase:
    return PublishedCase(
        case_id=case_id,
        leakage_group_id=f"group-{case_id}",
        role=role,
        primitive_coordinates={"width": 1.0},
        derived_coordinates={"net_section_ratio": 0.5},
        strata={"width": 0, "net_section_ratio": 1},
        status=status,
    )


def _manifest() -> PublishedCaseManifest:
    return PublishedCaseManifest(
        manifest_id="public-cases",
        primitive_axes=("width",),
        derived_axes=("net_section_ratio",),
        stratum_axes=("width", "net_section_ratio"),
        cases=(
            _case("train-1", EvidenceRole.TRAIN),
            _case("dev-1", EvidenceRole.DEVELOPMENT),
            _case(
                "reserve-1",
                EvidenceRole.TRAIN,
                status=PublishedCaseStatus.RESERVE,
            ),
        ),
    )


def _checkpoint(manifest: PublishedCaseManifest) -> CPEnvelope:
    budget = CampaignBudget(
        pilot=0,
        train=1,
        development=1,
        holdout=64,
        legacy=0,
        reserve=1,
    )
    binding = CampaignBinding(
        compiled_family_hash="a" * 64,
        pilot_evidence=(
            EvidenceReference(
                evidence_id="s1-closeout",
                content_hash="9" * 64,
                role=EvidenceRole.LEGACY,
            ),
        ),
        executable_envelope_hash="b" * 64,
        private_doe_plan_hash="c" * 64,
        role_ledger_hash="d" * 64,
        replacement_policy_hash="e" * 64,
        public_case_manifest_hash=manifest.content_hash,
        doe_seed_commitment="f" * 64,
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
            commitment="1" * 64,
            created_utc="2026-08-31T12:00:00Z",
        ),
        budget=budget,
        derived_axes=binding.derived_axes,
        created_utc="2026-08-31T12:01:00Z",
    )


def _build(tmp_path: Path) -> tuple[RoleDatasetSnapshot, ...]:
    manifest = _manifest()
    return _ingest(
        manifest,
        {"train-1": _success("train-1"), "dev-1": _failure("dev-1")},
        tmp_path,
    )


def _request_plan(manifest: PublishedCaseManifest) -> PublicFemRequestPlan:
    return build_public_fem_request_plan(
        _checkpoint(manifest),
        manifest,
        {"train-1": _request("train-1"), "dev-1": _request("dev-1")},
    )


def _put_success(store: ObjectStore, attempt: FemAttempt) -> None:
    assert attempt.result is not None
    store.put(
        canonical_json_bytes({"domain": "content", "payload": attempt.result}),
        recipe_hash=attempt.recipe_hash,
    )


def _ingest(
    manifest: PublishedCaseManifest,
    attempts: dict[str, FemAttempt],
    store_path: Path,
) -> tuple[RoleDatasetSnapshot, ...]:
    store = ObjectStore.create(store_path)
    for attempt in attempts.values():
        if attempt.success and attempt.result is not None:
            try:
                _put_success(store, attempt)
            except ValueError:
                # Invalid-result tests may deliberately be non-canonical; ingestion
                # must reject them before it attempts store resolution.
                pass
    return build_role_snapshots(
        _checkpoint(manifest),
        manifest,
        _request_plan(manifest),
        attempts,
        store,
    )


def test_snapshot_rejects_self_consistent_unplanned_request(tmp_path: Path) -> None:
    manifest = _manifest()
    checkpoint = _checkpoint(manifest)
    plan = _request_plan(manifest)
    original = _success("train-1")
    altered_request = replace(
        original.request,
        material=MaterialSpec(
            name="rubber",
            model="compressible_neo_hookean_plane_stress",
            properties={
                "elastic_modulus": CanonicalQuantity(999.0, Unit.MPA),
                "poisson_ratio": CanonicalQuantity(0.3, Unit.ONE),
            },
        ),
    )
    assert original.result is not None
    altered_result = replace(original.result, request_id=altered_request.request_id)
    altered = FemAttempt(
        request=altered_request,
        recipe_hash=altered_request.recipe_hash,
        success=True,
        result=altered_result,
        result_hash=hash_spec(altered_result, "content"),
        store_content_hash=store_content_hash(
            canonical_json_bytes({"domain": "content", "payload": altered_result})
        ),
        failure_code=None,
        message="backend details",
    )
    store = ObjectStore.create(tmp_path)
    _put_success(store, altered)

    with pytest.raises(DatasetSnapshotValidationError, match="request.*plan"):
        build_role_snapshots(
            checkpoint,
            manifest,
            plan,
            {"train-1": altered, "dev-1": _failure("dev-1")},
            store,
        )


def test_snapshot_requires_success_bytes_to_resolve_from_store(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    attempt = _success("train-1")
    empty_store = ObjectStore.create(tmp_path)

    with pytest.raises(DatasetSnapshotValidationError, match="ObjectStore"):
        build_role_snapshots(
            _checkpoint(manifest),
            manifest,
            _request_plan(manifest),
            {"train-1": attempt, "dev-1": _failure("dev-1")},
            empty_store,
        )


def test_snapshot_sanitizes_backend_failure_messages(tmp_path: Path) -> None:
    manifest = _manifest()
    train = _success("train-1")
    failed = replace(_failure("dev-1"), message="secret path C:/private/run")
    store = ObjectStore.create(tmp_path)
    _put_success(store, train)

    snapshots = build_role_snapshots(
        _checkpoint(manifest),
        manifest,
        _request_plan(manifest),
        {"train-1": train, "dev-1": failed},
        store,
    )

    assert snapshots[0].cases[0].message == "failed:NEWTON_NON_CONVERGENCE"
    assert b"private" not in canonical_json_bytes(snapshots[0])


def test_snapshots_are_per_role_immutable_and_keep_failed_cases_visible(
    tmp_path: Path,
) -> None:
    snapshots = _build(tmp_path)

    assert tuple(snapshot.role for snapshot in snapshots) == (
        EvidenceRole.DEVELOPMENT,
        EvidenceRole.TRAIN,
    )
    development, train = snapshots
    assert development.cases[0].case_id == "dev-1"
    assert development.cases[0].failure_code is FemFailureCode.NEWTON_NON_CONVERGENCE
    assert development.cases[0].sample_trace is None
    assert train.cases[0].leakage_group_id == "group-train-1"
    assert train.cases[0].role is EvidenceRole.TRAIN
    trace = train.cases[0].sample_trace
    assert trace is not None
    assert trace.fem_result_hash == _success("train-1").result_hash
    assert trace.sampler_stream_hash is None
    assert train.cases[0].store_content_hash != trace.fem_result_hash
    assert train.request_plan_hash == _request_plan(_manifest()).content_hash
    assert all(
        case.case_id != "reserve-1" for snapshot in snapshots for case in snapshot.cases
    )
    assert train.content_hash == hash_spec(train, "content")
    with pytest.raises(FrozenInstanceError):
        train.role = EvidenceRole.DEVELOPMENT  # type: ignore[misc]


def test_snapshot_identity_is_deterministic_under_input_permutation(
    tmp_path: object,
) -> None:
    manifest = _manifest()
    forward = {"train-1": _success("train-1"), "dev-1": _failure("dev-1")}
    reverse = dict(reversed(tuple(forward.items())))

    assert _ingest(manifest, forward, tmp_path / "forward") == _ingest(  # type: ignore[operator]
        manifest,
        reverse,
        tmp_path / "reverse",  # type: ignore[operator]
    )


def test_manifest_must_be_the_one_committed_by_cp_envelope(tmp_path: object) -> None:
    manifest = _manifest()
    altered = PublishedCaseManifest(
        manifest_id="altered",
        primitive_axes=manifest.primitive_axes,
        derived_axes=manifest.derived_axes,
        stratum_axes=manifest.stratum_axes,
        cases=manifest.cases,
    )

    with pytest.raises(DatasetSnapshotValidationError, match="manifest hash"):
        build_role_snapshots(
            _checkpoint(manifest),
            altered,
            _request_plan(manifest),
            {},
            ObjectStore.create(tmp_path),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("case_id", ("reserve-1", "unknown-1"))
def test_attempts_must_match_exactly_the_active_public_cases(
    case_id: str, tmp_path: object
) -> None:
    manifest = _manifest()
    attempts = {
        "train-1": _success("train-1"),
        "dev-1": _failure("dev-1"),
        case_id: _success(case_id),
    }

    with pytest.raises(DatasetSnapshotValidationError, match="attempt case IDs"):
        build_role_snapshots(
            _checkpoint(manifest),
            manifest,
            _request_plan(manifest),
            attempts,
            ObjectStore.create(tmp_path),  # type: ignore[arg-type]
        )


def test_holdout_role_is_rejected_even_if_a_manifest_object_is_tampered(
    tmp_path: object,
) -> None:
    manifest = _manifest()
    plan = _request_plan(manifest)
    object.__setattr__(manifest.cases[0], "role", EvidenceRole.HOLDOUT)

    with pytest.raises(DatasetSnapshotValidationError, match="HOLDOUT"):
        build_role_snapshots(
            _checkpoint(manifest),
            manifest,
            plan,
            {},
            ObjectStore.create(tmp_path),  # type: ignore[arg-type]
        )


def test_tampered_manifest_cannot_split_one_leakage_group_across_roles(
    tmp_path: object,
) -> None:
    manifest = _manifest()
    plan = _request_plan(manifest)
    object.__setattr__(
        manifest.cases[1],
        "leakage_group_id",
        manifest.cases[0].leakage_group_id,
    )

    with pytest.raises(DatasetSnapshotValidationError, match="leakage group"):
        build_role_snapshots(
            _checkpoint(manifest),
            manifest,
            plan,
            {},
            ObjectStore.create(tmp_path),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda attempt: replace(attempt, recipe_hash="0" * 64), "recipe_hash"),
        (lambda attempt: replace(attempt, result_hash="0" * 64), "result_hash"),
        (
            lambda attempt: replace(attempt, store_content_hash="0" * 64),
            "store_content_hash",
        ),
        (
            lambda attempt: replace(
                attempt,
                result=replace(attempt.result, request_id="another-case"),
                result_hash=hash_spec(
                    replace(attempt.result, request_id="another-case"), "content"
                ),
            ),
            "request identity",
        ),
    ),
)
def test_ingestion_recomputes_all_successful_attempt_identities(
    mutate: object, message: str, tmp_path: Path
) -> None:
    manifest = _manifest()
    attempt = mutate(_success("train-1"))  # type: ignore[operator]

    with pytest.raises(DatasetSnapshotValidationError, match=message):
        _ingest(
            manifest,
            {"train-1": attempt, "dev-1": _failure("dev-1")},
            tmp_path,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"displacement": ((0.0, 0.0),)}, "displacement"),
        ({"quadrature_weights": ((0.0,),)}, "quadrature weight"),
        ({"quadrature_weights": ((0.4,),)}, "integrate to physical cell area"),
        ({"quadrature_points": (((2.0, 2.0),),)}, "inside its mesh cell"),
        ({"first_piola_stress": ((),)}, "quadrature shapes"),
        ({"total_strain_energy": float("nan")}, "finite"),
    ),
)
def test_ingestion_rejects_invalid_result_shapes_weights_and_mesh_consistency(
    overrides: dict[str, object], message: str, tmp_path: Path
) -> None:
    manifest = _manifest()
    attempt = _success("train-1")
    invalid = _result(attempt.request, **overrides)
    invalid_hash = (
        attempt.result_hash
        if overrides.get("total_strain_energy") != overrides.get("total_strain_energy")
        else hash_spec(invalid, "content")
    )
    attempt = replace(attempt, result=invalid, result_hash=invalid_hash)

    with pytest.raises(DatasetSnapshotValidationError, match=message):
        _ingest(
            manifest,
            {"train-1": attempt, "dev-1": _failure("dev-1")},
            tmp_path,
        )


def test_failure_attempts_cannot_hide_result_material(tmp_path: Path) -> None:
    manifest = _manifest()
    failed = replace(_failure("dev-1"), result_hash="0" * 64)

    with pytest.raises(DatasetSnapshotValidationError, match="failed attempt"):
        _ingest(
            manifest,
            {"train-1": _success("train-1"), "dev-1": failed},
            tmp_path,
        )


def test_snapshot_is_persisted_as_canonical_content_addressed_bytes(
    tmp_path: Path,
) -> None:
    snapshot = _build(tmp_path / "results")[1]
    store = ObjectStore.create(tmp_path / "snapshots")

    reference = persist_role_snapshot(snapshot, store)

    assert reference.snapshot_content_hash == snapshot.content_hash
    assert reference.store_content_hash == store_content_hash(
        canonical_json_bytes(snapshot)
    )
    assert reference.store_content_hash != reference.snapshot_content_hash
    assert store.get(reference.store_content_hash) == canonical_json_bytes(snapshot)
