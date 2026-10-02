from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import rfc8785

from campaigns.dogbone_v1.family import compiled_family
from campaigns.dogbone_v1.s10_closeout import _load_run_execution_profile
from campaigns.dogbone_v1.s10_evidence import DOGBONE_S10_DATASET_POLICY
from campaigns.dogbone_v1.s10_production import (
    RUN_ID,
    AttemptOrigin,
    ProductionFemRunLedger,
    ProductionFemRunRecord,
)
from campaigns.dogbone_v1.s11_contingency import (
    BASELINE_CANDIDATE_ID,
    ArchitectureScreenMetrics,
    ArchitectureSelection,
    build_architecture_screen_protocol,
)
from campaigns.dogbone_v1.s11_contingency_run import (
    CandidateModelStateEvidence,
    complete_train_screen,
)
from campaigns.dogbone_v1.s11_contingency_source import (
    S10_PRODUCTION_REPORT_CONTENT_HASH,
    S10ReportIndex,
    SelectiveS10Source,
    VerifiedS10Case,
    _build_train_snapshot_from_public_metadata,
    _FixtureSelectiveS10Source,
    _rebuild_request,
    _verify_s10_source_bindings_core,
    load_s10_report_index,
    open_selective_s10_source,
    verify_s10_source_bindings,
)
from piginot.contracts import (
    ExecutionProfile,
    canonical_json_bytes,
    hash_spec,
    strict_json_loads,
)
from piginot.datasets import sample_request_collocation
from piginot.datasets.request_plan import (
    PublicFemRequestPlan,
    PublicFemRequestPlanEntry,
)
from piginot.doe import EvidenceRole, PublishedCaseManifest
from piginot.doe.publication import PublishedCaseStatus
from piginot.fem import FemAttempt, FemResult
from piginot.store import ObjectStore

_ROOT = Path(__file__).resolve().parents[2]
_CAMPAIGN_ROOT = _ROOT / "artifacts" / "campaigns" / "dogbone-v1"
_REPORT_PATH = _CAMPAIGN_ROOT / "s10-production-dataset.json"
_HASH = "0" * 64
_REVISION = "1" * 40


def _manifest() -> PublishedCaseManifest:
    return PublishedCaseManifest.from_bytes(
        (_CAMPAIGN_ROOT / "public-case-manifest.json").read_bytes()
    )


def _profile() -> ExecutionProfile:
    return ExecutionProfile(
        python_version="3.11.9",
        library_versions={"piginot": "0.1.0"},
        blas_threads=1,
        device="cpu",
        dtype="float64",
        rng_algorithm="PCG64",
        seed=123,
    )


def _plan_and_ledger(
    report_index,  # type: ignore[no-untyped-def]
    *,
    profile: ExecutionProfile,
    exact_case_id: str | None = None,
) -> tuple[PublicFemRequestPlan, ProductionFemRunLedger, S10ReportIndex]:
    manifest = _manifest()
    report_cases = {item.case_id: item for item in report_index.cases}
    active = tuple(
        case for case in manifest.cases if case.status is PublishedCaseStatus.ACTIVE
    )
    family = compiled_family() if exact_case_id is not None else None
    entries_list: list[PublicFemRequestPlanEntry] = []
    for case in active:
        request = (
            _rebuild_request(case, profile, family)
            if case.case_id == exact_case_id and family is not None
            else None
        )
        entries_list.append(
            PublicFemRequestPlanEntry(
                case_id=case.case_id,
                leakage_group_id=case.leakage_group_id,
                role=case.role,
                request_recipe_hash=_HASH if request is None else request.recipe_hash,
                request_content_hash=(
                    _HASH if request is None else hash_spec(request, "content")
                ),
                mesh_id="mesh" if request is None else request.mesh.mesh_id,
                mesh_content_hash=(
                    _HASH if request is None else hash_spec(request.mesh, "content")
                ),
            )
        )
    entries = tuple(entries_list)
    plan = PublicFemRequestPlan(
        campaign_hash=report_index.campaign_hash,
        public_case_manifest_hash=manifest.content_hash,
        execution_profile_hash=hash_spec(profile, "content"),
        entries=entries,
    )
    records = tuple(
        ProductionFemRunRecord(
            case_id=entry.case_id,
            request_recipe_hash=entry.request_recipe_hash,
            request_content_hash=entry.request_content_hash,
            success=True,
            result_hash=report_cases[entry.case_id].fem_result_hash,
            store_content_hash=report_cases[entry.case_id].store_content_hash,
            failure_code=None,
            message="succeeded",
            elapsed_seconds=0.0,
            origin=AttemptOrigin.RECOVERED_FROM_STORE,
        )
        for entry in entries
    )
    ledger = ProductionFemRunLedger(
        run_id=RUN_ID,
        campaign_hash=plan.campaign_hash,
        public_case_manifest_hash=plan.public_case_manifest_hash,
        request_plan_hash=plan.content_hash,
        execution_profile_hash=plan.execution_profile_hash,
        source_revision=_REVISION,
        backend_name="test-backend",
        planned_case_count=len(records),
        records=records,
    )
    bound_report = replace(
        report_index,
        request_plan_hash=plan.content_hash,
        execution_profile_hash=hash_spec(profile, "content"),
        fem_source_revision=ledger.source_revision,
    )
    return plan, ledger, bound_report


def _metrics(*, eligible: bool = True) -> tuple[ArchitectureScreenMetrics, ...]:
    candidate_ids = (
        BASELINE_CANDIDATE_ID,
        "B-derived-ratio-film",
        "C-local-stress-enrichment",
        "D-derived-local-stress",
    )
    result: list[ArchitectureScreenMetrics] = []
    for index, candidate_id in enumerate(candidate_ids):
        if index == 0:
            signed, absolute, p90 = -0.20, 0.20, 0.40
        elif eligible and index == 2:
            signed, absolute, p90 = -0.10, 0.15, 0.35
        else:
            signed, absolute, p90 = -0.19, 0.19, 0.41
        result.append(
            ArchitectureScreenMetrics(
                candidate_id,
                signed,
                absolute,
                p90,
                0.10,
                0.20,
                400_000 + index,
                1.0,
            )
        )
    return tuple(result)


def _decision(
    protocol,  # type: ignore[no-untyped-def]
    source_bindings_hash: str,
    *,
    eligible: bool = True,
):  # type: ignore[no-untyped-def]
    states = tuple(
        CandidateModelStateEvidence(
            candidate.candidate_id,
            f"{index + 1:x}" * 64,
            f"{index + 5:x}" * 64,
        )
        for index, candidate in enumerate(protocol.candidates)
    )
    return complete_train_screen(
        protocol,
        _metrics(eligible=eligible),
        source_bindings_hash=source_bindings_hash,
        training_schedule_hash="a" * 64,
        normalization_hash="b" * 64,
        implementation_revision=_REVISION,
        model_states=states,
    )


def _empty_result(request) -> FemResult:  # type: ignore[no-untyped-def]
    return FemResult(
        request_id=request.request_id,
        mesh_id=request.mesh.mesh_id,
        element_order=2,
        displacement_dof_coordinates=(),
        displacement=(),
        quadrature_points=(),
        quadrature_weights=(),
        first_piola_stress=(),
        deformation_gradient=(),
        recovered_first_piola_stress=(),
        reactions={},
        rigid_mode_multipliers=(),
        total_strain_energy=0.0,
        external_work=0.0,
        total_potential_energy=0.0,
        convergence_history=(),
        qois={},
    )


def _replace_report_case(
    report: S10ReportIndex,
    case_id: str,
    **changes: object,
) -> S10ReportIndex:
    return replace(
        report,
        cases=tuple(
            replace(item, **changes) if item.case_id == case_id else item
            for item in report.cases
        ),
    )


def _bind_fixture_train_snapshot(
    protocol,  # type: ignore[no-untyped-def]
    manifest: PublishedCaseManifest,
    plan: PublicFemRequestPlan,
    ledger: ProductionFemRunLedger,
    report: S10ReportIndex,
):  # type: ignore[no-untyped-def]
    snapshot = _build_train_snapshot_from_public_metadata(
        manifest=manifest,
        request_plan=plan,
        ledger=ledger,
        report=report,
    )
    snapshots = tuple(
        replace(item, snapshot_content_hash=snapshot.content_hash)
        if item.role is EvidenceRole.TRAIN
        else item
        for item in report.role_snapshots
    )
    return (
        replace(protocol, training_snapshot_hash=snapshot.content_hash),
        replace(
            report,
            training_snapshot_hash=snapshot.content_hash,
            role_snapshots=snapshots,
        ),
        snapshot,
    )


def test_tracked_s10_report_index_is_canonical_and_exactly_pinned() -> None:
    report = load_s10_report_index()

    assert report.content_hash == S10_PRODUCTION_REPORT_CONTENT_HASH
    assert report.case_count == 344
    assert report.failed_case_count == 0
    assert report.training_snapshot_hash == (
        "af04a059e0efb647a1ba768f722d2fde7500d06d11695d1132a0b26dded24845"
    )
    assert report.development_snapshot_hash == (
        "5c6c59c621341d35797bcef3bca8f44510fcbfa90b329858d2e9a3e407824a77"
    )
    first = report.cases[0]
    assert first.collocation_stream_hash == (
        "4a925d22d391a96ab409a4408b8c6eb0ad74e85df6224b1ef25bbdc9eee66759"
    )
    assert first.collocation_batch_content_hash == (
        "4339ecdcb3c4bdf02d4d622378089a25c123ddf5e7031260b318fa843b4185bb"
    )
    assert first.collocation_store_content_hash == (
        "4543187e273cd59cacd5aa3b58d6d819e4fe254bf07a282c6997ee932bfc9d10"
    )
    assert first.collocation_point_count == 256
    assert first.collocation_replay_verified is True


def test_train_snapshot_is_rehydrated_from_public_metadata_without_store_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import campaigns.dogbone_v1.s11_contingency_source as source_module

    report = load_s10_report_index()
    manifest = _manifest()
    protocol = build_architecture_screen_protocol(manifest)
    profile = _profile()
    plan, ledger, report = _plan_and_ledger(report, profile=profile)
    protocol, report, expected = _bind_fixture_train_snapshot(
        protocol,
        manifest,
        plan,
        ledger,
        report,
    )
    source = _FixtureSelectiveS10Source.from_components(
        protocol=protocol,
        manifest=manifest,
        execution_profile=profile,
        request_plan=plan,
        ledger=ledger,
        report=report,
        store=ObjectStore.create(tmp_path / "store"),
        development_access_record=tmp_path / "development-access.json",
    )

    def forbidden(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("FEM result bytes must remain closed")

    monkeypatch.setattr(source_module, "_attempt_from_record", forbidden)

    actual = source.train_snapshot

    assert actual == expected
    assert actual.role is EvidenceRole.TRAIN
    assert len(actual.cases) == 256
    assert actual.content_hash == report.training_snapshot_hash
    assert source.development_opened is False


def test_train_snapshot_rejects_retained_public_identity_drift(tmp_path: Path) -> None:
    report = load_s10_report_index()
    manifest = _manifest()
    protocol = build_architecture_screen_protocol(manifest)
    profile = _profile()
    plan, ledger, report = _plan_and_ledger(report, profile=profile)
    protocol, report, _ = _bind_fixture_train_snapshot(
        protocol,
        manifest,
        plan,
        ledger,
        report,
    )
    source = _FixtureSelectiveS10Source.from_components(
        protocol=protocol,
        manifest=manifest,
        execution_profile=profile,
        request_plan=plan,
        ledger=ledger,
        report=report,
        store=ObjectStore.create(tmp_path / "store"),
        development_access_record=tmp_path / "development-access.json",
    )
    source._request_plan = replace(plan, campaign_hash="f" * 64)

    with pytest.raises(ValueError, match="retained request plan"):
        _ = source.train_snapshot


def test_single_fit_loader_resolves_only_a_registered_train_fit_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = object.__new__(_FixtureSelectiveS10Source)
    source._protocol = SimpleNamespace(  # type: ignore[assignment]
        fit_case_ids=("fit-a", "fit-b"),
    )
    verified = object()
    collocation = object()
    calls: list[tuple[object, ...]] = []

    def load_case(
        _source: object,
        case_id: str,
        role: EvidenceRole,
    ) -> object:
        calls.append(("load", case_id, role))
        return verified

    def replay(_source: object, selected: object) -> object:
        calls.append(("replay", selected))
        return collocation

    monkeypatch.setattr(_FixtureSelectiveS10Source, "_load_case", load_case)
    monkeypatch.setattr(
        _FixtureSelectiveS10Source,
        "replay_train_collocation",
        replay,
    )

    assert source.load_fit_with_collocation("fit-b") == (verified, collocation)
    assert calls == [
        ("load", "fit-b", EvidenceRole.TRAIN),
        ("replay", verified),
    ]

    with pytest.raises(ValueError, match="registered TRAIN fit"):
        source.load_fit_with_collocation("screen-only")
    with pytest.raises(ValueError, match="non-empty"):
        source.load_fit_with_collocation("")
    with pytest.raises(TypeError, match="case_id"):
        source.load_fit_with_collocation(1)  # type: ignore[arg-type]
    assert len(calls) == 2


def test_train_collocation_replay_checks_report_and_store_identities(
    tmp_path: Path,
) -> None:
    report = load_s10_report_index()
    manifest = _manifest()
    protocol = build_architecture_screen_protocol(manifest)
    profile = _profile()
    case_id = protocol.fit_case_ids[0]
    plan, ledger, report = _plan_and_ledger(
        report,
        profile=profile,
        exact_case_id=case_id,
    )
    family = compiled_family()
    case = next(item for item in manifest.cases if item.case_id == case_id)
    request = _rebuild_request(case, profile, family)
    result = _empty_result(request)
    result_hash = hash_spec(result, "content")
    ledger = replace(
        ledger,
        records=tuple(
            replace(item, result_hash=result_hash) if item.case_id == case_id else item
            for item in ledger.records
        ),
    )
    report = _replace_report_case(
        report,
        case_id,
        fem_result_hash=result_hash,
    )
    protocol, report, snapshot = _bind_fixture_train_snapshot(
        protocol,
        manifest,
        plan,
        ledger,
        report,
    )
    batch = sample_request_collocation(
        snapshot,
        case_id,
        request,
        stream_index=DOGBONE_S10_DATASET_POLICY.stream_index,
        interior_count=DOGBONE_S10_DATASET_POLICY.interior_count,
        boundary_count=DOGBONE_S10_DATASET_POLICY.boundary_count,
        interface_count=DOGBONE_S10_DATASET_POLICY.interface_count,
        qoi_quotas=DOGBONE_S10_DATASET_POLICY.qoi_quotas,
    )
    store = ObjectStore.create(tmp_path / "store")
    store_hash = store.put(batch.to_bytes())
    report = _replace_report_case(
        report,
        case_id,
        collocation_stream_hash=batch.sampler_stream_hash,
        collocation_batch_content_hash=batch.content_hash,
        collocation_store_content_hash=store_hash,
        collocation_point_count=len(batch.points),
        collocation_replay_verified=True,
    )
    source = _FixtureSelectiveS10Source.from_components(
        protocol=protocol,
        manifest=manifest,
        execution_profile=profile,
        request_plan=plan,
        ledger=ledger,
        report=report,
        store=store,
        development_access_record=tmp_path / "development-access.json",
    )
    verified_case = VerifiedS10Case(case, request, result)

    assert source.replay_train_collocation(verified_case) == batch

    wrong_role = VerifiedS10Case(
        replace(case, role=EvidenceRole.DEVELOPMENT),
        request,
        result,
    )
    with pytest.raises(ValueError, match="TRAIN case"):
        source.replay_train_collocation(wrong_role)
    with pytest.raises(TypeError, match="VerifiedS10Case"):
        source.replay_train_collocation(object())  # type: ignore[arg-type]

    bad_report = _replace_report_case(
        report,
        case_id,
        collocation_batch_content_hash="f" * 64,
    )
    bad_source = _FixtureSelectiveS10Source.from_components(
        protocol=protocol,
        manifest=manifest,
        execution_profile=profile,
        request_plan=plan,
        ledger=ledger,
        report=bad_report,
        store=store,
        development_access_record=tmp_path / "development-access-2.json",
    )
    with pytest.raises(ValueError, match="collocation identities"):
        bad_source.replay_train_collocation(verified_case)

    missing_report = _replace_report_case(
        report,
        case_id,
        collocation_store_content_hash="e" * 64,
    )
    missing_source = _FixtureSelectiveS10Source.from_components(
        protocol=protocol,
        manifest=manifest,
        execution_profile=profile,
        request_plan=plan,
        ledger=ledger,
        report=missing_report,
        store=store,
        development_access_record=tmp_path / "development-access-3.json",
    )
    with pytest.raises(ValueError, match="collocation store bytes"):
        missing_source.replay_train_collocation(verified_case)


def test_s10_report_index_rejects_one_byte_semantic_drift(tmp_path: Path) -> None:
    document = strict_json_loads(_REPORT_PATH.read_bytes())
    assert isinstance(document, dict)
    document["run_id"] = "wrong-run"
    changed = tmp_path / "report.json"
    changed.write_bytes(rfc8785.dumps(document))

    with pytest.raises(ValueError, match="content hash"):
        load_s10_report_index(changed)


def test_source_bindings_cover_the_exact_active_population_and_snapshots() -> None:
    report = load_s10_report_index()
    manifest = _manifest()
    protocol = build_architecture_screen_protocol(manifest)
    profile = _profile()
    plan, ledger, report = _plan_and_ledger(report, profile=profile)

    bindings = _verify_s10_source_bindings_core(
        protocol,
        manifest=manifest,
        execution_profile=profile,
        request_plan=plan,
        ledger=ledger,
        report=report,
    )

    assert bindings.active_case_count == 344
    assert bindings.successful_case_count == 344
    assert bindings.report_content_hash == report.content_hash
    assert bindings.request_plan_hash == plan.content_hash
    assert bindings.execution_profile_hash == hash_spec(profile, "content")


def test_source_bindings_reject_case_hash_or_snapshot_drift() -> None:
    report = load_s10_report_index()
    manifest = _manifest()
    protocol = build_architecture_screen_protocol(manifest)
    profile = _profile()
    plan, ledger, report = _plan_and_ledger(report, profile=profile)
    first = ledger.records[0]
    bad_record = replace(first, result_hash="f" * 64)
    bad_ledger = replace(ledger, records=(bad_record, *ledger.records[1:]))

    with pytest.raises(ValueError, match="report result hash"):
        _verify_s10_source_bindings_core(
            protocol,
            manifest=manifest,
            execution_profile=profile,
            request_plan=plan,
            ledger=bad_ledger,
            report=report,
        )

    bad_report = replace(report, training_snapshot_hash="f" * 64)
    with pytest.raises(ValueError, match="TRAIN snapshot"):
        _verify_s10_source_bindings_core(
            protocol,
            manifest=manifest,
            execution_profile=profile,
            request_plan=plan,
            ledger=ledger,
            report=bad_report,
        )


def test_source_bindings_reject_protocol_report_and_request_record_forgery() -> None:
    report = load_s10_report_index()
    manifest = _manifest()
    protocol = build_architecture_screen_protocol(manifest)
    profile = _profile()
    plan, ledger, report = _plan_and_ledger(report, profile=profile)
    fit_id = protocol.fit_case_ids[0]
    screen_id = protocol.screen_case_ids[0]
    swapped_protocol = replace(
        protocol,
        fit_case_ids=tuple(sorted({*protocol.fit_case_ids[1:], screen_id})),
        screen_case_ids=tuple(sorted({*protocol.screen_case_ids[1:], fit_id})),
    )
    with pytest.raises(ValueError, match="committed protocol"):
        verify_s10_source_bindings(
            swapped_protocol,
            manifest=manifest,
            execution_profile=profile,
            request_plan=plan,
            ledger=ledger,
            report=report,
        )
    with pytest.raises(ValueError, match="report content hash"):
        verify_s10_source_bindings(
            protocol,
            manifest=manifest,
            execution_profile=profile,
            request_plan=plan,
            ledger=ledger,
            report=replace(report, content_hash="f" * 64),
        )
    first = ledger.records[0]
    forged_ledger = replace(
        ledger,
        records=(replace(first, request_content_hash="f" * 64), *ledger.records[1:]),
    )
    with pytest.raises(ValueError, match="request identities"):
        _verify_s10_source_bindings_core(
            protocol,
            manifest=manifest,
            execution_profile=profile,
            request_plan=plan,
            ledger=forged_ledger,
            report=report,
        )


def test_development_stream_is_one_shot_and_requires_train_selection(
    tmp_path: Path,
) -> None:
    report = load_s10_report_index()
    manifest = _manifest()
    protocol = build_architecture_screen_protocol(manifest)
    profile = _profile()
    plan, ledger, report = _plan_and_ledger(report, profile=profile)
    store = ObjectStore.create(tmp_path / "store")
    marker = tmp_path / "development-access.json"
    source = _FixtureSelectiveS10Source.from_components(
        protocol=protocol,
        manifest=manifest,
        execution_profile=profile,
        request_plan=plan,
        ledger=ledger,
        report=report,
        store=store,
        development_access_record=marker,
    )
    assert type(source) is not SelectiveS10Source

    with pytest.raises((TypeError, ValueError), match="TrainScreenDecision"):
        source._open_development()
    with pytest.raises(TypeError, match="TrainScreenDecision"):
        source._open_development(
            ArchitectureSelection(
                "C-local-stress-enrichment",
                ("C-local-stress-enrichment",),
                False,
            )  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="fallback"):
        source._open_development(
            _decision(protocol, source.bindings.content_hash, eligible=False)
        )

    decision = _decision(protocol, source.bindings.content_hash)
    tampered = replace(
        decision,
        selection=ArchitectureSelection(
            "D-derived-local-stress",
            ("D-derived-local-stress",),
            False,
        ),
    )
    with pytest.raises(ValueError, match="deterministic reduction"):
        source._open_development(tampered)

    stream = source._open_development(decision)
    assert source.development_opened is True
    assert marker.read_bytes() == decision.to_bytes() + b"\n"
    assert iter(stream) is stream
    with pytest.raises(RuntimeError, match="already been opened"):
        source._open_development(decision)

    reopened = _FixtureSelectiveS10Source.from_components(
        protocol=protocol,
        manifest=manifest,
        execution_profile=profile,
        request_plan=plan,
        ledger=ledger,
        report=report,
        store=store,
        development_access_record=marker,
    )
    with pytest.raises(RuntimeError, match="already been opened"):
        reopened._open_development(decision)


def test_source_exposes_only_exact_fit_and_train_screen_streams(tmp_path: Path) -> None:
    report = load_s10_report_index()
    manifest = _manifest()
    protocol = build_architecture_screen_protocol(manifest)
    profile = _profile()
    plan, ledger, report = _plan_and_ledger(report, profile=profile)
    source = _FixtureSelectiveS10Source.from_components(
        protocol=protocol,
        manifest=manifest,
        execution_profile=profile,
        request_plan=plan,
        ledger=ledger,
        report=report,
        store=ObjectStore.create(tmp_path / "store"),
        development_access_record=tmp_path / "development-access.json",
    )

    assert source.fit_case_ids == protocol.fit_case_ids
    assert source.train_screen_case_ids == protocol.screen_case_ids
    assert not hasattr(source, "iter_training")
    with pytest.raises(ValueError, match="runtime request differs"):
        next(source.iter_fit())
    with pytest.raises(ValueError, match="runtime request differs"):
        next(source.iter_train_screen())


def test_selected_case_is_request_bound_before_exactly_one_result_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import campaigns.dogbone_v1.s11_contingency_source as source_module

    report = load_s10_report_index()
    manifest = _manifest()
    protocol = build_architecture_screen_protocol(manifest)
    profile = _profile()
    case_id = protocol.fit_case_ids[0]
    plan, ledger, report = _plan_and_ledger(
        report,
        profile=profile,
        exact_case_id=case_id,
    )
    store = ObjectStore.create(tmp_path / "store")
    source = _FixtureSelectiveS10Source.from_components(
        protocol=protocol,
        manifest=manifest,
        execution_profile=profile,
        request_plan=plan,
        ledger=ledger,
        report=report,
        store=store,
        development_access_record=tmp_path / "development-access.json",
    )
    calls: list[str] = []

    def resolved(record, request, actual_store):  # type: ignore[no-untyped-def]
        calls.append(record.case_id)
        assert actual_store is store
        result = FemResult(
            request_id=request.request_id,
            mesh_id=request.mesh.mesh_id,
            element_order=2,
            displacement_dof_coordinates=(),
            displacement=(),
            quadrature_points=(),
            quadrature_weights=(),
            first_piola_stress=(),
            deformation_gradient=(),
            recovered_first_piola_stress=(),
            reactions={},
            rigid_mode_multipliers=(),
            total_strain_energy=0.0,
            external_work=0.0,
            total_potential_energy=0.0,
            convergence_history=(),
            qois={},
        )
        return FemAttempt(
            request=request,
            recipe_hash=request.recipe_hash,
            success=True,
            result=result,
            result_hash=hash_spec(result, "content"),
            store_content_hash=record.store_content_hash,
            failure_code=None,
            message="succeeded",
        )

    monkeypatch.setattr(source_module, "_attempt_from_record", resolved)

    loaded = next(source.iter_fit())

    assert loaded.case.case_id == case_id
    assert loaded.request.request_id == case_id
    assert loaded.result.request_id == case_id
    assert calls == [case_id]


def test_request_drift_is_rejected_before_result_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import campaigns.dogbone_v1.s11_contingency_source as source_module

    report = load_s10_report_index()
    manifest = _manifest()
    protocol = build_architecture_screen_protocol(manifest)
    profile = _profile()
    plan, ledger, report = _plan_and_ledger(report, profile=profile)
    source = _FixtureSelectiveS10Source.from_components(
        protocol=protocol,
        manifest=manifest,
        execution_profile=profile,
        request_plan=plan,
        ledger=ledger,
        report=report,
        store=ObjectStore.create(tmp_path / "store"),
        development_access_record=tmp_path / "development-access.json",
    )
    called = False

    def forbidden(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True
        raise AssertionError("result store must remain closed")

    monkeypatch.setattr(source_module, "_attempt_from_record", forbidden)

    with pytest.raises(ValueError, match="runtime request differs"):
        next(source.iter_fit())
    assert called is False


def test_execution_profile_fixture_parser_remains_the_s10_trust_boundary(
    tmp_path: Path,
) -> None:
    profile = _profile()
    (tmp_path / "execution-profile.json").write_bytes(canonical_json_bytes(profile))

    parsed = _load_run_execution_profile(tmp_path)

    assert parsed == profile


def test_production_entrypoint_accepts_the_platform_path_subclass(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "empty-s10-run"
    output_root = tmp_path / "output"
    run_root.mkdir()
    output_root.mkdir()

    with pytest.raises(FileNotFoundError):
        open_selective_s10_source(
            run_root,
            development_access_record=output_root / "development-access.json",
        )
