from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from campaigns.dogbone_v1.family import compiled_family
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
    DEVELOPMENT_PHASE_ID,
    TRAIN_FIT_PHASE_ID,
    TRAIN_SCREEN_PHASE_ID,
    CandidateModelStateEvidence,
    TrainScreenDecision,
    VerifiedScreenRunIndex,
    boundary_cloud_tensors,
    case_ids_for_phase,
    complete_train_screen,
    load_committed_protocol,
    qoi_vertex_indices,
    validate_train_screen_decision,
    verify_s10_screen_run_index,
)
from piginot.datasets.request_plan import (
    PublicFemRequestPlan,
    PublicFemRequestPlanEntry,
)
from piginot.doe import EvidenceRole, PublishedCaseManifest
from piginot.doe.publication import PublishedCaseStatus
from piginot.fem import FemFailureCode

_HASH = "0" * 64
_REVISION = "1" * 40


def _manifest() -> PublishedCaseManifest:
    root = Path(__file__).resolve().parents[2]
    return PublishedCaseManifest.from_bytes(
        (
            root
            / "artifacts"
            / "campaigns"
            / "dogbone-v1"
            / "public-case-manifest.json"
        ).read_bytes()
    )


def _protocol():  # type: ignore[no-untyped-def]
    return build_architecture_screen_protocol(_manifest())


def _metrics() -> tuple[ArchitectureScreenMetrics, ...]:
    values = (
        (BASELINE_CANDIDATE_ID, -0.20, 0.20, 0.40, 0.10, 0.20, 400_000),
        ("B-derived-ratio-film", -0.19, 0.19, 0.41, 0.10, 0.20, 405_000),
        ("C-local-stress-enrichment", -0.10, 0.15, 0.35, 0.10, 0.20, 410_000),
        ("D-derived-local-stress", -0.18, 0.18, 0.39, 0.10, 0.20, 415_000),
    )
    return tuple(
        ArchitectureScreenMetrics(
            candidate_id,
            signed,
            absolute,
            p90,
            u_l2,
            p_l2,
            parameters,
            1.0,
        )
        for candidate_id, signed, absolute, p90, u_l2, p_l2, parameters in values
    )


def _decision(
    *,
    metrics: tuple[ArchitectureScreenMetrics, ...] | None = None,
) -> TrainScreenDecision:
    protocol = _protocol()
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
        _metrics() if metrics is None else metrics,
        source_bindings_hash="9" * 64,
        training_schedule_hash="a" * 64,
        normalization_hash="b" * 64,
        implementation_revision=_REVISION,
        model_states=states,
    )


def _plan_and_ledger(
    *,
    corrupt_role_for: str | None = None,
    failed_case_id: str | None = None,
) -> tuple[PublicFemRequestPlan, ProductionFemRunLedger]:
    manifest = _manifest()
    active = tuple(
        case for case in manifest.cases if case.status is PublishedCaseStatus.ACTIVE
    )
    entries: list[PublicFemRequestPlanEntry] = []
    records: list[ProductionFemRunRecord] = []
    for case in active:
        role = case.role
        if corrupt_role_for == case.case_id:
            role = EvidenceRole.DEVELOPMENT
        entries.append(
            PublicFemRequestPlanEntry(
                case_id=case.case_id,
                leakage_group_id=case.leakage_group_id,
                role=role,
                request_recipe_hash=_HASH,
                request_content_hash=_HASH,
                mesh_id="mesh",
                mesh_content_hash=_HASH,
            )
        )
        success = case.case_id != failed_case_id
        records.append(
            ProductionFemRunRecord(
                case_id=case.case_id,
                request_recipe_hash=_HASH,
                request_content_hash=_HASH,
                success=success,
                result_hash=_HASH if success else None,
                store_content_hash=_HASH if success else None,
                failure_code=None if success else FemFailureCode.UNSUPPORTED_REQUEST,
                message="succeeded" if success else "failed:UNSUPPORTED_REQUEST",
                elapsed_seconds=0.0,
                origin=(
                    AttemptOrigin.RECOVERED_FROM_STORE
                    if success
                    else AttemptOrigin.SOLVED
                ),
            )
        )
    plan = PublicFemRequestPlan(
        campaign_hash=_HASH,
        public_case_manifest_hash=manifest.content_hash,
        execution_profile_hash=_HASH,
        entries=tuple(entries),
    )
    ledger = ProductionFemRunLedger(
        run_id=RUN_ID,
        campaign_hash=plan.campaign_hash,
        public_case_manifest_hash=plan.public_case_manifest_hash,
        request_plan_hash=plan.content_hash,
        execution_profile_hash=plan.execution_profile_hash,
        source_revision=_REVISION,
        backend_name="test-backend",
        planned_case_count=len(entries),
        records=tuple(records),
    )
    return plan, ledger


def test_committed_protocol_loader_is_exact_and_read_only(tmp_path: Path) -> None:
    protocol = _protocol()
    artifact = tmp_path / "S11-CONTINGENCY-PROTOCOL.json"
    artifact.write_bytes(protocol.to_bytes() + b"\n")

    loaded = load_committed_protocol(artifact)

    assert loaded == protocol
    artifact.write_bytes(protocol.to_bytes())
    with pytest.raises(ValueError, match="committed protocol"):
        load_committed_protocol(artifact)


def test_phase_case_ids_keep_development_closed_until_train_selection() -> None:
    protocol = _protocol()

    assert case_ids_for_phase(protocol, TRAIN_FIT_PHASE_ID) == protocol.fit_case_ids
    assert (
        case_ids_for_phase(protocol, TRAIN_SCREEN_PHASE_ID) == protocol.screen_case_ids
    )
    with pytest.raises(ValueError, match="decision"):
        case_ids_for_phase(protocol, DEVELOPMENT_PHASE_ID)
    with pytest.raises(ValueError, match="TRAIN fallback"):
        case_ids_for_phase(
            protocol,
            DEVELOPMENT_PHASE_ID,
            decision=_decision(
                metrics=tuple(
                    replace(
                        metric,
                        median_absolute_peak_error=0.30,
                        p90_absolute_peak_error=0.50,
                    )
                    if metric.candidate_id != BASELINE_CANDIDATE_ID
                    else metric
                    for metric in _metrics()
                )
            ),
        )
    assert (
        case_ids_for_phase(protocol, DEVELOPMENT_PHASE_ID, decision=_decision())
        == protocol.development_case_ids
    )


def test_train_screen_decision_recomputes_selection_and_binds_protocol() -> None:
    protocol = _protocol()

    decision = _decision()

    assert type(decision) is TrainScreenDecision
    assert decision.protocol_content_hash == protocol.content_hash
    assert decision.selection.selected_candidate_id == "C-local-stress-enrichment"
    assert decision.metrics == _metrics()
    assert validate_train_screen_decision(protocol, decision) is decision
    with pytest.raises(ValueError, match="deterministic reduction"):
        validate_train_screen_decision(
            protocol,
            replace(
                decision,
                selection=ArchitectureSelection(
                    "D-derived-local-stress",
                    ("D-derived-local-stress",),
                    False,
                ),
            ),
        )
    with pytest.raises((TypeError, ValueError)):
        TrainScreenDecision(  # type: ignore[call-arg]
            protocol.content_hash,
            "9" * 64,
            "a" * 64,
            "b" * 64,
            _REVISION,
            decision.model_states,
            _metrics(),
            decision.selection,
        )


def test_s10_screen_run_index_binds_roles_records_and_closed_development() -> None:
    protocol = _protocol()
    plan, ledger = _plan_and_ledger()

    index = verify_s10_screen_run_index(protocol, request_plan=plan, ledger=ledger)

    assert type(index) is VerifiedScreenRunIndex
    assert index.train_fit_case_ids == protocol.fit_case_ids
    assert index.train_screen_case_ids == protocol.screen_case_ids
    assert index.development_case_ids == protocol.development_case_ids
    assert index.development_opened is False
    assert index.request_plan_hash == plan.content_hash


def test_s10_screen_run_index_rejects_role_or_success_drift() -> None:
    protocol = _protocol()
    bad_role_plan, bad_role_ledger = _plan_and_ledger(
        corrupt_role_for=protocol.fit_case_ids[0]
    )
    with pytest.raises(ValueError, match="TRAIN"):
        verify_s10_screen_run_index(
            protocol, request_plan=bad_role_plan, ledger=bad_role_ledger
        )

    failed_plan, failed_ledger = _plan_and_ledger(
        failed_case_id=protocol.screen_case_ids[0]
    )
    with pytest.raises(ValueError, match="successful"):
        verify_s10_screen_run_index(
            protocol, request_plan=failed_plan, ledger=failed_ledger
        )


def test_qoi_vertices_are_right_arc_without_mixed_endpoint_vertices() -> None:
    mesh = compiled_family().reference_mesh
    indices = qoi_vertex_indices(
        mesh, semantic_tag="right_arc", excluded_tags=("right_grip",)
    )
    right_arc_vertices = {
        vertex
        for facet_index in mesh.semantic_tags["right_arc"]
        for vertex in mesh.boundary_facets[facet_index]
    }
    mixed_vertices = {
        vertex
        for facet_index in mesh.semantic_tags["right_grip"]
        for vertex in mesh.boundary_facets[facet_index]
    }

    assert set(indices) <= right_arc_vertices
    assert not set(indices) & mixed_vertices
    assert len(indices) > 0
    assert tuple(sorted(indices)) == indices


def test_boundary_cloud_tensors_are_tag_ordered_measure_weighted_and_deterministic() -> (
    None
):
    mesh = compiled_family().reference_mesh
    tag_names = tuple(mesh.semantic_tags)

    first = boundary_cloud_tensors(mesh, tag_names=tag_names, sample_count=64)
    second = boundary_cloud_tensors(mesh, tag_names=tag_names, sample_count=64)

    torch.testing.assert_close(first.xy, second.xy, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.features, second.features, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.weights, second.weights, rtol=0.0, atol=0.0)
    assert first.xy.shape == (64, 2)
    assert first.features.shape == (64, 2 + len(tag_names))
    assert first.weights.shape == (64,)
    assert first.mask.shape == (64,)
    assert torch.all(first.mask)
    assert torch.all(first.weights > 0.0)
    torch.testing.assert_close(
        first.features[:, 2:].sum(dim=1),
        torch.ones(64, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
