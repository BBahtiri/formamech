from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from campaigns.dogbone_v1.family import compiled_family
from campaigns.dogbone_v1.s11_contingency import (
    BASELINE_CANDIDATE_ID,
    CONTINGENCY_PROTOCOL_ID,
    ArchitectureScreenMetrics,
    ArchitectureSelection,
    build_architecture_screen_protocol,
    select_architecture_candidate,
)
from piginot.contracts import canonical_json_bytes
from piginot.doe import EvidenceRole, PublishedCaseManifest, PublishedCaseStatus
from piginot.operators import OperatorConfig, PiGinotOperator, module_digest


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


def _metric(
    candidate_id: str,
    *,
    signed: float,
    absolute: float,
    p90: float,
    u_l2: float,
    p_l2: float,
    parameters: int = 400_000,
) -> ArchitectureScreenMetrics:
    return ArchitectureScreenMetrics(
        candidate_id=candidate_id,
        median_signed_peak_error=signed,
        median_absolute_peak_error=absolute,
        p90_absolute_peak_error=p90,
        median_u_relative_l2=u_l2,
        median_p_relative_l2=p_l2,
        trainable_parameter_count=parameters,
        elapsed_seconds=10.0,
    )


def test_protocol_binds_sources_roles_candidates_and_fixed_compute() -> None:
    protocol = build_architecture_screen_protocol(_manifest())

    assert protocol.protocol_id == CONTINGENCY_PROTOCOL_ID
    assert protocol.public_case_manifest_hash == (
        "ce95c03f5b22308a7aa9bf627966a6b21300a4e7b0272d3affa7dd503af564a9"
    )
    assert protocol.training_snapshot_hash == (
        "af04a059e0efb647a1ba768f722d2fde7500d06d11695d1132a0b26dded24845"
    )
    assert protocol.development_snapshot_hash == (
        "5c6c59c621341d35797bcef3bca8f44510fcbfa90b329858d2e9a3e407824a77"
    )
    assert len(protocol.fit_case_ids) == 192
    assert len(protocol.screen_case_ids) == 64
    assert len(protocol.development_case_ids) == 64
    assert not set(protocol.fit_case_ids) & set(protocol.screen_case_ids)
    assert tuple(candidate.candidate_id for candidate in protocol.candidates) == (
        "A-s11-baseline",
        "B-derived-ratio-film",
        "C-local-stress-enrichment",
        "D-derived-local-stress",
    )
    assert protocol.optimizer_steps == 1_500
    assert protocol.case_batch_size == 4
    assert protocol.query_points_per_case == 256
    assert protocol.qoi_fraction == 0.25
    assert protocol.peak_supervised_p_weight == 1.0
    assert protocol.objective_policy_id == (
        "s11c-supervised-architecture-isolation-v1"
    )
    assert protocol.supervised_objective_terms == (
        "normalized_u_mse",
        "normalized_in_plane_P_mse",
        "root_qoi_peak_P_mse",
    )
    assert protocol.physics_objectives_deferred_to_step == "S12"
    assert protocol.deferred_physics_objective_terms == (
        "strong_equilibrium",
        "weak_equilibrium",
        "constitutive_in_plane",
        "angular_momentum",
        "traction_free",
    )
    assert protocol.data_preparation_policy_id == (
        "s11c-s10-p2-vertices-boundary-arclength-v1"
    )
    assert protocol.normalization_policy_id == (
        "s11c-fit-zscore-u-by-prescribed-displacement-p-by-760mpa-v1"
    )
    assert protocol.qoi_policy_id == "s11c-right-arc-minus-right-grip-root-peak-v1"
    assert protocol.metric_policy_id == (
        "s11c-relative-l2-root-frobenius-linear-quantile-v1"
    )
    assert protocol.optimizer_policy_id == "s11c-torch-adam-no-foreach-no-fused-v1"
    assert protocol.source_access_policy_id == (
        "s11c-verified-fit-screen-durable-development-v1"
    )
    assert protocol.development_confirmation_policy_id == (
        "s11c-baseline-vs-train-winner-same-guardrails-v1"
    )
    assert protocol.boundary_point_count == 320
    assert protocol.boundary_tag_names == (
        "bottom",
        "gauge_top",
        "left_symmetry",
        "right_arc",
        "right_grip",
    )
    assert protocol.stress_reference_mpa == 760.0
    assert protocol.operator_tensor_dtype == "torch.float64"
    assert protocol.initialization_policy_id == (
        "s11c-fresh-shared-state-template-state-hash-v2"
    )
    assert protocol.common_state_template_candidate_id == BASELINE_CANDIDATE_ID
    assert protocol.candidate_specific_initialization_seed == 11123
    assert (
        protocol.sampling_schedule_policy_id == "s11c-shared-case-and-query-schedule-v1"
    )
    assert protocol.to_bytes() == canonical_json_bytes(protocol)


def test_protocol_binds_each_candidate_operator_schema() -> None:
    protocol = build_architecture_screen_protocol(_manifest())
    tags = tuple(compiled_family().reference_mesh.semantic_tags)

    assert protocol.candidates[0].operator_module_digest == (
        "blake2b256:ac064556f53e573b4f9de63a2610077dfd38141b6eef695731302cea3a560f47"
    )
    for candidate in protocol.candidates:
        config = OperatorConfig(
            parameter_names=candidate.parameter_names,
            boundary_tag_names=tags,
            hidden_dim=64,
            latent_token_count=32,
            num_attention_heads=4,
            num_fourier_frequencies=6,
            encoder_self_attention_layers=3,
            decoder_cross_attention_layers=6,
            local_boundary_kernel_scales=candidate.local_boundary_kernel_scales,
            stress_enrichment_layers=candidate.stress_enrichment_layers,
        )
        model = PiGinotOperator(config).to(dtype=torch.float64)
        assert candidate.operator_module_digest == module_digest(model)


def test_committed_protocol_is_the_exact_preregistered_document() -> None:
    protocol = build_architecture_screen_protocol(_manifest())
    root = Path(__file__).resolve().parents[2]
    artifact = (
        root
        / "artifacts"
        / "campaigns"
        / "dogbone-v1"
        / "S11-CONTINGENCY-PROTOCOL.json"
    )

    assert artifact.read_bytes() == protocol.to_bytes() + b"\n"
    assert protocol.content_hash == (
        "f16a4f63c3a8cfc3ee321bc4e15162802cbc6f106ed807f5aa4bc59eceac81df"
    )


def test_split_is_whole_case_role_safe_and_hamilton_balanced() -> None:
    manifest = _manifest()
    protocol = build_architecture_screen_protocol(manifest)
    cases = {case.case_id: case for case in manifest.cases}
    selected = (*protocol.fit_case_ids, *protocol.screen_case_ids)

    assert len(selected) == len(set(selected)) == 256
    assert all(cases[case_id].role is EvidenceRole.TRAIN for case_id in selected)
    assert all(
        cases[case_id].status is PublishedCaseStatus.ACTIVE for case_id in selected
    )
    assert all(
        cases[case_id].role is EvidenceRole.DEVELOPMENT
        for case_id in protocol.development_case_ids
    )
    assert all(case.role is not EvidenceRole.HOLDOUT for case in manifest.cases)

    axes = protocol.derived_parameter_names
    populations: dict[tuple[int, ...], int] = {}
    screen: dict[tuple[int, ...], int] = {}
    for case_id in selected:
        key = tuple(cases[case_id].strata[axis] for axis in axes)
        populations[key] = populations.get(key, 0) + 1
        if case_id in protocol.screen_case_ids:
            screen[key] = screen.get(key, 0) + 1
    assert all(
        abs(screen.get(key, 0) - count / 4.0) < 1.0
        for key, count in populations.items()
    )


def test_protocol_validation_refuses_overlap_and_hidden_roles() -> None:
    protocol = build_architecture_screen_protocol(_manifest())
    with pytest.raises(ValueError, match="disjoint"):
        replace(
            protocol,
            screen_case_ids=(protocol.fit_case_ids[0], *protocol.screen_case_ids[1:]),
        )
    with pytest.raises(ValueError, match="64 DEVELOPMENT"):
        replace(protocol, development_case_ids=protocol.development_case_ids[:-1])
    with pytest.raises(ValueError, match="candidate"):
        replace(protocol, candidates=protocol.candidates[:-1])
    with pytest.raises(ValueError, match="initialization_policy_id"):
        replace(protocol, initialization_policy_id="independent-default-v1")
    with pytest.raises(ValueError, match="sampling_schedule_policy_id"):
        replace(protocol, sampling_schedule_policy_id="candidate-specific-v1")
    with pytest.raises(ValueError, match="objective_policy_id"):
        replace(protocol, objective_policy_id="full-S1-hybrid")
    with pytest.raises(ValueError, match="deferred_physics_objective_terms"):
        replace(protocol, deferred_physics_objective_terms=("traction_free",))
    with pytest.raises(ValueError, match="normalization_policy_id"):
        replace(protocol, normalization_policy_id="fit-on-all-splits")
    with pytest.raises(ValueError, match="boundary_point_count"):
        replace(protocol, boundary_point_count=319)
    with pytest.raises(ValueError, match="source_access_policy_id"):
        replace(protocol, source_access_policy_id="unverified-all-roles")
    with pytest.raises(ValueError, match="development_confirmation_policy_id"):
        replace(protocol, development_confirmation_policy_id="best-development-model")


def test_selection_uses_bias_and_field_guardrails_then_declared_ties() -> None:
    protocol = build_architecture_screen_protocol(_manifest())
    baseline = _metric(
        BASELINE_CANDIDATE_ID,
        signed=-0.20,
        absolute=0.25,
        p90=0.40,
        u_l2=0.10,
        p_l2=0.20,
    )
    derived = _metric(
        "B-derived-ratio-film",
        signed=-0.15,
        absolute=0.20,
        p90=0.38,
        u_l2=0.10,
        p_l2=0.20,
    )
    local = _metric(
        "C-local-stress-enrichment",
        signed=-0.10,
        absolute=0.18,
        p90=0.37,
        u_l2=0.101,
        p_l2=0.202,
        parameters=410_000,
    )
    combined = _metric(
        "D-derived-local-stress",
        signed=-0.10,
        absolute=0.18,
        p90=0.36,
        u_l2=0.101,
        p_l2=0.202,
        parameters=420_000,
    )

    selection = select_architecture_candidate(
        protocol,
        (baseline, derived, local, combined),
    )

    assert selection.selected_candidate_id == combined.candidate_id
    assert selection.fallback_to_baseline is False
    assert selection.eligible_candidate_ids == (
        derived.candidate_id,
        local.candidate_id,
        combined.candidate_id,
    )


def test_selection_falls_back_when_improvement_or_guardrails_fail() -> None:
    protocol = build_architecture_screen_protocol(_manifest())
    baseline = _metric(
        BASELINE_CANDIDATE_ID,
        signed=-0.20,
        absolute=0.25,
        p90=0.40,
        u_l2=0.10,
        p_l2=0.20,
    )
    insufficient = _metric(
        "B-derived-ratio-film",
        signed=-0.17,
        absolute=0.23,
        p90=0.39,
        u_l2=0.10,
        p_l2=0.20,
    )
    field_regression = _metric(
        "C-local-stress-enrichment",
        signed=-0.10,
        absolute=0.18,
        p90=0.35,
        u_l2=0.11,
        p_l2=0.20,
    )
    tail_regression = _metric(
        "D-derived-local-stress",
        signed=-0.10,
        absolute=0.18,
        p90=0.41,
        u_l2=0.10,
        p_l2=0.20,
    )

    selection = select_architecture_candidate(
        protocol,
        (baseline, insufficient, field_regression, tail_regression),
    )

    assert selection.selected_candidate_id == BASELINE_CANDIDATE_ID
    assert selection.fallback_to_baseline is True
    assert selection.eligible_candidate_ids == ()


def test_metric_and_protocol_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="finite"):
        _metric(
            BASELINE_CANDIDATE_ID,
            signed=float("nan"),
            absolute=0.2,
            p90=0.3,
            u_l2=0.1,
            p_l2=0.2,
        )
    with pytest.raises(ValueError, match="non-negative"):
        _metric(
            BASELINE_CANDIDATE_ID,
            signed=-0.1,
            absolute=-0.2,
            p90=0.3,
            u_l2=0.1,
            p_l2=0.2,
        )
    with pytest.raises(TypeError, match="ArchitectureScreenProtocol"):
        select_architecture_candidate(
            object(),  # type: ignore[arg-type]
            (),
        )


def test_selection_contract_refuses_inconsistent_decisions() -> None:
    with pytest.raises(ValueError, match="fallback"):
        ArchitectureSelection(BASELINE_CANDIDATE_ID, (), False)
    with pytest.raises(ValueError, match="eligible"):
        ArchitectureSelection(
            BASELINE_CANDIDATE_ID,
            ("C-local-stress-enrichment",),
            True,
        )
    with pytest.raises(ValueError, match="selected"):
        ArchitectureSelection("C-local-stress-enrichment", (), False)
    with pytest.raises(ValueError, match="eligible"):
        ArchitectureSelection(
            "C-local-stress-enrichment",
            ("C-local-stress-enrichment", "C-local-stress-enrichment"),
            False,
        )
