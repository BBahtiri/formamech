from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from campaigns.dogbone_v1.s11_contingency import (
    DERIVED_PARAMETER_NAMES,
    RAW_PARAMETER_NAMES,
)
from campaigns.dogbone_v1.s11_contingency_tensors import ArchitectureCaseTensors
from campaigns.dogbone_v1.s13c_evaluate import (
    _case_diagnostics,
    _verify_binding,
    failed_guardrails,
)
from campaigns.dogbone_v1.s13c_protocol import (
    DogboneS13cProtocol,
    build_dogbone_s13c_protocol,
    build_dogbone_s13c_sampler_plan,
    build_dogbone_s13c_trainer_config,
    build_initial_operator,
)
from campaigns.dogbone_v1.s13c_trial import build_s13c_binding
from piginot.doe import EvidenceRole, PublishedCaseStatus
from piginot.training import (
    CheckpointLineage,
    NoEligibleTrialError,
    SelectionMetrics,
    TrialResult,
    select_checkpoint,
)

_DTYPE = torch.float64
_TAGS = ("bottom", "gauge_top", "left_symmetry", "right_arc", "right_grip")


@pytest.fixture(scope="module")
def protocol() -> DogboneS13cProtocol:
    return build_dogbone_s13c_protocol()


def _metrics(**changes: float | int) -> SelectionMetrics:
    values: dict[str, float | int] = {
        "median_absolute_peak_error": 0.02,
        "p90_absolute_peak_error": 0.05,
        "median_signed_peak_error": -0.01,
        "p_relative_l2": 0.05,
        "u_relative_l2": 0.01,
        "hybrid_loss": 0.003,
        "invalid_collocation_points": 0,
    }
    values.update(changes)
    return SelectionMetrics(**values)  # type: ignore[arg-type]


def _result(
    protocol: DogboneS13cProtocol,
    metrics: SelectionMetrics,
    role: EvidenceRole = EvidenceRole.TRAIN,
) -> TrialResult:
    (trial_id,) = protocol.search_space.trial_ids
    return TrialResult(
        lineage=CheckpointLineage(
            trial_id=trial_id,
            run_hash="1" * 64,
            dataset_snapshot_hash=protocol.dataset_snapshot_hash,
            execution_profile_hash="2" * 64,
            weights_hash="3" * 64,
            model_state_hash="4" * 64,
            checkpoint_hash="5" * 64,
        ),
        role=role,
        metrics=metrics,
    )


def test_frozen_selection_admits_a_passing_trial(protocol: DogboneS13cProtocol) -> None:
    metrics = _metrics()
    decision = select_checkpoint(
        (_result(protocol, metrics),), policy=protocol.selection_policy
    )

    assert failed_guardrails(protocol, metrics) == ()
    assert decision.selected_result.metrics == metrics


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"invalid_collocation_points": 1}, ("invalid_collocation_points",)),
        ({"median_signed_peak_error": -0.0699}, ("median_signed_peak_error",)),
        ({"median_absolute_peak_error": 0.081}, ("median_absolute_peak_error",)),
        ({"p90_absolute_peak_error": 0.137}, ("p90_absolute_peak_error",)),
        ({"u_relative_l2": 0.0381}, ("u_relative_l2",)),
        ({"p_relative_l2": 0.1959}, ("p_relative_l2",)),
    ],
)
def test_each_frozen_guardrail_blocks_selection(
    protocol: DogboneS13cProtocol,
    changes: dict[str, float | int],
    expected: tuple[str, ...],
) -> None:
    metrics = _metrics(**changes)

    assert failed_guardrails(protocol, metrics) == expected
    with pytest.raises(NoEligibleTrialError):
        select_checkpoint(
            (_result(protocol, metrics),), policy=protocol.selection_policy
        )


def test_historical_s13_failures_stay_ineligible(protocol: DogboneS13cProtocol) -> None:
    s13_trial_zero = _metrics(
        median_absolute_peak_error=0.12254976158510011,
        p90_absolute_peak_error=0.17333798085407623,
        median_signed_peak_error=-0.12254976158510011,
        p_relative_l2=0.14235578996279002,
        u_relative_l2=0.03502205934491044,
    )
    s13b_peak_arm = _metrics(
        median_absolute_peak_error=0.04312717246626491,
        p90_absolute_peak_error=0.0764694613009652,
        median_signed_peak_error=-0.041109291783627606,
        p_relative_l2=0.16615034385916377,
        u_relative_l2=0.036733,
        invalid_collocation_points=1,
    )

    assert failed_guardrails(protocol, s13_trial_zero) == (
        "median_signed_peak_error",
        "median_absolute_peak_error",
        "p90_absolute_peak_error",
    )
    assert failed_guardrails(protocol, s13b_peak_arm) == ("invalid_collocation_points",)


def test_selection_refuses_non_train_evidence(protocol: DogboneS13cProtocol) -> None:
    with pytest.raises(PermissionError):
        select_checkpoint(
            (_result(protocol, _metrics(), EvidenceRole.HOLDOUT),),
            policy=protocol.selection_policy,
        )


def _tensors() -> ArchitectureCaseTensors:
    query_xy = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]], dtype=_DTYPE
    )
    target_p = torch.zeros((5, 2, 2), dtype=_DTYPE)
    target_p[:, 0, 0] = torch.tensor([0.1, 0.2, 0.3, 0.5, 0.4], dtype=_DTYPE)
    boundary_tags = torch.zeros((4, len(_TAGS)), dtype=_DTYPE)
    boundary_tags[torch.arange(4), torch.arange(4)] = 1.0
    return ArchitectureCaseTensors(
        case_id="candidate-test",
        leakage_group_id="group-test",
        role=EvidenceRole.TRAIN,
        status=PublishedCaseStatus.ACTIVE,
        parameter_names=(*RAW_PARAMETER_NAMES, *DERIVED_PARAMETER_NAMES),
        normalized_parameters=torch.zeros(11, dtype=_DTYPE),
        normalization_hash="6" * 64,
        boundary_tag_names=_TAGS,
        boundary_xy=torch.tensor(
            [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype=_DTYPE
        ),
        boundary_tags=boundary_tags,
        boundary_normals=torch.tensor(
            [[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=_DTYPE
        ),
        boundary_weights=torch.full((4,), 0.25, dtype=_DTYPE),
        boundary_mask=torch.ones(4, dtype=torch.bool),
        query_xy=query_xy,
        target_u=0.1 + query_xy,
        target_P=target_p,
        qoi_indices=torch.tensor([3, 4], dtype=torch.int64),
    )


def test_case_diagnostics_locate_the_peak_and_split_root_errors() -> None:
    tensors = _tensors()
    predicted_p = tensors.target_P.clone()
    predicted_p[3, 0, 0] = 0.3  # true root peak at vertex 3 is underpredicted
    predicted_p[4, 0, 0] = 0.45  # predicted root peak moves to vertex 4

    diagnostics = _case_diagnostics(
        tensors,
        tensors.target_u.clone(),
        predicted_p,
        torch.ones(5, dtype=_DTYPE),
    )

    assert diagnostics.reference_peak_vertex == 3
    assert diagnostics.predicted_peak_vertex == 4
    assert diagnostics.peak_vertex_distance_mm == pytest.approx(1.0)
    assert diagnostics.signed_peak_error == pytest.approx((0.45 - 0.5) / 0.5)
    assert diagnostics.u_relative_l2 == 0.0
    assert diagnostics.non_root_p_relative_l2 == 0.0
    assert diagnostics.root_p_relative_l2 > 0.0
    assert diagnostics.minimum_vertex_determinant == 1.0


def test_binding_verification_rejects_foreign_runs(
    protocol: DogboneS13cProtocol,
) -> None:
    binding = build_s13c_binding(
        protocol,
        model=build_initial_operator(),
        config=build_dogbone_s13c_trainer_config(protocol),
        plan=build_dogbone_s13c_sampler_plan(protocol),
        implementation_revision="0" * 40,
    )
    _verify_binding(binding, protocol)
    with pytest.raises(ValueError, match="objective_registry_hash"):
        _verify_binding(replace(binding, objective_registry_hash="7" * 64), protocol)
    with pytest.raises(ValueError, match="execution profile"):
        _verify_binding(
            replace(
                binding,
                execution_profile=replace(binding.execution_profile, blas_threads=1),
            ),
            protocol,
        )
