from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from piginot.contracts.errors import ContractValidationError
from piginot.doe import EvidenceRole
from piginot.training.search import GridSearchSpace
from piginot.training.selection import (
    CHOOSE_NONRANKED_CHECKPOINT_MUTANT,
    SELECT_ON_HOLDOUT_MUTANT,
    CheckpointLineage,
    NoEligibleTrialError,
    SelectionDecision,
    SelectionMetrics,
    SelectionPolicy,
    TrialResult,
    _choose_nonranked_checkpoint_mutant_for_test,
    _select_on_holdout_mutant_for_test,
    select_checkpoint,
)

_HASHES = tuple(f"{value:x}" * 64 for value in range(1, 7))


def _space() -> GridSearchSpace:
    return GridSearchSpace(
        learning_rates=(5e-5, 1e-4),
        physics_weight_scales=(0.1, 1.0),
        optimizer_steps=400,
        case_batch_size=4,
        maximum_trials=4,
    )


def _policy() -> SelectionPolicy:
    return SelectionPolicy(_space())


def _metrics(
    *,
    median_abs: float = 0.075,
    p90_abs: float = 0.13,
    median_signed: float = -0.065,
    p_relative: float = 0.18,
    u_relative: float = 0.037,
    hybrid_loss: float = 1.0,
    invalid_collocation_points: int = 0,
) -> SelectionMetrics:
    return SelectionMetrics(
        median_absolute_peak_error=median_abs,
        p90_absolute_peak_error=p90_abs,
        median_signed_peak_error=median_signed,
        p_relative_l2=p_relative,
        u_relative_l2=u_relative,
        hybrid_loss=hybrid_loss,
        invalid_collocation_points=invalid_collocation_points,
    )


def _result(
    index: int,
    *,
    metrics: SelectionMetrics | None = None,
    role: EvidenceRole = EvidenceRole.TRAIN,
) -> TrialResult:
    return TrialResult(
        lineage=CheckpointLineage(
            trial_id=_space().trial_ids[index],
            run_hash=_HASHES[0],
            dataset_snapshot_hash=_HASHES[1],
            execution_profile_hash=_HASHES[2],
            weights_hash=_HASHES[3],
            model_state_hash=_HASHES[4],
            checkpoint_hash=_HASHES[5],
        ),
        role=role,
        metrics=_metrics() if metrics is None else metrics,
    )


def test_policy_binds_exact_search_space_and_fixed_s11_guardrails() -> None:
    policy = _policy()
    guardrails = policy.guardrails

    assert policy.search_space_hash == _space().search_space_hash
    assert policy.registered_trial_ids == _space().trial_ids
    assert guardrails.incumbent_median_absolute_peak_error == 0.08085713846866197
    assert guardrails.incumbent_p90_absolute_peak_error == 0.13634630217960156
    assert guardrails.incumbent_median_signed_peak_error == -0.07762409959121884
    assert guardrails.incumbent_u_relative_l2 == 0.03730866424082414
    assert guardrails.incumbent_p_relative_l2 == 0.1919960359979199
    assert guardrails.minimum_underprediction_reduction == 0.10
    assert guardrails.maximum_field_relative_regression == 0.02

    with pytest.raises(ContractValidationError, match="fixed"):
        replace(guardrails, minimum_underprediction_reduction=0.05)
    with pytest.raises(ContractValidationError, match="search_space"):
        SelectionPolicy(object())  # type: ignore[arg-type]


def test_selection_is_input_order_independent_and_uses_full_ranking_tuple() -> None:
    results = (
        _result(0, metrics=_metrics(p90_abs=0.13, p_relative=0.17)),
        _result(1, metrics=_metrics(p90_abs=0.12, p_relative=0.19)),
        _result(
            2,
            metrics=_metrics(
                p90_abs=0.12,
                median_signed=-0.05,
                p_relative=0.18,
                u_relative=0.036,
                hybrid_loss=0.8,
            ),
        ),
        _result(
            3,
            metrics=_metrics(
                p90_abs=0.12,
                median_signed=-0.05,
                p_relative=0.18,
                u_relative=0.035,
                hybrid_loss=0.7,
            ),
        ),
    )

    forward = select_checkpoint(results, policy=_policy())
    reverse = select_checkpoint(tuple(reversed(results)), policy=_policy())

    assert forward == reverse
    assert forward.selected_result.trial_id == results[3].trial_id
    assert forward.ranked_eligible_trial_ids == tuple(
        result.trial_id for result in (results[3], results[2], results[1], results[0])
    )
    assert forward.role is EvidenceRole.TRAIN
    assert forward.selected_lineage.checkpoint_hash == _HASHES[5]
    assert forward.selected_result.run_hash == _HASHES[0]
    assert forward.selected_result.dataset_snapshot_hash == _HASHES[1]
    assert forward.selected_result.execution_profile_hash == _HASHES[2]
    assert forward.selected_result.weights_hash == _HASHES[3]
    assert forward.selected_result.model_state_hash == _HASHES[4]


@pytest.mark.parametrize(
    "metrics",
    (
        _metrics(median_signed=-0.075),
        _metrics(median_abs=0.081),
        _metrics(p90_abs=0.137),
        _metrics(u_relative=0.0381),
        _metrics(p_relative=0.196),
    ),
)
def test_each_guardrail_is_independently_load_bearing(
    metrics: SelectionMetrics,
) -> None:
    results = tuple(_result(index, metrics=metrics) for index in range(4))

    with pytest.raises(NoEligibleTrialError, match="CP-FREEZE"):
        select_checkpoint(results, policy=_policy())


def test_positive_signed_bias_has_zero_underprediction_magnitude() -> None:
    results = tuple(
        _result(index, metrics=_metrics(median_signed=0.01 + index * 0.001))
        for index in range(4)
    )

    selected = select_checkpoint(results, policy=_policy())

    assert selected.selected_result.metrics.median_signed_peak_error > 0.0


def test_folded_collocation_points_make_a_trial_ineligible() -> None:
    invalid = _metrics(invalid_collocation_points=1)
    results = tuple(_result(index, metrics=invalid) for index in range(4))

    with pytest.raises(NoEligibleTrialError, match="CP-FREEZE"):
        select_checkpoint(results, policy=_policy())

    with pytest.raises(ContractValidationError, match="invalid_collocation_points"):
        _metrics(invalid_collocation_points=-1)
    with pytest.raises(ContractValidationError, match="invalid_collocation_points"):
        _metrics(invalid_collocation_points=True)  # type: ignore[arg-type]
    with pytest.raises(ContractValidationError, match="invalid_collocation_points"):
        _metrics(invalid_collocation_points=0.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "role",
    (
        EvidenceRole.PILOT,
        EvidenceRole.DEVELOPMENT,
        EvidenceRole.HOLDOUT,
        EvidenceRole.LEGACY,
    ),
)
def test_selection_refuses_every_non_train_result(role: EvidenceRole) -> None:
    results = tuple(
        _result(index, role=role if index == 0 else EvidenceRole.TRAIN)
        for index in range(4)
    )

    with pytest.raises(PermissionError, match=role.value):
        select_checkpoint(results, policy=_policy())


def test_selection_requires_exactly_one_result_per_registered_trial() -> None:
    complete = tuple(_result(index) for index in range(4))
    with pytest.raises(ContractValidationError, match="exactly once"):
        select_checkpoint(complete[:-1], policy=_policy())
    with pytest.raises(ContractValidationError, match="duplicate"):
        select_checkpoint((*complete[:-1], complete[0]), policy=_policy())
    with pytest.raises(ContractValidationError, match="registered"):
        select_checkpoint(
            (
                *complete[:-1],
                replace(
                    complete[-1],
                    lineage=replace(
                        complete[-1].lineage,
                        trial_id="trial-outside",
                    ),
                ),
            ),
            policy=_policy(),
        )


def test_metrics_role_and_lineage_fail_closed() -> None:
    with pytest.raises(ContractValidationError, match="finite"):
        _metrics(hybrid_loss=float("nan"))
    with pytest.raises(ContractValidationError, match="non-negative"):
        _metrics(p_relative=-0.1)
    with pytest.raises(ContractValidationError, match="metrics"):
        replace(_result(0), metrics=object())  # type: ignore[arg-type]
    with pytest.raises(ContractValidationError, match="role"):
        replace(_result(0), role="TRAIN")  # type: ignore[arg-type]
    for field in (
        "run_hash",
        "dataset_snapshot_hash",
        "execution_profile_hash",
        "weights_hash",
        "model_state_hash",
        "checkpoint_hash",
    ):
        with pytest.raises(ContractValidationError, match=field):
            replace(_result(0).lineage, **{field: "NOT-A-DIGEST"})


def test_mutant_helpers_demonstrate_both_forbidden_selection_paths() -> None:
    train = _result(0, metrics=_metrics(median_abs=0.07))
    holdout = _result(
        1,
        metrics=_metrics(median_abs=0.01),
        role=EvidenceRole.HOLDOUT,
    )
    decision = select_checkpoint(
        (
            train,
            _result(1, metrics=_metrics(median_abs=0.071)),
            _result(2, metrics=_metrics(median_abs=0.072)),
            _result(3, metrics=_metrics(median_abs=0.073)),
        ),
        policy=_policy(),
    )

    assert SELECT_ON_HOLDOUT_MUTANT == "SELECT_ON_HOLDOUT"
    assert CHOOSE_NONRANKED_CHECKPOINT_MUTANT == "CHOOSE_NONRANKED_CHECKPOINT"
    assert (
        _select_on_holdout_mutant_for_test(
            _space(),
            (train, holdout),
            role=EvidenceRole.HOLDOUT,
        ).trial_id
        == holdout.trial_id
    )
    assert (
        _choose_nonranked_checkpoint_mutant_for_test(decision).trial_id
        != decision.selected_result.trial_id
    )
    with pytest.raises(PermissionError, match="HOLDOUT"):
        select_checkpoint(
            (train, holdout, _result(2), _result(3)),
            policy=_policy(),
        )


def test_decision_constructor_rejects_a_nonranked_order() -> None:
    valid = select_checkpoint(
        tuple(_result(index) for index in range(4)), policy=_policy()
    )

    with pytest.raises(ContractValidationError, match="declared ranking"):
        SelectionDecision(
            policy=valid.policy,
            results=valid.results,
            ranked_eligible_trial_ids=tuple(reversed(valid.ranked_eligible_trial_ids)),
        )


def test_selection_contracts_are_frozen() -> None:
    selected = select_checkpoint(
        tuple(_result(index) for index in range(4)), policy=_policy()
    )

    with pytest.raises(FrozenInstanceError):
        selected.results = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        selected.selected_result.lineage.trial_id = "other"  # type: ignore[misc]


def test_absolute_guardrails_admit_only_within_every_declared_limit() -> None:
    from piginot.training import AbsoluteSelectionGuardrails

    guardrails = AbsoluteSelectionGuardrails(0.05, 0.05, 0.02)
    policy = SelectionPolicy(_space(), guardrails)
    inside = _metrics(median_abs=0.04, p_relative=0.03, u_relative=0.01)
    assert guardrails.admits(inside)
    for change in (
        {"median_abs": 0.051},
        {"p_relative": 0.06},
        {"u_relative": 0.021},
        {"invalid_collocation_points": 1},
    ):
        values = {
            "median_abs": 0.04,
            "p_relative": 0.03,
            "u_relative": 0.01,
        } | change
        assert not guardrails.admits(_metrics(**values))  # type: ignore[arg-type]
    results = tuple(
        _result(index, metrics=replace(inside, hybrid_loss=float(index)))
        for index in range(4)
    )
    decision = select_checkpoint(results, policy=policy)
    assert decision.selected_result.trial_id == _space().trial_ids[0]
    assert policy.policy_hash != _policy().policy_hash
    with pytest.raises(NoEligibleTrialError):
        select_checkpoint(
            tuple(_result(i, metrics=_metrics()) for i in range(4)), policy=policy
        )
    with pytest.raises(ContractValidationError, match="positive"):
        AbsoluteSelectionGuardrails(0.0, 0.05, 0.02)
    with pytest.raises(ContractValidationError, match="finite"):
        AbsoluteSelectionGuardrails(float("nan"), 0.05, 0.02)
    with pytest.raises(TypeError):
        guardrails.admits(object())  # type: ignore[arg-type]
    with pytest.raises(ContractValidationError, match="guardrails"):
        SelectionPolicy(_space(), object())  # type: ignore[arg-type]
