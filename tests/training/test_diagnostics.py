from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from piginot.doe import EvidenceRole
from piginot.objectives import (
    EquilibriumForm,
    ObjectiveMode,
    ObjectiveRegistry,
    ObjectiveResiduals,
    ObjectiveTerm,
)
from piginot.training import AdaptiveWeightConfig, AdaptiveWeightState
from piginot.training.diagnostics import (
    AdaptiveTargetDiagnostic,
    ObjectiveGradientDiagnostic,
    PairwiseGradientDiagnostic,
    TermGradientDiagnostic,
    measure_objective_gradient_diagnostic,
)
from piginot.training.engine import TrainingPopulation

_HASH = "0123456789abcdef" * 4


class _VectorModel(torch.nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([1.0, 2.0], dtype=torch.float64)
        )
        self.unused = torch.nn.Parameter(torch.tensor([7.0], dtype=torch.float64))


class _AnalyticObjective:
    def __init__(self, *, zero_data_u: bool = False) -> None:
        self.calls: list[tuple[tuple[str, ...], int]] = []
        self.zero_data_u = zero_data_u

    def __call__(
        self,
        model: torch.nn.Module,
        sample_ids: tuple[str, ...],
        step: int,
    ) -> ObjectiveResiduals:
        self.calls.append((sample_ids, step))
        assert type(model) is _VectorModel
        left, right = model.weight.unbind()
        data_u = left * 0.0 if self.zero_data_u else left
        return ObjectiveResiduals(
            {
                ObjectiveTerm.DATA_U: data_u.reshape(1),
                ObjectiveTerm.DATA_P: right.reshape(1),
                ObjectiveTerm.DATA_PEAK: (left - right).reshape(1),
                ObjectiveTerm.EQUILIBRIUM_STRONG: (left + right).reshape(1),
                ObjectiveTerm.CONSTITUTIVE: left.reshape(1),
                ObjectiveTerm.ANGULAR: right.reshape(1),
                ObjectiveTerm.DIRICHLET: (2.0 * left).reshape(1),
                ObjectiveTerm.TRACTION_FREE: (left - right).reshape(1),
            }
        )


def _population() -> TrainingPopulation:
    return TrainingPopulation(
        dataset_snapshot_hash=_HASH,
        public_manifest_hash="1" * 64,
        role=EvidenceRole.TRAIN,
        sample_ids=("train-001", "train-002"),
    )


def _registry() -> ObjectiveRegistry:
    return ObjectiveRegistry(
        mode=ObjectiveMode.HYBRID,
        equilibrium_form=EquilibriumForm.STRONG,
        term_weights={ObjectiveTerm.DATA_PEAK: 1.0},
    )


def _adaptive_state() -> AdaptiveWeightState:
    return AdaptiveWeightState(
        multipliers={
            ObjectiveTerm.CONSTITUTIVE: 2.0,
            ObjectiveTerm.ANGULAR: 3.0,
            ObjectiveTerm.DIRICHLET: 4.0,
            ObjectiveTerm.TRACTION_FREE: 5.0,
        },
        update_count=1,
        last_update_step=100,
    )


def _measure(
    model: _VectorModel,
    objective: _AnalyticObjective,
    *,
    role: EvidenceRole = EvidenceRole.TRAIN,
) -> ObjectiveGradientDiagnostic:
    return measure_objective_gradient_diagnostic(
        model,
        population=_population(),
        sample_ids=("train-002",),
        step=800,
        base_registry=_registry(),
        physics_weight_scale=0.1,
        adaptive_config=AdaptiveWeightConfig(
            beta=0.1,
            warmup_steps=200,
            update_period=100,
            minimum_multiplier=0.5,
            maximum_multiplier=2.0,
        ),
        adaptive_state=_adaptive_state(),
        objective=objective,
        role=role,
    )


def _term(
    report: ObjectiveGradientDiagnostic,
    term: ObjectiveTerm,
) -> TermGradientDiagnostic:
    return next(item for item in report.terms if item.term is term)


def _pair(
    report: ObjectiveGradientDiagnostic,
    left: ObjectiveTerm,
    right: ObjectiveTerm,
) -> PairwiseGradientDiagnostic:
    return next(
        item
        for item in report.pairs
        if item.left_term is left and item.right_term is right
    )


def _target(
    report: ObjectiveGradientDiagnostic,
    term: ObjectiveTerm,
) -> AdaptiveTargetDiagnostic:
    return next(item for item in report.adaptive_targets if item.term is term)


def test_measurement_reports_losses_full_l2_gradients_and_conflicts() -> None:
    model = _VectorModel()

    report = _measure(model, _AnalyticObjective())

    data_u = _term(report, ObjectiveTerm.DATA_U)
    assert data_u.unweighted_loss == pytest.approx(1.0)
    assert data_u.effective_weight == pytest.approx(1.0)
    assert data_u.weighted_loss == pytest.approx(1.0)
    assert data_u.unweighted_gradient_l2 == pytest.approx(2.0)
    assert data_u.weighted_gradient_l2 == pytest.approx(2.0)

    constitutive = _term(report, ObjectiveTerm.CONSTITUTIVE)
    assert constitutive.effective_weight == pytest.approx(0.2)
    assert constitutive.unweighted_gradient_l2 == pytest.approx(2.0)
    assert constitutive.weighted_gradient_l2 == pytest.approx(0.4)

    orthogonal = _pair(report, ObjectiveTerm.DATA_U, ObjectiveTerm.DATA_P)
    assert orthogonal.cosine == pytest.approx(0.0)
    assert orthogonal.conflict is False
    conflict = _pair(report, ObjectiveTerm.DATA_U, ObjectiveTerm.DATA_PEAK)
    assert conflict.cosine == pytest.approx(-(2.0**-0.5))
    assert conflict.conflict is True
    assert conflict.weighted_dot_product < 0.0


def test_measurement_exposes_exact_adaptive_targets_and_cap_flags() -> None:
    report = _measure(_VectorModel(), _AnalyticObjective())

    constitutive = _target(report, ObjectiveTerm.CONSTITUTIVE)
    assert constitutive.reference_statistic == pytest.approx(6.0)
    assert constitutive.term_statistic == pytest.approx(1.0)
    assert constitutive.raw_target == pytest.approx(6.0)
    assert constitutive.bounded_target == pytest.approx(2.0)
    assert constitutive.hit_minimum is False
    assert constitutive.hit_maximum is True
    assert constitutive.current_multiplier == pytest.approx(2.0)
    assert constitutive.projected_multiplier == pytest.approx(2.0)

    dirichlet = _target(report, ObjectiveTerm.DIRICHLET)
    assert dirichlet.raw_target == pytest.approx(1.5)
    assert dirichlet.bounded_target == pytest.approx(1.5)
    assert dirichlet.hit_minimum is False
    assert dirichlet.hit_maximum is False
    assert dirichlet.projected_multiplier == pytest.approx(3.75)


def test_measurement_is_repeatable_and_restores_runtime_state() -> None:
    model = _VectorModel()
    model.train()
    model.weight.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
    before_parameters = tuple(parameter.detach().clone() for parameter in model.parameters())
    before_gradient = model.weight.grad.clone()
    torch.manual_seed(991)
    before_rng = torch.random.get_rng_state().clone()
    objective = _AnalyticObjective()

    first = _measure(model, objective)
    second = _measure(model, objective)

    assert first == second
    assert model.training
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    assert model.weight.grad is not None
    assert torch.equal(model.weight.grad, before_gradient)
    for parameter, expected in zip(model.parameters(), before_parameters, strict=True):
        assert torch.equal(parameter, expected)


@pytest.mark.parametrize("role", (EvidenceRole.DEVELOPMENT, EvidenceRole.HOLDOUT))  # type: ignore[untyped-decorator]
def test_measurement_refuses_protected_roles_before_objective_access(
    role: EvidenceRole,
) -> None:
    objective = _AnalyticObjective()

    with pytest.raises(PermissionError, match=role.value):
        _measure(_VectorModel(), objective, role=role)

    assert objective.calls == []


def test_measurement_refuses_unknown_ids_before_objective_access() -> None:
    objective = _AnalyticObjective()

    with pytest.raises(ValueError, match="sample_ids"):
        measure_objective_gradient_diagnostic(
            _VectorModel(),
            population=_population(),
            sample_ids=("unknown",),
            step=800,
            base_registry=_registry(),
            physics_weight_scale=0.1,
            adaptive_config=AdaptiveWeightConfig(),
            adaptive_state=_adaptive_state(),
            objective=objective,
            role=EvidenceRole.TRAIN,
        )

    assert objective.calls == []


def test_zero_data_gradient_has_defined_nonconflicting_pairs() -> None:
    report = _measure(_VectorModel(), _AnalyticObjective(zero_data_u=True))

    data_u = _term(report, ObjectiveTerm.DATA_U)
    assert data_u.unweighted_gradient_l2 == 0.0
    pair = _pair(report, ObjectiveTerm.DATA_U, ObjectiveTerm.DATA_P)
    assert pair.cosine is None
    assert pair.conflict is False


def test_diagnostic_contracts_are_frozen() -> None:
    report = _measure(_VectorModel(), _AnalyticObjective())

    with pytest.raises(FrozenInstanceError):
        report.step = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        report.terms[0].weighted_loss = 0.0  # type: ignore[misc]
