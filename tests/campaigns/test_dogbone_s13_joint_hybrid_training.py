from __future__ import annotations

import copy
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from campaigns.dogbone_v1 import s13_joint_hybrid_training as training
from campaigns.dogbone_v1 import s13_objective as historical
from campaigns.dogbone_v1.s11_contingency_models import model_state_hash
from campaigns.dogbone_v1.s13_joint_hybrid_protocol import JointHybridPlan
from campaigns.dogbone_v1.s13_overfit_diagnostic import (
    _build_vertex_sampler_plan,
    build_s13_overfit_plan,
)
from piginot.contracts import canonical_json_bytes, hash_spec
from piginot.datasets import (
    CollocationKind,
    SemanticCollocationQuota,
    sample_request_collocation,
)
from piginot.objectives import ObjectiveResiduals, ObjectiveTerm
from piginot.operators import LiftingVerification, module_digest
from piginot.training import DeterministicSamplerPlan
from tests.campaigns.test_dogbone_s13_objective import _model, _snapshot
from tests.campaigns.test_dogbone_s13_screen import _case


def _sampler() -> DeterministicSamplerPlan:
    return DeterministicSamplerPlan(
        sampler_id="synthetic-joint-vertices",
        population_size=4,
        batches=((2, 0), (1, 3)),
    )


def _run(arm_id: str = "control", *, evaluate=None):  # type: ignore[no-untyped-def]
    plan = JointHybridPlan()
    arms = training.clone_joint_arms(_model(), plan)
    state = arms[0 if arm_id == "control" else 1]
    result = training._train_arm_steps(  # noqa: SLF001
        state,
        training.SampledJointObjective(_case("candidate-002485")),
        _sampler(),
        plan,
        optimizer_steps=2,
        evaluation_steps=(0, 1, 2),
        evaluate=evaluate,
    )
    return state, result


def test_sampler_is_the_complete_pinned_v5_57_of_969_schedule() -> None:
    plan = JointHybridPlan()
    sampler = training.build_joint_vertex_sampler(plan)

    assert sampler == _build_vertex_sampler_plan(
        build_s13_overfit_plan(version="v5"), 969
    )
    assert sampler.plan_hash == plan.vertex_sampler_plan_hash
    assert len(sampler.batches) == 4000
    assert all(len(batch) == 57 for batch in sampler.batches)
    assert sorted(index for batch in sampler.batches[:17] for index in batch) == list(
        range(969)
    )


def test_selected_data_matches_historical_residuals_and_reuses_physics() -> None:
    case = _case("candidate-002485")
    model = _model()
    indices = torch.tensor((2, 0), dtype=torch.long)
    selected = training.SampledJointObjective(case)(model, indices, candidate=True)
    full = historical.DogboneS13ObjectiveBuilder((case,))(
        model, (case.tensors.case_id,), 1
    )

    assert tuple(selected.values) == historical.DOGBONE_S13_OBJECTIVE_TERMS
    for term in historical.DOGBONE_S13_OBJECTIVE_TERMS:
        expected = full.values[term]
        if term in (ObjectiveTerm.DATA_U, ObjectiveTerm.DATA_P):
            expected = expected.index_select(1, indices)
        torch.testing.assert_close(
            selected.values[term], expected, rtol=1e-12, atol=1e-14
        )
    assert selected.values[ObjectiveTerm.DATA_U].shape == (1, 2, 2)
    assert selected.values[ObjectiveTerm.DATA_P].shape == (1, 2, 2, 2)


def test_control_never_calls_physics_or_adds_orientation_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object) -> object:
        raise AssertionError("control must not invoke physics")

    monkeypatch.setattr(training, "_physics_residuals", forbidden)
    residuals = training.SampledJointObjective(_case("candidate-002485"))(
        _model(), torch.tensor((0, 3), dtype=torch.long), candidate=False
    )
    assert tuple(residuals.values) == (ObjectiveTerm.DATA_U, ObjectiveTerm.DATA_P)


@pytest.mark.parametrize(
    "indices",
    (
        torch.tensor((), dtype=torch.long),
        torch.tensor((0, 0), dtype=torch.long),
        torch.tensor((-1, 1), dtype=torch.long),
        torch.tensor((0, 4), dtype=torch.long),
        torch.tensor((0.0, 1.0), dtype=torch.float64),
        torch.tensor(((0, 1),), dtype=torch.long),
    ),
)
def test_sampled_objective_rejects_invalid_indices(indices: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="vertex_indices"):
        training.SampledJointObjective(_case("candidate-002485"))(
            _model(), indices, candidate=False
        )


def test_clones_are_exact_independent_all_trainable_with_fresh_adam() -> None:
    baseline = _model()
    for parameter in baseline.parameters():
        parameter.requires_grad_(False)
    initial_hash = model_state_hash(baseline)
    control, candidate = training.clone_joint_arms(baseline, JointHybridPlan())

    assert control.arm_id == "control" and candidate.arm_id == "candidate"
    assert (
        model_state_hash(control.model)
        == model_state_hash(candidate.model)
        == initial_hash
    )
    assert not any(parameter.requires_grad for parameter in baseline.parameters())
    for left, right, original in zip(
        control.model.parameters(),
        candidate.model.parameters(),
        baseline.parameters(),
        strict=True,
    ):
        assert left.requires_grad and right.requires_grad
        assert len({left.data_ptr(), right.data_ptr(), original.data_ptr()}) == 3
    for state in (control, candidate):
        assert not state.optimizer.state
        assert state.optimizer.defaults["lr"] == 2e-5
        assert state.optimizer.defaults["betas"] == (0.9, 0.999)
        assert state.optimizer.defaults["eps"] == 1e-8
        assert state.optimizer.defaults["weight_decay"] == 0.0
        assert state.optimizer.defaults["foreach"] is False
        assert state.optimizer.defaults["fused"] is False
        assert tuple(state.optimizer.param_groups[0]["params"]) == tuple(
            state.model.parameters()
        )


@pytest.mark.parametrize("arm_id", ("control", "candidate"))
def test_synthetic_training_records_shared_batches_static_terms_clipping_and_steps(
    arm_id: str,
) -> None:
    state, result = _run(arm_id)
    weights = dict(
        JointHybridPlan().control_term_weights
        if arm_id == "control"
        else JointHybridPlan().candidate_term_weights
    )

    assert result.optimizer_updates == 2
    assert result.arm_id == arm_id
    assert result.sampler_plan_hash == _sampler().plan_hash
    assert (
        tuple(record.vertex_indices for record in result.step_trace)
        == _sampler().batches
    )
    assert tuple(record.step for record in result.step_trace) == (1, 2)
    assert result.parameter_manifest == training.joint_parameter_manifest(state.model)
    for record in result.step_trace:
        unweighted = dict(record.unweighted_terms)
        weighted = dict(record.weighted_terms)
        assert tuple(unweighted) == tuple(weights)
        assert weighted == pytest.approx(
            {name: value * weights[name] for name, value in unweighted.items()}
        )
        assert record.total_loss == pytest.approx(sum(weighted.values()))
        assert record.preclip_gradient_norm >= 0.0
        assert record.clip_coefficient == pytest.approx(
            min(1.0, 1.0 / (record.preclip_gradient_norm + 1e-6))
        )
        assert record.optimizer_updates == record.step
        assert all(value for _, value in record.finite_checks)
    assert all(
        int(value["step"].item()) == 2 for value in state.optimizer.state.values()
    )


def test_candidate_changes_joint_u_and_shared_parameters_not_only_p_head() -> None:
    baseline = _model()
    initial = {
        name: value.detach().clone() for name, value in baseline.named_parameters()
    }
    state, _ = _run("candidate")
    changed = tuple(
        name
        for name, value in state.model.named_parameters()
        if not torch.equal(value, initial[name])
    )
    assert any(name.startswith("u_head") for name in changed)
    assert any(not name.startswith(("u_head", "p_head")) for name in changed)


def test_evaluation_cannot_change_training_rng_modes_or_progress() -> None:
    seen: list[int] = []

    def evaluate(model: torch.nn.Module, step: int) -> dict[str, int]:
        assert not model.training
        seen.append(step)
        torch.rand(7)
        random.random()
        np.random.random()
        model.train()
        return {"step": step}

    plain_state, plain_result = _run()
    evaluated_state, evaluated_result = _run(evaluate=evaluate)

    assert seen == [0, 1, 2]
    assert model_state_hash(plain_state.model) == model_state_hash(
        evaluated_state.model
    )
    assert plain_result.step_trace == evaluated_result.step_trace
    assert evaluated_result.evaluations == (
        (0, {"step": 0}),
        (1, {"step": 1}),
        (2, {"step": 2}),
    )
    assert evaluated_state.model.training


def test_parameter_mutating_evaluation_is_restored_and_aborts() -> None:
    state = training.clone_joint_arms(_model(), JointHybridPlan())[0]
    before = model_state_hash(state.model)

    def mutate(model: torch.nn.Module, step: int) -> None:
        with torch.no_grad():
            next(model.parameters()).add_(1.0)

    with pytest.raises(RuntimeError, match="evaluation.*mutated"):
        training._train_arm_steps(
            state,
            training.SampledJointObjective(_case("candidate-002485")),
            _sampler(),
            JointHybridPlan(),
            optimizer_steps=2,
            evaluation_steps=(0, 2),
            evaluate=mutate,
        )  # noqa: SLF001
    assert model_state_hash(state.model) == before
    assert not state.optimizer.state


def test_optimizer_mutating_evaluation_is_restored_and_aborts() -> None:
    state = training.clone_joint_arms(_model(), JointHybridPlan())[0]
    before = copy.deepcopy(state.optimizer.state_dict())

    def mutate(model: torch.nn.Module, step: int) -> None:
        state.optimizer.param_groups[0]["lr"] = 1.0

    with pytest.raises(RuntimeError, match="evaluation.*mutated"):
        training._train_arm_steps(
            state,
            training.SampledJointObjective(_case("candidate-002485")),
            _sampler(),
            JointHybridPlan(),
            optimizer_steps=2,
            evaluation_steps=(0, 2),
            evaluate=mutate,
        )  # noqa: SLF001
    assert state.optimizer.state_dict() == before


def test_evaluator_exception_still_restores_mode_and_rng() -> None:
    state = training.clone_joint_arms(_model(), JointHybridPlan())[0]
    model_mode = state.model.training
    rng = torch.get_rng_state().clone()

    def fail(model: torch.nn.Module, step: int) -> None:
        torch.rand(5)
        model.train()
        raise LookupError("synthetic evaluation failure")

    with pytest.raises(training.JointTrainingFailure, match="synthetic"):
        training._train_arm_steps(
            state,
            training.SampledJointObjective(_case("candidate-002485")),
            _sampler(),
            JointHybridPlan(),
            optimizer_steps=2,
            evaluation_steps=(0, 2),
            evaluate=fail,
        )  # noqa: SLF001
    assert state.model.training is model_mode
    assert torch.equal(torch.get_rng_state(), rng)


def test_common_nonfinite_gradient_aborts_before_optimizer_update() -> None:
    state = training.clone_joint_arms(_model(), JointHybridPlan())[0]
    next(state.model.parameters()).register_hook(
        lambda value: torch.full_like(value, float("nan"))
    )
    with pytest.raises(training.JointTrainingFailure, match="gradient") as caught:
        training._train_arm_steps(
            state,
            training.SampledJointObjective(_case("candidate-002485")),
            _sampler(),
            JointHybridPlan(),
            optimizer_steps=2,
            evaluation_steps=(0, 2),
        )  # noqa: SLF001
    assert not state.optimizer.state
    assert caught.value.cause_type == "FloatingPointError"
    assert caught.value.optimizer_updates == 0
    assert caught.value.attempted_step == 1
    assert caught.value.step_trace == ()


def test_common_nonfinite_loss_aborts_before_optimizer_update() -> None:
    state = training.clone_joint_arms(_model(), JointHybridPlan())[0]

    def huge(
        model: torch.nn.Module, indices: torch.Tensor, *, candidate: bool
    ) -> ObjectiveResiduals:
        return ObjectiveResiduals(
            {
                ObjectiveTerm.DATA_U: next(model.parameters()).reshape(-1) * 0 + 1e300,
                ObjectiveTerm.DATA_P: next(model.parameters()).reshape(-1) * 0,
            }
        )

    with pytest.raises(training.JointTrainingFailure, match="loss"):
        training._train_arm_steps(
            state,
            huge,
            _sampler(),
            JointHybridPlan(),
            optimizer_steps=2,
            evaluation_steps=(0, 2),
        )  # noqa: SLF001
    assert not state.optimizer.state


def test_valid_deformation_material_kernel_failure_is_not_repaired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object) -> torch.Tensor:
        raise ValueError("valid-F kernel failure")

    monkeypatch.setattr(historical, "constitutive_residual", fail)
    with pytest.raises(training.JointTrainingFailure, match="valid-F kernel failure"):
        _run("candidate")


def test_public_entrypoint_rejects_nonfixed_case_and_sampler_before_any_training() -> (
    None
):
    state = training.clone_joint_arms(_model(), JointHybridPlan())[0]
    with pytest.raises(ValueError, match="969|fixed|schedule"):
        training.train_joint_arm(
            state,
            training.SampledJointObjective(_case("candidate-002485")),
            _sampler(),
            JointHybridPlan(),
        )
    assert not state.optimizer.state


def test_invalid_arm_scope_and_reused_optimizer_fail_closed() -> None:
    state = training.clone_joint_arms(_model(), JointHybridPlan())[0]
    next(state.model.parameters()).requires_grad_(False)
    with pytest.raises(ValueError, match="trainable"):
        training._train_arm_steps(
            state,
            training.SampledJointObjective(_case("candidate-002485")),
            _sampler(),
            JointHybridPlan(),
            optimizer_steps=2,
            evaluation_steps=(0, 2),
        )  # noqa: SLF001
    fresh, _ = _run()
    with pytest.raises(ValueError, match="fresh"):
        training._train_arm_steps(
            fresh,
            training.SampledJointObjective(_case("candidate-002485")),
            _sampler(),
            JointHybridPlan(),
            optimizer_steps=2,
            evaluation_steps=(0, 2),
        )  # noqa: SLF001
    with pytest.raises(ValueError, match="arm_id"):
        replace(state, arm_id="third")


def test_failure_after_first_update_preserves_completed_trace_and_evaluations() -> None:
    state = training.clone_joint_arms(_model(), JointHybridPlan())[0]
    sampled = training.SampledJointObjective(_case("candidate-002485"))
    calls = 0

    def fail_second(
        model: torch.nn.Module, indices: torch.Tensor, *, candidate: bool
    ) -> ObjectiveResiduals:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("second-step kernel failure")
        return sampled(model, indices, candidate=candidate)

    with pytest.raises(training.JointTrainingFailure, match="second-step") as caught:
        training._train_arm_steps(
            state,
            fail_second,
            _sampler(),
            JointHybridPlan(),
            optimizer_steps=2,
            evaluation_steps=(0, 1, 2),
            evaluate=lambda model, step: step,
        )  # noqa: SLF001
    failure = caught.value
    assert failure.arm_id == "control"
    assert failure.cause_type == "ValueError"
    assert failure.attempted_step == 2
    assert failure.optimizer_updates == 1
    assert len(failure.step_trace) == 1
    assert failure.evaluations == ((0, 0), (1, 1))


def test_parameter_replacing_evaluation_restores_original_parameter_identity() -> None:
    state = training.clone_joint_arms(_model(), JointHybridPlan())[0]
    parameter = state.model.u_head.weight
    before = model_state_hash(state.model)

    def replace_parameter(model: torch.nn.Module, step: int) -> None:
        model.u_head.weight = torch.nn.Parameter(
            model.u_head.weight.detach().clone() + 1
        )

    with pytest.raises(training.JointTrainingFailure, match="evaluation.*mutated"):
        training._train_arm_steps(
            state,
            training.SampledJointObjective(_case("candidate-002485")),
            _sampler(),
            JointHybridPlan(),
            optimizer_steps=2,
            evaluation_steps=(0, 2),
            evaluate=replace_parameter,
        )  # noqa: SLF001
    assert state.model.u_head.weight is parameter
    assert model_state_hash(state.model) == before


def test_sampler_mutating_evaluation_is_restored_and_aborts() -> None:
    state = training.clone_joint_arms(_model(), JointHybridPlan())[0]
    sampler = _sampler()
    expected = copy.deepcopy(sampler)

    def mutate(model: torch.nn.Module, step: int) -> None:
        object.__setattr__(sampler, "batches", ((1, 0), (2, 3)))

    with pytest.raises(training.JointTrainingFailure, match="evaluation.*mutated"):
        training._train_arm_steps(
            state,
            training.SampledJointObjective(_case("candidate-002485")),
            sampler,
            JointHybridPlan(),
            optimizer_steps=2,
            evaluation_steps=(0, 2),
            evaluate=mutate,
        )  # noqa: SLF001
    assert sampler == expected


def test_poststep_nonfinite_parameter_records_the_performed_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = training.clone_joint_arms(_model(), JointHybridPlan())[0]
    original_step = state.optimizer.step

    def fail_after_update(*args: object, **kwargs: object) -> None:
        original_step(*args, **kwargs)
        with torch.no_grad():
            next(state.model.parameters()).fill_(float("nan"))

    monkeypatch.setattr(state.optimizer, "step", fail_after_update)
    with pytest.raises(training.JointTrainingFailure, match="parameter") as caught:
        training._train_arm_steps(  # noqa: SLF001
            state,
            training.SampledJointObjective(_case("candidate-002485")),
            _sampler(),
            JointHybridPlan(),
            optimizer_steps=2,
            evaluation_steps=(0, 2),
        )
    assert caught.value.optimizer_updates == 1
    assert caught.value.attempted_step == 1
    assert len(caught.value.step_trace) == 1
    assert dict(caught.value.step_trace[0].finite_checks)["parameters"] is False


def test_public_entrypoint_binds_fixed_budget_without_executing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case("candidate-002485")
    request = case.request
    collocation = sample_request_collocation(
        _snapshot(request),
        request.request_id,
        request,
        stream_index=0,
        interior_count=128,
        boundary_count=64,
        interface_count=0,
        qoi_quotas=(
            SemanticCollocationQuota(
                quota_id="stress-critical-right-arc-v1",
                kind=CollocationKind.BOUNDARY,
                count=64,
                semantic_tag="right_arc",
            ),
        ),
    )
    tensors = replace(
        case.tensors,
        query_xy=case.tensors.query_xy.repeat((243, 1))[:969],
        target_u=case.tensors.target_u.repeat((243, 1))[:969],
        target_P=case.tensors.target_P.repeat((243, 1, 1))[:969],
    )
    objective = training.SampledJointObjective(
        replace(case, tensors=tensors, collocation=collocation)
    )
    state = training.clone_joint_arms(_model(), JointHybridPlan())[0]
    plan = JointHybridPlan()
    calls: list[dict[str, object]] = []
    sentinel = object()

    def no_training(*args: object, **kwargs: object) -> object:
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(training, "_train_arm_steps", no_training)
    evaluate = lambda model, step: step  # noqa: E731
    actual = training.train_joint_arm(
        state, objective, training.build_joint_vertex_sampler(plan), plan, evaluate
    )
    assert actual is sentinel
    assert calls == [
        {
            "optimizer_steps": 4000,
            "evaluation_steps": (0, 1000, 2000, 3000, 4000),
            "evaluate": evaluate,
        }
    ]
    with pytest.raises(ValueError, match="evaluation callback"):
        training.train_joint_arm(
            state, objective, training.build_joint_vertex_sampler(plan), plan
        )
    result = training.JointArmTrainingResult(
        "control",
        (),
        (),
        "a" * 64,
        0,
        (),
        state.expected_config_hash,
        state.expected_module_digest,
    )
    assert (
        b'"$type":"piginot.campaigns.JointArmTrainingResult"'
        in canonical_json_bytes(result)
    )


def test_joint_clones_bind_baseline_config_and_module_digest() -> None:
    baseline = _model()
    expected_config = hash_spec(baseline.config, "content")
    expected_digest = module_digest(baseline)
    states = training.clone_joint_arms(baseline, JointHybridPlan())
    for state in states:
        assert state.expected_config_hash == expected_config
        assert state.expected_module_digest == expected_digest
        assert hash_spec(state.model.config, "content") == expected_config
        assert module_digest(state.model) == expected_digest


@pytest.mark.parametrize("change", ("config", "digest"))
def test_arm_architecture_mismatch_aborts_before_optimization(change: str) -> None:
    state = training.clone_joint_arms(_model(), JointHybridPlan())[0]
    if change == "config":
        state.model.config = replace(
            state.model.config, soft_dirichlet_residual_tolerance=2e-6
        )
    else:
        state.model.register_buffer(
            "unexpected_schema", torch.zeros(1, dtype=torch.float64)
        )
    with pytest.raises(ValueError, match="config|module digest"):
        training._train_arm_steps(  # noqa: SLF001
            state,
            training.SampledJointObjective(_case("candidate-002485")),
            _sampler(),
            JointHybridPlan(),
            optimizer_steps=2,
            evaluation_steps=(0, 2),
        )
    assert not state.optimizer.state


def test_evaluator_config_mutation_is_restored_and_failure_preserves_trace() -> None:
    state = training.clone_joint_arms(_model(), JointHybridPlan())[0]
    config = copy.deepcopy(state.model.config)

    def mutate_after_step(model: torch.nn.Module, step: int) -> int:
        if step == 1:
            model.config = replace(model.config, soft_dirichlet_residual_tolerance=2e-6)
        return step

    with pytest.raises(
        training.JointTrainingFailure, match="evaluation.*mutated"
    ) as caught:
        training._train_arm_steps(  # noqa: SLF001
            state,
            training.SampledJointObjective(_case("candidate-002485")),
            _sampler(),
            JointHybridPlan(),
            optimizer_steps=2,
            evaluation_steps=(0, 1, 2),
            evaluate=mutate_after_step,
        )
    assert state.model.config == config
    assert caught.value.attempted_step == caught.value.optimizer_updates == 1
    assert len(caught.value.step_trace) == 1
    assert caught.value.evaluations == ((0, 0),)


def test_postflight_config_mismatch_fails_with_completed_evidence() -> None:
    state = training.clone_joint_arms(_model(), JointHybridPlan())[0]
    objective = training.SampledJointObjective(_case("candidate-002485"))
    calls = 0

    def mutate_at_final_step(
        model: torch.nn.Module, indices: torch.Tensor, *, candidate: bool
    ) -> ObjectiveResiduals:
        nonlocal calls
        residuals = objective(model, indices, candidate=candidate)
        calls += 1
        if calls == 2:
            model.config = replace(model.config, soft_dirichlet_residual_tolerance=2e-6)
        return residuals

    with pytest.raises(training.JointTrainingFailure, match="config") as caught:
        training._train_arm_steps(  # noqa: SLF001
            state,
            mutate_at_final_step,
            _sampler(),
            JointHybridPlan(),
            optimizer_steps=2,
            evaluation_steps=(0, 2),
        )
    assert caught.value.optimizer_updates == 2
    assert caught.value.attempted_step == 2
    assert len(caught.value.step_trace) == 2


def test_training_result_records_exact_verified_config_and_module_digest() -> None:
    state, result = _run()
    assert result.config_hash == hash_spec(state.model.config, "content")
    assert result.module_digest == module_digest(state.model)
    assert result.config_hash == state.expected_config_hash
    assert result.module_digest == state.expected_module_digest


def test_evaluator_semantic_digest_mutation_is_restored_without_parameter_change() -> (
    None
):
    state = training.clone_joint_arms(_model(), JointHybridPlan())[0]
    before = model_state_hash(state.model)

    def mutate(model: torch.nn.Module, step: int) -> None:
        model.lifting_verification = LiftingVerification(
            verification_id="unexpected-evaluation-change",
            parameter_range_hash="a" * 64,
            verification_hash="b" * 64,
            boundary_refinement_levels=(1, 2),
            full_parameter_range_verified=True,
            max_boundary_residual=0.0,
        )

    with pytest.raises(
        training.JointTrainingFailure, match="evaluation.*mutated"
    ) as caught:
        training._train_arm_steps(  # noqa: SLF001
            state,
            training.SampledJointObjective(_case("candidate-002485")),
            _sampler(),
            JointHybridPlan(),
            optimizer_steps=2,
            evaluation_steps=(0, 2),
            evaluate=mutate,
        )
    assert caught.value.optimizer_updates == 0
    assert state.model.lifting_verification is None
    assert model_state_hash(state.model) == before
    assert module_digest(state.model) == state.expected_module_digest


def test_evaluator_mutation_and_error_preserve_both_failure_reasons() -> None:
    state = training.clone_joint_arms(_model(), JointHybridPlan())[0]
    before = model_state_hash(state.model)

    def mutate_and_fail(model: torch.nn.Module, step: int) -> None:
        with torch.no_grad():
            next(model.parameters()).add_(1)
        raise LookupError("synthetic metric failure")

    with pytest.raises(
        training.JointTrainingFailure, match="evaluation.*mutated"
    ) as caught:
        training._train_arm_steps(  # noqa: SLF001
            state,
            training.SampledJointObjective(_case("candidate-002485")),
            _sampler(),
            JointHybridPlan(),
            optimizer_steps=2,
            evaluation_steps=(0, 2),
            evaluate=mutate_and_fail,
        )
    assert "LookupError: synthetic metric failure" in caught.value.reason
    assert caught.value.optimizer_updates == 0
    assert model_state_hash(state.model) == before
