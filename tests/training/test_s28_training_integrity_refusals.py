"""Integrity failures stop before objective use, weight mutation or publication.

These are contract and synthetic-checkpoint tests; no campaign is fitted.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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
from piginot.training import adaptive, checkpoint, diagnostics, engine, state
from tests.training import test_checkpoint as cp
from tests.training import test_diagnostics as dg
from tests.training import test_engine as en


@pytest.mark.parametrize(
    "changes",
    [
        {"dataset_snapshot_hash": "g" * 64},
        {"public_manifest_hash": "A" * 64},
        {"dataset_snapshot_hash": 1},
        {"role": "TRAIN"},
        {"sample_ids": []},
        {"sample_ids": (" train",)},
        {"sample_ids": ("b", "a")},
        {"sample_ids": ("a", "a")},
    ],
)
def test_population_refuses_ambiguous_identity_or_roster(changes):
    with pytest.raises((TypeError, ValueError)):
        replace(en._population(), **changes)


@pytest.mark.parametrize(
    "field,value",
    [
        ("search_space", object()),
        ("trial", object()),
        ("adaptive", object()),
        ("scheduler", object()),
        ("adam_betas", []),
        ("adam_betas", (-0.1, 0.9)),
        ("adam_weight_decay", -0.1),
        ("gradient_clip_norm", float("inf")),
        ("adam_eps", 0.0),
    ],
)
def test_trainer_policy_refuses_unbound_optimizer_settings(field, value):
    with pytest.raises((TypeError, ValueError)):
        replace(en._config(), **{field: value})


@pytest.mark.parametrize("field", ["optimizer_steps", "case_batch_size"])
def test_trial_cannot_override_bound_search_budget(field):
    config = en._config()
    trial = replace(config.trial, **{field: getattr(config.trial, field) + 1})
    with pytest.raises(ValueError, match="differ"):
        replace(config, trial=trial)


@pytest.mark.parametrize(
    "change",
    [
        "population-type",
        "plan-type",
        "binding-type",
        "config-type",
        "objective-type",
        "snapshot",
        "manifest",
        "sampler",
        "trainer",
        "objective",
        "weights",
        "population-size",
        "plan-length",
        "batch-size",
        "target-zero",
        "target-type",
        "target-over-budget",
        "path-type",
        "resume-type",
        "missing-parent",
        "parent-file",
        "parent-link",
        "destination-link",
        "model-dtype",
        "profile-dtype",
    ],
)
def test_training_preflight_refuses_without_objective_or_weight_mutation(
    tmp_path, change
):
    model = en._model()
    binding = en._binding(model)
    arguments = {
        "model": model,
        "population": en._population(),
        "plan": en._plan(),
        "binding": binding,
        "config": en._config(),
        "base_registry": en._registry(),
    }
    calls = []

    def forbidden(*args):
        calls.append(args)
        pytest.fail("refused preflight invoked the objective")

    arguments["objective"] = forbidden
    target = tmp_path / "checkpoint"
    field = {
        "population-type": "population",
        "plan-type": "plan",
        "binding-type": "binding",
        "config-type": "config",
        "objective-type": "objective",
    }.get(change)
    if field is not None:
        arguments[field] = object()
    elif change in {
        "snapshot",
        "manifest",
        "sampler",
        "trainer",
        "objective",
        "weights",
    }:
        name = {
            "snapshot": "dataset_snapshot_hash",
            "manifest": "public_manifest_hash",
            "sampler": "sampler_plan_hash",
            "trainer": "trainer_config_hash",
            "objective": "objective_registry_hash",
            "weights": "initial_weights_hash",
        }[change]
        arguments["binding"] = replace(binding, **{name: "f" * 64})
    elif change in {"population-size", "plan-length", "batch-size"}:
        plan = arguments["plan"]
        if change == "population-size":
            plan = replace(plan, population_size=7)
        elif change == "plan-length":
            plan = replace(plan, batches=plan.batches[:-1])
        else:
            plan = replace(plan, batches=tuple((batch[0],) for batch in plan.batches))
        arguments["plan"] = plan
        arguments["binding"] = replace(binding, sampler_plan_hash=plan.plan_hash)
    elif change.startswith("target-"):
        arguments["target_step"] = {
            "target-zero": 0,
            "target-type": True,
            "target-over-budget": 7,
        }[change]
    elif change == "path-type":
        target = str(target)
    elif change == "resume-type":
        arguments["resume_from"] = "not-a-Path"
    elif change == "missing-parent":
        target = tmp_path / "missing" / "checkpoint"
    elif change == "parent-file":
        parent = tmp_path / "parent"
        parent.write_text("file")
        target = parent / "checkpoint"
    elif change in {"parent-link", "destination-link"}:
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "linked"
        link.symlink_to(real, target_is_directory=True)
        target = link / "checkpoint" if change == "parent-link" else link
    elif change == "model-dtype":
        model.to(torch.float32)
        arguments["binding"] = replace(
            binding, initial_weights_hash=checkpoint.model_state_hash(model)
        )
    else:
        arguments["binding"] = replace(
            binding,
            execution_profile=replace(binding.execution_profile, dtype="float16"),
        )
    before = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with pytest.raises((TypeError, ValueError, FileNotFoundError, FileExistsError)):
        engine.train_to_checkpoint(target, **arguments)
    assert not calls
    assert all(
        torch.equal(value, model.state_dict()[name]) for name, value in before.items()
    )
    assert torch.equal(rng, torch.get_rng_state())
    if isinstance(target, Path):
        assert not target.exists() or target.is_symlink()


@pytest.mark.parametrize(
    "changes",
    [
        {"optimizer_id": "another-optimizer"},
        {"parameter_names": ("weight::bad",)},
        {"parameter_names": ("weight", "weight")},
        {"state_parameter_names": ("foreign",)},
        {"state_parameter_names": ("bias", "weight")},
        {"betas": [0.9, 0.999]},
        {"betas": (-0.1, 0.9)},
        {"weight_decay": -0.1},
        {"amsgrad": 1},
        {"maximize": 1},
    ],
)
def test_optimizer_metadata_refuses_ambiguous_restore(changes):
    valid = state.AdamOptimizerState(
        ("weight", "bias"), (), 0.01, (0.9, 0.999), 1e-8, 0.0, False, False
    )
    with pytest.raises((TypeError, ValueError)):
        replace(valid, **changes)


@pytest.mark.parametrize(
    "change",
    [
        "model-type",
        "optimizer-type",
        "empty-model",
        "groups",
        "order",
        "device",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
        "decoupled_weight_decay",
        "maximize",
    ],
)
def test_capture_optimizer_refuses_nonreplayable_structure(change):
    model = cp._model()
    optimizer = cp._optimizer(model)
    if change == "model-type":
        model = object()
    elif change == "optimizer-type":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    elif change == "empty-model":
        model = torch.nn.Identity()
    elif change == "groups":
        optimizer = torch.optim.Adam(
            [{"params": [model.weight]}, {"params": [model.bias]}],
            foreach=False,
            fused=False,
        )
    elif change == "order":
        optimizer = torch.optim.Adam(
            [model.bias, model.weight], foreach=False, fused=False
        )
    elif change == "device":
        model = torch.nn.Linear(1, 1, device="meta")
        optimizer = cp._optimizer(model)
    else:
        optimizer.param_groups[0][change] = True
    with pytest.raises((TypeError, ValueError)):
        state.capture_adam_optimizer_state(model, optimizer)


@pytest.mark.parametrize(
    "call",
    [
        lambda: state.DeterministicSamplerPlan("bad", 2, []),
        lambda: state.DeterministicSamplerPlan("bad", 2, ((0,), (0, 1))),
        lambda: state.DeterministicSamplerPlan("bad", 2, ([],)),
        lambda: state.DeterministicSamplerPlan("bad", 2, ((2,),)),
        lambda: state.DeterministicSamplerPlan("bad", 2, ((0, 0),)),
        lambda: state.DeterministicSamplerState("bad", 0),
        lambda: state.DeterministicSamplerPlan(" bad", 2, ((0,),)),
        lambda: state.DeterministicSamplerPlan("bad", 2**53, ((0,),)),
        lambda: state.next_sampler_batch(object(), cp._plan().initial_state),
        lambda: state.next_sampler_batch(cp._plan(), object()),
    ],
)
def test_sampler_refuses_noncanonical_rosters_and_cursors(call):
    with pytest.raises((TypeError, ValueError)):
        call()


@pytest.mark.parametrize(
    "call",
    [
        lambda: adaptive.AdaptiveWeightConfig(warmup_steps=-1),
        lambda: adaptive.GradientStatistic(-1.0),
        lambda: adaptive.AdaptiveWeightState(multipliers=[]),
        lambda: adaptive.AdaptiveWeightState(multipliers={"constitutive": 1.0}),
        lambda: adaptive.AdaptiveWeightState(multipliers={}),
        lambda: adaptive.AdaptiveWeightState(update_count=1),
        lambda: adaptive.AdaptiveWeightState(last_update_step=0),
        lambda: adaptive.AdaptiveWeightState.initial(multipliers=[]),
        lambda: adaptive.AdaptiveWeightState.initial(
            multipliers={ObjectiveTerm.DATA_U: 2.0}
        ),
        lambda: adaptive.AdaptiveWeightUpdate(object(), en._registry()),
        lambda: adaptive.AdaptiveWeightUpdate(
            adaptive.AdaptiveWeightState.initial(), object()
        ),
    ],
)
def test_adaptive_resume_metadata_refuses_inconsistent_state(call):
    with pytest.raises((TypeError, ValueError)):
        call()


@pytest.mark.parametrize(
    "change",
    [
        "mapping",
        "keys",
        "parameters-empty",
        "parameters-duplicate",
        "parameters-type",
        "no-grad",
        "sparse-parameter",
        "loss-type",
        "loss-integer",
        "loss-shape",
        "loss-nonfinite",
        "loss-detached",
        "disconnected",
        "nonfinite-gradient",
        "zero-gradient",
        "freed-graph",
    ],
)
def test_gradient_measurement_refuses_invalid_graph_without_mutating_grad(change):
    parameter = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    parameter.grad = torch.tensor([7.0], dtype=torch.float64)
    parameters = (parameter,)
    loss = parameter.square().sum()
    losses = {
        term: loss
        for term in (adaptive.REFERENCE_PHYSICS_TERM, *adaptive.ADAPTIVE_PHYSICS_TERMS)
    }
    if change == "mapping":
        losses = []
    elif change == "keys":
        losses = {"constitutive": loss}
    elif change == "parameters-empty":
        parameters = ()
    elif change == "parameters-duplicate":
        parameters += parameters
    elif change == "parameters-type":
        parameters = (object(),)
    elif change == "no-grad":
        parameters = (parameter.detach(),)
    elif change == "sparse-parameter":
        parameters = (parameter.detach().to_sparse().requires_grad_(True),)
    else:
        if change == "loss-type":
            invalid = 1.0
        elif change == "loss-integer":
            invalid = torch.tensor(1)
        elif change == "loss-shape":
            invalid = parameter.square()
        elif change == "loss-nonfinite":
            invalid = loss * float("inf")
        elif change == "loss-detached":
            invalid = loss.detach()
        elif change == "disconnected":
            invalid = torch.tensor(2.0, requires_grad=True)
        elif change == "nonfinite-gradient":
            invalid = (parameter - 1.0).sqrt().sum()
        elif change == "zero-gradient":
            invalid = (parameter * 0.0).sum()
        else:
            loss.backward()
            invalid = loss
        losses[adaptive.REFERENCE_PHYSICS_TERM] = invalid
    gradient = parameter.grad.clone()
    with pytest.raises((TypeError, ValueError)):
        adaptive.measure_unweighted_gradient_statistics(losses, parameters)
    assert torch.equal(parameter.grad, gradient)


@pytest.mark.parametrize(
    "change",
    [
        "registry",
        "weak",
        "state",
        "config",
        "mapping",
        "keys",
        "missing",
        "statistic",
        "zero",
        "past-step",
    ],
)
def test_adaptive_update_refuses_mixed_registry_or_gradient_identity(change):
    arguments = {
        "base_registry": en._registry(),
        "state": adaptive.AdaptiveWeightState.initial(),
        "gradients": {
            term: adaptive.GradientStatistic(1.0)
            for term in (
                adaptive.REFERENCE_PHYSICS_TERM,
                *adaptive.ADAPTIVE_PHYSICS_TERMS,
            )
        },
        "config": adaptive.AdaptiveWeightConfig(warmup_steps=0),
        "step": 0,
    }
    if change in {"registry", "state", "config"}:
        arguments[
            {"registry": "base_registry", "state": "state", "config": "config"}[change]
        ] = object()
    elif change == "weak":
        arguments["base_registry"] = ObjectiveRegistry(
            mode=ObjectiveMode.HYBRID, equilibrium_form=EquilibriumForm.WEAK
        )
    elif change == "mapping":
        arguments["gradients"] = []
    elif change == "keys":
        arguments["gradients"] = {"constitutive": adaptive.GradientStatistic(1.0)}
    elif change == "missing":
        arguments["gradients"] = {}
    elif change in {"statistic", "zero"}:
        arguments["gradients"][adaptive.REFERENCE_PHYSICS_TERM] = (
            object() if change == "statistic" else adaptive.GradientStatistic(0.0)
        )
    else:
        arguments["state"] = adaptive.AdaptiveWeightState(
            update_count=1, last_update_step=0
        )
    with pytest.raises((TypeError, ValueError)):
        adaptive.update_adaptive_weights(**arguments)


@pytest.mark.parametrize(
    "change",
    [
        "role-type",
        "population-type",
        "model-type",
        "empty-model",
        "sample-type",
        "sample-duplicate",
        "sample-noncanonical",
        "sample-foreign",
        "step",
        "registry-type",
        "registry-mode",
        "config-type",
        "state-type",
        "scale",
        "clip",
        "snapshot-type",
        "objective-type",
    ],
)
def test_diagnostics_preflight_has_no_objective_or_model_side_effect(change):
    model = dg._VectorModel()
    calls = []
    arguments = {
        "population": dg._population(),
        "sample_ids": ("train-002",),
        "step": 1,
        "base_registry": dg._registry(),
        "physics_weight_scale": 1.0,
        "adaptive_config": adaptive.AdaptiveWeightConfig(),
        "adaptive_state": adaptive.AdaptiveWeightState.initial(),
        "objective": lambda *args: calls.append(args),
        "role": EvidenceRole.TRAIN,
        "gradient_clip_norm": 1.0,
        "post_checkpoint_snapshot": True,
    }
    invalid = {
        "role-type": ("role", "TRAIN"),
        "population-type": ("population", object()),
        "sample-type": ("sample_ids", []),
        "sample-duplicate": ("sample_ids", ("train-002", "train-002")),
        "sample-noncanonical": ("sample_ids", (" train-002",)),
        "sample-foreign": ("sample_ids", ("development",)),
        "step": ("step", 0),
        "registry-type": ("base_registry", object()),
        "registry-mode": (
            "base_registry",
            ObjectiveRegistry(mode=ObjectiveMode.DATA_ONLY),
        ),
        "config-type": ("adaptive_config", object()),
        "state-type": ("adaptive_state", object()),
        "scale": ("physics_weight_scale", 0.0),
        "clip": ("gradient_clip_norm", float("nan")),
        "snapshot-type": ("post_checkpoint_snapshot", 1),
        "objective-type": ("objective", object()),
    }
    before = {name: value.clone() for name, value in model.state_dict().items()}
    if change == "model-type":
        model = object()
    elif change == "empty-model":
        model = torch.nn.Identity()
    else:
        key, value = invalid[change]
        arguments[key] = value
    with pytest.raises((TypeError, ValueError)):
        diagnostics.measure_objective_gradient_diagnostic(model, **arguments)
    assert not calls
    if type(model) is dg._VectorModel:
        assert all(
            torch.equal(value, model.state_dict()[name])
            for name, value in before.items()
        )


@pytest.mark.parametrize(
    "change",
    [
        "return-type",
        "missing",
        "nonfinite-loss",
        "disconnected",
        "zero-reference",
        "zero-adapted",
        "nonfinite-gradient",
    ],
)
def test_failed_diagnostics_restores_rng_gradients_buffers_and_mode(change):
    model = dg._VectorModel()
    model.register_buffer("audit_buffer", torch.tensor([17.0], dtype=torch.float64))
    model.eval()
    model.weight.grad = torch.tensor([11.0, 12.0], dtype=torch.float64)
    before_grad = model.weight.grad.clone()
    rng = torch.get_rng_state().clone()

    def objective(module, ids, step):
        module.audit_buffer.add_(1.0)
        torch.rand(1)
        if change == "return-type":
            return object()
        residuals = dg._AnalyticObjective()(module, ids, step)
        values = dict(residuals.values)
        if change == "missing":
            values.pop(ObjectiveTerm.CONSTITUTIVE)
        else:
            if change == "nonfinite-loss":
                value = module.weight[0] * 1e308
            elif change == "disconnected":
                value = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
            elif change == "nonfinite-gradient":
                value = (module.weight[0] - 1.0).sqrt() + 1.0
            else:
                value = module.weight[0] * 0.0
            term = (
                ObjectiveTerm.CONSTITUTIVE
                if change == "zero-adapted"
                else ObjectiveTerm.EQUILIBRIUM_STRONG
            )
            values[term] = value.reshape(1)
        return ObjectiveResiduals(values)

    with pytest.raises((TypeError, ValueError)):
        dg._measure(model, objective)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(model.weight.grad, before_grad)
    assert torch.equal(model.audit_buffer, torch.tensor([17.0], dtype=torch.float64))
    assert not model.training


@pytest.mark.parametrize(
    "change",
    [
        "integer",
        "identity",
        "config-type",
        "initial-floor",
        "state-type",
        "state-identity",
        "clock",
        "floor",
        "cooldown",
    ],
)
def test_scheduler_refuses_unbound_identity_or_resume_counters(change):
    from piginot.training.scheduler import (
        PlateauSchedulerConfig,
        advance_plateau_scheduler,
        initial_scheduler_state,
    )

    config = PlateauSchedulerConfig()
    original = initial_scheduler_state(config, learning_rate=0.01)
    if change == "integer":
        with pytest.raises(TypeError):
            PlateauSchedulerConfig(patience=True)
        return
    if change == "identity":
        with pytest.raises(ValueError):
            PlateauSchedulerConfig(scheduler_id="another-policy")
        return
    if change in {"config-type", "initial-floor"}:
        with pytest.raises((TypeError, ValueError)):
            initial_scheduler_state(
                object() if change == "config-type" else config, learning_rate=1e-20
            )
        return
    invalid = {
        "state-type": object(),
        "state-identity": replace(original, scheduler_id="another-policy"),
        "clock": replace(original, step=1),
        "floor": replace(original, learning_rate=1e-20),
        "cooldown": replace(original, cooldown_remaining=1),
    }[change]
    with pytest.raises((TypeError, ValueError)):
        advance_plateau_scheduler(config, invalid, metric=1.0)
    assert original.step == original.observation_count == 0


@pytest.mark.parametrize("change", ["type", "missing-terms", "overflow"])
def test_invalid_objective_stops_without_an_optimizer_step(tmp_path, change):
    model = en._model()
    original = checkpoint.model_state_hash(model)
    target = tmp_path / "never-published"
    calls = []

    def objective(module, ids, step):
        calls.append(step)
        if change == "type":
            return object()
        if change == "missing-terms":
            return ObjectiveResiduals({})
        scalar = module.weight.sum() * 1e308
        return ObjectiveResiduals(
            {term: scalar.reshape(1) for term in en._registry().required_terms}
        )

    with torch.random.fork_rng(devices=[]):
        with pytest.raises((TypeError, ValueError)):
            engine.train_to_checkpoint(
                target,
                model=model,
                population=en._population(),
                plan=en._plan(),
                binding=en._binding(model),
                config=en._config(),
                base_registry=en._registry(),
                objective=objective,
                target_step=1,
            )
    assert calls == [1]
    assert checkpoint.model_state_hash(model) == original
    assert not target.exists()
