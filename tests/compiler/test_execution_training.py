"""TRAIN-only preparation and checkpoint orchestration use synthetic operators."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from piginot.compiler import energy_training
from piginot.compiler.planning import build_plan
from tests.compiler.plan_fixtures import inputs
from tests.problem_helpers import plate_mesh


class Adapter:
    def geometry(self, spec, case, level=1):
        return plate_mesh(columns=1, rows=1, mesh_id=spec.geometry.reference_mesh_id)


@pytest.fixture
def context(tmp_path):
    spec, config, _, descriptor = inputs(tmp_path)
    plan = build_plan(spec, config, tmp_path / "output", descriptor=descriptor)
    model = replace(
        plan.config.model,
        hidden_dim=8,
        parameter_hidden_dim=8,
        latent_token_count=2,
        num_attention_heads=2,
        decoder_layers=1,
        positional_frequencies=1,
    )
    plan = replace(plan, config=replace(plan.config, model=model))
    return energy_training.EnergyTraining(plan, Adapter(), "c" * 40)


def test_normalization_fits_only_fit_ids_and_keeps_declared_feature_order(context):
    case = context.plan.manifest.cases[0]
    outlier = replace(
        case,
        case_id="screen-001",
        primitive_coordinates={**case.primitive_coordinates, "width": 9999.0},
    )
    cases = tuple(
        outlier if c.case_id == outlier.case_id else c
        for c in context.plan.manifest.cases
    )
    context.plan = replace(
        context.plan, manifest=replace(context.plan.manifest, cases=cases)
    )
    preparation = context.prepare()
    assert preparation["normalization"]["fit_case_ids"] == ["fit-001", "fit-002"]
    names = preparation["normalization"]["parameter_names"]
    assert names == list(context.plan.config.model.parameter_names)
    assert preparation["normalization"]["mean"][names.index("width")] == 12.0
    assert preparation["normalization"]["standard_deviation"] == [1.0] * len(names)
    assert preparation["training_labels"] == "NONE"
    assert context.binding.problem_spec_hash == context.plan.problem.problem_hash
    assert context.population.sample_ids == ("fit-001", "fit-002")


def test_prepare_is_deterministic_and_does_not_change_global_rng(context):
    torch.manual_seed(52)
    before = torch.get_rng_state().clone()
    first = context.prepare()
    second = energy_training.EnergyTraining(
        context.plan, context.adapter, "c" * 40
    ).prepare()
    assert first == second
    assert torch.equal(before, torch.get_rng_state())


def test_preparation_publishes_complete_canonical_contracts_as_bound_stage(context):
    from piginot.compiler.execution_records import load_bound, save_bound
    from piginot.contracts import canonical_json_bytes

    preparation = context.prepare()
    path = context.plan.output_path.parent / "preparation.json"
    save_bound(path, context.plan.plan_sha256, preparation)
    assert load_bound(path, context.plan.plan_sha256) == preparation
    assert preparation["run_binding_utf8"].encode() == canonical_json_bytes(
        context.binding
    )
    assert preparation["sampler_plan_utf8"].encode() == canonical_json_bytes(
        context.sampler
    )


def test_segment_targets_resume_last_checkpoint_and_never_reselect(
    context, monkeypatch
):
    assert context.last_checkpoint is None
    context.prepare()
    calls = []
    stop = {"target": 2}

    def train(path, **kwargs):
        calls.append((kwargs["target_step"], kwargs["resume_from"]))
        if kwargs["target_step"] == stop["target"]:
            raise KeyboardInterrupt()
        path.mkdir()
        (path / "marker").write_text(str(kwargs["target_step"]))
        return SimpleNamespace(
            step=kwargs["target_step"],
            model_state_hash="d" * 64,
            history=(),
            checkpoint=SimpleNamespace(
                path=path, manifest=SimpleNamespace(manifest_hash="e" * 64)
            ),
        )

    def load(path, **kwargs):
        step = int((path / "marker").read_text())
        return SimpleNamespace(
            manifest=SimpleNamespace(
                step=step, model_state_hash="d" * 64, manifest_hash="e" * 64
            ),
            sampler_state=SimpleNamespace(next_step=step),
        )

    monkeypatch.setattr(energy_training, "train_energy_to_checkpoint", train)
    monkeypatch.setattr(energy_training, "load_training_checkpoint", load)
    with pytest.raises(KeyboardInterrupt):
        context.train(resume=False)
    assert context.last_checkpoint == context.checkpoint_path(1)
    stop["target"] = 3
    with pytest.raises(KeyboardInterrupt):
        context.train(resume=True)
    assert context.last_checkpoint == context.checkpoint_path(2)
    stop["target"] = 99
    result = context.train(resume=True)
    assert result["step"] == 4
    assert result["selection_rule"] == "final"
    assert context.last_checkpoint == context.checkpoint_path(4)
    assert calls == [
        (1, None),
        (2, context.checkpoint_path(1)),
        (2, context.checkpoint_path(1)),
        (3, context.checkpoint_path(2)),
        (3, context.checkpoint_path(2)),
        (4, context.checkpoint_path(3)),
    ]


def test_checkpoint_directories_without_resume_refuse(context):
    context.prepare()
    context.checkpoint_path(1).mkdir(parents=True)
    with pytest.raises(FileExistsError):
        context.train(resume=False)


def test_objective_uses_generic_problem_material_and_lifting(context, monkeypatch):
    context.prepare()
    seen = {}

    def energy(gradient, weights, material, **kwargs):
        seen.update(material=material, gradient=gradient, weights=weights)
        return "terms"

    monkeypatch.setattr(energy_training, "plane_stress_potential_energy", energy)
    result = context.objective(context.model, "fit-001", 1)
    assert result == "terms"
    assert seen["material"].model == context.plan.problem.material.model
    assert seen["gradient"].shape[-2:] == (2, 2)
    assert seen["weights"].numel() == 2 * context.plan.config.objective.points_per_cell


def test_model_parameters_and_inference_use_bound_dtypes(context):
    context.prepare()
    assert next(context.model.parameters()).dtype == torch.float32
    with pytest.raises(KeyError):
        context.objective(context.model, "screen-001", 1)


def test_batch_larger_than_fit_population_refuses(context):
    training = replace(context.plan.config.training, case_batch_size=3)
    context.plan = replace(
        context.plan, config=replace(context.plan.config, training=training)
    )
    with pytest.raises(ValueError, match="batch"):
        context.prepare()


def test_train_requires_preparation_and_objective_rejects_wrong_model(context):
    with pytest.raises(ValueError, match="preparation"):
        context.train(resume=False)
    context.prepare()
    with pytest.raises(TypeError, match="declared"):
        context.objective(torch.nn.Linear(1, 1), "fit-001", 1)


@pytest.mark.parametrize(
    "kind", ["folder-symlink", "mixed", "file", "symlink", "wrong-step"]
)
def test_changed_or_corrupt_checkpoint_roster_refuses(context, monkeypatch, kind):
    context.prepare()
    context.last_checkpoint = context.checkpoint_path(99)
    folder = context.checkpoint_path(1).parent
    if kind == "folder-symlink":
        target = context.plan.output_path / "elsewhere"
        target.mkdir(parents=True)
        folder.symlink_to(target, target_is_directory=True)
    else:
        folder.mkdir(parents=True)
        if kind == "mixed":
            context.checkpoint_path(2).mkdir()
        elif kind == "file":
            context.checkpoint_path(1).write_text("bad")
        elif kind == "symlink":
            target = context.plan.output_path / "elsewhere"
            target.mkdir()
            context.checkpoint_path(1).symlink_to(target, target_is_directory=True)
        else:
            context.checkpoint_path(1).mkdir()
            monkeypatch.setattr(
                context,
                "_load",
                lambda *args: SimpleNamespace(
                    manifest=SimpleNamespace(step=2),
                    sampler_state=SimpleNamespace(next_step=2),
                ),
            )
    with pytest.raises(ValueError):
        context.train(resume=True)
    assert context.last_checkpoint is None


@pytest.mark.parametrize("invalid_step", [1, 2])
@pytest.mark.parametrize("invalid", ["loader", "manifest-step", "sampler-cursor"])
def test_checkpoint_trust_advances_only_after_loader_and_cursor_validation(
    context, monkeypatch, invalid_step, invalid
):
    context.prepare()
    for step in (1, 2):
        context.checkpoint_path(step).mkdir(parents=True)

    def load(path, model):
        step = int(path.name.removeprefix("step-"))
        bad = step == invalid_step
        if bad and invalid == "loader":
            raise ValueError("corrupt checkpoint")
        return SimpleNamespace(
            manifest=SimpleNamespace(
                step=99 if bad and invalid == "manifest-step" else step
            ),
            sampler_state=SimpleNamespace(
                next_step=99 if bad and invalid == "sampler-cursor" else step
            ),
        )

    monkeypatch.setattr(context, "_load", load)
    context.last_checkpoint = context.checkpoint_path(99)
    with pytest.raises(ValueError):
        context.train(resume=True)
    assert context.last_checkpoint == (
        context.checkpoint_path(1) if invalid_step == 2 else None
    )


def test_published_checkpoint_remains_trusted_if_progress_write_fails(
    context, monkeypatch
):
    context.prepare()
    target = context.checkpoint_path(1)
    monkeypatch.setattr(
        energy_training,
        "train_energy_to_checkpoint",
        lambda *args, **kwargs: SimpleNamespace(
            checkpoint=SimpleNamespace(path=target),
            step=1,
            model_state_hash="d" * 64,
            history=(),
        ),
    )
    monkeypatch.setattr(
        energy_training,
        "atomic_write",
        lambda *args: (_ for _ in ()).throw(RuntimeError("progress write failed")),
    )
    with pytest.raises(RuntimeError, match="progress write failed"):
        context.train(resume=False)
    assert context.last_checkpoint == target


def test_fixed_final_checkpoint_rejects_foreign_or_changed_evidence(
    context, monkeypatch
):
    context.prepare()
    training = {
        "checkpoint_path": str(context.checkpoint_path(4)),
        "run_hash": context.binding.run_hash,
        "checkpoint_sha256": "a" * 64,
        "model_state_hash": "b" * 64,
    }
    with pytest.raises(ValueError, match="selection"):
        context.load_final({**training, "checkpoint_path": "foreign"})
    monkeypatch.setattr(
        context,
        "_load",
        lambda *args: SimpleNamespace(
            manifest=SimpleNamespace(manifest_hash="c" * 64, model_state_hash="b" * 64)
        ),
    )
    with pytest.raises(ValueError, match="evidence"):
        context.load_final(training)
    monkeypatch.setattr(
        context,
        "_load",
        lambda *args: SimpleNamespace(
            manifest=SimpleNamespace(manifest_hash="a" * 64, model_state_hash="b" * 64)
        ),
    )
    context.load_final(training)


def test_checkpoint_loader_uses_original_binding_and_optimizer(context, monkeypatch):
    context.prepare()
    seen = []
    monkeypatch.setattr(
        energy_training,
        "load_training_checkpoint",
        lambda path, **kwargs: seen.append(kwargs) or "loaded",
    )
    assert context._load(context.checkpoint_path(4), context.model) == "loaded"
    assert seen[0]["binding"] == context.binding
    assert isinstance(seen[0]["optimizer"], torch.optim.Adam)


def test_float64_prediction_preserves_training_weights_dtype_and_rng(context):
    context.prepare()
    before = torch.get_rng_state().clone()
    predictions = []
    for _ in range(2):
        prediction = context.predict("screen-001", torch.zeros((2, 2)))
        assert (
            prediction.u.dtype
            == prediction.P.dtype
            == prediction.gradient.dtype
            == torch.float64
        )
        assert prediction.gradient.shape == (1, 2, 2, 2)
        predictions.append(prediction)
    assert torch.equal(predictions[0].u, predictions[1].u)
    assert torch.equal(predictions[0].P, predictions[1].P)
    assert torch.equal(predictions[0].gradient, predictions[1].gradient)
    assert next(context.model.parameters()).dtype == torch.float32
    assert torch.equal(before, torch.get_rng_state())


def test_reference_prediction_bounds_query_batches_and_preserves_fields(
    context, monkeypatch
):
    from piginot.operators.energy_operator import constitutive_in_plane_stress
    from piginot.problem import build_lifting, material_spec, query_frame

    context.prepare()
    points = torch.stack(
        (torch.linspace(0.0, 10.0, 1025), torch.linspace(0.1, 0.9, 1025)), dim=-1
    ).to(torch.float64)
    context.predict("screen-001", points[:1])
    model = context.inference_model
    values = context._values("screen-001")
    expected_u, expected_gradient = model.displacement_and_gradient(
        query_xy=points.unsqueeze(0),
        parameters=context.parameters(values, torch.float64),
        frame=query_frame(context.plan.problem, [values]),
        lifting=build_lifting(context.plan.problem, [values]),
        create_graph=False,
    )
    expected_p = constitutive_in_plane_stress(
        expected_gradient[0], material_spec(context.plan.problem, values)
    ).unsqueeze(0)
    original = model.displacement_and_gradient
    sizes = []

    def bounded(**kwargs):
        count = kwargs["query_xy"].shape[1]
        sizes.append(count)
        assert count <= 512
        return original(**kwargs)

    monkeypatch.setattr(model, "displacement_and_gradient", bounded)
    actual = context.predict("screen-001", points)
    assert sizes == [512, 512, 1]
    torch.testing.assert_close(actual.u, expected_u, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(actual.P, expected_p, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(
        actual.gradient, expected_gradient, rtol=1e-11, atol=1e-12
    )


def test_resume_completed_final_checkpoint_performs_no_new_optimizer_step(
    context, monkeypatch
):
    context.prepare()
    for step in range(1, 5):
        context.checkpoint_path(step).mkdir(parents=True)
    monkeypatch.setattr(
        context,
        "_load",
        lambda path, model: SimpleNamespace(
            manifest=SimpleNamespace(
                step=int(path.name.removeprefix("step-")),
                manifest_hash="a" * 64,
                model_state_hash="b" * 64,
            ),
            sampler_state=SimpleNamespace(
                next_step=int(path.name.removeprefix("step-"))
            ),
        ),
    )
    monkeypatch.setattr(
        energy_training,
        "train_energy_to_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("must not retrain completed checkpoint")
        ),
    )
    assert context.train(resume=True)["step"] == 4
