"""The S13b arm is resumable, TRAIN-only, and bound to its own recipe."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from campaigns.dogbone_v1 import s13_pilot, s13b_run
from campaigns.dogbone_v1.s13b_protocol import build_dogbone_s13b_pilot_plan
from piginot.contracts import hash_spec
from piginot.doe import EvidenceRole
from piginot.objectives import ObjectiveTerm


def _roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    s10 = tmp_path / "s10"
    s11 = tmp_path / "s11"
    output_parent = tmp_path / "runs"
    for path in (s10, s11, output_parent):
        path.mkdir()
    return s10, s11, output_parent / "s13b-peak-arm"


def test_binding_separates_peak_arm_from_legacy_s13(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot = build_dogbone_s13b_pilot_plan()
    model = torch.nn.Linear(1, 1, dtype=torch.float64)
    monkeypatch.setattr(
        s13_pilot,
        "model_state_hash",
        lambda model: pilot.base_protocol.initial_trainer_weights_hash,
    )

    binding = s13b_run._build_s13b_binding(
        pilot,
        model=model,
        config=pilot.trainer_config,
        plan=pilot.sampler_plan,
        implementation_revision="a" * 40,
    )
    legacy = s13_pilot._build_pilot_binding(
        pilot.base_protocol,
        model=model,
        config=pilot.trainer_config,
        plan=pilot.sampler_plan,
        implementation_revision="a" * 40,
    )

    assert binding.objective_registry_hash == hash_spec(
        pilot.objective_registry, "recipe"
    )
    assert binding.objective_registry_hash != legacy.objective_registry_hash
    assert binding.run_hash != legacy.run_hash
    assert binding.run_hash != pilot.control_run_hash
    assert binding.trainer_config_hash == pilot.trainer_config.config_hash
    assert binding.sampler_plan_hash == pilot.sampler_plan.plan_hash
    assert binding.implementation_revision == "a" * 40


def test_runner_requires_clean_revision_before_opening_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s10, s11, output = _roots(tmp_path)
    monkeypatch.setattr(
        s13b_run,
        "_git_provenance",
        lambda: s13_pilot.PilotGitProvenance("a" * 40, True, "b" * 64),
    )
    monkeypatch.setattr(
        s13b_run,
        "load_selected_s11_initialization",
        lambda path: pytest.fail("initialization opened before clean-revision check"),
    )
    monkeypatch.setattr(
        s13b_run,
        "open_selective_s10_source",
        lambda *args, **kwargs: pytest.fail("S10 opened before clean-revision check"),
    )

    with pytest.raises(RuntimeError, match="clean revision"):
        s13b_run.run_s13b_peak_pilot(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            run_root=output,
        )

    assert not output.exists()


@pytest.mark.parametrize("prior_step", (0, 400))
def test_runner_uses_peak_objective_and_only_train_fit_for_800_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prior_step: int
) -> None:
    s10, s11, output = _roots(tmp_path)
    pilot = build_dogbone_s13b_pilot_plan()
    monkeypatch.setattr(s13b_run, "build_dogbone_s13b_pilot_plan", lambda: pilot)
    monkeypatch.setattr(
        s13b_run,
        "_git_provenance",
        lambda: s13_pilot.PilotGitProvenance("a" * 40, False, "b" * 64),
    )
    models: list[object] = []

    def initialize(path: Path) -> SimpleNamespace:
        assert path == s11
        model = object()
        models.append(model)
        return SimpleNamespace(model=model)

    monkeypatch.setattr(s13b_run, "load_selected_s11_initialization", initialize)
    binding = SimpleNamespace(run_hash="c" * 64)
    monkeypatch.setattr(
        s13b_run, "_build_s13b_binding", lambda *args, **kwargs: binding
    )
    source = SimpleNamespace(
        bindings=SimpleNamespace(content_hash=pilot.base_protocol.source_bindings_hash),
        development_opened=False,
        iter_train_screen=lambda: pytest.fail("TRAIN-screen opened during fitting"),
    )
    access_records: list[Path] = []

    def open_source(path: Path, *, development_access_record: Path) -> object:
        assert path == s10
        access_records.append(development_access_record)
        return source

    monkeypatch.setattr(s13b_run, "open_selective_s10_source", open_source)
    population = SimpleNamespace(role=EvidenceRole.TRAIN, sample_ids=("fit-001",))
    cases = (object(),)
    data = SimpleNamespace(population=population, cases=cases)
    monkeypatch.setattr(
        s13b_run,
        "materialize_dogbone_s13_training_data",
        lambda actual_source, initialization: data,
    )
    monkeypatch.setattr(s13b_run, "_verify_training_data", lambda *args: None)
    objective = object()
    objective_calls: list[tuple[object, bool]] = []

    def build_objective(actual_cases: object, *, include_peak_loss: bool) -> object:
        objective_calls.append((actual_cases, include_peak_loss))
        return objective

    monkeypatch.setattr(s13b_run, "DogboneS13ObjectiveBuilder", build_objective)
    calls: list[dict[str, object]] = []

    def train(path: Path, **kwargs: object) -> SimpleNamespace:
        calls.append({"path": path, **kwargs})
        return SimpleNamespace(
            step=kwargs["target_step"],
            model_state_hash="d" * 64,
            checkpoint=SimpleNamespace(
                path=path,
                manifest=SimpleNamespace(manifest_hash="e" * 64),
            ),
            final_losses={ObjectiveTerm.DATA_PEAK: 0.1},
        )

    monkeypatch.setattr(s13b_run, "train_to_checkpoint", train)
    progress: list[str] = []
    if prior_step:
        previous = output / "checkpoint-step-0400"
        previous.mkdir(parents=True)
        monkeypatch.setattr(
            s13b_run, "_latest_checkpoint", lambda *args, **kwargs: (previous, 400)
        )

    result = s13b_run.run_s13b_peak_pilot(
        s10_run_root=s10,
        s11_checkpoint_root=s11,
        run_root=output,
        segment_steps=400,
        progress=progress.append,
    )

    assert result.step == 800
    assert access_records == [output / "development-access.json"]
    assert objective_calls == [(cases, True)]
    assert [call["target_step"] for call in calls] == (
        [400, 800] if prior_step == 0 else [800]
    )
    assert [call["path"] for call in calls] == (
        [output / "checkpoint-step-0400", output / "checkpoint-step-0800"]
        if prior_step == 0
        else [output / "checkpoint-step-0800"]
    )
    assert calls[0]["resume_from"] == (
        None if prior_step == 0 else output / "checkpoint-step-0400"
    )
    assert calls[0]["model"] is models[0]
    if prior_step == 0:
        assert calls[1]["resume_from"] == output / "checkpoint-step-0400"
        assert calls[1]["model"] is models[1]
    assert all(call["population"] is population for call in calls)
    assert all(call["objective"] is objective for call in calls)
    assert all(call["base_registry"] is pilot.objective_registry for call in calls)
    assert all(call["binding"] is binding for call in calls)
    assert all(call["plan"] == pilot.sampler_plan for call in calls)
    assert all(call["config"] == pilot.trainer_config for call in calls)
    assert any("non_gate=true" in line for line in progress)
    assert not (output / "development-access.json").exists()


def test_runner_rejects_invalid_segment_before_source_access(
    tmp_path: Path,
) -> None:
    s10, s11, output = _roots(tmp_path)

    with pytest.raises(ValueError, match="segment_steps"):
        s13b_run.run_s13b_peak_pilot(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            run_root=output,
            segment_steps=0,
        )


def test_runner_stops_if_materialization_opens_development(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s10, s11, output = _roots(tmp_path)
    monkeypatch.setattr(
        s13b_run,
        "_git_provenance",
        lambda: s13_pilot.PilotGitProvenance("a" * 40, False, "b" * 64),
    )
    monkeypatch.setattr(
        s13b_run,
        "load_selected_s11_initialization",
        lambda path: SimpleNamespace(model=object()),
    )
    monkeypatch.setattr(
        s13b_run,
        "_build_s13b_binding",
        lambda *args, **kwargs: SimpleNamespace(run_hash="c" * 64),
    )
    source = SimpleNamespace(
        bindings=SimpleNamespace(content_hash="d" * 64),
        development_opened=False,
    )
    monkeypatch.setattr(
        s13b_run,
        "open_selective_s10_source",
        lambda *args, **kwargs: source,
    )

    def leak(*args: object) -> object:
        source.development_opened = True
        return object()

    monkeypatch.setattr(s13b_run, "materialize_dogbone_s13_training_data", leak)
    monkeypatch.setattr(
        s13b_run,
        "_verify_training_data",
        lambda *args: pytest.fail("verification reached after DEVELOPMENT opened"),
    )

    with pytest.raises(RuntimeError, match="DEVELOPMENT labels opened"):
        s13b_run.run_s13b_peak_pilot(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            run_root=output,
        )


def test_runner_cli_reports_non_gate_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    s10, s11, output = _roots(tmp_path)
    checkpoint = output / "checkpoint-step-0800"
    monkeypatch.setattr(
        s13b_run,
        "run_s13b_peak_pilot",
        lambda **kwargs: SimpleNamespace(checkpoint=SimpleNamespace(path=checkpoint)),
    )

    code = s13b_run.main(
        (
            "--s10-run-root",
            str(s10),
            "--s11-checkpoint-root",
            str(s11),
            "--run-root",
            str(output),
        )
    )

    assert code == 0
    assert f"pilot_checkpoint={checkpoint}" in capsys.readouterr().out
