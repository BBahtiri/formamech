from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from campaigns.dogbone_v1 import s13_trial
from campaigns.dogbone_v1.s13_pilot import PilotGitProvenance


def _roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    s10 = tmp_path / "s10"
    s11 = tmp_path / "s11"
    output_parent = tmp_path / "runs"
    for path in (s10, s11, output_parent):
        path.mkdir()
    return s10, s11, output_parent / "registered-trial"


def test_segment_targets_cover_remaining_budget_without_skipping() -> None:
    assert s13_trial._segment_targets(0, 250, 100) == (100, 200, 250)
    assert s13_trial._segment_targets(100, 250, 100) == (200, 250)
    with pytest.raises(ValueError, match="already complete"):
        s13_trial._segment_targets(250, 250, 100)
    with pytest.raises(ValueError, match="segment_steps"):
        s13_trial._segment_targets(0, 250, 0)


def test_registered_trial_refuses_dirty_code_before_opening_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s10, s11, output = _roots(tmp_path)
    monkeypatch.setattr(
        s13_trial,
        "_git_provenance",
        lambda: PilotGitProvenance("a" * 40, True, "b" * 64),
    )
    opened = False

    def open_source(*args: object, **kwargs: object) -> None:
        nonlocal opened
        opened = True

    monkeypatch.setattr(s13_trial, "open_selective_s10_source", open_source)

    with pytest.raises(RuntimeError, match="clean revision"):
        s13_trial.run_registered_s13_trial(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            run_root=output,
            trial_index=0,
            segment_steps=100,
        )

    assert not opened
    assert not output.exists()


def test_registered_trial_rejects_unknown_or_existing_source_output(
    tmp_path: Path,
) -> None:
    s10, s11, output = _roots(tmp_path)
    with pytest.raises(ValueError, match="outside source roots"):
        s13_trial._prepare_run_root(s10, s11, s10 / "run")
    output.mkdir()
    (output / "unexpected.txt").write_text("foreign")
    with pytest.raises(ValueError, match="unexpected"):
        s13_trial._latest_checkpoint(output, expected_run_hash="a" * 64)


def test_registered_trial_resume_requires_matching_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, output = _roots(tmp_path)
    output.mkdir()
    checkpoint = output / "checkpoint-step-0100"
    checkpoint.mkdir()
    (checkpoint / "checkpoint.json").write_bytes(b"fixture")
    monkeypatch.setattr(
        s13_trial.TrainingCheckpointManifest,
        "from_bytes",
        lambda data: SimpleNamespace(step=100, run_hash="b" * 64),
    )

    with pytest.raises(ValueError, match="run_hash"):
        s13_trial._latest_checkpoint(output, expected_run_hash="a" * 64)

    path, step = s13_trial._latest_checkpoint(output, expected_run_hash="b" * 64)
    assert path == checkpoint
    assert step == 100


def test_trial_index_and_step_inputs_refuse_before_run(
    tmp_path: Path,
) -> None:
    s10, s11, output = _roots(tmp_path)
    with pytest.raises(ValueError, match="trial_index"):
        s13_trial.run_registered_s13_trial(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            run_root=output,
            trial_index=6,
        )
    with pytest.raises(ValueError, match="segment_steps"):
        s13_trial.run_registered_s13_trial(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            run_root=output,
            trial_index=0,
            segment_steps=0,
        )


def test_registered_trial_reloads_initial_model_and_resumes_second_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s10, s11, output = _roots(tmp_path)
    monkeypatch.setattr(
        s13_trial,
        "_git_provenance",
        lambda: PilotGitProvenance("a" * 40, False, "b" * 64),
    )
    models: list[object] = []

    def initialize(path: Path) -> SimpleNamespace:
        assert path == s11
        model = object()
        models.append(model)
        return SimpleNamespace(model=model)

    monkeypatch.setattr(s13_trial, "load_selected_s11_initialization", initialize)
    monkeypatch.setattr(
        s13_trial,
        "_build_pilot_binding",
        lambda *args, **kwargs: SimpleNamespace(run_hash="c" * 64),
    )
    source = SimpleNamespace(
        bindings=SimpleNamespace(content_hash="d" * 64),
        development_opened=False,
    )
    monkeypatch.setattr(
        s13_trial,
        "open_selective_s10_source",
        lambda *args, **kwargs: source,
    )
    materializations: list[object] = []
    objective = object()
    data = SimpleNamespace(
        population=SimpleNamespace(sample_ids=("fit-001",)),
        objective_builder=lambda: objective,
    )

    def materialize(actual_source: object, initialization: object) -> object:
        assert actual_source is source
        materializations.append(initialization)
        return data

    monkeypatch.setattr(s13_trial, "materialize_dogbone_s13_training_data", materialize)
    monkeypatch.setattr(s13_trial, "_verify_training_data", lambda *args: None)
    calls: list[dict[str, object]] = []

    def train(path: Path, **kwargs: object) -> SimpleNamespace:
        calls.append({"path": path, **kwargs})
        target = kwargs["target_step"]
        return SimpleNamespace(
            step=target,
            model_state_hash="e" * 64,
            checkpoint=SimpleNamespace(
                manifest=SimpleNamespace(manifest_hash="f" * 64)
            ),
            final_losses={},
        )

    monkeypatch.setattr(s13_trial, "train_to_checkpoint", train)
    progress: list[str] = []

    result = s13_trial.run_registered_s13_trial(
        s10_run_root=s10,
        s11_checkpoint_root=s11,
        run_root=output,
        trial_index=0,
        segment_steps=750,
        progress=progress.append,
    )

    assert result.step == 1500
    assert len(materializations) == 1
    assert len(models) == 2
    assert [call["target_step"] for call in calls] == [750, 1500]
    assert calls[0]["resume_from"] is None
    assert calls[1]["resume_from"] == output / "checkpoint-step-0750"
    assert calls[0]["model"] is models[0]
    assert calls[1]["model"] is models[1]
    assert all(call["objective"] is objective for call in calls)
    assert any("phase=CHECKPOINT" in line for line in progress)


def test_trial_main_reports_final_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    s10, s11, output = _roots(tmp_path)
    checkpoint = output / "checkpoint-step-1500"
    monkeypatch.setattr(
        s13_trial,
        "run_registered_s13_trial",
        lambda **kwargs: SimpleNamespace(checkpoint=SimpleNamespace(path=checkpoint)),
    )

    exit_code = s13_trial.main(
        (
            "--s10-run-root",
            str(s10),
            "--s11-checkpoint-root",
            str(s11),
            "--run-root",
            str(output),
            "--trial-index",
            "0",
        )
    )

    assert exit_code == 0
    assert f"final_checkpoint={checkpoint}" in capsys.readouterr().out
