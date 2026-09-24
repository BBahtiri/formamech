from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from campaigns.dogbone_v1 import s13_evaluate
from piginot.doe import EvidenceRole
from piginot.training import CheckpointLineage, SelectionMetrics, TrialResult


def _roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    s10 = tmp_path / "s10"
    s11 = tmp_path / "s11"
    run = tmp_path / "registered-trial"
    checkpoint = run / "checkpoint-step-1500"
    for path in (s10, s11, run, checkpoint):
        path.mkdir()
    return s10, s11, checkpoint


def _metrics() -> SelectionMetrics:
    return SelectionMetrics(
        median_absolute_peak_error=0.05,
        p90_absolute_peak_error=0.1,
        median_signed_peak_error=-0.04,
        p_relative_l2=0.15,
        u_relative_l2=0.03,
        hybrid_loss=0.002,
        invalid_collocation_points=0,
    )


def _batch_roots(tmp_path: Path) -> tuple[Path, Path, tuple[Path, ...]]:
    s10, s11, first = _roots(tmp_path)
    checkpoints = [first]
    for index in range(1, 6):
        checkpoint = tmp_path / f"registered-trial-{index}" / "checkpoint-step-1500"
        checkpoint.mkdir(parents=True)
        checkpoints.append(checkpoint)
    return s10, s11, tuple(checkpoints)


def _mock_batch_dependencies(
    monkeypatch: pytest.MonkeyPatch, checkpoints: tuple[Path, ...]
) -> SimpleNamespace:
    configs = s13_evaluate.build_dogbone_s13_trainer_configs(
        s13_evaluate.build_dogbone_s13_protocol()
    )
    trial_indices = {
        config.trial.trial_id: index for index, config in enumerate(configs)
    }
    models: list[torch.nn.Module] = []
    events: list[str] = []
    source = SimpleNamespace(
        bindings=SimpleNamespace(content_hash="c" * 64),
        development_opened=False,
    )
    data = SimpleNamespace(population=SimpleNamespace(sample_ids=("fit",)))

    def initialize(path: Path) -> SimpleNamespace:
        model = torch.nn.Linear(2, 2, dtype=torch.float64)
        models.append(model)
        events.append("initialize")
        return SimpleNamespace(model=model)

    monkeypatch.setattr(s13_evaluate, "load_selected_s11_initialization", initialize)

    def build_binding(*args: object, **kwargs: object) -> SimpleNamespace:
        assert kwargs["implementation_revision"] == (
            "5473cdeebd06597d4ec0bdd50df7d230d00306cd"
        )
        config = kwargs["config"]
        return SimpleNamespace(
            run_hash=f"{trial_indices[config.trial.trial_id] + 1:064x}"
        )

    monkeypatch.setattr(s13_evaluate, "_build_pilot_binding", build_binding)

    def load(path: Path, **kwargs: object) -> SimpleNamespace:
        index = checkpoints.index(path)
        assert kwargs["binding"].run_hash == f"{index + 1:064x}"
        assert kwargs["model"] is models[-1]
        assert isinstance(kwargs["optimizer"], torch.optim.Adam)
        events.append(f"load-{index}")
        return SimpleNamespace(
            manifest=SimpleNamespace(
                step=1500,
                run_hash=f"{index + 1:064x}",
                dataset_snapshot_hash="b" * 64,
                execution_profile_hash="d" * 64,
                model_export_hash=f"{index + 10:064x}",
                model_state_hash=f"{index + 20:064x}",
                manifest_hash=f"{index + 30:064x}",
            )
        )

    monkeypatch.setattr(s13_evaluate, "load_training_checkpoint", load)

    def open_source(path: Path, *, development_access_record: Path) -> object:
        assert (
            development_access_record
            == checkpoints[0].parent / "development-access.json"
        )
        events.append("source")
        return source

    monkeypatch.setattr(s13_evaluate, "open_selective_s10_source", open_source)

    def materialize(actual_source: object, initialization: object) -> object:
        assert actual_source is source
        assert initialization.model is models[-6]
        events.append("fit")
        return data

    monkeypatch.setattr(
        s13_evaluate, "materialize_dogbone_s13_training_data", materialize
    )
    monkeypatch.setattr(s13_evaluate, "_verify_training_data", lambda *args: None)

    def screen(**kwargs: object) -> SelectionMetrics:
        index = trial_indices[kwargs["trial"].trial_id]
        assert kwargs["source"] is source
        assert kwargs["training_data"] is data
        assert kwargs["model"] is models[-6 + index]
        events.append(f"screen-{index}")
        return _metrics()

    monkeypatch.setattr(s13_evaluate, "evaluate_dogbone_s13_train_screen", screen)
    return SimpleNamespace(events=events, source=source, configs=configs)


def _mock_evaluation_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: Path,
) -> SimpleNamespace:
    model = torch.nn.Linear(2, 2, dtype=torch.float64)
    initialization = SimpleNamespace(model=model)
    source = SimpleNamespace(
        bindings=SimpleNamespace(content_hash="c" * 64),
        development_opened=False,
    )
    data = SimpleNamespace(population=SimpleNamespace(sample_ids=("fit",)))
    manifest = SimpleNamespace(
        step=1500,
        run_hash="a" * 64,
        dataset_snapshot_hash="b" * 64,
        execution_profile_hash="d" * 64,
        model_export_hash="e" * 64,
        model_state_hash="f" * 64,
        manifest_hash="1" * 64,
    )
    loaded = SimpleNamespace(manifest=manifest)
    events: list[str] = []
    monkeypatch.setattr(
        s13_evaluate, "load_selected_s11_initialization", lambda path: initialization
    )

    def build_binding(*args: object, **kwargs: object) -> SimpleNamespace:
        assert kwargs["implementation_revision"] == (
            "5473cdeebd06597d4ec0bdd50df7d230d00306cd"
        )
        return SimpleNamespace(run_hash="a" * 64)

    monkeypatch.setattr(s13_evaluate, "_build_pilot_binding", build_binding)

    def load(path: Path, **kwargs: object) -> SimpleNamespace:
        assert path == checkpoint
        assert kwargs["model"] is model
        assert isinstance(kwargs["optimizer"], torch.optim.Adam)
        events.append("load")
        return loaded

    monkeypatch.setattr(s13_evaluate, "load_training_checkpoint", load)

    def open_source(path: Path, *, development_access_record: Path) -> object:
        assert (
            development_access_record == checkpoint.parent / "development-access.json"
        )
        events.append("source")
        return source

    monkeypatch.setattr(s13_evaluate, "open_selective_s10_source", open_source)

    def materialize(actual_source: object, actual_initialization: object) -> object:
        assert actual_source is source
        assert actual_initialization is initialization
        events.append("fit")
        return data

    monkeypatch.setattr(
        s13_evaluate, "materialize_dogbone_s13_training_data", materialize
    )
    monkeypatch.setattr(s13_evaluate, "_verify_training_data", lambda *args: None)

    def screen(**kwargs: object) -> SelectionMetrics:
        assert kwargs["source"] is source
        assert kwargs["training_data"] is data
        assert kwargs["model"] is model
        events.append("screen")
        return _metrics()

    monkeypatch.setattr(s13_evaluate, "evaluate_dogbone_s13_train_screen", screen)
    return SimpleNamespace(source=source, manifest=manifest, events=events)


def test_trial_index_and_checkpoint_preflight_fail_before_external_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s10, s11, checkpoint = _roots(tmp_path)
    monkeypatch.setattr(
        s13_evaluate,
        "load_selected_s11_initialization",
        lambda path: pytest.fail("S11 must not open"),
    )
    with pytest.raises(ValueError, match="trial_index"):
        s13_evaluate.evaluate_registered_s13_trial(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            checkpoint_path=checkpoint,
            trial_index=6,
        )
    with pytest.raises(ValueError, match="final registered checkpoint"):
        s13_evaluate.evaluate_registered_s13_trial(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            checkpoint_path=checkpoint.parent,
            trial_index=0,
        )
    (checkpoint.parent / "development-access.json").write_text("opened")
    with pytest.raises(RuntimeError, match="DEVELOPMENT"):
        s13_evaluate.evaluate_registered_s13_trial(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            checkpoint_path=checkpoint,
            trial_index=0,
        )


def test_integrity_load_precedes_train_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s10, s11, checkpoint = _roots(tmp_path)
    context = _mock_evaluation_dependencies(monkeypatch, checkpoint)

    def reject_load(path: Path, **kwargs: object) -> None:
        raise ValueError("model export hash mismatch")

    monkeypatch.setattr(s13_evaluate, "load_training_checkpoint", reject_load)
    with pytest.raises(ValueError, match="model export hash mismatch"):
        s13_evaluate.evaluate_registered_s13_trial(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            checkpoint_path=checkpoint,
            trial_index=0,
        )
    assert context.events == []


def test_incomplete_checkpoint_cannot_be_screened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s10, s11, checkpoint = _roots(tmp_path)
    context = _mock_evaluation_dependencies(monkeypatch, checkpoint)
    context.manifest.step = 1499
    with pytest.raises(ValueError, match="optimizer_steps"):
        s13_evaluate.evaluate_registered_s13_trial(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            checkpoint_path=checkpoint,
            trial_index=0,
        )
    assert context.events == ["load"]


def test_wrong_registered_run_cannot_be_screened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s10, s11, checkpoint = _roots(tmp_path)
    context = _mock_evaluation_dependencies(monkeypatch, checkpoint)
    context.manifest.run_hash = "0" * 64
    with pytest.raises(ValueError, match="run_hash"):
        s13_evaluate.evaluate_registered_s13_trial(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            checkpoint_path=checkpoint,
            trial_index=0,
        )
    assert context.events == ["load"]


def test_full_screen_returns_exact_train_result_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s10, s11, checkpoint = _roots(tmp_path)
    context = _mock_evaluation_dependencies(monkeypatch, checkpoint)
    result = s13_evaluate.evaluate_registered_s13_trial(
        s10_run_root=s10,
        s11_checkpoint_root=s11,
        checkpoint_path=checkpoint,
        trial_index=0,
    )
    assert type(result) is TrialResult
    assert result.role is EvidenceRole.TRAIN
    assert result.metrics == _metrics()
    assert context.events == ["load", "source", "fit", "screen"]
    assert result.lineage == CheckpointLineage(
        trial_id=s13_evaluate.build_dogbone_s13_trainer_configs(
            s13_evaluate.build_dogbone_s13_protocol()
        )[0].trial.trial_id,
        run_hash="a" * 64,
        dataset_snapshot_hash="b" * 64,
        execution_profile_hash="d" * 64,
        weights_hash="e" * 64,
        model_state_hash="f" * 64,
        checkpoint_hash="1" * 64,
    )
    assert list(checkpoint.iterdir()) == []
    assert not (checkpoint.parent / "development-access.json").exists()


def test_opened_development_source_aborts_before_train_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s10, s11, checkpoint = _roots(tmp_path)
    context = _mock_evaluation_dependencies(monkeypatch, checkpoint)
    context.source.development_opened = True
    with pytest.raises(RuntimeError, match="DEVELOPMENT"):
        s13_evaluate.evaluate_registered_s13_trial(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            checkpoint_path=checkpoint,
            trial_index=0,
        )
    assert context.events == ["load", "source"]


def test_development_opened_during_fit_aborts_before_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s10, s11, checkpoint = _roots(tmp_path)
    context = _mock_evaluation_dependencies(monkeypatch, checkpoint)

    def materialize(source: SimpleNamespace, initialization: object) -> object:
        context.events.append("fit")
        source.development_opened = True
        return SimpleNamespace(population=SimpleNamespace(sample_ids=("fit",)))

    monkeypatch.setattr(
        s13_evaluate, "materialize_dogbone_s13_training_data", materialize
    )
    with pytest.raises(RuntimeError, match="DEVELOPMENT"):
        s13_evaluate.evaluate_registered_s13_trial(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            checkpoint_path=checkpoint,
            trial_index=0,
        )
    assert context.events == ["load", "source", "fit"]


def test_development_opened_during_screen_aborts_before_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s10, s11, checkpoint = _roots(tmp_path)
    context = _mock_evaluation_dependencies(monkeypatch, checkpoint)

    def screen(**kwargs: object) -> SelectionMetrics:
        context.events.append("screen")
        context.source.development_opened = True
        return _metrics()

    monkeypatch.setattr(s13_evaluate, "evaluate_dogbone_s13_train_screen", screen)
    with pytest.raises(RuntimeError, match="DEVELOPMENT"):
        s13_evaluate.evaluate_registered_s13_trial(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            checkpoint_path=checkpoint,
            trial_index=0,
        )
    assert context.events == ["load", "source", "fit", "screen"]


def test_cli_prints_canonical_single_train_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    s10, s11, checkpoint = _roots(tmp_path)
    result = TrialResult(
        lineage=CheckpointLineage(
            trial_id="trial-test",
            run_hash="a" * 64,
            dataset_snapshot_hash="b" * 64,
            execution_profile_hash="c" * 64,
            weights_hash="d" * 64,
            model_state_hash="e" * 64,
            checkpoint_hash="f" * 64,
        ),
        role=EvidenceRole.TRAIN,
        metrics=_metrics(),
    )
    monkeypatch.setattr(
        s13_evaluate, "evaluate_registered_s13_trial", lambda **kwargs: result
    )
    assert (
        s13_evaluate.main(
            (
                "--s10-run-root",
                str(s10),
                "--s11-checkpoint-root",
                str(s11),
                "--checkpoint-path",
                str(checkpoint),
                "--trial-index",
                "0",
            )
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["role"] == "TRAIN"
    assert printed["lineage"]["checkpoint_hash"] == "f" * 64
    assert "CP-FREEZE" not in printed
    assert list(checkpoint.iterdir()) == []


def test_batch_validates_all_six_before_one_shared_train_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s10, s11, checkpoints = _batch_roots(tmp_path)
    context = _mock_batch_dependencies(monkeypatch, checkpoints)

    results = s13_evaluate.evaluate_registered_s13_trials(
        s10_run_root=s10,
        s11_checkpoint_root=s11,
        checkpoint_paths=checkpoints,
    )

    assert type(results) is tuple and len(results) == 6
    assert all(type(result) is TrialResult for result in results)
    assert all(result.role is EvidenceRole.TRAIN for result in results)
    assert tuple(result.trial_id for result in results) == tuple(
        config.trial.trial_id for config in context.configs
    )
    assert tuple(result.weights_hash for result in results) == tuple(
        f"{index + 10:064x}" for index in range(6)
    )
    assert context.events.count("fit") == 1
    assert context.events.count("source") == 1
    assert context.events.index("source") > context.events.index("load-5")
    assert context.events[-6:] == [f"screen-{index}" for index in range(6)]
    assert all(list(path.iterdir()) == [] for path in checkpoints)


def test_batch_bad_sixth_checkpoint_blocks_all_s10_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s10, s11, checkpoints = _batch_roots(tmp_path)
    context = _mock_batch_dependencies(monkeypatch, checkpoints)
    original_load = s13_evaluate.load_training_checkpoint

    def reject_sixth(path: Path, **kwargs: object) -> object:
        if path == checkpoints[5]:
            raise ValueError("model export hash mismatch")
        return original_load(path, **kwargs)

    monkeypatch.setattr(s13_evaluate, "load_training_checkpoint", reject_sixth)
    with pytest.raises(ValueError, match="model export hash mismatch"):
        s13_evaluate.evaluate_registered_s13_trials(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            checkpoint_paths=checkpoints,
        )
    assert "source" not in context.events
    assert "fit" not in context.events
    assert not any(event.startswith("screen") for event in context.events)


def test_batch_preflights_all_paths_and_development_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s10, s11, checkpoints = _batch_roots(tmp_path)
    context = _mock_batch_dependencies(monkeypatch, checkpoints)
    with pytest.raises(ValueError, match="six"):
        s13_evaluate.evaluate_registered_s13_trials(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            checkpoint_paths=checkpoints[:5],
        )
    with pytest.raises(ValueError, match="checkpoint_path"):
        s13_evaluate.evaluate_registered_s13_trials(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            checkpoint_paths=(*checkpoints[:5], tmp_path / "missing"),
        )
    (checkpoints[5].parent / "development-access.json").write_text("opened")
    with pytest.raises(RuntimeError, match="DEVELOPMENT"):
        s13_evaluate.evaluate_registered_s13_trials(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            checkpoint_paths=checkpoints,
        )
    with pytest.raises(ValueError, match="distinct run root"):
        s13_evaluate.evaluate_registered_s13_trials(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            checkpoint_paths=(*checkpoints[:4], checkpoints[4], checkpoints[4]),
        )
    assert context.events == []


def test_batch_refuses_development_opened_by_source_or_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s10, s11, checkpoints = _batch_roots(tmp_path)
    context = _mock_batch_dependencies(monkeypatch, checkpoints)
    context.source.development_opened = True
    with pytest.raises(RuntimeError, match="DEVELOPMENT"):
        s13_evaluate.evaluate_registered_s13_trials(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            checkpoint_paths=checkpoints,
        )
    assert "fit" not in context.events

    context.source.development_opened = False
    original_materialize = s13_evaluate.materialize_dogbone_s13_training_data

    def opening_fit(source: object, initialization: object) -> object:
        data = original_materialize(source, initialization)
        context.source.development_opened = True
        return data

    monkeypatch.setattr(
        s13_evaluate, "materialize_dogbone_s13_training_data", opening_fit
    )
    with pytest.raises(RuntimeError, match="DEVELOPMENT"):
        s13_evaluate.evaluate_registered_s13_trials(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            checkpoint_paths=checkpoints,
        )
    assert not any(event.startswith("screen") for event in context.events)

    context.source.development_opened = False
    monkeypatch.setattr(
        s13_evaluate, "materialize_dogbone_s13_training_data", original_materialize
    )

    def opening_screen(**kwargs: object) -> SelectionMetrics:
        context.events.append("screen")
        context.source.development_opened = True
        return _metrics()

    monkeypatch.setattr(
        s13_evaluate, "evaluate_dogbone_s13_train_screen", opening_screen
    )
    with pytest.raises(RuntimeError, match="DEVELOPMENT"):
        s13_evaluate.evaluate_registered_s13_trials(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            checkpoint_paths=checkpoints,
        )
    assert context.events.count("screen") == 1


def test_batch_cli_prints_canonical_results_in_registered_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    s10, s11, checkpoints = _batch_roots(tmp_path)
    results = tuple(
        TrialResult(
            lineage=CheckpointLineage(
                trial_id=f"trial-{index}",
                run_hash=f"{index + 1:064x}",
                dataset_snapshot_hash="b" * 64,
                execution_profile_hash="c" * 64,
                weights_hash="d" * 64,
                model_state_hash="e" * 64,
                checkpoint_hash="f" * 64,
            ),
            role=EvidenceRole.TRAIN,
            metrics=_metrics(),
        )
        for index in range(6)
    )
    monkeypatch.setattr(
        s13_evaluate, "evaluate_registered_s13_trials", lambda **kwargs: results
    )
    arguments = [
        "--s10-run-root",
        str(s10),
        "--s11-checkpoint-root",
        str(s11),
    ]
    for checkpoint in checkpoints:
        arguments.extend(("--batch-checkpoint-path", str(checkpoint)))
    assert s13_evaluate.main(arguments) == 0
    printed = json.loads(capsys.readouterr().out)
    assert [record["lineage"]["trial_id"] for record in printed] == [
        f"trial-{index}" for index in range(6)
    ]
    assert all(record["role"] == "TRAIN" for record in printed)
    assert all(list(path.iterdir()) == [] for path in checkpoints)


def test_cli_rejects_mixed_and_incomplete_modes(tmp_path: Path) -> None:
    s10, s11, checkpoints = _batch_roots(tmp_path)
    shared = ("--s10-run-root", str(s10), "--s11-checkpoint-root", str(s11))
    with pytest.raises(SystemExit) as mixed:
        s13_evaluate.main(
            (
                *shared,
                "--checkpoint-path",
                str(checkpoints[0]),
                "--trial-index",
                "0",
                "--batch-checkpoint-path",
                str(checkpoints[0]),
            )
        )
    assert mixed.value.code == 2
    with pytest.raises(SystemExit) as incomplete:
        s13_evaluate.main(shared)
    assert incomplete.value.code == 2
