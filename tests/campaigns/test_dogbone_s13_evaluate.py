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
        assert development_access_record == checkpoint.parent / "development-access.json"
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
