"""The S13b pilot comparison is TRAIN-only and cannot promote a gate."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from campaigns.dogbone_v1 import s13b_evaluate
from campaigns.dogbone_v1.s13b_evaluate import compare_s13b_train_results
from campaigns.dogbone_v1.s13b_protocol import build_dogbone_s13b_pilot_plan
from piginot.doe import EvidenceRole
from piginot.training import CheckpointLineage, SelectionMetrics, TrialResult


def _result(*, control: bool, metrics: SelectionMetrics) -> TrialResult:
    pilot = build_dogbone_s13b_pilot_plan()
    lineage = CheckpointLineage(
        trial_id=(
            pilot.control_trial_id if control else pilot.search_space.trials[0].trial_id
        ),
        run_hash=pilot.control_run_hash if control else "1" * 64,
        dataset_snapshot_hash=pilot.base_protocol.dataset_snapshot_hash,
        execution_profile_hash="2" * 64,
        weights_hash="3" * 64,
        model_state_hash="4" * 64,
        checkpoint_hash=pilot.control_manifest_hash if control else "5" * 64,
    )
    return TrialResult(lineage=lineage, role=EvidenceRole.TRAIN, metrics=metrics)


def _control_metrics() -> SelectionMetrics:
    return SelectionMetrics(
        median_absolute_peak_error=0.12,
        p90_absolute_peak_error=0.17,
        median_signed_peak_error=-0.12,
        p_relative_l2=0.15,
        u_relative_l2=0.03,
        hybrid_loss=0.001,
        invalid_collocation_points=0,
    )


def _peak_metrics() -> SelectionMetrics:
    return replace(
        _control_metrics(),
        median_absolute_peak_error=0.08,
        p90_absolute_peak_error=0.13,
        median_signed_peak_error=-0.10,
        p_relative_l2=0.151,
        u_relative_l2=0.0303,
        hybrid_loss=100.0,
    )


def test_pilot_signal_uses_peak_and_field_criteria_not_hybrid_loss() -> None:
    pilot = build_dogbone_s13b_pilot_plan()

    comparison = compare_s13b_train_results(
        pilot,
        control=_result(control=True, metrics=_control_metrics()),
        peak=_result(control=False, metrics=_peak_metrics()),
    )

    assert comparison.pilot_signal is True
    assert all(passed for _, passed in comparison.criteria)
    assert comparison.plan_hash == pilot.plan_hash
    assert comparison.peak.metrics.hybrid_loss == 100.0


@pytest.mark.parametrize(
    ("change", "value", "criterion"),
    (
        ("median_absolute_peak_error", 0.12, "median_absolute_peak_improved"),
        ("p90_absolute_peak_error", 0.17, "p90_absolute_peak_improved"),
        ("median_signed_peak_error", -0.109, "underprediction_reduced_10_percent"),
        ("u_relative_l2", 0.0307, "u_field_nonregression"),
        ("p_relative_l2", 0.154, "p_field_nonregression"),
        ("invalid_collocation_points", 1, "zero_invalid_collocation"),
    ),
)
def test_each_failed_criterion_rejects_pilot_signal(
    change: str, value: float | int, criterion: str
) -> None:
    metrics = replace(_peak_metrics(), **{change: value})  # type: ignore[arg-type]

    comparison = compare_s13b_train_results(
        build_dogbone_s13b_pilot_plan(),
        control=_result(control=True, metrics=_control_metrics()),
        peak=_result(control=False, metrics=metrics),
    )

    assert comparison.pilot_signal is False
    assert dict(comparison.criteria)[criterion] is False


def test_comparison_rejects_non_train_or_wrong_control_lineage() -> None:
    pilot = build_dogbone_s13b_pilot_plan()
    control = _result(control=True, metrics=_control_metrics())
    peak = _result(control=False, metrics=_peak_metrics())

    with pytest.raises(PermissionError, match="TRAIN"):
        compare_s13b_train_results(
            pilot, control=replace(control, role=EvidenceRole.DEVELOPMENT), peak=peak
        )
    with pytest.raises(ValueError, match="control checkpoint"):
        compare_s13b_train_results(
            pilot,
            control=replace(
                control,
                lineage=replace(control.lineage, checkpoint_hash="0" * 64),
            ),
            peak=peak,
        )


def test_checkpoint_loader_rejects_wrong_control_manifest_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pilot = build_dogbone_s13b_pilot_plan()
    checkpoint = tmp_path / "checkpoint-step-0800"
    checkpoint.mkdir()
    manifest = SimpleNamespace(
        step=800, run_hash=pilot.control_run_hash, manifest_hash="0" * 64
    )
    monkeypatch.setattr(
        s13b_evaluate,
        "load_training_checkpoint",
        lambda *args, **kwargs: SimpleNamespace(manifest=manifest),
    )
    model = torch.nn.Linear(1, 1, dtype=torch.float64)
    binding = SimpleNamespace(run_hash=pilot.control_run_hash)

    with pytest.raises(ValueError, match="manifest hash"):
        s13b_evaluate._load_checked_checkpoint(
            checkpoint,
            model=model,
            config=pilot.trainer_config,
            binding=binding,  # type: ignore[arg-type]
            expected_manifest_hash=pilot.control_manifest_hash,
        )


def test_evaluator_loads_both_checkpoints_before_opening_train_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s10 = tmp_path / "s10"
    s11 = tmp_path / "s11"
    control_root = tmp_path / "control"
    peak_root = tmp_path / "peak"
    for path in (s10, s11, control_root, peak_root):
        path.mkdir()
    control_cp = control_root / "checkpoint-step-0800"
    peak_cp = peak_root / "checkpoint-step-0800"
    control_cp.mkdir()
    peak_cp.mkdir()
    pilot = build_dogbone_s13b_pilot_plan()
    models: list[object] = []

    def initialize(path: Path) -> SimpleNamespace:
        assert path == s11
        model = object()
        models.append(model)
        return SimpleNamespace(model=model)

    monkeypatch.setattr(s13b_evaluate, "load_selected_s11_initialization", initialize)
    monkeypatch.setattr(
        s13b_evaluate,
        "_build_pilot_binding",
        lambda *args, **kwargs: SimpleNamespace(run_hash=pilot.control_run_hash),
    )
    monkeypatch.setattr(
        s13b_evaluate,
        "_build_s13b_binding",
        lambda *args, **kwargs: SimpleNamespace(run_hash="1" * 64),
    )
    loaded: list[Path] = []

    def load(checkpoint: Path, **kwargs: object) -> SimpleNamespace:
        loaded.append(checkpoint)
        return SimpleNamespace(
            step=800,
            run_hash=kwargs["binding"].run_hash,  # type: ignore[attr-defined]
            manifest_hash=(
                pilot.control_manifest_hash if checkpoint == control_cp else "5" * 64
            ),
        )

    monkeypatch.setattr(s13b_evaluate, "_load_checked_checkpoint", load)
    source = SimpleNamespace(
        development_opened=False,
        bindings=SimpleNamespace(content_hash=pilot.base_protocol.source_bindings_hash),
    )

    def open_source(path: Path, *, development_access_record: Path) -> object:
        assert path == s10
        assert development_access_record == peak_root / "development-access.json"
        assert loaded == [control_cp, peak_cp]
        return source

    monkeypatch.setattr(s13b_evaluate, "open_selective_s10_source", open_source)
    monkeypatch.setattr(
        s13b_evaluate,
        "materialize_dogbone_s13_training_data",
        lambda *args: object(),
    )
    monkeypatch.setattr(s13b_evaluate, "_verify_training_data", lambda *args: None)
    screened: list[object] = []

    def screen(**kwargs: object) -> SelectionMetrics:
        screened.append(kwargs["model"])
        assert kwargs["trial"] == pilot.base_protocol.search_space.trials[0]
        return _control_metrics() if len(screened) == 1 else _peak_metrics()

    monkeypatch.setattr(s13b_evaluate, "evaluate_dogbone_s13_train_screen", screen)
    monkeypatch.setattr(
        s13b_evaluate,
        "_trial_result",
        lambda config, manifest, metrics: _result(
            control=config.trial.optimizer_steps == 1500, metrics=metrics
        ),
    )

    comparison = s13b_evaluate.evaluate_s13b_pilot(
        s10_run_root=s10,
        s11_checkpoint_root=s11,
        control_checkpoint_path=control_cp,
        peak_checkpoint_path=peak_cp,
        peak_training_revision="a" * 40,
    )

    assert loaded == [control_cp, peak_cp]
    assert screened == models
    assert comparison.pilot_signal is True
    assert source.development_opened is False


def test_evaluator_refuses_bad_training_revision_before_source_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        s13b_evaluate,
        "open_selective_s10_source",
        lambda *args, **kwargs: pytest.fail("source opened before revision check"),
    )

    with pytest.raises(ValueError, match="peak_training_revision"):
        s13b_evaluate.evaluate_s13b_pilot(
            s10_run_root=tmp_path,
            s11_checkpoint_root=tmp_path,
            control_checkpoint_path=tmp_path,
            peak_checkpoint_path=tmp_path,
            peak_training_revision="not-a-commit",
        )


def test_cli_emits_canonical_non_gate_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pilot = build_dogbone_s13b_pilot_plan()
    comparison = compare_s13b_train_results(
        pilot,
        control=_result(control=True, metrics=_control_metrics()),
        peak=_result(control=False, metrics=_peak_metrics()),
    )
    received: list[dict[str, object]] = []

    def evaluate(**kwargs: object) -> object:
        received.append(kwargs)
        return comparison

    monkeypatch.setattr(s13b_evaluate, "evaluate_s13b_pilot", evaluate)
    code = s13b_evaluate.main(
        (
            "--s10-run-root",
            str(tmp_path / "s10"),
            "--s11-checkpoint-root",
            str(tmp_path / "s11"),
            "--control-checkpoint-path",
            str(tmp_path / "control"),
            "--peak-checkpoint-path",
            str(tmp_path / "peak"),
            "--peak-training-revision",
            "a" * 40,
        )
    )

    assert code == 0
    assert received[0]["peak_training_revision"] == "a" * 40
    output = capsys.readouterr().out
    assert '"pilot_signal":true' in output
    assert '"role":"TRAIN"' in output
    assert "CP-FREEZE" not in output
