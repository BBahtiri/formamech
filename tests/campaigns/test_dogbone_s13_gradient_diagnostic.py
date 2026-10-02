"""TRAIN-only matched gradient diagnostic for the two S13 step-800 arms."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from campaigns.dogbone_v1 import s13_gradient_diagnostic as diagnostic
from campaigns.dogbone_v1.s13b_protocol import build_dogbone_s13b_pilot_plan
from piginot.contracts import canonical_json_bytes
from piginot.doe import EvidenceRole
from piginot.objectives import ObjectiveTerm
from piginot.training import (
    AdaptiveWeightState,
    TrainingPopulation,
    TrainingSchedulerState,
)
from piginot.training.diagnostics import (
    ObjectiveGradientDiagnostic,
    TermGradientDiagnostic,
)


def _measurement(sample_ids: tuple[str, ...]) -> ObjectiveGradientDiagnostic:
    return ObjectiveGradientDiagnostic(
        role=EvidenceRole.TRAIN,
        dataset_snapshot_hash="0" * 64,
        public_manifest_hash="1" * 64,
        step=800,
        sample_ids=sample_ids,
        physics_weight_scale=1.0e-4,
        gradient_clip_norm=1.0,
        total_weighted_gradient_l2=0.5,
        clip_coefficient=1.0,
        gradient_space="raw_loss_gradient_pre_adam",
        post_checkpoint_snapshot=True,
        adaptive_state_last_update_step=800,
        terms=(
            TermGradientDiagnostic(
                term=ObjectiveTerm.DATA_U,
                unweighted_loss=1.0,
                effective_weight=1.0,
                weighted_loss=1.0,
                unweighted_gradient_l2=2.0,
                weighted_gradient_l2=2.0,
            ),
        ),
        pairs=(),
        adaptive_targets=(),
    )


def test_matched_diagnostic_validates_checkpoints_before_train_source_and_shares_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot = build_dogbone_s13b_pilot_plan()
    s10 = tmp_path / "s10"
    s11 = tmp_path / "s11"
    control = tmp_path / "control" / "checkpoint-step-0800"
    peak = tmp_path / "peak" / "checkpoint-step-0800"
    for path in (s10, s11, control, peak):
        path.mkdir(parents=True)
    access_record = peak.parent / "development-access.json"
    calls: list[str] = []

    monkeypatch.setattr(diagnostic, "build_dogbone_s13b_pilot_plan", lambda: pilot)
    monkeypatch.setattr(
        diagnostic,
        "_preflight_paths",
        lambda s10_path, s11_path, checkpoint, **kwargs: (
            s10_path,
            s11_path,
            checkpoint,
            checkpoint.parent / "development-access.json",
        ),
    )

    models = iter((torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)))
    monkeypatch.setattr(
        diagnostic,
        "load_selected_s11_initialization",
        lambda path: SimpleNamespace(model=next(models)),
    )
    monkeypatch.setattr(
        diagnostic,
        "_build_pilot_binding",
        lambda *args, **kwargs: SimpleNamespace(
            run_hash=pilot.control_run_hash,
            sampler_plan_hash="c" * 64,
        ),
    )
    monkeypatch.setattr(
        diagnostic,
        "_build_s13b_binding",
        lambda *args, **kwargs: SimpleNamespace(
            run_hash="9" * 64,
            sampler_plan_hash="d" * 64,
        ),
    )

    def load_arm(checkpoint: Path, **kwargs: object) -> object:
        calls.append(f"checkpoint:{checkpoint.parent.name}")
        binding = cast(Any, kwargs["binding"])
        run_hash = binding.run_hash
        manifest_hash = "a" * 64 if checkpoint == control else "b" * 64
        expected_manifest = (
            pilot.control_manifest_hash
            if checkpoint == control
            else (
                diagnostic._PEAK_MANIFEST_HASH  # noqa: SLF001
            )
        )
        assert kwargs["expected_manifest_hash"] == expected_manifest
        return SimpleNamespace(
            manifest=SimpleNamespace(
                step=800,
                run_hash=run_hash,
                manifest_hash=manifest_hash,
            ),
            adaptive_state=AdaptiveWeightState(
                update_count=7,
                last_update_step=800,
            ),
            scheduler_state=TrainingSchedulerState(
                scheduler_id="test-scheduler",
                step=800,
                learning_rate=1.0e-5 if checkpoint == control else 5.0e-6,
                best_metric=1.0,
                bad_observations=0,
                cooldown_remaining=0,
                observation_count=800,
            ),
        )

    monkeypatch.setattr(diagnostic, "_load_arm_checkpoint", load_arm)
    source = SimpleNamespace(
        development_opened=False,
        bindings=SimpleNamespace(content_hash=pilot.base_protocol.source_bindings_hash),
    )

    def open_source(path: Path, *, development_access_record: Path) -> object:
        assert calls == ["checkpoint:control", "checkpoint:peak"]
        assert development_access_record == access_record
        calls.append("source")
        return source

    monkeypatch.setattr(diagnostic, "open_selective_s10_source", open_source)
    population = TrainingPopulation(
        dataset_snapshot_hash=pilot.base_protocol.dataset_snapshot_hash,
        public_manifest_hash=pilot.base_protocol.public_manifest_hash,
        role=EvidenceRole.TRAIN,
        sample_ids=pilot.base_protocol.fit_case_ids,
    )
    data = SimpleNamespace(population=population, cases=(object(),))
    monkeypatch.setattr(
        diagnostic,
        "materialize_dogbone_s13_training_data",
        lambda *args: data,
    )
    monkeypatch.setattr(diagnostic, "_verify_training_data", lambda *args: None)

    objectives: list[tuple[bool, object]] = []

    def build_objective(cases: object, *, include_peak_loss: bool) -> object:
        objective = object()
        objectives.append((include_peak_loss, objective))
        return objective

    monkeypatch.setattr(diagnostic, "DogboneS13ObjectiveBuilder", build_objective)
    measurements: list[dict[str, object]] = []

    def measure(model: object, **kwargs: object) -> ObjectiveGradientDiagnostic:
        measurements.append({"model": model, **kwargs})
        return _measurement(kwargs["sample_ids"])  # type: ignore[arg-type]

    monkeypatch.setattr(diagnostic, "measure_objective_gradient_diagnostic", measure)

    report = diagnostic.run_s13_gradient_diagnostic(
        s10_run_root=s10,
        s11_checkpoint_root=s11,
        control_checkpoint_path=control,
        peak_checkpoint_path=peak,
        peak_training_revision="6" * 40,
    )

    base_plan = diagnostic.build_dogbone_s13_sampler_plan(pilot.base_protocol)
    expected_batches = tuple(
        tuple(population.sample_ids[index] for index in base_plan.batches[step - 1])
        for step in (753, 800, 801, 802)
    )
    assert calls == ["checkpoint:control", "checkpoint:peak", "source"]
    assert [item[0] for item in objectives] == [False, True]
    assert len(measurements) == 12
    assert tuple(item["sample_ids"] for item in measurements[:4]) == expected_batches
    assert tuple(item["sample_ids"] for item in measurements[4:8]) == expected_batches
    assert tuple(item["sample_ids"] for item in measurements[8:]) == expected_batches
    assert all(item["role"] is EvidenceRole.TRAIN for item in measurements)
    assert all(item["objective"] is objectives[0][1] for item in measurements[:4])
    assert all(item["objective"] is objectives[1][1] for item in measurements[4:])
    assert report.role is EvidenceRole.TRAIN
    assert report.non_gate is True
    assert tuple(batch.sample_ids for batch in report.panel) == expected_batches
    assert report.control.checkpoint_manifest_hash == "a" * 64
    assert report.peak.checkpoint_manifest_hash == "b" * 64
    assert report.control.scheduler_learning_rate == pytest.approx(1.0e-5)
    assert report.peak.scheduler_learning_rate == pytest.approx(5.0e-6)
    assert len(report.control.faithful_snapshots) == 4
    assert len(report.control.common_peak_snapshots) == 4
    assert report.peak.faithful_snapshots == report.peak.common_peak_snapshots
    assert source.development_opened is False


def test_matched_diagnostic_rejects_overlapping_run_roots_before_loading(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run" / "checkpoint-step-0800"
    checkpoint.mkdir(parents=True)

    with pytest.raises(ValueError, match="separate run roots"):
        diagnostic.run_s13_gradient_diagnostic(
            s10_run_root=tmp_path,
            s11_checkpoint_root=tmp_path,
            control_checkpoint_path=checkpoint,
            peak_checkpoint_path=checkpoint,
            peak_training_revision="6" * 40,
        )


def test_checkpoint_loader_verifies_binding_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint-step-0800"
    checkpoint.mkdir()
    model = torch.nn.Linear(1, 1)
    config = SimpleNamespace(
        trial=SimpleNamespace(learning_rate=1.0e-5),
        adam_betas=(0.9, 0.999),
        adam_eps=1.0e-8,
        adam_weight_decay=0.0,
    )
    binding = SimpleNamespace(run_hash="1" * 64)
    optimizer = object()
    monkeypatch.setattr(torch.optim, "Adam", lambda *args, **kwargs: optimizer)
    loaded = SimpleNamespace(
        manifest=SimpleNamespace(
            step=800,
            run_hash=binding.run_hash,
            manifest_hash="2" * 64,
        )
    )

    def load(path: Path, **kwargs: object) -> object:
        assert path == checkpoint
        assert kwargs == {"model": model, "optimizer": optimizer, "binding": binding}
        return loaded

    monkeypatch.setattr(diagnostic, "load_training_checkpoint", load)

    result = diagnostic._load_arm_checkpoint(  # noqa: SLF001
        checkpoint,
        model=model,
        config=config,
        binding=binding,
        expected_manifest_hash="2" * 64,
    )

    assert result is loaded

    with pytest.raises(ValueError, match="pinned arm"):
        diagnostic._load_arm_checkpoint(  # noqa: SLF001
            checkpoint,
            model=model,
            config=config,
            binding=binding,
            expected_manifest_hash="3" * 64,
        )


def test_matched_diagnostic_rejects_bad_revision_before_path_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="40-character"):
        diagnostic.run_s13_gradient_diagnostic(
            s10_run_root=tmp_path / "missing-s10",
            s11_checkpoint_root=tmp_path / "missing-s11",
            control_checkpoint_path=tmp_path / "missing-control",
            peak_checkpoint_path=tmp_path / "missing-peak",
            peak_training_revision="not-a-revision",
        )


def test_cli_emits_canonical_train_only_non_gate_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sample_ids = ("train-001",)
    panel = diagnostic.GradientPanelBatch(
        snapshot_id="fresh-step-801",
        plan_step=801,
        sample_ids=sample_ids,
    )
    snapshot = diagnostic.GradientSnapshotReport(
        panel=panel,
        diagnostic=_measurement(sample_ids),
    )
    arm = diagnostic.GradientArmReport(
        arm_id="control",
        run_hash="1" * 64,
        checkpoint_manifest_hash="2" * 64,
        checkpoint_sampler_plan_hash="b" * 64,
        objective_registry_hash="3" * 64,
        scheduler_learning_rate=1.0e-5,
        scheduler_state_hash="9" * 64,
        faithful_snapshots=(snapshot,),
        common_peak_snapshots=(snapshot,),
    )
    report = diagnostic.S13GradientDiagnosticReport(
        role=EvidenceRole.TRAIN,
        non_gate=True,
        pilot_id="pilot-v1",
        plan_hash="4" * 64,
        panel_sampler_plan_hash="5" * 64,
        panel=(panel,),
        control=arm,
        peak=diagnostic.GradientArmReport(
            arm_id="peak",
            run_hash="6" * 64,
            checkpoint_manifest_hash="7" * 64,
            checkpoint_sampler_plan_hash="c" * 64,
            objective_registry_hash="8" * 64,
            scheduler_learning_rate=5.0e-6,
            scheduler_state_hash="a" * 64,
            faithful_snapshots=(snapshot,),
            common_peak_snapshots=(snapshot,),
        ),
    )
    monkeypatch.setattr(
        diagnostic, "run_s13_gradient_diagnostic", lambda **kwargs: report
    )

    result = diagnostic.main(
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
            "6" * 40,
        )
    )

    assert result == 0
    output = capsys.readouterr().out
    assert '"role":"TRAIN"' in output
    assert '"non_gate":true' in output
    assert '"snapshot_id":"fresh-step-801"' in output


def test_cli_writes_canonical_report_without_overwriting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sample_ids = ("train-001",)
    panel = diagnostic.GradientPanelBatch(
        snapshot_id="fresh-step-801",
        plan_step=801,
        sample_ids=sample_ids,
    )
    snapshot = diagnostic.GradientSnapshotReport(
        panel=panel,
        diagnostic=_measurement(sample_ids),
    )
    arm = diagnostic.GradientArmReport(
        arm_id="control",
        run_hash="1" * 64,
        checkpoint_manifest_hash="2" * 64,
        checkpoint_sampler_plan_hash="3" * 64,
        objective_registry_hash="4" * 64,
        scheduler_learning_rate=1.0e-5,
        scheduler_state_hash="5" * 64,
        faithful_snapshots=(snapshot,),
        common_peak_snapshots=(snapshot,),
    )
    report = diagnostic.S13GradientDiagnosticReport(
        role=EvidenceRole.TRAIN,
        non_gate=True,
        pilot_id="pilot-v1",
        plan_hash="6" * 64,
        panel_sampler_plan_hash="7" * 64,
        panel=(panel,),
        control=arm,
        peak=arm,
    )
    monkeypatch.setattr(
        diagnostic, "run_s13_gradient_diagnostic", lambda **kwargs: report
    )
    output_parent = tmp_path / "output"
    output_parent.mkdir()
    output_path = output_parent / "report.json"
    arguments = (
        "--s10-run-root",
        str(tmp_path / "s10"),
        "--s11-checkpoint-root",
        str(tmp_path / "s11"),
        "--control-checkpoint-path",
        str(tmp_path / "control-run" / "checkpoint-step-0800"),
        "--peak-checkpoint-path",
        str(tmp_path / "peak-run" / "checkpoint-step-0800"),
        "--peak-training-revision",
        "6" * 40,
        "--output-path",
        str(output_path),
    )

    assert diagnostic.main(arguments) == 0
    assert output_path.read_bytes() == canonical_json_bytes(report) + b"\n"
    assert f"report_path={output_path}" in capsys.readouterr().out

    with pytest.raises(FileExistsError):
        diagnostic.main(arguments)
