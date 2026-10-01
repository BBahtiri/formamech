from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from campaigns.dogbone_v1 import s13_pilot
from campaigns.dogbone_v1.s13_protocol import build_dogbone_s13_protocol
from piginot.contracts import hash_spec
from piginot.doe import EvidenceRole
from piginot.training import TrainingPopulation


def _roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    s10 = tmp_path / "s10"
    s11 = tmp_path / "s11"
    output_parent = tmp_path / ".piginot-runs"
    for path in (s10, s11, output_parent):
        path.mkdir()
    return s10, s11, output_parent / "pilot"


def test_pilot_rejects_existing_output_before_opening_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s10, s11, output = _roots(tmp_path)
    output.mkdir()
    opened = False

    def unexpected_open(*args: object, **kwargs: object) -> None:
        nonlocal opened
        opened = True

    monkeypatch.setattr(s13_pilot, "open_selective_s10_source", unexpected_open)

    with pytest.raises(FileExistsError, match="output_root"):
        s13_pilot.run_s13_fit_pilot(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            output_root=output,
        )

    assert not opened


def test_pilot_rejects_output_inside_s10_source(tmp_path: Path) -> None:
    s10, s11, _ = _roots(tmp_path)

    with pytest.raises(ValueError, match="outside source roots"):
        s13_pilot.run_s13_fit_pilot(
            s10_run_root=s10,
            s11_checkpoint_root=s11,
            output_root=s10 / "diagnostic",
        )


def test_pilot_preflight_rejects_relative_or_existing_paths(tmp_path: Path) -> None:
    s10, s11, output = _roots(tmp_path)

    with pytest.raises(ValueError, match="absolute"):
        s13_pilot._preflight_paths(Path("relative"), s11, output)
    with pytest.raises(ValueError, match="absolute"):
        s13_pilot._preflight_paths(s10, s11, Path("relative"))
    with pytest.raises(ValueError, match="outside source roots"):
        s13_pilot._preflight_paths(s10, s11, s11 / "diagnostic")


def test_pilot_git_provenance_fingerprints_status_and_tracked_diff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        ("rev-parse", "HEAD"): b"a" * 40 + b"\n",
        ("status", "--porcelain=v1", "--untracked-files=all"): b" M objective.py\n",
        ("diff", "HEAD", "--binary"): b"diff --git a/objective.py b/objective.py\n",
    }
    monkeypatch.setattr(s13_pilot, "_git", lambda *args: values[args])

    provenance = s13_pilot._git_provenance()

    assert provenance.head == "a" * 40
    assert provenance.dirty_worktree
    assert (
        provenance.status_and_tracked_diff_blake2b256
        == hashlib.blake2b(
            values[("status", "--porcelain=v1", "--untracked-files=all")]
            + b"\0"
            + values[("diff", "HEAD", "--binary")],
            digest_size=32,
        ).hexdigest()
    )


def test_pilot_rejects_initial_weight_drift() -> None:
    protocol = build_dogbone_s13_protocol()
    model = torch.nn.Linear(1, 1, dtype=torch.float64)
    plan = s13_pilot.build_dogbone_s13_sampler_plan(protocol)
    config = s13_pilot.build_dogbone_s13_trainer_configs(protocol)[0]

    with pytest.raises(ValueError, match="initial trainer weights"):
        s13_pilot._build_pilot_binding(
            protocol,
            model=model,
            config=config,
            plan=plan,
            implementation_revision="a" * 40,
        )


def test_pilot_binding_uses_exact_registered_trial_and_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = build_dogbone_s13_protocol()
    model = torch.nn.Linear(1, 1, dtype=torch.float64)
    plan = s13_pilot.build_dogbone_s13_sampler_plan(protocol)
    config = s13_pilot.build_dogbone_s13_trainer_configs(protocol)[0]
    monkeypatch.setattr(
        s13_pilot,
        "model_state_hash",
        lambda model: protocol.initial_trainer_weights_hash,
    )

    binding = s13_pilot._build_pilot_binding(
        protocol,
        model=model,
        config=config,
        plan=plan,
        implementation_revision="a" * 40,
    )

    assert binding.initial_weights_hash == protocol.initial_trainer_weights_hash
    assert binding.trainer_config_hash == config.config_hash
    assert binding.sampler_plan_hash == plan.plan_hash
    assert binding.objective_registry_hash == hash_spec(
        protocol.objective_registry, "recipe"
    )
    assert binding.execution_profile.blas_threads == 1


def test_pilot_training_data_identity_checks() -> None:
    protocol = build_dogbone_s13_protocol()
    population = TrainingPopulation(
        dataset_snapshot_hash=protocol.dataset_snapshot_hash,
        public_manifest_hash=protocol.public_manifest_hash,
        role=EvidenceRole.TRAIN,
        sample_ids=protocol.fit_case_ids,
    )
    data = object.__new__(s13_pilot.DogboneS13TrainingData)
    object.__setattr__(data, "population", population)
    object.__setattr__(
        data,
        "normalization",
        SimpleNamespace(content_hash=protocol.normalization_hash),
    )
    object.__setattr__(data, "source_bindings_hash", protocol.source_bindings_hash)
    object.__setattr__(data, "selected_candidate_id", protocol.selected_candidate_id)
    object.__setattr__(
        data,
        "selected_checkpoint_file_hash",
        protocol.initial_checkpoint_file_hash,
    )

    s13_pilot._verify_training_data(data, protocol, protocol.source_bindings_hash)
    with pytest.raises(ValueError, match="source bindings"):
        s13_pilot._verify_training_data(data, protocol, "0" * 64)


def test_pilot_only_trains_first_registered_trial_for_twenty_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s10, s11, output = _roots(tmp_path)
    protocol = build_dogbone_s13_protocol()
    model = torch.nn.Linear(1, 1, dtype=torch.float64)
    initialization = SimpleNamespace(model=model)
    source = SimpleNamespace(
        bindings=SimpleNamespace(content_hash=protocol.source_bindings_hash),
        development_opened=False,
    )
    objective = object()
    data = SimpleNamespace(
        population=object(),
        objective_builder=lambda: objective,
    )
    binding = SimpleNamespace(run_hash="c" * 64)
    result = SimpleNamespace(step=20, model_state_hash="d" * 64)
    calls: dict[str, Any] = {}

    monkeypatch.setattr(s13_pilot, "build_dogbone_s13_protocol", lambda: protocol)
    monkeypatch.setattr(
        s13_pilot, "load_selected_s11_initialization", lambda path: initialization
    )

    def open_source(path: Path, *, development_access_record: Path) -> object:
        calls["source_root"] = path
        calls["access_record"] = development_access_record
        return source

    monkeypatch.setattr(s13_pilot, "open_selective_s10_source", open_source)
    monkeypatch.setattr(
        s13_pilot,
        "materialize_dogbone_s13_training_data",
        lambda actual_source, actual_initialization: data,
    )
    monkeypatch.setattr(s13_pilot, "_verify_training_data", lambda *args: None)
    monkeypatch.setattr(
        s13_pilot,
        "_git_provenance",
        lambda: s13_pilot.PilotGitProvenance("a" * 40, True, "b" * 64),
    )
    monkeypatch.setattr(s13_pilot, "_build_pilot_binding", lambda *args, **kw: binding)

    def fake_train(checkpoint_path: Path, **kwargs: object) -> object:
        calls["checkpoint_path"] = checkpoint_path
        calls.update(kwargs)
        return result

    monkeypatch.setattr(s13_pilot, "train_to_checkpoint", fake_train)
    progress: list[str] = []

    returned = s13_pilot.run_s13_fit_pilot(
        s10_run_root=s10,
        s11_checkpoint_root=s11,
        output_root=output,
        progress=progress.append,
    )

    assert returned is result
    assert calls["source_root"] == s10
    assert calls["access_record"] == output / "development-access.json"
    assert calls["checkpoint_path"] == output / "checkpoint"
    assert calls["config"] == s13_pilot.build_dogbone_s13_trainer_configs(protocol)[0]
    assert calls["target_step"] == 20
    assert calls["objective"] is objective
    assert calls["binding"] is binding
    assert any("non_gate=true" in line for line in progress)
    assert not (output / "development-access.json").exists()


def test_pilot_main_reports_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    s10, s11, output = _roots(tmp_path)
    checkpoint = output / "checkpoint"
    monkeypatch.setattr(
        s13_pilot,
        "run_s13_fit_pilot",
        lambda **kwargs: SimpleNamespace(checkpoint=SimpleNamespace(path=checkpoint)),
    )

    exit_code = s13_pilot.main(
        (
            "--s10-run-root",
            str(s10),
            "--s11-checkpoint-root",
            str(s11),
            "--output-root",
            str(output),
        )
    )

    assert exit_code == 0
    assert f"checkpoint={checkpoint}" in capsys.readouterr().out
