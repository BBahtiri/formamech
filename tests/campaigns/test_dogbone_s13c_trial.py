from __future__ import annotations

from pathlib import Path

import pytest
import torch

from campaigns.dogbone_v1 import s13c_trial
from campaigns.dogbone_v1.s13c_evaluate import load_run_binding
from campaigns.dogbone_v1.s13c_protocol import (
    build_dogbone_s13c_protocol,
    build_dogbone_s13c_sampler_plan,
    build_dogbone_s13c_trainer_config,
    build_initial_operator,
)
from campaigns.dogbone_v1.s13c_trial import (
    RUN_BINDING_NAME,
    TRAINING_LOG_NAME,
    build_parser,
    build_s13c_binding,
    execution_profile,
    latest_checkpoint,
    segment_targets,
)
from piginot.contracts import canonical_json_bytes

_REVISION = "0123456789abcdef0123456789abcdef01234567"


def _binding():  # type: ignore[no-untyped-def]
    protocol = build_dogbone_s13c_protocol()
    return build_s13c_binding(
        protocol,
        model=build_initial_operator(),
        config=build_dogbone_s13c_trainer_config(protocol),
        plan=build_dogbone_s13c_sampler_plan(protocol),
        implementation_revision=_REVISION,
    )


def test_segment_targets_cover_the_budget_and_honour_stop_after() -> None:
    assert segment_targets(0, 2400, 400) == (400, 800, 1200, 1600, 2000, 2400)
    assert segment_targets(1000, 2400, 400) == (1400, 1800, 2200, 2400)
    assert segment_targets(0, 2400, 400, stop_after=1000) == (400, 800, 1000)
    assert segment_targets(2400, 2400, 400) == ()
    with pytest.raises(ValueError):
        segment_targets(0, 2400, 0)
    with pytest.raises(ValueError):
        segment_targets(0, 2400, 400, stop_after=2401)
    with pytest.raises(ValueError):
        segment_targets(-1, 2400, 400)


def test_binding_is_derived_from_the_protocol() -> None:
    protocol = build_dogbone_s13c_protocol()
    binding = _binding()
    profile = execution_profile(protocol)

    assert binding.architecture_id == protocol.architecture_id
    assert binding.objective_registry_hash == protocol.objective_hash
    assert binding.initial_weights_hash == protocol.initial_weights_hash
    assert binding.seed == protocol.training_seed
    assert (profile.dtype, profile.blas_threads, profile.device) == (
        "float32",
        2,
        "cpu",
    )
    model = build_initial_operator()
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    with pytest.raises(ValueError, match="initial operator"):
        build_s13c_binding(
            protocol,
            model=model,
            config=build_dogbone_s13c_trainer_config(protocol),
            plan=build_dogbone_s13c_sampler_plan(protocol),
            implementation_revision=_REVISION,
        )


def test_run_binding_file_round_trips_and_pins_the_run_root(tmp_path: Path) -> None:
    binding = _binding()
    s13c_trial._bind_run_root(tmp_path, binding)
    s13c_trial._bind_run_root(tmp_path, binding)

    assert load_run_binding(tmp_path / RUN_BINDING_NAME) == binding
    other = build_s13c_binding(
        build_dogbone_s13c_protocol(),
        model=build_initial_operator(),
        config=build_dogbone_s13c_trainer_config(build_dogbone_s13c_protocol()),
        plan=build_dogbone_s13c_sampler_plan(build_dogbone_s13c_protocol()),
        implementation_revision="f" * 40,
    )
    with pytest.raises(ValueError, match="different S13c run binding"):
        s13c_trial._bind_run_root(tmp_path, other)
    (tmp_path / "pretty.json").write_bytes(
        canonical_json_bytes(binding).replace(b",", b", ", 1)
    )
    with pytest.raises(ValueError, match="canonical"):
        load_run_binding(tmp_path / "pretty.json")


def test_latest_checkpoint_accepts_only_registered_entries(tmp_path: Path) -> None:
    (tmp_path / RUN_BINDING_NAME).write_text("{}")
    (tmp_path / TRAINING_LOG_NAME).write_text("")
    assert latest_checkpoint(tmp_path, expected_run_hash="0" * 64) == (None, 0)

    (tmp_path / ".checkpoint-step-0400.staging-abc").mkdir()
    with pytest.raises(ValueError, match="staging"):
        latest_checkpoint(tmp_path, expected_run_hash="0" * 64)
    (tmp_path / ".checkpoint-step-0400.staging-abc").rmdir()
    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(ValueError, match="unexpected entry"):
        latest_checkpoint(tmp_path, expected_run_hash="0" * 64)


def test_run_root_must_stay_outside_sources(tmp_path: Path) -> None:
    s10 = tmp_path / "s10"
    s10.mkdir()
    with pytest.raises(ValueError, match="absolute"):
        s13c_trial._prepare_run_root(s10, Path("relative"))
    with pytest.raises(ValueError, match="outside"):
        s13c_trial._prepare_run_root(s10, s10 / "run")
    repository = Path(s13c_trial.__file__).resolve().parents[2]
    with pytest.raises(ValueError, match="outside"):
        s13c_trial._prepare_run_root(s10, repository / "run")
    created = s13c_trial._prepare_run_root(s10, tmp_path / "run")
    assert created.is_dir()
    assert s13c_trial._prepare_run_root(s10, tmp_path / "run") == created


def test_registered_training_requires_a_clean_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _Dirty:
        head = _REVISION
        dirty_worktree = True
        status_and_tracked_diff_blake2b256 = "0" * 64

    monkeypatch.setattr(s13c_trial, "_git_provenance", lambda: _Dirty())
    with pytest.raises(RuntimeError, match="clean revision"):
        s13c_trial.run_registered_s13c_trial(
            s10_run_root=tmp_path, run_root=tmp_path / "run"
        )


def test_parser_requires_both_roots() -> None:
    options = build_parser().parse_args(["--s10-run-root", "/a", "--run-root", "/b"])
    assert (options.segment_steps, options.stop_after) == (400, None)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--run-root", "/b"])
