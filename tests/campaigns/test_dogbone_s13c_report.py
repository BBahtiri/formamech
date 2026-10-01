from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from campaigns.dogbone_v1.s13c_report import (
    _historical_metrics,
    _training_summary,
    compare_replay,
)
from campaigns.dogbone_v1.s13c_trial import TRAINING_LOG_NAME
from piginot.contracts import strict_json_loads


def _engine_helpers() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "training" / "test_energy_engine.py"
    spec = importlib.util.spec_from_file_location("_s13c_engine_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_log(root: Path, steps: list[int]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "barrier": 0.0,
                "energy": 0.01 + 0.001 * step,
                "gradient_norm": 0.1,
                "inadmissible_points": 0,
                "learning_rate": 3e-4,
                "minimum_determinant": 0.9,
                "step": step,
                "total": 0.01,
            }
        )
        for step in steps
    ]
    (root / TRAINING_LOG_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_training_summary_keeps_replayed_records_once(tmp_path: Path) -> None:
    _write_log(tmp_path, [*range(1, 151), *range(101, 201)])
    summary = _training_summary(tmp_path)

    assert summary["logged_steps"] == 200
    assert summary["replayed_log_records"] == 50
    assert summary["maximum_barrier"] == 0.0
    assert summary["mean_energy_last_100_steps"] > summary["mean_energy_first_100_steps"]


def test_training_summary_rejects_missing_steps(tmp_path: Path) -> None:
    _write_log(tmp_path, [1, 2, 4])
    with pytest.raises(ValueError, match="every optimizer step"):
        _training_summary(tmp_path)
    (tmp_path / TRAINING_LOG_NAME).write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        _training_summary(tmp_path)


def test_historical_metrics_come_from_the_tracked_failure_evidence() -> None:
    history = _historical_metrics()

    assert history["s13_trial_0"]["median_absolute_peak_error"] == pytest.approx(
        0.1225497615851001
    )
    assert history["s13b_peak_arm"]["invalid_collocation_points"] == 0 or (
        history["s13b_peak_arm"]["invalid_collocation_points"] == 1
    )
    assert history["guardrail_limits"]["required_invalid_collocation_points"] == 0


def test_replay_probe_reports_determinism_and_resume(tmp_path: Path) -> None:
    helpers = _engine_helpers()
    registered = tmp_path / "registered"
    probe = tmp_path / "probe"
    registered.mkdir()
    probe.mkdir()
    helpers._train(registered / "checkpoint-step-0006")
    helpers._train(probe / "checkpoint-step-0003", target_step=3)
    helpers._train(
        probe / "checkpoint-step-0006",
        resume_from=probe / "checkpoint-step-0003",
    )

    report = compare_replay(
        registered_run_root=registered,
        probe_run_root=probe,
        target_step=6,
        output=tmp_path / "probe.json",
    )

    assert report["g13_determinism"] == "PASS"
    assert report["g13_resume"] == "PASS"
    assert all(report["byte_identical_files"].values())
    assert strict_json_loads((tmp_path / "probe.json").read_bytes())["target_step"] == 6

    tampered = probe / "checkpoint-step-0006" / "model.safetensors"
    payload = bytearray(tampered.read_bytes())
    payload[-1] ^= 1
    tampered.chmod(0o644)
    tampered.write_bytes(bytes(payload))
    failed = compare_replay(
        registered_run_root=registered,
        probe_run_root=probe,
        target_step=6,
        output=tmp_path / "probe-failed.json",
    )
    assert failed["g13_resume"] == "FAIL"


def test_replay_probe_requires_an_interrupted_replay(tmp_path: Path) -> None:
    helpers = _engine_helpers()
    registered = tmp_path / "registered"
    probe = tmp_path / "probe"
    registered.mkdir()
    probe.mkdir()
    helpers._train(registered / "checkpoint-step-0006")
    helpers._train(probe / "checkpoint-step-0006")
    with pytest.raises(RuntimeError, match="interrupted"):
        compare_replay(
            registered_run_root=registered,
            probe_run_root=probe,
            target_step=6,
            output=tmp_path / "probe.json",
        )
