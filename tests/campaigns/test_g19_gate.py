"""G19 gate logic: NOT_RUN without evidence, refusals on bad evidence."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from campaigns import g19_gate as gate


def _runner(outputs: dict[tuple[str, ...], tuple[int, str]]) -> gate.Runner:
    def run(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        key = tuple(command[3:]) if command[:1] == ["git"] else tuple(command)
        code, text = outputs.get(key, (0, ""))
        return subprocess.CompletedProcess(list(command), code, text, "")

    return run


def test_isolation_passes_only_for_a_lone_clone(tmp_path: Path) -> None:
    (tmp_path / ".git" / "objects" / "info").mkdir(parents=True)
    lone = _runner({("worktree", "list", "--porcelain"): (0, "worktree /x\nHEAD a\n")})
    assert gate.check_isolation(tmp_path, lone)["status"] == "PASS"
    remote = _runner(
        {
            ("remote",): (0, "origin\n"),
            ("worktree", "list", "--porcelain"): (0, "worktree /x\n"),
        }
    )
    assert gate.check_isolation(tmp_path, remote)["status"] == "FAIL"
    (tmp_path / ".git" / "objects" / "info" / "alternates").write_text("/shared\n")
    assert gate.check_isolation(tmp_path, lone)["status"] == "FAIL"


def test_identity_refuses_foreign_authors(tmp_path: Path) -> None:
    log = ("log", "--format=%an", f"{gate.BASE}..HEAD")
    good = _runner(
        {log: (0, "Betim Bahtiri\n"), ("config", "user.name"): (0, "Betim Bahtiri")}
    )
    assert gate.check_identity(tmp_path, good)["status"] == "PASS"
    bad = _runner(
        {
            log: (0, "S9 test\nBetim Bahtiri\n"),
            ("config", "user.name"): (0, "Betim Bahtiri"),
        }
    )
    assert gate.check_identity(tmp_path, bad)["status"] == "FAIL"


def test_missing_evidence_is_not_run_never_pass(tmp_path: Path) -> None:
    assert gate.check_suite(tmp_path)["status"] == "NOT_RUN"
    assert gate.check_coverage(tmp_path)["status"] == "NOT_RUN"
    assert gate.check_rescore(tmp_path, None)["status"] == "NOT_RUN"
    probe = tmp_path / gate.PROBE_PATH
    probe.parent.mkdir(parents=True)
    probe.write_text(
        json.dumps({"target_step": 400, "registered_model_state_hash": "a" * 64})
    )
    assert gate.check_resume(tmp_path, None, None)["status"] == "NOT_RUN"


def _checkpoint(directory: Path, step: int, weights: bytes) -> Path:
    directory.mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(weights)
    (directory / "checkpoint.json").write_bytes(f"{step}".encode())
    return directory


def test_resume_needs_both_runs_and_equal_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from piginot.training import checkpoint as module

    class Stub:
        def __init__(self, payload: bytes) -> None:
            self.step = int(payload.decode())
            self.model_state_hash = "m" * 64
            self.manifest_hash = "h" * 64
            self.run_hash = "r" * 64

    monkeypatch.setattr(
        module.TrainingCheckpointManifest,
        "from_bytes",
        classmethod(lambda cls, b: Stub(b)),
    )
    probe = tmp_path / gate.PROBE_PATH
    probe.parent.mkdir(parents=True)
    probe.write_text(
        json.dumps({"target_step": 400, "registered_model_state_hash": "a" * 64})
    )
    replay = _checkpoint(tmp_path / "replay" / "checkpoint-step-0400", 400, b"w")
    straight = _checkpoint(tmp_path / "straight" / "checkpoint-step-0400", 400, b"w")
    assert gate.check_resume(tmp_path, replay, None)["status"] == "NOT_RUN"
    # not interrupted: no step-200 checkpoint beside the replay
    assert gate.check_resume(tmp_path, replay, straight)["status"] == "FAIL"
    (tmp_path / "replay" / "checkpoint-step-0200").mkdir()
    result = gate.check_resume(tmp_path, replay, straight)
    assert result["status"] == "PASS"
    comparison = result["measured"]["registered_profile_comparison"]
    assert comparison["matches_registered"] is False
    assert comparison["expected_to_match"] is False
    (straight / "model.safetensors").write_bytes(b"x")
    assert gate.check_resume(tmp_path, replay, straight)["status"] == "FAIL"


def test_rescore_ignores_only_the_evaluator_revision(tmp_path: Path) -> None:
    recorded = {"evaluator_revision": "old", "run_hash": "r", "metrics": [1, 2]}
    screen = tmp_path / gate.SCREEN_PATH
    screen.parent.mkdir(parents=True)
    screen.write_text(json.dumps(recorded))
    evaluation = tmp_path / "rescore.json"
    same = {**recorded, "evaluator_revision": "new"}
    evaluation.write_text(json.dumps(same) + "\nselected\n", encoding="utf-16")
    assert gate.check_rescore(tmp_path, evaluation)["status"] == "PASS"
    evaluation.write_text(json.dumps({**same, "metrics": [1, 3]}) + "\n")
    result = gate.check_rescore(tmp_path, evaluation)
    assert result["status"] == "FAIL"
    assert result["measured"]["differing_fields"] == ["metrics"]


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16"])
def test_text_from_any_shell_is_read(tmp_path: Path, encoding: str) -> None:
    path = tmp_path / "log.txt"
    path.write_text("3075 passed, 1 skipped in 9.0s\n", encoding=encoding)
    assert gate.read_text_any(path).strip() == "3075 passed, 1 skipped in 9.0s"


def test_bundle_check_needs_a_golden_for_the_profile(tmp_path: Path) -> None:
    served = {
        "bundles": {
            "dogbone-v1-s13c-cp-freeze": {
                "manifest_sha256": "009db9d379d6f94ccda5395762e22c9c810c466a686626d500f1575d45b1bb4c",
                "claim_level": "HOLDOUT_VERIFIED",
            },
            "notched-v1-cp-freeze": {
                "manifest_sha256": "3749f4be4761a55582dd07f90e3cc5950b8231357162366d79dc47463dddcc99",
                "claim_level": "DEVELOPMENT_VERIFIED",
            },
        },
        "outputs": {
            "dogbone-v1-s13c-cp-freeze/dogbone-in-domain.json": {
                "verdict": "IN_DOMAIN",
                "sha256": "1",
                "repeat_identical": True,
            },
            "notched-v1-cp-freeze/notched-in-domain.json": {
                "verdict": "IN_DOMAIN",
                "sha256": "2",
                "repeat_identical": True,
            },
            "notched-v1-cp-freeze/notched-out-of-envelope.json": {
                "verdict": "REFUSED",
                "sha256": "3",
                "repeat_identical": True,
            },
        },
    }
    profile = {
        "python": "3.11",
        "system": "Linux",
        "machine": "x",
        "torch": "t",
        "cpu_capability": "c",
    }
    assert gate.check_bundles(tmp_path, served, profile)["status"] == "NOT_RUN"
    (tmp_path / gate.GOLDEN_PATH).parent.mkdir(parents=True)
    (tmp_path / gate.GOLDEN_PATH).write_text(
        json.dumps(
            {
                "profiles": {
                    gate.profile_key(profile): {
                        name: output["sha256"]
                        for name, output in served["outputs"].items()
                    }
                }
            }
        )
    )
    assert gate.check_bundles(tmp_path, served, profile)["status"] == "PASS"
    served["outputs"]["notched-v1-cp-freeze/notched-in-domain.json"]["sha256"] = "9"
    assert gate.check_bundles(tmp_path, served, profile)["status"] == "FAIL"
