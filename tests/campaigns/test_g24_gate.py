"""G24 gate logic and the S24 replay comparison on synthetic records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from campaigns import g24_gate as gate
from campaigns import s24_replay as replay
from tests.git_helpers import run_git

# ------------------------------------------------------------------- replay


def _checkpoint(
    root: Path,
    *,
    run_hash: str = "1" * 64,
    weight: float = 1.0,
    extra_metadata: str = "x",
    model_state_hash: str = "m" * 64,
) -> Path:
    root.mkdir(parents=True)
    tensors = {"w": torch.full((2, 2), weight), "b": torch.zeros(3)}
    save_file(tensors, str(root / "model.safetensors"), {"run_hash": run_hash})
    save_file(
        {**tensors, "rng": torch.arange(4, dtype=torch.uint8)},
        str(root / "state.safetensors"),
        {"run_hash": run_hash, "step": "400", "extra": extra_metadata},
    )
    (root / "optimizer.json").write_text('{"lr": 0.0003}')
    (root / "adaptive.json").write_text('{"weights": []}')
    manifest = {
        "run_hash": run_hash,
        "step": 400,
        "model_state_hash": model_state_hash,
        "model_export_hash": run_hash[:8],
        "tensor_payload_hash": run_hash[:8],
    }
    (root / "checkpoint.json").write_text(json.dumps(manifest))
    return root


def test_a_replay_that_differs_only_in_its_run_hash_passes(tmp_path: Path) -> None:
    registered = _checkpoint(tmp_path / "a")
    replayed = _checkpoint(tmp_path / "b", run_hash="2" * 64)
    record = replay.compare(registered, replayed)
    assert record["passed"], record
    assert record["manifest_differences"] == [
        "model_export_hash",
        "run_hash",
        "tensor_payload_hash",
    ]
    assert record["tensor_files"]["model.safetensors"]["differing_metadata"] == [
        "run_hash"
    ]


@pytest.mark.parametrize(
    "change",
    [{"weight": 2.0}, {"extra_metadata": "y"}, {"model_state_hash": "n" * 64}],
    ids=["tensor", "metadata", "manifest"],
)
def test_any_other_difference_fails(tmp_path: Path, change: dict[str, Any]) -> None:
    registered = _checkpoint(tmp_path / "a")
    replayed = _checkpoint(tmp_path / "b", run_hash="2" * 64, **change)
    assert not replay.compare(registered, replayed)["passed"]


def test_optimizer_files_must_be_byte_identical(tmp_path: Path) -> None:
    registered = _checkpoint(tmp_path / "a")
    replayed = _checkpoint(tmp_path / "b")
    (replayed / "optimizer.json").write_text('{"lr": 0.0003 }')
    record = replay.compare(registered, replayed)
    assert not record["passed"] and record["exact_files"]["optimizer.json"] is False


# --------------------------------------------------------------------- gate


def _repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git(repo, "init", "-q")
    (repo / "a.txt").write_text("a")
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-q", "-m", "a")
    return repo


def test_clean_ignores_only_the_gate_record_files(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    assert gate.clean(repo)
    suite = repo / gate.SUITE_PATH
    suite.parent.mkdir(parents=True)
    suite.write_text("{}")
    assert gate.clean(repo)
    (repo / "a.txt").write_text("b")
    assert not gate.clean(repo)


def test_module_coverage_counts_lines_and_branches() -> None:
    report = {
        "files": {
            "src/piginot/lifting/adf.py": {
                "summary": {
                    "num_statements": 8,
                    "num_branches": 2,
                    "covered_lines": 8,
                    "covered_branches": 1,
                }
            },
            "src\\piginot\\problem\\fem.py": {
                "summary": {"num_statements": 4, "covered_lines": 2}
            },
        }
    }
    coverage = gate._module_coverage(report)
    assert coverage["src/piginot/lifting/"] == 90.0
    assert coverage["src/piginot/problem/"] == 50.0
    assert coverage["src/piginot/contracts/problem_spec.py"] is None


def test_missing_run_records_are_not_run(tmp_path: Path) -> None:
    assert gate.check_adf_viability(tmp_path, "0" * 64)["status"] == "NOT_RUN"
    committed = {"dogbone-v1": {"development": {}}}
    fresh = {"dogbone-v1": {"development": {}, "envelope": {}}}
    record = gate.check_regression(tmp_path, "dogbone-v1", committed, fresh)
    assert record["status"] == "NOT_RUN" and record["id"] == "G24.regression-dogbone"
    assert gate.check_portability(tmp_path)["status"] == "NOT_RUN"


def test_the_protocol_must_be_committed_first(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    assert gate.check_protocol_first(repo)["status"] == "NOT_RUN"


def test_lifting_verified_needs_committed_and_recomputed_passes() -> None:
    def family(passed: bool) -> dict[str, Any]:
        return {
            "lifting": {
                kind: {"passed": passed, "parameter_vectors": 89}
                for kind in ("quarter", "adf")
            }
        }

    good = {campaign: family(True) for campaign in ("dogbone-v1", "notched-v1")}
    bad = {**good, "notched-v1": family(False)}
    assert gate.check_lifting_verified(good, good)["status"] == "PASS"
    assert gate.check_lifting_verified(good, bad)["status"] == "FAIL"
