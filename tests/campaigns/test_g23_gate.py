"""G23 gate logic on synthetic records: width, coverage recount, records."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from campaigns import g23_gate as gate
from campaigns import s23_calibration as s23
from campaigns import s23_reveal as reveal
from piginot.bounds import (
    IN_CALIBRATION,
    OUT_OF_CALIBRATION,
    CalibrationRecord,
    MahalanobisGate,
    case_ids_sha256,
)
from piginot.contracts import canonical_json_bytes

QUANTITIES = ["p_relative_l2", "absolute_peak_error", "u_relative_l2"]


def _protocol() -> dict[str, Any]:
    return {
        "protocol_id": s23.PROTOCOL_ID,
        "alpha": 0.1,
        "quantities": QUANTITIES,
        "families": {
            "dogbone-v1": {
                "bundle": "dogbone-bundle",
                "calibration_case_ids": [],
                "h1_case_ids": [f"h1-{i:02d}" for i in range(64)],
            }
        },
        "coverage_test": {
            "gated": [
                f"dogbone-v1:{kind}:{q}"
                for kind in ("budget_total", "reference_terms")
                for q in QUANTITIES[:2]
            ],
            "per_check_level": 0.05 / 4,
            "rule": "beta-binomial",
        },
        "thresholds": {
            "dogbone_budget_total_at_most": {
                "p_relative_l2": 0.05,
                "absolute_peak_error": 0.05,
                "u_relative_l2": 0.02,
            }
        },
    }


def _calibration(peak: float = 0.04) -> CalibrationRecord:
    return CalibrationRecord(
        bundle="dogbone-bundle",
        bundle_manifest_sha256="d" * 64,
        campaign_id="dogbone-v1",
        protocol_sha256="e" * 64,
        alpha=0.1,
        n=64,
        k=59,
        calibration_case_ids_sha256=case_ids_sha256(["c"]),
        bounds={
            "budget_total": {
                "p_relative_l2": 0.03,
                "absolute_peak_error": peak,
                "u_relative_l2": 0.002,
            },
            "reference_terms": {q: 0.02 for q in QUANTITIES},
        },
        gate=MahalanobisGate(("x",), (0.0,), ((1.0,),), 2.0),
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / s23.sidecar_path("dogbone-bundle")
    path.parent.mkdir(parents=True)
    path.write_bytes(canonical_json_bytes(_calibration().document()) + b"\n")
    return tmp_path


def test_dogbone_widths_must_stay_within_the_tolerances(root: Path) -> None:
    assert gate.check_width(root, _protocol())["status"] == "PASS"
    wide = _calibration(peak=0.051)
    (root / s23.sidecar_path("dogbone-bundle")).write_bytes(
        canonical_json_bytes(wide.document()) + b"\n"
    )
    assert gate.check_width(root, _protocol())["status"] == "FAIL"


def _reveal(root: Path, covered: int) -> None:
    protocol = _protocol()
    calibration = _calibration()
    ids = protocol["families"]["dogbone-v1"]["h1_case_ids"]
    records = []
    for index, case_id in enumerate(ids):
        error = 0.001 if index < covered else 0.5
        records.append(
            {
                "case_id": case_id,
                "slices": ["all"],
                "reported_slices": [],
                "surrogate": {q: error for q in QUANTITIES},
                "components": {
                    q: {"discretisation": error / 2, "projection": 0.0}
                    for q in QUANTITIES
                },
            }
        )
    statuses = [
        {
            "case_id": case_id,
            "distance_to_train": 0.0 if index < 63 else 9.0,
            "status": IN_CALIBRATION if index < 63 else OUT_OF_CALIBRATION,
        }
        for index, case_id in enumerate(ids)
    ]
    coverage = reveal.count_coverage(protocol, calibration, records, statuses)
    path = root / reveal.reveal_path("dogbone-v1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cases": records, "coverage": coverage}))


def test_coverage_is_recounted_and_judged(root: Path) -> None:
    protocol = _protocol()
    assert gate.check_coverage(root, protocol)["status"] == "NOT_RUN"
    _reveal(root, covered=63)
    check = gate.check_coverage(root, protocol)
    assert check["status"] == "PASS"
    counts = check["measured"]["dogbone-v1"]["counts"]
    assert counts["budget_total:p_relative_l2"]["m"] == 63
    assert counts["budget_total:p_relative_l2"]["covered"] == 63
    _reveal(root, covered=40)
    assert gate.check_coverage(root, protocol)["status"] == "FAIL"


def test_a_tampered_count_fails_the_recount(root: Path) -> None:
    _reveal(root, covered=63)
    path = root / reveal.reveal_path("dogbone-v1")
    document = json.loads(path.read_text())
    document["coverage"]["counts"]["budget_total:p_relative_l2"]["covered"] = 64
    path.write_text(json.dumps(document))
    check = gate.check_coverage(root, _protocol())
    assert check["status"] == "FAIL"
    assert check["measured"]["dogbone-v1"]["recounts"] is False


def test_records_need_the_same_cases_in_order() -> None:
    with pytest.raises(ValueError, match="same cases"):
        reveal.count_coverage(
            _protocol(),
            _calibration(),
            [{"case_id": "a"}],
            [{"case_id": "b", "status": IN_CALIBRATION}],
        )


def test_missing_records_are_not_run(tmp_path: Path) -> None:
    assert gate.check_portability(tmp_path)["status"] == "NOT_RUN"
    assert gate.check_quality(tmp_path)["status"] == "NOT_RUN"
    protocol = _protocol()
    assert gate.check_reveal_once(tmp_path, protocol, "0" * 64)["status"] == "NOT_RUN"


def test_gate_records_do_not_change_the_code_tree(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.name", "t"],
        ["git", "config", "user.email", "t@example.invalid"],
    ):
        subprocess.run(command, cwd=root, check=True)
    (root / "code.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "code"], cwd=root, check=True)
    tree = gate._tree(root)
    record = root / gate.SUITE_PATH
    record.parent.mkdir(parents=True)
    record.write_text("{}\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "record"], cwd=root, check=True)
    assert gate._same_code(root, tree)
    (root / "code.py").write_text("x = 2\n")
    subprocess.run(["git", "commit", "-qam", "change"], cwd=root, check=True)
    assert not gate._same_code(root, tree)


def test_the_windows_record_is_judged(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.name", "t"],
        ["git", "config", "user.email", "t@example.invalid"],
    ):
        subprocess.run(command, cwd=root, check=True)
    (root / "code.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "code"], cwd=root, check=True)
    log = tmp_path / "windows.log"
    log.write_text(
        "=========================== short test summary info ===========\n"
        "SKIPPED [1] tests/a.py:1: the OS refuses to create symbolic links\n"
        "========== 3200 passed, 1 skipped in 10.0s ==========\n"
    )
    (root / gate.WINDOWS_PATH).parent.mkdir(parents=True)
    gate.record_windows(root, log, root)
    assert gate.check_portability(root)["status"] == "PASS"


def _documents(scale: float = 0.0, bound: float = 0.03) -> tuple[dict, dict]:
    calibration = _calibration()
    sidecar = calibration.document()
    sidecar["gate"]["precision"] = [[1.0 * (1.0 + scale)]]
    sidecar["gate"]["max_distance"] = 2.0 * (1.0 - scale)
    sidecar["bounds"]["budget_total"]["p_relative_l2"] = bound
    evidence = {
        "schema_version": "1.0",
        "campaign_id": "dogbone-v1",
        "protocol_sha256": "e" * 64,
        "case_ids": ["c"],
        "scores": {},
        "bounds": sidecar["bounds"],
        "n": 64,
        "k": 59,
        "dogbone_width_check": None,
        "gate": sidecar["gate"],
    }
    return sidecar, evidence


def test_decision_8_accepts_last_bit_gate_differences_only() -> None:
    committed = _documents()
    same = gate.compare_calibration(
        *committed[:1], *_documents(1e-14)[:1], committed[1], _documents(1e-14)[1]
    )
    assert same["sidecar_exact_fields"] and same["evidence_exact_fields"]
    assert same["gate_within_tolerance"]
    assert 0.0 < same["gate_relative_difference"] < 1e-13
    drifted = gate.compare_calibration(
        committed[0], _documents(1e-6)[0], committed[1], _documents(1e-6)[1]
    )
    assert not drifted["gate_within_tolerance"]
    other_bound = gate.compare_calibration(
        committed[0], _documents(0.0, 0.031)[0], committed[1], _documents(0.0, 0.031)[1]
    )
    assert not other_bound["sidecar_exact_fields"]
    assert not other_bound["evidence_exact_fields"]


def test_gates_on_other_axes_are_not_comparable() -> None:
    first = {"axes": ["x"], "mean": [0.0], "precision": [[1.0]], "max_distance": 1.0}
    second = {"axes": ["y"], "mean": [0.0], "precision": [[1.0]], "max_distance": 1.0}
    assert gate.gate_difference(first, second) is None
    assert gate.gate_difference(first, first) == 0.0
