"""S22 evaluation statistics and the G22 family checks on synthetic records."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from campaigns import g22_gate as gate
from campaigns import s22_serving as s22

PROTOCOL = json.loads((s22.REPOSITORY / s22.PROTOCOL_PATH).read_text("utf-8"))


def _record(index: int, *, path: str = "warm_newton") -> dict[str, Any]:
    return {
        "case_id": f"case-{index:03d}",
        "cold_success": True,
        "path": path,
        "attempts": [],
        "certificate_verdict": "OK",
        "seconds": {"cold": 100.0, "exact": 4.0},
        "operator_peak_error": 0.01,
        "served_peak_error": 0.015,
        "corrected_peak_error": 1e-4,
        "equivalence": {
            "u_max_relative": 1e-15,
            "energy_relative": 0.0,
            "p_relative_l2": 1e-14,
        },
        "exact_result_hash": "h",
        "certified_document_sha256": "c",
    }


def _setup(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    protocol = copy.deepcopy(PROTOCOL)
    results = {}
    for campaign in s22.FAMILIES:
        protocol["families"][campaign]["case_ids"] = [r["case_id"] for r in records]
        results[campaign] = {
            "records": records,
            "summary": s22.summarize(records),
            "platform_fingerprint": PROTOCOL["reference_machine"],
        }
    return protocol, results


def _statuses(
    protocol: dict[str, Any], results: dict[str, dict[str, Any]]
) -> dict[str, str]:
    return {c["id"]: c["status"] for c in gate.family_checks(protocol, results)}


def test_summary() -> None:
    summary = s22.summarize([_record(i) for i in range(64)])
    assert summary["median_speedup"] == pytest.approx(25.0)
    assert summary["paths"] == {"warm_newton": 64}
    assert summary["silent_results"] == 0
    assert summary["equivalence_max_relative"] == pytest.approx(1e-14)


def test_good_records_pass() -> None:
    statuses = _statuses(*_setup([_record(i) for i in range(64)]))
    assert set(statuses.values()) == {"PASS"}, statuses


@pytest.mark.parametrize(
    ("defect", "failing"),
    [
        ("slow", "G22.speedup"),
        ("different", "G22.equivalence"),
        ("silent", "G22.never-silent"),
        ("uncorrected", "G22.correction"),
        ("cold_failed", "G22.integrity"),
        ("tampered", "G22.recompute"),
        ("other_machine", "G22.reference-machine"),
    ],
)
def test_each_defect_fails_its_check(defect: str, failing: str) -> None:
    records = [_record(i) for i in range(64)]
    if defect == "slow":
        for r in records:
            r["seconds"]["exact"] = 20.0
    elif defect == "different":
        records[3]["equivalence"]["p_relative_l2"] = 1e-4
    elif defect == "silent":
        records[4]["equivalence"] = None
    elif defect == "uncorrected":
        for r in records:
            r["corrected_peak_error"] = 0.005
    elif defect == "cold_failed":
        records[7] = {
            "case_id": records[7]["case_id"],
            "cold_success": False,
            "cold_message": "x",
        }
    protocol, results = _setup(records)
    if defect == "tampered":
        results["dogbone-v1"]["summary"] = {
            **results["dogbone-v1"]["summary"],
            "median_speedup": 99.0,
        }
    if defect == "other_machine":
        results["notched-v1"]["platform_fingerprint"] = {
            **PROTOCOL["reference_machine"],
            "os": "Windows",
        }
    assert _statuses(protocol, results)[failing] == "FAIL"


def test_missing_results_are_not_run(tmp_path: Any) -> None:
    assert gate._results(tmp_path) is None
    assert gate.check_quality(tmp_path)["status"] == "NOT_RUN"
