"""S21 evaluation statistics and the G21 checks on synthetic records."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from campaigns import g21_gate as gate
from campaigns import s21_certificate as s21

PROTOCOL = json.loads((s21.REPOSITORY / s21.PROTOCOL_PATH).read_text("utf-8"))


def _record(
    index: int, error: float, *, verdict: str = "OK", reproduce: bool = False
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "case_id": f"case-{index:03d}",
        "mesh_id_equal": True,
        "vertex_gap": 0.0,
        "certificate": {
            "verdict": verdict,
            "eta_energy": error * 1.02,
            "dirichlet_deviation": 0.0,
            "indicators": {"p_l2_change": error * 0.01, "energy_ratio": 1.0},
        },
        "floor_eta": 1e-12,
        "true": {
            "energy": error,
            "p_relative_l2": error * 0.01,
            "u_relative_l2": error * 0.001,
        },
        "s14_indicator_median": 1.0 + (index % 3),
        "seconds": {"cold_fem": 100.0, "certificate": 0.6},
    }
    if reproduce:
        record["reproduction"] = {"coarse_p_difference": 0.01}
    return record


def _records() -> list[dict[str, Any]]:
    return [_record(i, 0.1 + 0.01 * i, reproduce=i < 8) for i in range(64)]


def _terms(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {r["case_id"]: {"coarse_p_difference": 0.01} for r in records}


def test_summary_statistics() -> None:
    records = _records()
    summary = s21.summarize(records, _terms(records), PROTOCOL)
    assert summary["certificates_ok"] == 64
    assert summary["spearman_eta_energy"] == pytest.approx(1.0)
    assert summary["effectivity_within_band"] == 1.0
    assert summary["effectivity_median"] == pytest.approx(1.02)
    assert summary["spearman_stress"] == pytest.approx(1.0)
    assert summary["cost_ratio"] == pytest.approx(0.006)
    assert summary["reproduction_cases"] == 8
    assert summary["reproduction_max_relative"] == 0.0


def _results(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for campaign in s21.FAMILIES:
        out[campaign] = {
            "records": records,
            "summary": s21.summarize(records, _terms(records), PROTOCOL),
            "platform_fingerprint": PROTOCOL["reference_machine"],
        }
    return out


def _protocol_for(records: list[dict[str, Any]]) -> dict[str, Any]:
    protocol = copy.deepcopy(PROTOCOL)
    for campaign in s21.FAMILIES:
        protocol["families"][campaign]["case_ids"] = [r["case_id"] for r in records]
        protocol["families"][campaign]["reproduction_case_ids"] = [
            r["case_id"] for r in records[:8]
        ]
    return protocol


def _statuses(checks: list[dict[str, Any]]) -> dict[str, str]:
    return {c["id"]: c["status"] for c in checks}


def test_good_records_pass_every_family_check(monkeypatch: pytest.MonkeyPatch) -> None:
    records = _records()
    monkeypatch.setattr(
        s21,
        "development_records",
        lambda c: [
            {"case_id": r["case_id"], "terms": {"coarse_p_difference": 0.01}}
            for r in records
        ],
    )
    statuses = _statuses(gate._family_checks(_protocol_for(records), _results(records)))
    assert set(statuses.values()) == {"PASS"}, statuses


@pytest.mark.parametrize(
    ("defect", "failing"),
    [
        ("indefinite", "G21.integrity"),
        ("slow", "G21.cost"),
        ("biased", "G21.effectivity"),
        ("tampered", "G21.recompute"),
        ("other_machine", "G21.reference-machine"),
        ("s14_better", "G21.versus-s14"),
    ],
)
def test_each_defect_fails_its_check(
    monkeypatch: pytest.MonkeyPatch, defect: str, failing: str
) -> None:
    records = _records()
    monkeypatch.setattr(
        s21,
        "development_records",
        lambda c: [
            {"case_id": r["case_id"], "terms": {"coarse_p_difference": 0.01}}
            for r in records
        ],
    )
    if defect == "indefinite":
        records[5]["certificate"]["verdict"] = "INDEFINITE"
    elif defect == "slow":
        for r in records:
            r["seconds"]["certificate"] = 20.0
    elif defect == "biased":
        for r in records:
            r["certificate"]["eta_energy"] = r["true"]["energy"] * 3.0
    elif defect == "s14_better":
        for i, r in enumerate(records):
            r["s14_indicator_median"] = r["true"]["p_relative_l2"]
            r["certificate"]["indicators"]["p_l2_change"] = 0.01 * ((i * 37) % 64)
    results = _results(records)
    if defect == "tampered":
        results["notched-v1"]["summary"] = {
            **results["notched-v1"]["summary"],
            "cost_ratio": 0.0,
        }
    if defect == "other_machine":
        results["dogbone-v1"]["platform_fingerprint"] = {
            **PROTOCOL["reference_machine"],
            "cpu_capability": "AVX512",
        }
    statuses = _statuses(gate._family_checks(_protocol_for(records), results))
    assert statuses[failing] == "FAIL", statuses


def test_a_reissued_certificate_is_deterministic_and_single_threaded() -> None:
    import torch

    case_id = PROTOCOL["families"]["notched-v1"]["case_ids"][0]
    before = torch.get_num_threads()
    seen: list[int] = []
    original = gate.s21._bundle

    def spy(name: str) -> object:
        seen.append(torch.get_num_threads())
        return original(name)

    gate.s21._bundle = spy  # type: ignore[assignment]
    try:
        first = gate.reissue("notched-v1", case_id)
    finally:
        gate.s21._bundle = original  # type: ignore[assignment]
    assert seen == [1]
    assert torch.get_num_threads() == before
    assert first["verdict"] == "OK"
    assert gate.reissue("notched-v1", case_id) == first


def test_missing_results_are_not_run(tmp_path: Any) -> None:
    assert gate._results(tmp_path) is None
    assert gate.check_quality(tmp_path)["status"] == "NOT_RUN"
