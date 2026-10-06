"""The notched bundle's limitation items follow the S14 evidence as measured."""

from __future__ import annotations

import json
from pathlib import Path

from campaigns.notched_v1 import s14
from campaigns.notched_v1.s15 import limitation_items


def _gated(statistics: dict[str, float], tolerance: dict[str, float]) -> list[dict]:
    return [
        {
            "slice": "all",
            "quantity": quantity,
            "statistic": value,
            "status": "PASS" if value <= tolerance[quantity] else "FAIL",
        }
        for quantity, value in statistics.items()
    ]


TOLERANCE = {"p_relative_l2": 0.05, "absolute_peak_error": 0.05, "u_relative_l2": 0.02}


def _development(run_root: Path) -> None:
    document = {
        "case_count": 64,
        "budget": {
            "gated": _gated(
                {
                    "p_relative_l2": 0.061,
                    "absolute_peak_error": 0.072,
                    "u_relative_l2": 0.004,
                },
                TOLERANCE,
            ),
            "gated_status": "FAIL",
        },
        "confidence": {"calibration": {"in_sample_coverage": 0.9}},
    }
    (run_root / s14.DEVELOPMENT_RESULT).write_text(json.dumps(document))


def test_unopened_holdout_is_stated_as_not_verified(tmp_path: Path) -> None:
    _development(tmp_path)
    items = {item["id"]: item["statement"] for item in limitation_items(tmp_path)}
    assert items["not-verified"].startswith("NOT_VERIFIED: the S14 DEVELOPMENT")
    assert "2 of 3 gated rows" in items["not-verified"]
    assert "stress field 6.10%, peak 7.20%, displacement 0.40%" in items["not-verified"]
    assert "HOLDOUT was never opened" in items["not-verified"]
    assert "64 cases" in items["finite-sampling"]
    assert "no HOLDOUT coverage measurement" in items["confidence-calibration"]
    assert "HOLDOUT coverage was" not in items["confidence-calibration"]


def _holdout(run_root: Path, statistics: dict[str, float]) -> None:
    gated = _gated(statistics, TOLERANCE)
    document = {
        "budget": {
            "gated": gated,
            "gated_status": "FAIL"
            if any(row["status"] == "FAIL" for row in gated)
            else "PASS",
        },
        "confidence": {"calibration_coverage": 0.871},
    }
    (run_root / s14.HOLDOUT_RESULT).write_text(json.dumps(document))


def test_opened_holdout_reports_its_coverage(tmp_path: Path) -> None:
    _development(tmp_path)
    _holdout(
        tmp_path,
        {"p_relative_l2": 0.03, "absolute_peak_error": 0.04, "u_relative_l2": 0.003},
    )
    items = {item["id"]: item["statement"] for item in limitation_items(tmp_path)}
    assert "not-verified" not in items
    assert "holdout-budget-failed" not in items
    assert "HOLDOUT coverage was 87.1%" in items["confidence-calibration"]
    assert "HOLDOUT budget was checked" in items["finite-sampling"]
    assert "ADR 0030" in items["development-budget-accepted"]
    assert "2 of 3 gated rows fail" in items["development-budget-accepted"]


def test_failed_holdout_is_stated_with_its_numbers(tmp_path: Path) -> None:
    _development(tmp_path)
    _holdout(
        tmp_path,
        {
            "p_relative_l2": 0.0601,
            "absolute_peak_error": 0.064,
            "u_relative_l2": 0.0053,
        },
    )
    items = {item["id"]: item["statement"] for item in limitation_items(tmp_path)}
    statement = items["holdout-budget-failed"]
    assert "stress field 6.01%, peak 6.40%, displacement 0.53%" in statement
    assert "at most DEVELOPMENT_VERIFIED" in statement
