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
            )
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


def test_opened_holdout_reports_its_coverage(tmp_path: Path) -> None:
    _development(tmp_path)
    holdout = {"confidence": {"calibration_coverage": 0.871}}
    (tmp_path / s14.HOLDOUT_RESULT).write_text(json.dumps(holdout))
    items = {item["id"]: item["statement"] for item in limitation_items(tmp_path)}
    assert "not-verified" not in items
    assert "HOLDOUT coverage was 87.1%" in items["confidence-calibration"]
    assert "HOLDOUT budget was checked" in items["finite-sampling"]
