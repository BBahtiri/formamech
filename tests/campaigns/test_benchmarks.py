"""The benchmark page renders every population, including an unopened HOLDOUT."""

from __future__ import annotations

from typing import Any

from campaigns.benchmarks import render


def _population(cases: int, status: str, spearman: float | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cases": cases,
        "p90_including_fem_terms": {
            "p_relative_l2": 0.04,
            "absolute_peak_error": 0.045,
            "u_relative_l2": 0.003,
        },
        "gated_status": status,
    }
    if spearman is not None:
        row["pooled_spearman"] = spearman
    return row


def _family(name: str, holdout: dict[str, Any] | None) -> dict[str, Any]:
    development = _population(64, "FAIL", 0.61)
    development.update(
        latency_u_and_p_median_seconds=0.12,
        fem_median_seconds=129.0,
        speedup=1100.0,
    )
    return {
        "family": name,
        "train_screen": {
            "cases": 32,
            "p_relative_l2": 0.017,
            "median_absolute_peak_error": 0.012,
            "u_relative_l2": 0.0017,
        },
        "train_rehearsal": _population(32, "FAIL", None),
        "development": development,
        "holdout": holdout,
        "claim_level": "NOT_VERIFIED" if holdout is None else "HOLDOUT_VERIFIED",
        "development_budget_decision": None
        if holdout is None
        else "accepted by the decision owner (ADR 0000)",
    }


def test_render_reports_an_unopened_holdout_without_numbers() -> None:
    opened = _population(64, "PASS", 0.63)
    opened["calibration_coverage"] = 0.87
    text = render({"families": [_family("a-v1", opened), _family("b-v1", None)]})
    assert text.count("| TRAIN rehearsal (32 cases, p90 with FEM terms) |") == 2
    assert "| HOLDOUT (64 cases, p90 with FEM terms) | 4.00% | 4.50% | 0.30% |" in text
    assert "| HOLDOUT | not opened: DEVELOPMENT failed the gated budget" in text
    assert "HOLDOUT was not opened" in text
    assert "HOLDOUT calibration coverage of the 90% confidence bound: 87.00%" in text
    assert "Claim level: `NOT_VERIFIED`." in text
    assert text.count("DEVELOPMENT budget: accepted by the decision owner") == 1
