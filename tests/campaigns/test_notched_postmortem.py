"""S20 notched post-mortem: diagnostics and preregistered verdict logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from campaigns.notched_v1 import postmortem as pm

TOL = {"p_relative_l2": 0.05, "absolute_peak_error": 0.05, "u_relative_l2": 0.02}


def _case(index: int, surrogate: float, discretisation: float, radius: int) -> dict:
    terms = {
        q: {"discretisation": discretisation, "projection": 0.001}
        for q in pm.QUANTITIES
    }
    return {
        "case_id": f"c{index}",
        "surrogate": {q: surrogate for q in pm.QUANTITIES},
        "components": terms,
        "slices": ["all", f"notch_radius:{radius}"],
    }


def test_decomposition_separates_the_reference_term() -> None:
    cases = [_case(i, 0.02, 0.04, i % 4) for i in range(12)]
    d1 = pm.decomposition(cases, TOL)["p_relative_l2"]
    row = d1["slices"]["all"]
    assert row["total"] == pytest.approx(0.061)
    assert row["without_discretisation"] == pytest.approx(0.021)
    assert d1["failing_slices"]["total"] == [
        "all",
        "notch_radius:0",
        "notch_radius:1",
        "notch_radius:2",
        "notch_radius:3",
    ]
    assert d1["failing_slices"]["without_discretisation"] == []
    d2 = pm.reference_dominated(cases, TOL)["p_relative_l2"]
    assert d2["at_least_half_tolerance"] == 12 and d2["at_least_tolerance"] == 0


def test_concentration_and_fragility() -> None:
    cases = [_case(i, 0.04 if i % 4 == 0 else 0.02, 0.0, i % 4) for i in range(16)]
    d3 = pm.slice_concentration(cases)["absolute_peak_error"]
    assert d3["sharpest_over_bluntest"] == pytest.approx(2.0)
    spikes = [_case(i, 0.01, 0.0, 0) for i in range(13)] + [
        _case(20 + i, 0.2, 0.0, 0) for i in range(3)
    ]
    d5 = pm.metric_fragility(spikes, TOL)["p_relative_l2"]
    assert not d5["all"] <= 0.05
    assert d5["passes_without_three_largest"]


def test_conjunction_never_counts_missing_evidence_as_support() -> None:
    assert pm._verdict(True, True) == "SUPPORTED"
    assert pm._verdict(True, None) == "NOT_RUN"
    assert pm._verdict(False, None) == "NOT_SUPPORTED"
    assert pm._verdict(None, False) == "NOT_SUPPORTED"


def test_the_committed_public_analysis_reproduces(tmp_path: Path) -> None:
    document = pm.analyse(pm.REPOSITORY)
    recorded = (pm.REPOSITORY / pm.OUTPUT_PATH).read_text("utf-8")
    import json

    assert json.loads(recorded) == json.loads(json.dumps(document))


def test_a_changed_input_is_refused(tmp_path: Path) -> None:
    import shutil

    for relative in ("artifacts/campaigns/notched-v1",):
        shutil.copytree(pm.REPOSITORY / relative, tmp_path / relative)
    target = tmp_path / pm.CAMPAIGN / "S14-HOLDOUT.json"
    target.write_text(target.read_text("utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the protocol"):
        pm.analyse(tmp_path)
