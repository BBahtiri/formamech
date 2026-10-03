from __future__ import annotations

import os
from pathlib import Path

import pytest

from campaigns.dogbone_v1.s15_bundle import (
    BUNDLE_NAME,
    LIMITATION_ITEMS,
    identities,
    model_spec,
    verification_report,
)
from piginot.bundle.specs import decode_model_spec

WEIGHTS = Path(os.environ.get("PIGINOT_CP_FREEZE_WEIGHTS", "/nonexistent"))


def test_model_spec_reproduces_the_s14_serving_arithmetic() -> None:
    spec = model_spec()
    assert decode_model_spec(spec.to_bytes()) == spec
    assert [c.name for c in spec.conditioning] == [
        "L_total",
        "W_grip",
        "W_gauge",
        "R_fillet",
        "nominal_axial_strain",
    ]
    values = {"L_total": 54.0, "nominal_axial_strain": 0.15, "W_grip": 20.0}
    assert spec.lifting_displacement.evaluate(values) == 0.15 * 54.0 / 2.0
    assert [e.evaluate(values) for e in spec.frame_extent] == [27.0, 10.0]
    assert spec.confidence.variant == "patch-mean"
    assert spec.confidence.quantile == 0.9
    assert spec.query_chunk == 2_048


def test_verification_report_lists_every_outstanding_item() -> None:
    report = verification_report()
    assert report["claim_level"] == "HOLDOUT_VERIFIED"
    assert report["verdict"] == "PASS"
    assert len(report["upstream_gates"]) == 14
    outstanding = {item["id"]: item for item in report["outstanding_work"]}
    assert outstanding["G14.budget"]["status"] == "WAIVED"
    assert "ADR 0021" in outstanding["G14.budget"]["reason"]
    for gate in ("G15", "G16", "G17", "experimental-validation"):
        assert outstanding[gate]["status"] == "NOT_RUN"
    assert {item["id"] for item in LIMITATION_ITEMS} >= {
        "development-budget-waiver",
        "confidence-calibration",
        "served-stress",
    }
    assert identities()["run_hash"]
    assert BUNDLE_NAME


@pytest.mark.skipif(
    not WEIGHTS.is_file(),
    reason="set PIGINOT_CP_FREEZE_WEIGHTS to the CP-FREEZE model export",
)
def test_the_dogbone_bundle_builds_and_loads(tmp_path: Path) -> None:
    from campaigns.dogbone_v1.s15_bundle import build
    from piginot.bundle import load_bundle

    build(tmp_path / "bundle", WEIGHTS, created_utc="2026-10-03T00:00:00Z")
    bundle = load_bundle(tmp_path / "bundle")
    assert bundle.serving.parameter_count == 573_474
    assert bundle.claim_level.value == "HOLDOUT_VERIFIED"
