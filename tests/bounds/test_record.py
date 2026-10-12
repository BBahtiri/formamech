"""S23 calibration sidecars: binding, round trip and evaluation (ADR 0104)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from piginot.bounds import (
    IN_CALIBRATION,
    OUT_OF_CALIBRATION,
    CalibrationError,
    CalibrationRecord,
    MahalanobisGate,
    calibration_path,
    case_ids_sha256,
)
from piginot.contracts import canonical_json_bytes

DIGEST = "a" * 64


def gate(max_distance: float = 2.0) -> MahalanobisGate:
    return MahalanobisGate(
        ("width", "radius"), (20.0, 4.0), ((0.25, 0.0), (0.0, 1.0)), max_distance
    )


def record(**changes: Any) -> CalibrationRecord:
    fields: dict[str, Any] = {
        "bundle": "plate",
        "bundle_manifest_sha256": DIGEST,
        "campaign_id": "plate-v1",
        "protocol_sha256": "b" * 64,
        "alpha": 0.1,
        "n": 64,
        "k": 59,
        "calibration_case_ids_sha256": case_ids_sha256(["c1", "c2"]),
        "bounds": {
            "budget_total": {"p_relative_l2": 0.03, "absolute_peak_error": 0.04},
            "reference_terms": {"p_relative_l2": 0.012, "absolute_peak_error": 0.02},
        },
        "gate": gate(),
    }
    fields.update(changes)
    return CalibrationRecord(**fields)


def test_the_document_round_trips_through_canonical_json() -> None:
    original = record()
    copy = CalibrationRecord.from_document(
        json.loads(canonical_json_bytes(original.document()))
    )
    assert copy == original
    assert copy.sha256() == original.sha256()


def test_a_sidecar_is_bound_to_its_bundle(tmp_path: Path) -> None:
    path = tmp_path / "calibration" / "plate.json"
    path.parent.mkdir()
    path.write_bytes(canonical_json_bytes(record().document()))
    assert CalibrationRecord.load(path, bundle="plate", manifest_sha256=DIGEST)
    with pytest.raises(CalibrationError, match="another bundle"):
        CalibrationRecord.load(path, bundle="plate", manifest_sha256="c" * 64)
    with pytest.raises(CalibrationError, match="another bundle"):
        CalibrationRecord.load(path, bundle="other", manifest_sha256=DIGEST)
    with pytest.raises(CalibrationError, match="cannot read"):
        CalibrationRecord.load(tmp_path / "x.json", bundle="p", manifest_sha256=DIGEST)
    assert calibration_path(tmp_path / "plate") == path


def test_inside_and_outside_the_calibration_envelope() -> None:
    calibration = record()
    inside = calibration.evaluate({"width": 21.0, "radius": 4.5}, "budget_total")
    assert inside["status"] == IN_CALIBRATION
    assert inside["upper_bounds"] == {
        "absolute_peak_error": 0.04,
        "p_relative_l2": 0.03,
    }
    assert inside["distance_to_train"] == pytest.approx(0.70710678, rel=1e-6)
    outside = calibration.evaluate({"width": 30.0, "radius": 4.0}, "reference_terms")
    assert outside["status"] == OUT_OF_CALIBRATION
    assert outside["upper_bounds"] is None
    assert outside["kind"] == "reference_terms"
    with pytest.raises(CalibrationError, match="kind"):
        calibration.evaluate({"width": 21.0, "radius": 4.5}, "other")


@pytest.mark.parametrize(
    "changes",
    [
        {"bundle": ""},
        {"bundle_manifest_sha256": "xyz"},
        {"protocol_sha256": "A" * 64},
        {"alpha": 1.0},
        {"k": 65},
        {"k": 0},
        {"bounds": {"budget_total": {"p": 0.1}}},
        {"bounds": {"budget_total": {}, "reference_terms": {"p": 0.1}}},
        {"bounds": {"budget_total": {"p": -0.1}, "reference_terms": {"p": 0.1}}},
        {"bounds": {"budget_total": {"p": float("inf")}, "reference_terms": {"p": 1}}},
    ],
)
def test_malformed_records_are_refused(changes: dict[str, Any]) -> None:
    with pytest.raises(CalibrationError):
        record(**changes)


def test_malformed_documents_are_refused() -> None:
    with pytest.raises(CalibrationError, match="not a"):
        CalibrationRecord.from_document({"schema": "other"})
    document = record().document()
    del document["gate"]
    with pytest.raises(CalibrationError, match="malformed"):
        CalibrationRecord.from_document(document)


def test_case_id_digests_keep_the_order() -> None:
    assert case_ids_sha256(["a", "b"]) != case_ids_sha256(["b", "a"])
