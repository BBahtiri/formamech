"""H1 confirmation of calibrated bounds: binding and withholding (ADR 0104)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from piginot.bounds import (
    OUT_OF_CALIBRATION,
    CalibrationError,
    ConfirmedCalibration,
    H1Confirmation,
    confirmation_path,
)
from piginot.contracts import canonical_json_bytes
from tests.bounds.test_record import record


def confirmation(**changes: Any) -> H1Confirmation:
    calibration = record()
    fields: dict[str, Any] = {
        "bundle": calibration.bundle,
        "bundle_manifest_sha256": calibration.bundle_manifest_sha256,
        "calibration_sha256": calibration.sha256(),
        "reveal_sha256": "f" * 64,
        "confirmed": {
            "budget_total": {"p_relative_l2": True, "absolute_peak_error": False},
            "reference_terms": {"p_relative_l2": True, "absolute_peak_error": True},
        },
    }
    fields.update(changes)
    return H1Confirmation(**fields)


def test_unconfirmed_bounds_are_withheld() -> None:
    served = ConfirmedCalibration(record(), confirmation())
    inside = served.evaluate({"width": 21.0, "radius": 4.5}, "budget_total")
    assert inside["upper_bounds"] == {"p_relative_l2": 0.03}
    assert inside["not_confirmed"] == ["absolute_peak_error"]
    reference = served.evaluate({"width": 21.0, "radius": 4.5}, "reference_terms")
    assert reference["not_confirmed"] == []
    assert len(reference["upper_bounds"]) == 2
    outside = served.evaluate({"width": 30.0, "radius": 4.0}, "budget_total")
    assert outside["status"] == OUT_OF_CALIBRATION
    assert outside["upper_bounds"] is None
    assert served.bundle_manifest_sha256 == record().bundle_manifest_sha256


def test_a_confirmation_is_bound_to_its_calibration(tmp_path: Path) -> None:
    path = tmp_path / "h1.json"
    path.write_bytes(canonical_json_bytes(confirmation().document()))
    assert H1Confirmation.load(path, record()) == confirmation()
    other = record(
        bounds={
            "budget_total": {"p_relative_l2": 0.031, "absolute_peak_error": 0.04},
            "reference_terms": {"p_relative_l2": 0.012, "absolute_peak_error": 0.02},
        }
    )
    with pytest.raises(CalibrationError, match="another calibration"):
        H1Confirmation.load(path, other)
    with pytest.raises(CalibrationError, match="cannot read"):
        H1Confirmation.load(tmp_path / "missing.json", record())
    assert confirmation_path(tmp_path / "plate") == tmp_path / "h1-results/plate.json"


@pytest.mark.parametrize(
    "changes",
    [
        {"bundle": ""},
        {"reveal_sha256": "x"},
        {"confirmed": {"budget_total": {"p": True}}},
        {"confirmed": {"budget_total": {}, "reference_terms": {"p": True}}},
        {"confirmed": {"budget_total": {"p": 1}, "reference_terms": {"p": True}}},
    ],
)
def test_malformed_confirmations_are_refused(changes: dict[str, Any]) -> None:
    with pytest.raises(CalibrationError):
        confirmation(**changes)


def test_documents_round_trip_and_bad_ones_are_refused() -> None:
    document = json.loads(canonical_json_bytes(confirmation().document()))
    assert H1Confirmation.from_document(document) == confirmation()
    with pytest.raises(CalibrationError, match="not a"):
        H1Confirmation.from_document({"schema": "other"})
    del document["confirmed"]
    with pytest.raises(CalibrationError, match="malformed"):
        H1Confirmation.from_document(document)
