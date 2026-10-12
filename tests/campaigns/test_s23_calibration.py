"""S23 calibration builder on synthetic records (ADR 0104, G23.isolation).

The real calibration runs only after the decision owner accepts the protocol;
these tests use a synthetic family so that no DEVELOPMENT bound is computed
before then.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from campaigns import s23_calibration as s23
from piginot.bounds import IN_CALIBRATION, CalibrationRecord
from piginot.contracts import canonical_json_bytes
from piginot.evaluation import BurnedCaseError

CAMPAIGN = "dogbone-v1"
AXES = ["length", "width", "elastic_modulus", "nominal_axial_strain"]


def _case(rng: random.Random, case_id: str, role: str) -> dict[str, Any]:
    length = rng.uniform(40.0, 70.0)
    return {
        "case_id": case_id,
        "role": role,
        "status": "ACTIVE",
        "primitive_coordinates": {
            "length": length,
            "width": 0.2 * length + rng.uniform(0.0, 3.0),
            "elastic_modulus": 760,
            "nominal_axial_strain": rng.uniform(0.1, 0.2),
        },
    }


def _development_case(rng: random.Random, case_id: str) -> dict[str, Any]:
    def parts(scale: float) -> dict[str, float]:
        return {
            "discretisation": rng.uniform(0.2, 1.0) * scale,
            "projection": rng.uniform(0.0, 0.3) * scale,
        }

    return {
        "case_id": case_id,
        "slices": ["all"],
        "reported_slices": [],
        "surrogate": {
            "p_relative_l2": rng.uniform(0.0, 0.02),
            "absolute_peak_error": rng.uniform(0.0, 0.03),
            "u_relative_l2": rng.uniform(0.0, 0.002),
        },
        "components": {
            "p_relative_l2": parts(0.01),
            "absolute_peak_error": parts(0.015),
            "u_relative_l2": parts(0.001),
        },
    }


def _write(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), "utf-8")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    rng = random.Random(23)
    train = [_case(rng, f"train-{i:03d}", "TRAIN") for i in range(96)]
    development = [_case(rng, f"dev-{i:03d}", "DEVELOPMENT") for i in range(64)]
    campaign = tmp_path / "artifacts/campaigns" / CAMPAIGN
    _write(
        campaign / "public-case-manifest.json",
        {"primitive_axes": AXES, "cases": train + development},
    )
    _write(
        campaign / "S14-DEVELOPMENT.json",
        {
            "evidence_role": "DEVELOPMENT",
            "budget": {},
            "cases": [_development_case(rng, c["case_id"]) for c in development],
        },
    )
    _write(
        campaign / "FRESH-HOLDOUTS.json",
        {
            "sets": {
                "H1": {"case_ids": [f"fresh-h1-{i:03d}" for i in range(64)]},
                "H2": {"case_ids": [f"fresh-h2-{i:03d}" for i in range(64)]},
            }
        },
    )
    burned = {
        "campaign_id": CAMPAIGN,
        "case_ids": ["holdout-000"],
        "sources": ["S14-HOLDOUT.json"],
    }
    from piginot.evaluation import BurnedCaseRegistry

    registry = BurnedCaseRegistry(
        CAMPAIGN, frozenset(burned["case_ids"]), tuple(burned["sources"])
    )
    _write(
        tmp_path / "artifacts/evaluation" / f"BURNED-HOLDOUT-{CAMPAIGN}.json",
        registry.to_document(),
    )
    return tmp_path


@pytest.fixture
def families(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(s23, "FAMILIES", {CAMPAIGN: "dogbone-v1-s13c-cp-freeze"})


def _accept(root: Path) -> None:
    document = s23.protocol_document(root, "decision owner, test")
    _write(root / s23.PROTOCOL_PATH, document)


def test_the_protocol_needs_the_owners_acceptance(root: Path, families: None) -> None:
    with pytest.raises(s23.ProtocolNotAccepted):
        s23.protocol_document(root, "  ")
    document = s23.protocol_document(root, "decision owner, test")
    family = document["families"][CAMPAIGN]
    assert len(family["calibration_case_ids"]) == 64
    assert family["h1_case_ids"][0] == "fresh-h1-000"
    assert document["coverage_test"]["per_check_level"] == pytest.approx(0.05 / 4)
    assert document["thresholds_accepted"] == "decision owner, test"


def test_calibration_refuses_to_run_before_acceptance(
    root: Path, families: None
) -> None:
    with pytest.raises(s23.ProtocolNotAccepted, match="does not exist"):
        s23.calibrate(root, CAMPAIGN, "0" * 64)
    unaccepted = s23.protocol_document(root, "x")
    unaccepted["thresholds_accepted"] = ""
    _write(root / s23.PROTOCOL_PATH, unaccepted)
    with pytest.raises(s23.ProtocolNotAccepted, match="not accepted"):
        s23.calibrate(root, CAMPAIGN, "0" * 64)


def test_a_calibration_is_deterministic_and_admits_its_cases(
    root: Path, families: None
) -> None:
    _accept(root)
    calibration, evidence = s23.calibrate(root, CAMPAIGN, "0" * 64)
    again, evidence_again = s23.calibrate(root, CAMPAIGN, "0" * 64)
    assert canonical_json_bytes(calibration.document()) == canonical_json_bytes(
        again.document()
    )
    assert canonical_json_bytes(evidence) == canonical_json_bytes(evidence_again)
    assert (calibration.n, calibration.k) == (64, 59)
    assert calibration.gate.axes == ("length", "width", "nominal_axial_strain")
    for kind in ("budget_total", "reference_terms"):
        for quantity, bound in calibration.bounds[kind].items():
            ranked = sorted(evidence["scores"][kind][quantity])
            assert bound == ranked[58]
    manifest = json.loads(
        (
            root / "artifacts/campaigns" / CAMPAIGN / "public-case-manifest.json"
        ).read_text()
    )
    for case in manifest["cases"]:
        if case["role"] == "DEVELOPMENT":
            report = calibration.evaluate(case["primitive_coordinates"], "budget_total")
            assert report["status"] == IN_CALIBRATION
    copy = CalibrationRecord.from_document(
        json.loads(canonical_json_bytes(calibration.document()))
    )
    assert copy == calibration
    assert set(evidence["dogbone_width_check"]) == {
        "p_relative_l2",
        "absolute_peak_error",
        "u_relative_l2",
    }


def test_budget_totals_are_the_s14_sums(root: Path, families: None) -> None:
    record = s23.development_record(root, CAMPAIGN)
    per_case = s23.scores(record)
    case = record["cases"][0]
    for quantity in s23.QUANTITIES:
        parts = case["components"][quantity]
        reference = parts["discretisation"] + parts["projection"]
        assert per_case["reference_terms"][quantity][0] == reference
        assert per_case["budget_total"][quantity][0] == pytest.approx(
            case["surrogate"][quantity] + reference, rel=1e-15
        )


def test_a_burned_holdout_case_is_refused(root: Path, families: None) -> None:
    """Mutant: a burned HOLDOUT case slips into the calibration set."""

    _accept(root)
    path = root / "artifacts/campaigns" / CAMPAIGN / "S14-DEVELOPMENT.json"
    record = json.loads(path.read_text())
    record["cases"][3]["case_id"] = "holdout-000"
    _write(path, record)
    protocol = json.loads((root / s23.PROTOCOL_PATH).read_text())
    protocol["families"][CAMPAIGN]["calibration_case_ids"][3] = "holdout-000"
    _write(root / s23.PROTOCOL_PATH, protocol)
    with pytest.raises(BurnedCaseError):
        s23.calibrate(root, CAMPAIGN, "0" * 64)


def test_a_sealed_case_is_refused(root: Path, families: None) -> None:
    """Mutant: a sealed H1 case slips into the calibration set."""

    with pytest.raises(s23.SealedCaseError, match="fresh-h1-007"):
        s23.guard(root, CAMPAIGN, ["dev-000", "fresh-h1-007"])
    with pytest.raises(s23.SealedCaseError, match="fresh-h2-001"):
        s23.guard(root, CAMPAIGN, ["fresh-h2-001"])
    s23.guard(root, CAMPAIGN, ["dev-000", "dev-001"])


def test_cases_must_match_the_protocol(root: Path, families: None) -> None:
    _accept(root)
    protocol = json.loads((root / s23.PROTOCOL_PATH).read_text())
    protocol["families"][CAMPAIGN]["calibration_case_ids"].reverse()
    _write(root / s23.PROTOCOL_PATH, protocol)
    with pytest.raises(ValueError, match="does not match"):
        s23.calibrate(root, CAMPAIGN, "0" * 64)


def test_the_protocol_is_written_once(
    root: Path, families: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(s23, "REPOSITORY", root)
    assert s23.main(["protocol", "--accepted", "decision owner, test"]) == 0
    assert s23.main(["protocol", "--accepted", "again"]) == 1
