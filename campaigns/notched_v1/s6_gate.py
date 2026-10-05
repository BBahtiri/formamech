"""The G6 gate of the notched campaign: the morphing framework on this family.

The shared G6 artifact (``artifacts/gates/S6.json``) predates the
``BlendedNotchMap``; this gate measures the G6 checks again on the notched
family and its new map:

- ``G6.stress``: 500 public shapes over the geometry box widened by 25% on
  every side; every accepted morph satisfies every validity criterion and
  every rejection carries a reason code;
- ``G6.ledger``: accepted + rejected equals attempted, with no silent drop;
- ``G6.tags``: tag names and facet counts of 500 accepted in-box shapes equal
  the reference mesh;
- ``G6.determinism``: an identical parameter point gives a byte-identical
  canonical mesh;
- ``G6.membership``: a labelled set of 204 points at, and one ulp beyond,
  every primitive bound;
- ``G6.injectivity``: the framework's deliberately folded maps are refused
  with ``NONINJECTIVE_MORPH`` (the framework tests);
- ``G6.coverage``: branch coverage of ``piginot.morphing`` over its tests.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from campaigns.notched_v1.family import (
    BOUNDS,
    GEOMETRY,
    MORPH_MAP,
    VALIDITY,
    compiled_family,
    parameter_point,
)
from campaigns.notched_v1.gate_io import (
    check_document,
    gate_document,
    run_pytest_with_coverage,
    write_gate_document,
)
from campaigns.notched_v1.pilot import public_samples
from campaigns.notched_v1.s9 import GATES
from piginot.contracts import canonical_json_bytes
from piginot.morphing import MorphingError, apply_morph
from piginot.morphing.validation import validate_deformation

REPOSITORY = Path(__file__).resolve().parents[2]
GATE_PATH = GATES / "S6.json"
STRESS_SEED = "notched-g6-boundary-spanning-sweep-v1"
_WIDENING = 0.25
_INJECTIVITY_TESTS = (
    "tests/morphing/test_validation_and_ledger.py::"
    "test_deliberately_noninjective_map_fires_the_required_mutant",
    "tests/morphing/test_validation_and_ledger.py::"
    "test_higher_order_rule_detects_an_interior_fold_with_valid_mesh_vertices",
)


def _widened(values: tuple[float, ...]) -> tuple[float, ...]:
    widened = []
    for name, value in zip(GEOMETRY, values, strict=True):
        lower, upper = BOUNDS[name]
        span = upper - lower
        fraction = (value - lower) / span
        widened.append(lower - _WIDENING * span + fraction * (1 + 2 * _WIDENING) * span)
    return tuple(widened)


def _valid(report: Any) -> bool:
    return bool(
        report.minimum_jacobian_determinant is not None
        and report.minimum_jacobian_determinant > VALIDITY.jacobian_margin
        and report.minimum_feature_size_mm >= VALIDITY.minimum_feature_size_mm
        and report.minimum_angle_degrees >= VALIDITY.minimum_angle_degrees
        and report.maximum_aspect_ratio <= VALIDITY.maximum_aspect_ratio
    )


def stress_sweep() -> dict[str, Any]:
    """The widened sweep, through the validator alone and through membership."""

    family = compiled_family()
    ledgers: dict[str, dict[str, Any]] = {}
    for path in ("validator", "membership_and_validator"):
        accepted = rejected = unreasoned = invalid_accepted = 0
        reasons: Counter[str] = Counter()
        for values in public_samples(500, seed=STRESS_SEED):
            theta = parameter_point(*_widened(values))
            try:
                if path == "validator":
                    mesh, report = validate_deformation(
                        family.reference_mesh, MORPH_MAP.bind(theta), VALIDITY
                    )
                    codes = tuple(code.value for code in report.reason_codes)
                else:
                    result = apply_morph(
                        family.reference_mesh,
                        family.morph_map,
                        family.membership,
                        theta,
                    )
                    mesh, report = (
                        result.mesh if result.accepted else None,
                        result.report,
                    )
                    codes = tuple(code.value for code in result.reason_codes)
            except MorphingError as error:
                mesh, report, codes = None, None, (error.reason_code.value,)
            if mesh is not None:
                accepted += 1
                invalid_accepted += not _valid(report)
            else:
                rejected += 1
                unreasoned += not codes
                reasons.update(codes)
        ledgers[path] = {
            "attempted": 500,
            "accepted": accepted,
            "rejected": rejected,
            "rejection_reason_occurrences": dict(sorted(reasons.items())),
            "accepted_violating_validity": invalid_accepted,
            "rejected_without_reason": unreasoned,
            "silent_drops": 500 - accepted - rejected,
        }
    return ledgers


def tag_sweep() -> dict[str, Any]:
    family = compiled_family()
    reference = family.reference_mesh
    expected = {name: len(facets) for name, facets in reference.semantic_tags.items()}
    mismatched = accepted = 0
    first: bytes | None = None
    repeat: bytes | None = None
    for index, values in enumerate(public_samples(500)):
        theta = parameter_point(*values)
        result = apply_morph(reference, family.morph_map, family.membership, theta)
        if not result.accepted or result.mesh is None:
            continue
        accepted += 1
        observed = {
            name: len(facets) for name, facets in result.mesh.semantic_tags.items()
        }
        if (
            observed != expected
            or result.mesh.boundary_facets != reference.boundary_facets
        ):
            mismatched += 1
        if index == 173:
            first = canonical_json_bytes(result.mesh)
            again = apply_morph(reference, family.morph_map, family.membership, theta)
            repeat = None if again.mesh is None else canonical_json_bytes(again.mesh)
    return {
        "accepted": accepted,
        "tag_mismatches": mismatched,
        "reference_tags": expected,
        "determinism_sample": 173,
        "byte_identical": first is not None and first == repeat,
    }


def membership_fixtures() -> dict[str, Any]:
    family = compiled_family()
    bases = public_samples(17, seed="notched-g6-membership-bases-v1")
    labelled = correct = 0
    for base in bases:
        for axis, name in enumerate(GEOMETRY):
            lower, upper = BOUNDS[name]
            for value, inside in (
                (lower, True),
                (upper, True),
                (math.nextafter(lower, -math.inf), False),
                (math.nextafter(upper, math.inf), False),
            ):
                values = list(base)
                values[axis] = value
                decision = family.membership.evaluate(parameter_point(*values))
                labelled += 1
                correct += decision.accepted is inside
    return {
        "labelled_instances": labelled,
        "correct": correct,
        "accuracy_percent": 100.0 * correct / labelled,
        "inclusive_boundary_cases": True,
        "boundary_adjacent_cases": True,
    }


def write_gate() -> dict[str, Any]:
    stress = stress_sweep()
    tags = tag_sweep()
    membership = membership_fixtures()
    injectivity_summary, _, injectivity_passed = run_pytest_with_coverage(
        REPOSITORY, _INJECTIVITY_TESTS, ("piginot.morphing",)
    )
    summary, coverage, passed = run_pytest_with_coverage(
        REPOSITORY, ("tests/morphing",), ("piginot.morphing",)
    )
    checks = [
        check_document(
            "G6.stress",
            all(
                ledger["accepted"] > 0
                and ledger["rejected"] > 0
                and ledger["accepted_violating_validity"] == 0
                and ledger["rejected_without_reason"] == 0
                for ledger in stress.values()
            ),
            stress,
            "500-sample boundary-spanning sweep: every accepted instance satisfies all "
            "validity criteria; every rejected instance has a recorded reason",
            "all",
        ),
        check_document(
            "G6.ledger",
            all(ledger["silent_drops"] == 0 for ledger in stress.values()),
            {path: ledger["silent_drops"] for path, ledger in stress.items()},
            "accepted + rejected = attempted; no silent drops",
            "eq",
        ),
        check_document(
            "G6.tags",
            tags["accepted"] == 500 and tags["tag_mismatches"] == 0,
            {"accepted": tags["accepted"], "tag_mismatches": tags["tag_mismatches"]},
            "tag identity preserved across 500 accepted instances",
            "all",
        ),
        check_document(
            "G6.determinism",
            tags["byte_identical"],
            tags["byte_identical"],
            "identical theta produces a byte-identical canonical mesh",
            "eq",
        ),
        check_document(
            "G6.membership",
            membership["labelled_instances"] >= 200
            and membership["accuracy_percent"] == 100.0,
            membership,
            "100% correct on a labelled in/out set of at least 200 instances",
            "all",
        ),
        check_document(
            "G6.injectivity",
            injectivity_passed,
            {"reason_code": "NONINJECTIVE_MORPH", "tests": injectivity_summary},
            "deliberately non-injective maps are rejected with NONINJECTIVE_MORPH",
            "eq",
        ),
        check_document(
            "G6.coverage",
            passed and coverage >= 85.0,
            {"tests": summary, "coverage_percent": coverage},
            85.0,
            "gte",
        ),
    ]
    document = gate_document(
        "G6",
        "S6",
        REPOSITORY,
        checks,
        morph_map="piginot.morphing.BlendedNotchMap",
        stress_sweep_seed=STRESS_SEED,
        stress_widening_fraction=_WIDENING,
    )
    write_gate_document(REPOSITORY / GATE_PATH, document)
    return document


if __name__ == "__main__":
    result = write_gate()
    print(result["verdict"], result["verdict_reason"])
    sys.exit(0 if result["verdict"] == "PASS" else 1)
