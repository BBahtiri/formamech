from __future__ import annotations

import copy

import pytest

from piginot.mutation import (
    EXIT_CODES,
    ComparisonStatus,
    Detector,
    DetectorOutcome,
    Mutant,
    SourceEdit,
    compare_signatures,
    run_detector,
    run_suite,
    signature_document,
)
from piginot.verification import CheckStatus
from tests.mutation import toy_targets as toy

MODULE = "tests.mutation.toy_targets"


def _detectors() -> tuple[Detector, ...]:
    def adds() -> DetectorOutcome:
        return DetectorOutcome(toy.add(2, 3) == 5, {"value": toy.add(2, 3)})

    def counts() -> DetectorOutcome:
        return DetectorOutcome(toy.Counter(1).next() == 2)

    def squares() -> DetectorOutcome:
        if toy.cached_square(3) != 9:
            raise ValueError("square changed")
        return DetectorOutcome(True)

    return (
        Detector("G0.add", "addition", adds),
        Detector("G0.count", "counting", counts),
        Detector("G0.square", "squares", squares),
    )


MUTANTS = (
    Mutant(
        "ADD_TO_SUB",
        "G0.add",
        "subtraction",
        (SourceEdit(f"{MODULE}:add", "left + right", "left - right"),),
    ),
    Mutant(
        "SQUARE_PLUS_ONE",
        "G0.square",
        "square off by one",
        (SourceEdit(f"{MODULE}:cached_square", "value * value", "value * value + 1"),),
    ),
)


def test_detector_statuses() -> None:
    passing, failing, crashing = (
        Detector("a", "", lambda: DetectorOutcome(True, {"x": 1})),
        Detector("b", "", lambda: DetectorOutcome(False)),
        Detector("c", "", lambda: 1 / 0),  # type: ignore[arg-type,return-value]
    )
    assert run_detector(passing).status is CheckStatus.PASS
    assert run_detector(failing).status is CheckStatus.FAIL
    result = run_detector(crashing)
    assert result.status is CheckStatus.ERROR and result.fired
    assert "ZeroDivisionError" in (result.error or "")
    assert result.document()["status"] == "ERROR"


def test_suite_records_signatures_with_clear_caches() -> None:
    progress: list[str] = []
    report = run_suite(
        MUTANTS, _detectors(), caches=(toy.cached_square,), progress=progress.append
    )
    assert report.control_passed
    assert report.primary_count == 2
    assert [run.fired for run in report.runs] == [("G0.add",), ("G0.square",)]
    assert report.runs[1].results[2].status is CheckStatus.ERROR
    assert progress[0] == "control: 3/3"
    document = report.document()
    assert document["primary_fired"] == 2 and document["mutant_count"] == 2
    baseline = signature_document(report)
    assert baseline["mutants"]["ADD_TO_SUB"]["signature"] == ["G0.add"]
    assert compare_signatures(baseline, baseline).status is ComparisonStatus.PASS


def test_stale_cache_would_hide_a_mutant() -> None:
    toy.cached_square(3)
    report = run_suite(MUTANTS[1:], _detectors()[2:])
    assert report.runs[0].fired == ()  # no cache cleared: the cached 9 is served
    toy.cached_square.cache_clear()


def test_suite_inputs_are_validated() -> None:
    detectors = _detectors()
    with pytest.raises(ValueError, match="unique"):
        run_suite(MUTANTS, (*detectors, detectors[0]))
    with pytest.raises(ValueError, match="unique"):
        run_suite((MUTANTS[0], MUTANTS[0]), detectors)
    with pytest.raises(ValueError, match="without a detector"):
        run_suite(MUTANTS, detectors[1:])


def test_a_failing_control_records_no_baseline() -> None:
    broken = Detector("G0.broken", "", lambda: DetectorOutcome(False))
    report = run_suite(MUTANTS[:1], (*_detectors()[:1], broken))
    assert not report.control_passed
    with pytest.raises(ValueError, match="control"):
        signature_document(report)


def _baseline() -> dict:
    return {
        "schema": "piginot-mutation-signatures-v1",
        "detectors": ["G1.a", "G1.b", "G1.c"],
        "mutants": {
            "M1": {
                "primary": "G1.a",
                "signature": ["G1.a", "G1.b"],
                "edits_sha256": "x",
            },
            "M2": {"primary": "G1.c", "signature": ["G1.c"], "edits_sha256": "y"},
        },
    }


def test_comparison_classifies_every_change() -> None:
    baseline = _baseline()

    lost_secondary = copy.deepcopy(baseline)
    lost_secondary["mutants"]["M1"]["signature"] = ["G1.a"]
    result = compare_signatures(baseline, lost_secondary)
    assert result.status is ComparisonStatus.COVERAGE_REGRESSION
    assert result.exit_code == EXIT_CODES[ComparisonStatus.COVERAGE_REGRESSION] == 3
    assert result.mutants[0].lost_secondary == ("G1.b",)

    lost_primary = copy.deepcopy(baseline)
    lost_primary["mutants"]["M2"]["signature"] = []
    assert compare_signatures(baseline, lost_primary).status is (
        ComparisonStatus.PRIMARY_LOST
    )

    new_detection = copy.deepcopy(baseline)
    new_detection["mutants"]["M2"]["signature"] = ["G1.a", "G1.c"]
    result = compare_signatures(baseline, new_detection)
    assert result.status is ComparisonStatus.PASS
    assert result.mutants[1].new_detections == ("G1.a",)

    changed = copy.deepcopy(baseline)
    changed["mutants"]["M2"]["edits_sha256"] = "z"
    assert compare_signatures(baseline, changed).status is (
        ComparisonStatus.MUTANT_CHANGED
    )

    missing = copy.deepcopy(baseline)
    del missing["mutants"]["M2"]
    missing["mutants"]["M3"] = {
        "primary": "G1.b",
        "signature": ["G1.b"],
        "edits_sha256": "w",
    }
    missing["detectors"] = ["G1.a", "G1.b", "G1.d"]
    result = compare_signatures(baseline, missing)
    assert result.status is ComparisonStatus.PRIMARY_LOST
    assert result.removed_detectors == ("G1.c",)
    assert result.added_detectors == ("G1.d",)
    statuses = {item.name: item.status for item in result.mutants}
    assert statuses["M2"] is ComparisonStatus.PRIMARY_LOST
    assert statuses["M3"] is ComparisonStatus.MUTANT_CHANGED
    assert result.document()["status"] == "PRIMARY_LOST"

    with pytest.raises(ValueError):
        compare_signatures({"schema": "other"}, baseline)
