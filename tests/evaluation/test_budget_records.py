"""G20.harness: recorded S14 budgets are reproduced exactly; mutants are caught."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from piginot.evaluation import (
    BudgetRecordMismatch,
    budget_entries,
    compare_budget_record,
    recompute_budget_record,
)

ROOT = Path(__file__).resolve().parents[2]
RECORDS = [
    (family, split)
    for family in ("dogbone-v1", "notched-v1")
    for split in ("DEVELOPMENT", "HOLDOUT")
]


def _record(family: str, split: str) -> dict[str, Any]:
    path = ROOT / "artifacts" / "campaigns" / family / f"S14-{split}.json"
    return json.loads(path.read_text("utf-8"))


@pytest.mark.parametrize(("family", "split"), RECORDS)
def test_every_recorded_row_is_reproduced_exactly(family: str, split: str) -> None:
    record = _record(family, split)
    assert compare_budget_record(record) == []
    rebuilt = recompute_budget_record(record)
    assert len(rebuilt["gated"]) == len(record["budget"]["gated"])
    assert len(rebuilt["reported"]) == len(record["budget"].get("reported", []))


def test_an_off_by_one_case_is_detected() -> None:
    """Mutant: each case's components attributed to its neighbour."""

    record = _record("notched-v1", "HOLDOUT")
    mutant = copy.deepcopy(record)
    cases = mutant["cases"]
    shifted = [cases[(i + 1) % len(cases)]["components"] for i in range(len(cases))]
    for case, components in zip(cases, shifted, strict=True):
        case["components"] = components
    mismatches = compare_budget_record(mutant)
    assert any(m.field == "statistic" for m in mismatches)


def test_a_dropped_case_is_detected() -> None:
    record = _record("dogbone-v1", "DEVELOPMENT")
    mutant = copy.deepcopy(record)
    mutant["cases"] = mutant["cases"][1:]
    assert any(m.field == "cases" for m in compare_budget_record(mutant))


def test_swapped_splits_are_detected() -> None:
    """Mutant: DEVELOPMENT cases checked against the HOLDOUT rows."""

    development = _record("notched-v1", "DEVELOPMENT")
    holdout = _record("notched-v1", "HOLDOUT")
    swapped = {**development, "budget": holdout["budget"]}
    mismatches = compare_budget_record(swapped)
    assert mismatches
    assert all(isinstance(m, BudgetRecordMismatch) for m in mismatches)


def test_missing_and_extra_rows_are_reported() -> None:
    record = _record("dogbone-v1", "HOLDOUT")
    mutant = copy.deepcopy(record)
    removed = mutant["budget"]["gated"].pop()
    found = compare_budget_record(mutant)
    assert any(
        m.field == "row" and m.slice_id == removed["slice"] and m.recorded is False
        for m in found
    )


def test_a_changed_status_or_tolerance_is_detected() -> None:
    record = _record("dogbone-v1", "DEVELOPMENT")
    mutant = copy.deepcopy(record)
    mutant["budget"]["gated"][0]["status"] = "FAIL"
    assert [m.field for m in compare_budget_record(mutant)] == ["status"]


def test_inputs_are_validated() -> None:
    record = _record("dogbone-v1", "DEVELOPMENT")
    with pytest.raises(ValueError, match="kind"):
        budget_entries(record, "other")
    mutant = copy.deepcopy(record)
    mutant["budget"]["gated"][1]["tolerance"] = 0.07
    with pytest.raises(ValueError, match="two tolerances"):
        recompute_budget_record(mutant)
    empty = {**record, "budget": {"gated": record["budget"]["gated"]}}
    assert recompute_budget_record(empty)["reported"] == ()
