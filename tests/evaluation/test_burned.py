"""G20.burned: burned HOLDOUT cases never reach selection or calibration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from piginot.evaluation import BurnedCaseError, BurnedCaseRegistry, guard_all

ROOT = Path(__file__).resolve().parents[2]


def _registry(family: str) -> BurnedCaseRegistry:
    record = json.loads(
        (ROOT / "artifacts" / "campaigns" / family / "S14-HOLDOUT.json").read_text(
            "utf-8"
        )
    )
    return BurnedCaseRegistry.from_holdout_records(family, {"S14-HOLDOUT.json": record})


@pytest.mark.parametrize(
    "purpose", ["selection", "tuning", "calibration", "acquisition", "compiler"]
)
def test_every_guarded_purpose_refuses_a_burned_case(purpose: str) -> None:
    registry = _registry("notched-v1")
    burned = sorted(registry.case_ids)[0]
    with pytest.raises(BurnedCaseError, match=purpose):
        registry.guard(["fresh-001", burned], purpose=purpose)
    registry.guard(["fresh-001", "fresh-002"], purpose=purpose)


def test_descriptive_use_is_explicit_and_allowed() -> None:
    registry = _registry("notched-v1")
    assert len(registry.descriptive(registry.case_ids)) == 64
    with pytest.raises(ValueError, match="descriptive"):
        registry.guard(["x"], purpose="analysis")


def test_the_unguarded_selection_mutant_is_caught() -> None:
    """Mutant ``SELECT_ON_BURNED``: a selector that skips the guard."""

    registry = _registry("dogbone-v1")
    candidates = ["fresh-001", *sorted(registry.case_ids)[:2]]

    def guarded_selector(case_ids: list[str]) -> list[str]:
        guard_all([registry], case_ids, purpose="selection")
        return case_ids

    def unguarded_selector(case_ids: list[str]) -> list[str]:
        return case_ids

    with pytest.raises(BurnedCaseError):
        guarded_selector(candidates)
    assert unguarded_selector(candidates) == candidates  # what the guard prevents


def test_the_registry_document_round_trips_and_detects_edits() -> None:
    registry = _registry("dogbone-v1")
    document = registry.to_document()
    assert BurnedCaseRegistry.from_document(document) == registry
    edited = {**document, "case_ids": document["case_ids"][:-1]}
    with pytest.raises(ValueError, match="altered"):
        BurnedCaseRegistry.from_document(edited)
    with pytest.raises(ValueError, match="schema"):
        BurnedCaseRegistry.from_document({**document, "schema_version": "0"})


def test_only_holdout_records_can_burn_and_inputs_are_validated() -> None:
    development = json.loads(
        (ROOT / "artifacts/campaigns/dogbone-v1/S14-DEVELOPMENT.json").read_text(
            "utf-8"
        )
    )
    with pytest.raises(ValueError, match="not a HOLDOUT"):
        BurnedCaseRegistry.from_holdout_records("dogbone-v1", {"dev": development})
    with pytest.raises(ValueError):
        BurnedCaseRegistry("", frozenset({"a"}), ("s",))
    with pytest.raises(ValueError):
        BurnedCaseRegistry("c", frozenset(), ("s",))
    with pytest.raises(ValueError):
        BurnedCaseRegistry("c", frozenset({"a"}), ())
    with pytest.raises(ValueError):
        BurnedCaseRegistry("c", frozenset({" a"}), ("s",))
