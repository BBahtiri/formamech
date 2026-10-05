"""Fired-gate signatures and their per-release comparison (``G16.regression``).

The signature of a mutant is the sorted set of gate checks that fire under
it. The committed baseline is compared with a new run:

- the primary gate no longer firing, or a baseline mutant missing, is
  ``PRIMARY_LOST``;
- any other baseline gate no longer firing is a ``COVERAGE_REGRESSION``,
  reported even when the primary still fires;
- a gate that fires now but not in the baseline is a new detection, reported
  so the baseline can be updated deliberately;
- a mutant whose edits changed is ``MUTANT_CHANGED``: its signature is not
  comparable until the baseline is re-recorded.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from piginot.mutation.runner import SuiteReport

SCHEMA = "piginot-mutation-signatures-v1"


class ComparisonStatus(StrEnum):
    PASS = "PASS"
    MUTANT_CHANGED = "MUTANT_CHANGED"
    COVERAGE_REGRESSION = "COVERAGE_REGRESSION"
    PRIMARY_LOST = "PRIMARY_LOST"


_SEVERITY = {
    ComparisonStatus.PASS: 0,
    ComparisonStatus.MUTANT_CHANGED: 1,
    ComparisonStatus.COVERAGE_REGRESSION: 2,
    ComparisonStatus.PRIMARY_LOST: 3,
}
EXIT_CODES = {
    ComparisonStatus.PASS: 0,
    ComparisonStatus.MUTANT_CHANGED: 2,
    ComparisonStatus.COVERAGE_REGRESSION: 3,
    ComparisonStatus.PRIMARY_LOST: 4,
}


def _edits_digest(edits: list[Mapping[str, str]]) -> str:
    payload = json.dumps(edits, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def signature_document(report: SuiteReport) -> dict[str, Any]:
    """The baseline to commit for a suite run whose control passed."""

    if not report.control_passed:
        raise ValueError("a baseline needs a control run in which nothing fires")
    return {
        "schema": SCHEMA,
        "detectors": list(report.detectors),
        "mutants": {
            run.mutant.name: {
                "primary": run.mutant.primary,
                "signature": list(run.fired),
                "edits_sha256": _edits_digest(run.mutant.document()["edits"]),
            }
            for run in report.runs
        },
    }


@dataclass(frozen=True, slots=True)
class MutantComparison:
    name: str
    status: ComparisonStatus
    primary: str
    lost_secondary: tuple[str, ...]
    new_detections: tuple[str, ...]
    detail: str = ""

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "primary": self.primary,
            "lost_secondary": list(self.lost_secondary),
            "new_detections": list(self.new_detections),
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class Comparison:
    status: ComparisonStatus
    mutants: tuple[MutantComparison, ...]
    removed_detectors: tuple[str, ...]
    added_detectors: tuple[str, ...]

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.status]

    def document(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "removed_detectors": list(self.removed_detectors),
            "added_detectors": list(self.added_detectors),
            "mutants": [item.document() for item in self.mutants],
        }


def compare_signatures(
    baseline: Mapping[str, Any], current: Mapping[str, Any]
) -> Comparison:
    """Compare a new signature document with the committed baseline."""

    for label, document in (("baseline", baseline), ("current", current)):
        if document.get("schema") != SCHEMA:
            raise ValueError(f"{label} is not a {SCHEMA} document")
    base_mutants: Mapping[str, Any] = baseline["mutants"]
    new_mutants: Mapping[str, Any] = current["mutants"]
    items: list[MutantComparison] = []
    for name in sorted(set(base_mutants) | set(new_mutants)):
        before = base_mutants.get(name)
        after = new_mutants.get(name)
        if before is None:
            assert after is not None  # the name came from one of the two
            items.append(
                MutantComparison(
                    name,
                    ComparisonStatus.MUTANT_CHANGED,
                    after["primary"],
                    (),
                    tuple(after["signature"]),
                    "mutant is not in the baseline",
                )
            )
            continue
        if after is None:
            items.append(
                MutantComparison(
                    name,
                    ComparisonStatus.PRIMARY_LOST,
                    before["primary"],
                    tuple(s for s in before["signature"] if s != before["primary"]),
                    (),
                    "baseline mutant did not run",
                )
            )
            continue
        fired = set(after["signature"])
        recorded = set(before["signature"])
        primary = before["primary"]
        lost_secondary = tuple(sorted(recorded - fired - {primary}))
        new = tuple(sorted(fired - recorded))
        if after["primary"] != primary or primary not in fired:
            status = ComparisonStatus.PRIMARY_LOST
            detail = "the primary gate no longer fires"
        elif lost_secondary:
            status = ComparisonStatus.COVERAGE_REGRESSION
            detail = "secondary detections lost while the primary still fires"
        elif after["edits_sha256"] != before["edits_sha256"]:
            status = ComparisonStatus.MUTANT_CHANGED
            detail = "the mutant's edits changed; re-record the baseline"
        else:
            status = ComparisonStatus.PASS
            detail = ""
        items.append(
            MutantComparison(name, status, primary, lost_secondary, new, detail)
        )
    removed = tuple(sorted(set(baseline["detectors"]) - set(current["detectors"])))
    added = tuple(sorted(set(current["detectors"]) - set(baseline["detectors"])))
    worst = max(
        (item.status for item in items),
        key=_SEVERITY.__getitem__,
        default=ComparisonStatus.PASS,
    )
    return Comparison(worst, tuple(items), removed, added)


__all__ = (
    "EXIT_CODES",
    "SCHEMA",
    "Comparison",
    "ComparisonStatus",
    "MutantComparison",
    "compare_signatures",
    "signature_document",
)
