"""Recompute a recorded S14 budget from its own per-case records.

A recorded DEVELOPMENT or HOLDOUT evaluation holds, per case, the surrogate
term, the discretisation and projection components and the slice memberships,
and, per slice and quantity, the gated and reported budget rows. This module
rebuilds the rows from the cases with ``piginot.verification.budget`` and
compares them with the recorded rows exactly: same slices, case counts,
tolerances, statuses and bit-identical statistics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from piginot.verification.budget import (
    ALL_CASES_SLICE,
    BudgetTolerance,
    CaseBudgetEntry,
    SliceBudgetResult,
    evaluate_budget,
)

ROW_KINDS = ("gated", "reported")


@dataclass(frozen=True, slots=True)
class BudgetRecordMismatch:
    """One recorded row that the recomputation does not reproduce."""

    kind: str
    slice_id: str
    quantity: str
    field: str
    recorded: object
    recomputed: object


def _tolerances(rows: Sequence[Mapping[str, Any]]) -> tuple[BudgetTolerance, ...]:
    by_quantity: dict[str, float] = {}
    for row in rows:
        quantity, tolerance = row["quantity"], float(row["tolerance"])
        if by_quantity.setdefault(quantity, tolerance) != tolerance:
            raise ValueError(f"{quantity} carries two tolerances")
    return tuple(BudgetTolerance(q, t) for q, t in by_quantity.items())


def _slice_ids(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    ordered: dict[str, None] = {ALL_CASES_SLICE: None}
    for row in rows:
        ordered.setdefault(row["slice"], None)
    return tuple(ordered)


def budget_entries(
    record: Mapping[str, Any], kind: str = "gated"
) -> tuple[CaseBudgetEntry, ...]:
    """The per-case budget entries of a record, for gated or reported slices."""

    if kind not in ROW_KINDS:
        raise ValueError(f"kind must be one of {ROW_KINDS}")
    entries = []
    for case in record["cases"]:
        if kind == "gated":
            slices = tuple(case["slices"])
        else:
            slices = (ALL_CASES_SLICE, *case["reported_slices"])
        components = {
            quantity: {
                "surrogate": case["surrogate"][quantity],
                "discretisation": parts["discretisation"],
                "projection": parts["projection"],
            }
            for quantity, parts in case["components"].items()
        }
        entries.append(CaseBudgetEntry(case["case_id"], slices, components))
    return tuple(entries)


def recompute_budget_record(
    record: Mapping[str, Any],
) -> dict[str, tuple[SliceBudgetResult, ...]]:
    """Recompute the gated and reported rows of a recorded evaluation."""

    budget = record["budget"]
    result: dict[str, tuple[SliceBudgetResult, ...]] = {}
    for kind in ROW_KINDS:
        rows = budget.get(kind, [])
        if not rows:
            result[kind] = ()
            continue
        # evaluate_budget always needs the "all" slice; drop its rows again
        # when the record does not list them (the reported rows do not).
        recorded_all = any(row["slice"] == ALL_CASES_SLICE for row in rows)
        result[kind] = tuple(
            row
            for row in evaluate_budget(
                budget_entries(record, kind), _tolerances(rows), _slice_ids(rows)
            )
            if recorded_all or row.slice_id != ALL_CASES_SLICE
        )
    return result


def compare_budget_record(record: Mapping[str, Any]) -> list[BudgetRecordMismatch]:
    """Every difference between the recorded and the recomputed rows (exact)."""

    recomputed = recompute_budget_record(record)
    mismatches: list[BudgetRecordMismatch] = []
    for kind in ROW_KINDS:
        recorded_rows = {
            (row["slice"], row["quantity"]): row
            for row in record["budget"].get(kind, [])
        }
        rebuilt = {(row.slice_id, row.quantity): row for row in recomputed[kind]}
        for key in sorted(set(recorded_rows) | set(rebuilt)):
            slice_id, quantity = key
            old, new = recorded_rows.get(key), rebuilt.get(key)
            if old is None or new is None:
                mismatches.append(
                    BudgetRecordMismatch(
                        kind,
                        slice_id,
                        quantity,
                        "row",
                        old is not None,
                        new is not None,
                    )
                )
                continue
            pairs = (
                ("cases", old["cases"], new.case_count),
                ("statistic", old["statistic"], new.statistic),
                ("tolerance", old["tolerance"], new.tolerance),
                ("status", old["status"], new.status.value),
            )
            mismatches.extend(
                BudgetRecordMismatch(kind, slice_id, quantity, name, a, b)
                for name, a, b in pairs
                if a != b
            )
    return mismatches
