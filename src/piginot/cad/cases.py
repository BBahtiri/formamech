"""Seeded public TRAIN-fit/screen cases for one fixed CAD geometry."""

from __future__ import annotations

import hashlib
import math
import random

from piginot.contracts import ProblemSpec, canonical_json_bytes
from piginot.doe import (
    EvidenceRole,
    PublishedCase,
    PublishedCaseManifest,
    PublishedCaseStatus,
)
from piginot.morphing.membership import _evaluate_tree

ATTEMPTS_PER_CASE = 1000


def generate_cases(
    spec: ProblemSpec, fit_cases: int, screen_cases: int, seed: int
) -> tuple[PublishedCaseManifest, tuple[str, ...], tuple[str, ...]]:
    generator = random.Random(seed)
    total = fit_cases + screen_cases
    accepted: list[PublishedCase] = []
    groups: set[str] = set()
    for _ in range(ATTEMPTS_PER_CASE * total):
        values = {
            p.name: generator.uniform(p.lo, p.hi) if p.lo < p.hi else p.lo
            for p in spec.parameters
        }
        try:
            derived = {
                c.name: _evaluate_tree(c.expression.tree, spec.scope(values))
                for c in spec.constraints
            }
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        if not all(
            math.isfinite(derived[c.name]) and c.lo <= derived[c.name] <= c.hi
            for c in spec.constraints
        ):
            continue
        group = hashlib.sha256(canonical_json_bytes(values)).hexdigest()
        if group in groups:
            continue
        index = len(accepted)
        case_id = (
            f"fit-{index + 1:06d}"
            if index < fit_cases
            else f"screen-{index - fit_cases + 1:06d}"
        )
        accepted.append(
            PublishedCase(
                case_id,
                group,
                EvidenceRole.TRAIN,
                values,
                derived,
                {},
                PublishedCaseStatus.ACTIVE,
            )
        )
        groups.add(group)
        if len(accepted) == total:
            manifest = PublishedCaseManifest(
                f"{spec.name}-cad-train-v1",
                tuple(p.name for p in spec.parameters),
                tuple(c.name for c in spec.constraints),
                (),
                tuple(accepted),
            )
            return (
                manifest,
                tuple(case.case_id for case in accepted[:fit_cases]),
                tuple(case.case_id for case in accepted[fit_cases:]),
            )
    raise ValueError(
        "CAD sampling exhausted its finite attempt budget; ranges or constraints cannot supply distinct cases"
    )
