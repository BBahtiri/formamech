"""Guard the complete selected public roster before geometry or label access."""

from __future__ import annotations

from piginot.compiler.config import CampaignConfig, CompilerConfigError
from piginot.contracts import ProblemSpec
from piginot.doe import (
    EvidenceRole,
    PublishedCase,
    PublishedCaseManifest,
    PublishedCaseStatus,
)
from piginot.morphing.membership import _evaluate_tree


def select_public_cases(
    spec: ProblemSpec, config: CampaignConfig, manifest: PublishedCaseManifest
) -> dict[str, tuple[PublishedCase, ...]]:
    parameters = {parameter.name: parameter for parameter in spec.parameters}
    if set(manifest.primitive_axes) != set(parameters):
        raise CompilerConfigError(
            "source", "manifest parameter axes differ from ProblemSpec"
        )
    by_id = {case.case_id: case for case in manifest.cases}
    phases = {
        "fit": config.source.fit_case_ids,
        "screen": config.source.screen_case_ids,
        "development": config.source.development_case_ids,
    }
    selected: dict[str, tuple[PublishedCase, ...]] = {}
    for phase, ids in phases.items():
        expected = (
            EvidenceRole.DEVELOPMENT if phase == "development" else EvidenceRole.TRAIN
        )
        members = []
        for case_id in ids:
            case = by_id.get(case_id)
            if (
                case is None
                or case.status is not PublishedCaseStatus.ACTIVE
                or case.role is not expected
            ):
                raise CompilerConfigError(
                    "source",
                    f"{case_id}: unavailable or unauthorized public case for {phase}",
                )
            values = dict(case.primitive_coordinates)
            if any(
                not parameter.lo <= values[name] <= parameter.hi
                for name, parameter in parameters.items()
            ):
                raise CompilerConfigError(
                    "source", f"{case_id}: coordinates outside the ProblemSpec domain"
                )
            scope = spec.scope(values)
            try:
                feasible = all(
                    constraint.lo
                    <= _evaluate_tree(constraint.expression.tree, scope)
                    <= constraint.hi
                    for constraint in spec.constraints
                )
            except (ValueError, ZeroDivisionError, OverflowError) as error:
                raise CompilerConfigError(
                    "source", f"{case_id}: invalid constraint values"
                ) from error
            if not feasible:
                raise CompilerConfigError(
                    "source", f"{case_id}: coordinates violate a declared constraint"
                )
            members.append(case)
        selected[phase] = tuple(members)
    groups = [
        set(case.leakage_group_id for case in members) for members in selected.values()
    ]
    if any(left & right for i, left in enumerate(groups) for right in groups[i + 1 :]):
        raise CompilerConfigError(
            "source", "a leakage group crosses fit, screen or development"
        )
    return selected
