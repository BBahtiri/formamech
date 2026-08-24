"""Solved primary-FEM convergence studies for the S8 manufactured cases."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from piginot.fem.error_estimation import ObservedOrderResult, observed_loglog_order
from piginot.fem.manufactured import MANUFACTURED_CASE_NAMES
from piginot.fem.mms_runner import MmsLevelResult, solve_mms_level
from piginot.fem.verification_meshes import (
    VERIFICATION_LEVELS,
    VerificationMeshKind,
)


@dataclass(frozen=True, slots=True)
class MmsConvergenceStudy:
    """Four refinement levels and fitted true-error orders for one case/order."""

    case_name: str
    element_order: int
    mesh_kind: VerificationMeshKind
    levels: tuple[MmsLevelResult, ...]
    displacement_l2_order: ObservedOrderResult
    displacement_h1_order: ObservedOrderResult
    first_piola_l2_order: ObservedOrderResult
    all_level_displacement_l2_fit: ObservedOrderResult
    all_level_displacement_h1_fit: ObservedOrderResult
    all_level_first_piola_l2_fit: ObservedOrderResult


@lru_cache(maxsize=16)
def run_mms_convergence_study(
    case_name: str,
    element_order: int,
    mesh_kind: VerificationMeshKind = VerificationMeshKind.STRUCTURED,
) -> MmsConvergenceStudy:
    """Run the declared four-level nonlinear convergence population."""

    if element_order not in (1, 2):
        raise ValueError("element_order must be 1 or 2")
    levels = tuple(
        solve_mms_level(
            case_name,
            level=level,
            element_order=element_order,
            mesh_kind=mesh_kind,
        )
        for level in VERIFICATION_LEVELS
    )
    mesh_sizes = tuple(level.h_max_mm for level in levels)
    displacement_l2_errors = tuple(
        level.norms.displacement_l2_error for level in levels
    )
    displacement_h1_errors = tuple(
        level.norms.displacement_h1_seminorm_error for level in levels
    )
    first_piola_errors = tuple(level.norms.first_piola_l2_error for level in levels)
    return MmsConvergenceStudy(
        case_name=case_name,
        element_order=element_order,
        mesh_kind=mesh_kind,
        levels=levels,
        displacement_l2_order=observed_loglog_order(
            mesh_sizes[-2:], displacement_l2_errors[-2:]
        ),
        displacement_h1_order=observed_loglog_order(
            mesh_sizes[-2:], displacement_h1_errors[-2:]
        ),
        first_piola_l2_order=observed_loglog_order(
            mesh_sizes[-2:], first_piola_errors[-2:]
        ),
        all_level_displacement_l2_fit=observed_loglog_order(
            mesh_sizes, displacement_l2_errors
        ),
        all_level_displacement_h1_fit=observed_loglog_order(
            mesh_sizes, displacement_h1_errors
        ),
        all_level_first_piola_l2_fit=observed_loglog_order(
            mesh_sizes, first_piola_errors
        ),
    )


@lru_cache(maxsize=1)
def run_all_mms_convergence_studies() -> tuple[MmsConvergenceStudy, ...]:
    """Run P1 and P2 studies for all four required manufactured families."""

    return tuple(
        run_mms_convergence_study(case_name, element_order)
        for case_name in MANUFACTURED_CASE_NAMES
        for element_order in (1, 2)
    )


__all__ = (
    "MmsConvergenceStudy",
    "run_all_mms_convergence_studies",
    "run_mms_convergence_study",
)
