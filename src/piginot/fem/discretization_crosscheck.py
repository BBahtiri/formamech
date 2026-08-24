"""Cross-discretization evidence that does not claim solver independence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

from piginot.contracts.specs import FrozenMap
from piginot.fem.manufactured import MANUFACTURED_CASE_NAMES
from piginot.fem.mms_runner import solve_mms_level
from piginot.fem.verification_meshes import VerificationMeshKind


@dataclass(frozen=True, slots=True)
class DiscretizationCaseCrosscheck:
    """Same exact problem measured over two orders and mesh families."""

    case_name: str
    level: int
    displacement_l2_errors: Mapping[str, float]
    structured_unstructured_relative_delta: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "displacement_l2_errors",
            FrozenMap.copy_of(self.displacement_l2_errors),
        )


@dataclass(frozen=True, slots=True)
class DiscretizationCrosscheck:
    """A discretization-error cross-check, not an implementation cross-check."""

    cases: tuple[DiscretizationCaseCrosscheck, ...]
    claim_scope: str = "discretization_error_not_implementation_error"


@lru_cache(maxsize=1)
def run_discretization_crosscheck() -> DiscretizationCrosscheck:
    """Compare P1/P2 on deterministic structured and unstructured meshes."""

    level = 3
    cases: list[DiscretizationCaseCrosscheck] = []
    for case_name in MANUFACTURED_CASE_NAMES:
        errors = {
            f"{kind.value}-p{order}": solve_mms_level(
                case_name,
                level=level,
                element_order=order,
                mesh_kind=kind,
            ).norms.displacement_l2_error
            for kind in VerificationMeshKind
            for order in (1, 2)
        }
        structured = errors["structured-p2"]
        unstructured = errors["unstructured-p2"]
        cases.append(
            DiscretizationCaseCrosscheck(
                case_name=case_name,
                level=level,
                displacement_l2_errors=errors,
                structured_unstructured_relative_delta=(
                    abs(structured - unstructured)
                    / max(abs(structured), abs(unstructured))
                ),
            )
        )
    return DiscretizationCrosscheck(cases=tuple(cases))


__all__ = (
    "DiscretizationCaseCrosscheck",
    "DiscretizationCrosscheck",
    "run_discretization_crosscheck",
)
