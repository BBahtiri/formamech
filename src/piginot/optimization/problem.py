"""Declarative design optimization contracts (Feature E / P5).

Pure standard-library + canonical JSON contracts (zero FEM/ML imports) that
specify bounded design variables, objectives, inequality/equality constraints,
and trust-region / FEM-verification budgets as immutable, hash-bound artifacts.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal

from piginot.contracts import canonical_json_bytes

DESIGN_PROBLEM_SCHEMA: Final = "piginot.design-problem-v1"
DESIGN_REPORT_SCHEMA: Final = "piginot.design-optimization-report-v1"
CONSTRAINT_KINDS: Final = ("inequality_le", "inequality_ge", "equality")


def _sha256_doc(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _finite(
    value: object, label: str, *, positive: bool = False, nonnegative: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    num = float(value)
    if not math.isfinite(num):
        raise ValueError(f"{label} must be finite")
    if positive and num <= 0.0:
        raise ValueError(f"{label} must be strictly positive")
    if nonnegative and num < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return num


@dataclass(frozen=True, slots=True)
class DesignParameter:
    """One bounded design variable with explicit physical unit and initial point."""

    name: str
    lower: float
    upper: float
    initial: float
    unit: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError(f"parameter name {self.name!r} must be an identifier")
        lo = _finite(self.lower, f"{self.name}.lower")
        hi = _finite(self.upper, f"{self.name}.upper")
        init = _finite(self.initial, f"{self.name}.initial")
        if hi <= lo:
            raise ValueError(f"{self.name}: upper must strictly exceed lower")
        if not (lo <= init <= hi):
            raise ValueError(f"{self.name}: initial must lie within [lower, upper]")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError(f"{self.name}: unit must be non-empty text")
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)
        object.__setattr__(self, "initial", init)

    @property
    def span(self) -> float:
        return self.upper - self.lower

    def clamp(self, value: float) -> float:
        return min(max(float(value), self.lower), self.upper)

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lower": self.lower,
            "upper": self.upper,
            "initial": self.initial,
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class DesignConstraint:
    """Scalar inequality or equality constraint on design response quantities."""

    name: str
    kind: Literal["inequality_le", "inequality_ge", "equality"]
    limit: float
    tolerance: float = 1e-6
    unit: str = "1"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("constraint name must be non-empty text")
        if self.kind not in CONSTRAINT_KINDS:
            raise ValueError(f"constraint kind must be one of {CONSTRAINT_KINDS}")
        object.__setattr__(self, "limit", _finite(self.limit, f"{self.name}.limit"))
        object.__setattr__(
            self,
            "tolerance",
            _finite(self.tolerance, f"{self.name}.tolerance", nonnegative=True),
        )
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("constraint unit must be non-empty text")

    def violation(self, observed: float) -> float:
        """Return signed violation (> 0 means violated beyond tolerance)."""
        val = _finite(observed, f"observed.{self.name}")
        if self.kind == "inequality_le":
            return max(0.0, val - self.limit - self.tolerance)
        if self.kind == "inequality_ge":
            return max(0.0, self.limit - val - self.tolerance)
        return max(0.0, abs(val - self.limit) - self.tolerance)

    def satisfied(self, observed: float) -> bool:
        return self.violation(observed) == 0.0

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "limit": self.limit,
            "tolerance": self.tolerance,
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class DesignProblem:
    """Declarative FEM-guarded design optimization problem contract."""

    name: str
    geometry_version: str
    problem_spec_hash: str
    parameters: tuple[DesignParameter, ...]
    objective_name: str
    objective_unit: str = "1"
    minimize: bool = True
    constraints: tuple[DesignConstraint, ...] = ()
    max_iterations: int = 25
    max_fem_evaluations: int = 12
    initial_trust_radius: float = 0.25
    min_trust_radius: float = 0.01
    gradient_check_interval: int = 4
    gradient_cos_min: float = 0.25
    accept_eta_tolerance: float = 0.05
    constraint_penalty_weight: float = 50.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be non-empty text")
        if (
            not isinstance(self.geometry_version, str)
            or not self.geometry_version.strip()
        ):
            raise ValueError("geometry_version must be non-empty text")
        if (
            not isinstance(self.problem_spec_hash, str)
            or not self.problem_spec_hash.strip()
        ):
            raise ValueError("problem_spec_hash must be non-empty text")
        if not self.parameters:
            raise ValueError("parameters must contain at least one DesignParameter")
        p_names = [p.name for p in self.parameters]
        if len(set(p_names)) != len(p_names):
            raise ValueError("design parameter names must be unique")
        c_names = [c.name for c in self.constraints]
        if len(set(c_names)) != len(c_names):
            raise ValueError("design constraint names must be unique")
        if type(self.max_iterations) is not int or self.max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")
        if type(self.max_fem_evaluations) is not int or self.max_fem_evaluations < 1:
            raise ValueError("max_fem_evaluations must be at least 1")
        r_init = _finite(
            self.initial_trust_radius, "initial_trust_radius", positive=True
        )
        r_min = _finite(self.min_trust_radius, "min_trust_radius", positive=True)
        if r_min > r_init:
            raise ValueError("min_trust_radius cannot exceed initial_trust_radius")
        if (
            type(self.gradient_check_interval) is not int
            or self.gradient_check_interval <= 0
        ):
            raise ValueError("gradient_check_interval must be a positive integer")
        cos_min = _finite(self.gradient_cos_min, "gradient_cos_min")
        if not (-1.0 <= cos_min <= 1.0):
            raise ValueError("gradient_cos_min must lie in [-1, 1]")
        _finite(self.accept_eta_tolerance, "accept_eta_tolerance", positive=True)
        _finite(
            self.constraint_penalty_weight, "constraint_penalty_weight", positive=True
        )
        object.__setattr__(self, "initial_trust_radius", r_init)
        object.__setattr__(self, "min_trust_radius", r_min)
        object.__setattr__(self, "gradient_cos_min", cos_min)

    def initial_point(self) -> dict[str, float]:
        return {p.name: p.initial for p in self.parameters}

    def clamp_point(self, point: Mapping[str, float]) -> dict[str, float]:
        return {p.name: p.clamp(point[p.name]) for p in self.parameters}

    def check_constraints(
        self, values: Mapping[str, float]
    ) -> tuple[bool, dict[str, float]]:
        violations: dict[str, float] = {}
        for con in self.constraints:
            if con.name not in values:
                raise KeyError(f"missing constraint response {con.name!r}")
            violations[con.name] = con.violation(values[con.name])
        return all(v == 0.0 for v in violations.values()), violations

    def document(self) -> dict[str, Any]:
        return {
            "schema": DESIGN_PROBLEM_SCHEMA,
            "name": self.name,
            "geometry_version": self.geometry_version,
            "problem_spec_hash": self.problem_spec_hash,
            "parameters": [p.document() for p in self.parameters],
            "objective_name": self.objective_name,
            "objective_unit": self.objective_unit,
            "minimize": self.minimize,
            "constraints": [c.document() for c in self.constraints],
            "max_iterations": self.max_iterations,
            "max_fem_evaluations": self.max_fem_evaluations,
            "initial_trust_radius": self.initial_trust_radius,
            "min_trust_radius": self.min_trust_radius,
            "gradient_check_interval": self.gradient_check_interval,
            "gradient_cos_min": self.gradient_cos_min,
            "accept_eta_tolerance": self.accept_eta_tolerance,
            "constraint_penalty_weight": self.constraint_penalty_weight,
        }

    @property
    def problem_hash(self) -> str:
        return _sha256_doc(self.document())


__all__ = (
    "CONSTRAINT_KINDS",
    "DESIGN_PROBLEM_SCHEMA",
    "DESIGN_REPORT_SCHEMA",
    "DesignConstraint",
    "DesignParameter",
    "DesignProblem",
)
