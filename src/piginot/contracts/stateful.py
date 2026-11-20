"""Additive pure-Python contracts for stateful / path-dependent problems (Feature F0).

This module intentionally keeps ``piginot.contracts.problem_spec`` (v1) unchanged
and imports no numerical libraries (no ``torch``, ``numpy``, ``scipy``, or ``skfem``).
It wraps a frozen ``ProblemSpec`` with explicit time grids, piecewise load paths
parsed via the safe ``DerivedExpression`` grammar (never ``eval``), material
internal-state schemas, integrator settings, verification policies, and
content-hashed trajectory state bindings.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, cast

from piginot.contracts.canonical import canonical_json_bytes
from piginot.contracts.expressions import DerivedExpression, ExpressionTree
from piginot.contracts.problem_spec import ProblemSpec

STATEFUL_PROBLEM_SCHEMA: Final = "piginot.stateful-problem-spec-v1"
BOUND_STATE_SCHEMA: Final = "piginot.bound-trajectory-state-v1"
STATE_SCHEMA_VERSION: Final = "piginot.material-state-schema-v1"
TIME_CONTROLLERS: Final = ("fixed", "adaptive")
LOAD_SEGMENT_KINDS: Final = ("ramp", "hold", "cycle", "function")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _sha256_canonical(document: object) -> str:
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


def _eval_tree(tree: ExpressionTree, env: Mapping[str, float]) -> float:
    kind = tree[0]
    if kind == "number":
        return float(cast(float, tree[1]))
    if kind == "name":
        name = cast(str, tree[1])
        if name not in env:
            raise KeyError(f"load expression references undefined symbol {name!r}")
        return float(env[name])
    if kind == "unary":
        op = cast(str, tree[1])
        operand = _eval_tree(cast(ExpressionTree, tree[2]), env)
        return operand if op == "+" else -operand
    if kind == "binary":
        op = cast(str, tree[1])
        left = _eval_tree(cast(ExpressionTree, tree[2]), env)
        right = _eval_tree(cast(ExpressionTree, tree[3]), env)
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            if right == 0.0:
                raise ZeroDivisionError("division by zero in load expression")
            return left / right
    raise ValueError(f"unsupported expression AST node {kind!r}")


@dataclass(frozen=True, slots=True)
class TimeGrid:
    """Bounded time horizon and step-size limits for incremental integration."""

    t0: float
    t1: float
    dt_initial: float
    dt_min: float
    dt_max: float
    controller: Literal["fixed", "adaptive"] = "fixed"

    def __post_init__(self) -> None:
        t0 = _finite(self.t0, "t0", nonnegative=True)
        t1 = _finite(self.t1, "t1", positive=True)
        if t1 <= t0:
            raise ValueError("t1 must strictly exceed t0")
        dt_min = _finite(self.dt_min, "dt_min", positive=True)
        dt_init = _finite(self.dt_initial, "dt_initial", positive=True)
        dt_max = _finite(self.dt_max, "dt_max", positive=True)
        if not (dt_min <= dt_init <= dt_max):
            raise ValueError("dt_min <= dt_initial <= dt_max must hold")
        if dt_min > (t1 - t0):
            raise ValueError("dt_min cannot exceed total time horizon t1 - t0")
        if self.controller not in TIME_CONTROLLERS:
            raise ValueError(f"controller must be one of {TIME_CONTROLLERS}")
        object.__setattr__(self, "t0", t0)
        object.__setattr__(self, "t1", t1)
        object.__setattr__(self, "dt_min", dt_min)
        object.__setattr__(self, "dt_initial", dt_init)
        object.__setattr__(self, "dt_max", dt_max)

    def fixed_times(self) -> tuple[float, ...]:
        """Deterministic uniform-plus-final time nodes starting at ``t0``."""
        times = [self.t0]
        step = 1
        while True:
            t = self.t0 + step * self.dt_initial
            if t >= self.t1 - 1e-12 * max(1.0, self.t1 - self.t0):
                times.append(self.t1)
                break
            times.append(t)
            step += 1
        return tuple(times)

    def document(self) -> dict[str, Any]:
        return {
            "t0": self.t0,
            "t1": self.t1,
            "dt_initial": self.dt_initial,
            "dt_min": self.dt_min,
            "dt_max": self.dt_max,
            "controller": self.controller,
        }


@dataclass(frozen=True, slots=True)
class LoadSegment:
    """One piecewise segment of a time-dependent boundary load path.

    Boundary expressions use the safe :class:`DerivedExpression` AST and may
    reference ``t`` (current time), ``s`` (local normalized progress in ``[0, 1]``),
    and declared problem parameters.
    """

    start: float
    end: float
    kind: Literal["ramp", "hold", "cycle", "function"]
    boundary_values: Mapping[str, DerivedExpression]

    def __post_init__(self) -> None:
        start = _finite(self.start, "segment.start", nonnegative=True)
        end = _finite(self.end, "segment.end", positive=True)
        if end <= start:
            raise ValueError("segment.end must strictly exceed segment.start")
        if self.kind not in LOAD_SEGMENT_KINDS:
            raise ValueError(f"segment.kind must be one of {LOAD_SEGMENT_KINDS}")
        if not isinstance(self.boundary_values, Mapping) or not self.boundary_values:
            raise ValueError("boundary_values must be a non-empty mapping")
        cleaned: dict[str, DerivedExpression] = {}
        for key, expr in sorted(self.boundary_values.items()):
            if not isinstance(key, str) or not key.strip():
                raise ValueError("boundary_values keys must be non-empty strings")
            if isinstance(expr, str):
                parsed = DerivedExpression.parse(expr)
            elif isinstance(expr, DerivedExpression):
                parsed = expr
            else:
                raise TypeError(
                    f"boundary expression for {key!r} must be DerivedExpression or str"
                )
            cleaned[key] = parsed
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "boundary_values", cleaned)

    def load_at(
        self, t: float, parameters: Mapping[str, float] | None = None
    ) -> dict[str, float]:
        """Safely compute all boundary expressions at time ``t``."""
        t_val = _finite(t, "t")
        tol = 1e-12 * max(1.0, self.end - self.start)
        if t_val < self.start - tol or t_val > self.end + tol:
            raise ValueError(
                f"time t={t_val} lies outside segment [{self.start}, {self.end}]"
            )
        clamped = min(max(t_val, self.start), self.end)
        local_s = (clamped - self.start) / (self.end - self.start)
        env: dict[str, float] = {
            **(
                {k: _finite(v, f"param.{k}") for k, v in parameters.items()}
                if parameters
                else {}
            ),
            "t": clamped,
            "s": local_s,
        }
        return {
            name: _eval_tree(expr.canonical_ast, env)
            for name, expr in self.boundary_values.items()
        }

    def document(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "kind": self.kind,
            "boundary_values": {
                k: v.canonical_ast for k, v in sorted(self.boundary_values.items())
            },
        }


def evaluate_load_path(
    segments: Sequence[LoadSegment],
    t: float,
    parameters: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Evaluate a validated contiguous sequence of ``LoadSegment``s at time ``t``."""
    if not segments:
        raise ValueError("load_path must contain at least one LoadSegment")
    t_val = _finite(t, "t")
    tol = 1e-12 * max(1.0, segments[-1].end - segments[0].start)
    for seg in segments:
        if seg.start - tol <= t_val <= seg.end + tol:
            return seg.load_at(t_val, parameters)
    raise ValueError(
        f"time t={t_val} outside load path [{segments[0].start}, {segments[-1].end}]"
    )


@dataclass(frozen=True, slots=True)
class StateVariableSpec:
    """Declarative specification of one constitutive internal variable."""

    name: str
    shape: tuple[int, ...]
    unit: str
    initial_rule: Literal["zeros", "identity"] = "zeros"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError(f"state variable name {self.name!r} must be an identifier")
        if type(self.shape) is not tuple or any(
            type(d) is not int or d <= 0 for d in self.shape
        ):
            raise ValueError("state variable shape must be a tuple of positive ints")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("unit must be non-empty text")
        if self.initial_rule not in ("zeros", "identity"):
            raise ValueError("initial_rule must be 'zeros' or 'identity'")

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "unit": self.unit,
            "initial_rule": self.initial_rule,
        }


@dataclass(frozen=True, slots=True)
class StateSchema:
    """Versioned schema of all quadrature-point internal state variables."""

    model_id: str
    variables: tuple[StateVariableSpec, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be non-empty text")
        names = [v.name for v in self.variables]
        if len(set(names)) != len(names):
            raise ValueError("state variable names must be unique")

    def document(self) -> dict[str, Any]:
        return {
            "schema": STATE_SCHEMA_VERSION,
            "model_id": self.model_id,
            "variables": [v.document() for v in self.variables],
        }

    @property
    def schema_hash(self) -> str:
        return _sha256_canonical(self.document())


@dataclass(frozen=True, slots=True)
class IntegratorSpec:
    """Incremental nonlinear solver and time-integration policy."""

    method: Literal["backward-euler", "exponential-map"] = "backward-euler"
    residual_tolerance: float = 1e-9
    max_newton_iterations: int = 25
    max_substep_halvings: int = 4
    plane_stress_max_iterations: int = 25

    def __post_init__(self) -> None:
        if self.method not in ("backward-euler", "exponential-map"):
            raise ValueError(f"unsupported integrator method {self.method!r}")
        _finite(self.residual_tolerance, "residual_tolerance", positive=True)
        for label in (
            "max_newton_iterations",
            "max_substep_halvings",
            "plane_stress_max_iterations",
        ):
            val = getattr(self, label)
            if type(val) is not int or val < 0:
                raise ValueError(f"{label} must be a nonnegative integer")
        if self.max_newton_iterations < 1:
            raise ValueError("max_newton_iterations must be at least 1")

    def document(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "residual_tolerance": self.residual_tolerance,
            "max_newton_iterations": self.max_newton_iterations,
            "max_substep_halvings": self.max_substep_halvings,
            "plane_stress_max_iterations": self.plane_stress_max_iterations,
        }


@dataclass(frozen=True, slots=True)
class StatefulVerificationPolicy:
    """Thermodynamic, admissibility, and fail-closed verification rules."""

    dissipation_tolerance: float = 1e-12
    min_det_f: float = 1e-6
    tangent_rel_tolerance: float = 1e-4
    time_refinement_rel_tolerance: float = 0.02
    allow_fem_fallback: bool = True
    fail_closed_on_violation: bool = True

    def __post_init__(self) -> None:
        _finite(self.dissipation_tolerance, "dissipation_tolerance", nonnegative=True)
        _finite(self.min_det_f, "min_det_f", positive=True)
        _finite(self.tangent_rel_tolerance, "tangent_rel_tolerance", positive=True)
        _finite(
            self.time_refinement_rel_tolerance,
            "time_refinement_rel_tolerance",
            positive=True,
        )

    def document(self) -> dict[str, Any]:
        return {
            "dissipation_tolerance": self.dissipation_tolerance,
            "min_det_f": self.min_det_f,
            "tangent_rel_tolerance": self.tangent_rel_tolerance,
            "time_refinement_rel_tolerance": self.time_refinement_rel_tolerance,
            "allow_fem_fallback": self.allow_fem_fallback,
            "fail_closed_on_violation": self.fail_closed_on_violation,
        }


@dataclass(frozen=True, slots=True)
class StatefulProblemSpec:
    """Additive stateful wrapper around a frozen v1 ``ProblemSpec``."""

    base: ProblemSpec
    time_grid: TimeGrid
    load_path: tuple[LoadSegment, ...]
    material_state_schema: StateSchema
    integrator: IntegratorSpec = IntegratorSpec()
    verification: StatefulVerificationPolicy = StatefulVerificationPolicy()

    def __post_init__(self) -> None:
        if not isinstance(self.base, ProblemSpec):
            raise TypeError("base must be a ProblemSpec")
        if not isinstance(self.time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid")
        if not self.load_path:
            raise ValueError("load_path must contain at least one LoadSegment")
        tol = 1e-12 * max(1.0, self.time_grid.t1 - self.time_grid.t0)
        if abs(self.load_path[0].start - self.time_grid.t0) > tol:
            raise ValueError("load_path must begin at time_grid.t0")
        if abs(self.load_path[-1].end - self.time_grid.t1) > tol:
            raise ValueError("load_path must end at time_grid.t1")
        for prev, curr in zip(self.load_path, self.load_path[1:], strict=False):
            if abs(curr.start - prev.end) > tol:
                raise ValueError("load_path segments must be contiguous without gaps")

    @property
    def load_path_hash(self) -> str:
        return _sha256_canonical([seg.document() for seg in self.load_path])

    @property
    def material_schema_hash(self) -> str:
        return self.material_state_schema.schema_hash

    def document(self) -> dict[str, Any]:
        return {
            "schema": STATEFUL_PROBLEM_SCHEMA,
            "base_problem_hash": self.base.problem_hash,
            "time_grid": self.time_grid.document(),
            "load_path": [seg.document() for seg in self.load_path],
            "load_path_hash": self.load_path_hash,
            "material_state_schema": self.material_state_schema.document(),
            "material_schema_hash": self.material_schema_hash,
            "integrator": self.integrator.document(),
            "verification": self.verification.document(),
        }

    @property
    def stateful_problem_hash(self) -> str:
        return _sha256_canonical(self.document())


@dataclass(frozen=True, slots=True)
class BoundTrajectoryState:
    """Immutable provenance identity binding an inelastic history to its exact prefix."""

    problem_spec_hash: str
    mesh_hash: str
    material_schema_hash: str
    load_path_hash: str
    time_index: int
    time_value: float
    step_controller_version: str
    state_hash: str

    def __post_init__(self) -> None:
        for label in (
            "problem_spec_hash",
            "mesh_hash",
            "material_schema_hash",
            "load_path_hash",
            "state_hash",
        ):
            val = getattr(self, label)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{label} must be non-empty text")
        if type(self.time_index) is not int or self.time_index < 0:
            raise ValueError("time_index must be a nonnegative integer")
        _finite(self.time_value, "time_value", nonnegative=True)
        if (
            not isinstance(self.step_controller_version, str)
            or not self.step_controller_version.strip()
        ):
            raise ValueError("step_controller_version must be non-empty text")

    def verify_compatible(
        self,
        *,
        problem_spec_hash: str,
        mesh_hash: str,
        material_schema_hash: str,
        load_path_hash: str,
        step_controller_version: str,
    ) -> tuple[bool, str]:
        """Refuse resuming a history if geometry, material, load prefix, or policy changed."""
        checks = (
            ("problem_spec_hash", self.problem_spec_hash, problem_spec_hash),
            ("mesh_hash", self.mesh_hash, mesh_hash),
            ("material_schema_hash", self.material_schema_hash, material_schema_hash),
            ("load_path_hash", self.load_path_hash, load_path_hash),
            (
                "step_controller_version",
                self.step_controller_version,
                step_controller_version,
            ),
        )
        for label, expected, actual in checks:
            if expected != actual:
                return (
                    False,
                    f"trajectory binding mismatch on {label}: {expected!r} != {actual!r}",
                )
        return True, "binding verified"

    def document(self) -> dict[str, Any]:
        return {
            "schema": BOUND_STATE_SCHEMA,
            "problem_spec_hash": self.problem_spec_hash,
            "mesh_hash": self.mesh_hash,
            "material_schema_hash": self.material_schema_hash,
            "load_path_hash": self.load_path_hash,
            "time_index": self.time_index,
            "time_value": self.time_value,
            "step_controller_version": self.step_controller_version,
            "state_hash": self.state_hash,
        }

    @property
    def binding_hash(self) -> str:
        return _sha256_canonical(self.document())


__all__ = (
    "BOUND_STATE_SCHEMA",
    "LOAD_SEGMENT_KINDS",
    "STATEFUL_PROBLEM_SCHEMA",
    "STATE_SCHEMA_VERSION",
    "TIME_CONTROLLERS",
    "BoundTrajectoryState",
    "IntegratorSpec",
    "LoadSegment",
    "StateSchema",
    "StateVariableSpec",
    "StatefulProblemSpec",
    "StatefulVerificationPolicy",
    "TimeGrid",
    "evaluate_load_path",
)
