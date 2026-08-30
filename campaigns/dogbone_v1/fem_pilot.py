"""Public FEM cost and solvability pilot for the ratified DogBone family."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

from campaigns.dogbone_v1.family import compiled_family, parameter_point
from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalQuantity,
    ExecutionProfile,
    MaterialSpec,
    Unit,
    canonical_json_bytes,
    hash_spec,
)
from piginot.contracts.errors import NonFiniteNumberError
from piginot.fem import (
    FemAttempt,
    FemBackend,
    FemFailureCode,
    FemRequest,
    FemSolverConfig,
)
from piginot.morphing import apply_morph

PILOT_ID = "dogbone-fem-solvability-20260829-v1"
SELECTION_POLICY_ID = "reference-s6-extremes-and-boundary-sentinels-v1"
_REPORTED_TIME_DECIMALS = 6
_ALLOWED_SENTINELS = frozenset({"reference", "difficult", "boundary"})


@dataclass(frozen=True, slots=True)
class PilotGeometryCase:
    """One public geometry selected before any FEM outcome is observed."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    case_id: str
    parameters: tuple[float, float, float, float]
    sentinel: str

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("case_id must be non-empty text")
        parameters = tuple(float(value) for value in self.parameters)
        if len(parameters) != 4:
            raise ValueError("pilot geometry parameters must contain four values")
        if any(not math.isfinite(value) for value in parameters):
            raise NonFiniteNumberError("pilot geometry parameters must be finite")
        if self.sentinel not in _ALLOWED_SENTINELS:
            raise ValueError("sentinel must be reference, difficult, or boundary")
        object.__setattr__(self, "parameters", parameters)


@dataclass(frozen=True, slots=True)
class FemPilotRecord:
    """Visible outcome and cost diagnostics for one pilot request."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    case_id: str
    parameters: tuple[float, float, float, float]
    sentinel: str
    request_id: str
    mesh_id: str
    recipe_hash: str
    success: bool
    result_hash: str | None
    failure_code: FemFailureCode | None
    message: str
    elapsed_seconds: float
    scalar_dof_count: int | None
    canonical_result_bytes: int | None
    newton_iterations_by_load_step: tuple[int, ...]
    final_residual_norm: float | None

    def __post_init__(self) -> None:
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0:
            raise NonFiniteNumberError(
                "pilot elapsed time must be finite and non-negative"
            )
        if self.success:
            if self.result_hash is None or self.failure_code is not None:
                raise ValueError(
                    "successful pilot record has inconsistent outcome fields"
                )
            if self.scalar_dof_count is None or self.scalar_dof_count <= 0:
                raise ValueError("successful pilot record must declare scalar DOFs")
            if self.canonical_result_bytes is None or self.canonical_result_bytes <= 0:
                raise ValueError("successful pilot record must declare result bytes")
            if not self.newton_iterations_by_load_step:
                raise ValueError(
                    "successful pilot record must retain Newton diagnostics"
                )
            if self.final_residual_norm is None:
                raise ValueError("successful pilot record must retain final residual")
        elif (
            self.result_hash is not None
            or self.failure_code is None
            or self.scalar_dof_count is not None
            or self.canonical_result_bytes is not None
            or self.newton_iterations_by_load_step
            or self.final_residual_norm is not None
        ):
            raise ValueError("failed pilot record has inconsistent outcome fields")


@dataclass(frozen=True, slots=True)
class FemPilotSummary:
    """Finite-population solvability and wall-time summary."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    attempted_count: int
    success_count: int
    failure_counts: Mapping[str, int]
    total_runtime_seconds: float
    median_runtime_seconds: float
    p95_runtime_seconds: float
    max_runtime_seconds: float
    total_result_bytes: int
    max_result_bytes: int
    max_newton_iterations_observed: int
    max_final_residual_norm: float | None

    def __post_init__(self) -> None:
        failures = dict(sorted(self.failure_counts.items()))
        if self.attempted_count <= 0:
            raise ValueError("pilot must contain at least one attempt")
        if self.success_count < 0 or self.success_count > self.attempted_count:
            raise ValueError("pilot success count is outside the attempted population")
        if sum(failures.values()) != self.attempted_count - self.success_count:
            raise ValueError("pilot failure counts do not reconcile")
        object.__setattr__(self, "failure_counts", MappingProxyType(failures))


@dataclass(frozen=True, slots=True)
class FemPilotReport:
    """Canonical public evidence from the non-secret DogBone FEM pilot."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    pilot_id: str
    selection_policy_id: str
    compiled_family_hash: str
    reference_mesh_id: str
    validity_policy_hash: str
    material: MaterialSpec
    prescribed_displacement: CanonicalQuantity
    solver_config: FemSolverConfig
    execution_profile: ExecutionProfile
    runtime_environment: Mapping[str, str]
    backend_name: str
    records: tuple[FemPilotRecord, ...]
    summary: FemPilotSummary
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        records = tuple(self.records)
        runtime_environment = dict(sorted(self.runtime_environment.items()))
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in runtime_environment.items()
        ):
            raise ValueError("runtime environment entries must be non-empty text")
        if len(records) != self.summary.attempted_count:
            raise ValueError("pilot records and summary population do not match")
        if tuple(record.case_id for record in records) != tuple(
            case.case_id for case in pilot_geometry_cases()
        ):
            raise ValueError("pilot record order does not match the selection policy")
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported FEM pilot schema {self.schema_version!r}")
        object.__setattr__(self, "records", records)
        object.__setattr__(
            self,
            "runtime_environment",
            MappingProxyType(runtime_environment),
        )


_PILOT_GEOMETRIES = (
    PilotGeometryCase("reference", (54.0, 20.0, 10.0, 12.0), "reference"),
    PilotGeometryCase(
        "validation_worst_angle",
        (
            68.15490487612537,
            22.834462189194603,
            6.10723695220005,
            11.549413934481429,
        ),
        "difficult",
    ),
    PilotGeometryCase(
        "validation_worst_jacobian",
        (
            42.45915219995253,
            24.61720261327944,
            6.381824787250642,
            19.47941478408407,
        ),
        "difficult",
    ),
    PilotGeometryCase(
        "validation_worst_feature",
        (
            40.62180736726733,
            25.363441376354437,
            7.337443874975097,
            18.042085735135643,
        ),
        "difficult",
    ),
    PilotGeometryCase("wide_gauge_short", (40.0, 16.0, 14.0, 20.0), "boundary"),
    PilotGeometryCase("narrow_gauge_long", (70.0, 26.0, 6.0, 20.0), "boundary"),
    PilotGeometryCase("narrow_gauge_short", (40.0, 26.0, 6.0, 20.0), "boundary"),
    PilotGeometryCase("wide_gauge_long", (70.0, 16.0, 14.0, 8.0), "boundary"),
    PilotGeometryCase("low_dimensions", (40.0, 16.0, 6.0, 8.0), "boundary"),
    PilotGeometryCase("arc_margin", (70.0, 26.0, 14.0, 8.11), "difficult"),
    PilotGeometryCase("high_concentration_long", (70.0, 26.0, 6.0, 14.0), "difficult"),
    PilotGeometryCase("high_concentration_short", (40.0, 24.0, 6.0, 14.0), "difficult"),
)


def pilot_geometry_cases() -> tuple[PilotGeometryCase, ...]:
    """Return the fixed public pilot population in preregistered order."""

    return _PILOT_GEOMETRIES


def _pilot_material() -> MaterialSpec:
    return MaterialSpec(
        "dogbone-pinned-reference-material-v1",
        "finite_strain_plane_stress_neo_hookean",
        {
            "elastic_modulus": CanonicalQuantity(760.0, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(0.23, Unit.ONE),
        },
    )


def _pilot_boundary_conditions() -> tuple[BoundaryConditionSpec, ...]:
    return (
        BoundaryConditionSpec(
            "left_x_symmetry",
            "left_symmetry",
            "symmetry",
            ("x",),
        ),
        BoundaryConditionSpec(
            "bottom_y_symmetry",
            "bottom",
            "symmetry",
            ("y",),
        ),
        BoundaryConditionSpec(
            "right_grip_displacement",
            "right_grip",
            "dirichlet",
            ("x",),
            (CanonicalQuantity(1.0, Unit.MM),),
        ),
    )


def _pilot_solver_config() -> FemSolverConfig:
    return FemSolverConfig(
        element_order=2,
        solver_tolerance=1.0e-10,
        max_newton_iterations=20,
        plane_stress_max_iterations=32,
        load_steps=8,
        max_dofs=100_000,
    )


def build_pilot_requests(
    *, execution_profile: ExecutionProfile
) -> tuple[FemRequest, ...]:
    """Build one S7 request for every accepted public pilot geometry."""

    if type(execution_profile) is not ExecutionProfile:
        raise TypeError("execution_profile must be an ExecutionProfile")
    family = compiled_family()
    requests: list[FemRequest] = []
    for index, case in enumerate(pilot_geometry_cases()):
        theta = parameter_point(*case.parameters)
        membership = family.membership.evaluate(theta)
        if not membership.accepted:
            raise RuntimeError(
                f"pilot geometry {case.case_id!r} left family membership"
            )
        morph = apply_morph(
            family.reference_mesh,
            family.morph_map,
            family.membership,
            theta,
        )
        if not morph.accepted or morph.mesh is None:
            raise RuntimeError(f"pilot geometry {case.case_id!r} failed morphing")
        requests.append(
            FemRequest(
                request_id=f"{PILOT_ID}-{index:02d}-{case.case_id}",
                mesh=morph.mesh,
                material=_pilot_material(),
                boundary_conditions=_pilot_boundary_conditions(),
                loads=(),
                qois=(),
                config=_pilot_solver_config(),
                execution_profile=execution_profile,
            )
        )
    return tuple(requests)


def _reported_seconds(value: float) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise NonFiniteNumberError("pilot elapsed time must be finite and non-negative")
    rounded = round(value, _REPORTED_TIME_DECIMALS)
    return 0.0 if rounded == 0.0 else rounded


def _iteration_counts(attempt: FemAttempt) -> tuple[int, ...]:
    result = attempt.result
    if result is None:
        return ()
    by_step: dict[int, int] = {}
    for entry in result.convergence_history:
        by_step[entry.load_step] = max(by_step.get(entry.load_step, 0), entry.iteration)
    expected_steps = tuple(range(1, attempt.request.config.load_steps + 1))
    if tuple(sorted(by_step)) != expected_steps:
        raise ValueError("successful FEM result omitted load-step diagnostics")
    return tuple(by_step[step] for step in expected_steps)


def _record(
    case: PilotGeometryCase,
    request: FemRequest,
    attempt: FemAttempt,
    elapsed_seconds: float,
) -> FemPilotRecord:
    if attempt.recipe_hash != request.recipe_hash:
        raise ValueError("backend returned an attempt for a different recipe")
    if attempt.request.recipe_hash != request.recipe_hash:
        raise ValueError("backend returned an attempt for a different request")
    if attempt.success:
        if attempt.result is None or attempt.result_hash is None:
            raise ValueError("successful backend attempt omitted its result")
        if attempt.result.request_id != request.request_id:
            raise ValueError("backend result request identity does not match")
        if attempt.result.mesh_id != request.mesh.mesh_id:
            raise ValueError("backend result mesh identity does not match")
        if hash_spec(attempt.result, "content") != attempt.result_hash:
            raise ValueError("backend result hash does not match its canonical result")
        history = attempt.result.convergence_history
        if not history:
            raise ValueError("successful backend result omitted convergence history")
        return FemPilotRecord(
            case_id=case.case_id,
            parameters=case.parameters,
            sentinel=case.sentinel,
            request_id=request.request_id,
            mesh_id=request.mesh.mesh_id,
            recipe_hash=request.recipe_hash,
            success=True,
            result_hash=attempt.result_hash,
            failure_code=None,
            message=attempt.message,
            elapsed_seconds=_reported_seconds(elapsed_seconds),
            scalar_dof_count=2 * len(attempt.result.displacement),
            canonical_result_bytes=len(canonical_json_bytes(attempt.result)),
            newton_iterations_by_load_step=_iteration_counts(attempt),
            final_residual_norm=history[-1].residual_norm,
        )
    if attempt.result is not None or attempt.result_hash is not None:
        raise ValueError("failed backend attempt retained a result")
    if attempt.failure_code is None:
        raise ValueError("failed backend attempt omitted a reason code")
    return FemPilotRecord(
        case_id=case.case_id,
        parameters=case.parameters,
        sentinel=case.sentinel,
        request_id=request.request_id,
        mesh_id=request.mesh.mesh_id,
        recipe_hash=request.recipe_hash,
        success=False,
        result_hash=None,
        failure_code=attempt.failure_code,
        message=attempt.message,
        elapsed_seconds=_reported_seconds(elapsed_seconds),
        scalar_dof_count=None,
        canonical_result_bytes=None,
        newton_iterations_by_load_step=(),
        final_residual_norm=None,
    )


def _summary(records: tuple[FemPilotRecord, ...]) -> FemPilotSummary:
    times = tuple(sorted(record.elapsed_seconds for record in records))
    middle = len(times) // 2
    median = (
        times[middle] if len(times) % 2 else (times[middle - 1] + times[middle]) / 2.0
    )
    p95_index = max(0, math.ceil(0.95 * len(times)) - 1)
    failures = Counter(
        record.failure_code.value
        for record in records
        if record.failure_code is not None
    )
    result_bytes = tuple(
        record.canonical_result_bytes
        for record in records
        if record.canonical_result_bytes is not None
    )
    iteration_counts = tuple(
        iteration
        for record in records
        for iteration in record.newton_iterations_by_load_step
    )
    final_residuals = tuple(
        record.final_residual_norm
        for record in records
        if record.final_residual_norm is not None
    )
    return FemPilotSummary(
        attempted_count=len(records),
        success_count=sum(record.success for record in records),
        failure_counts=failures,
        total_runtime_seconds=_reported_seconds(sum(times)),
        median_runtime_seconds=_reported_seconds(median),
        p95_runtime_seconds=_reported_seconds(times[p95_index]),
        max_runtime_seconds=_reported_seconds(times[-1]),
        total_result_bytes=sum(result_bytes),
        max_result_bytes=max(result_bytes, default=0),
        max_newton_iterations_observed=max(iteration_counts, default=0),
        max_final_residual_norm=max(final_residuals, default=None),
    )


def run_fem_pilot(
    *,
    backend: FemBackend,
    execution_profile: ExecutionProfile,
    clock: Callable[[], float],
    runtime_environment: Mapping[str, str] | None = None,
) -> FemPilotReport:
    """Execute every pilot case independently and retain every outcome."""

    requests = build_pilot_requests(execution_profile=execution_profile)
    records: list[FemPilotRecord] = []
    for case, request in zip(pilot_geometry_cases(), requests, strict=True):
        started = clock()
        attempts = backend.solve_many((request,))
        elapsed = clock() - started
        if len(attempts) != 1:
            raise ValueError("FEM backend must return exactly one attempt per request")
        records.append(_record(case, request, attempts[0], elapsed))
    frozen_records = tuple(records)
    family = compiled_family()
    return FemPilotReport(
        pilot_id=PILOT_ID,
        selection_policy_id=SELECTION_POLICY_ID,
        compiled_family_hash=hash_spec(family, "content"),
        reference_mesh_id=family.reference_mesh.mesh_id,
        validity_policy_hash=family.membership.validity_policy_hash,
        material=_pilot_material(),
        prescribed_displacement=CanonicalQuantity(1.0, Unit.MM),
        solver_config=_pilot_solver_config(),
        execution_profile=execution_profile,
        runtime_environment={} if runtime_environment is None else runtime_environment,
        backend_name=backend.backend_name,
        records=frozen_records,
        summary=_summary(frozen_records),
    )


__all__ = (
    "FemPilotRecord",
    "FemPilotReport",
    "FemPilotSummary",
    "PILOT_ID",
    "PilotGeometryCase",
    "SELECTION_POLICY_ID",
    "build_pilot_requests",
    "pilot_geometry_cases",
    "run_fem_pilot",
)
