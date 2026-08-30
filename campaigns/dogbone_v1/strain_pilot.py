"""Public FEM evidence for a strain-controlled DogBone load proposal."""

from __future__ import annotations

import math
import os
import platform
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import ClassVar

from campaigns.dogbone_v1.family import compiled_family
from campaigns.dogbone_v1.fem_pilot import (
    FemPilotRecord,
    FemPilotSummary,
    PilotGeometryCase,
    build_pilot_requests,
    pilot_geometry_cases,
    record_pilot_attempt,
    summarize_pilot_records,
)
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
from piginot.fem import FemAttempt, FemBackend, FemRequest, FemResult, FemSolverConfig
from piginot.fem.backend import ScikitFemBackend

STRAIN_PILOT_ID = "dogbone-strain-envelope-20260830-v1"
LOAD_POLICY_ID = "nominal-axial-strain-times-half-length-v1"
NOMINAL_STRAIN_LEVELS = (0.10, 0.20)
STRAIN_PILOT_LOAD_STEPS = 16
_RIGHT_DISPLACEMENT_NAME = "right_grip_displacement"
_ARTIFACT_RELATIVE_PATH = Path(
    "artifacts/campaigns/dogbone-v1/fem-strain-envelope-pilot.json"
)


def _finite(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise NonFiniteNumberError(f"{label} must be finite")
    return number


def prescribed_displacement_mm(
    total_length_mm: float,
    nominal_strain: float,
) -> float:
    """Map a full-length nominal strain to this half-model's right displacement."""

    length = _finite(total_length_mm, "total_length_mm")
    strain = _finite(nominal_strain, "nominal_strain")
    if length <= 0.0:
        raise ValueError("total_length_mm must be positive")
    if strain <= 0.0:
        raise ValueError("nominal_strain must be positive")
    return strain * length / 2.0


@dataclass(frozen=True, slots=True)
class StrainPilotRecord:
    """One load-explicit outcome in the public strain-envelope pilot."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    case_id: str
    nominal_axial_strain: float
    prescribed_displacement: CanonicalQuantity
    outcome: FemPilotRecord
    right_reaction_x: float | None
    minimum_deformation_jacobian: float | None

    def __post_init__(self) -> None:
        strain = _finite(self.nominal_axial_strain, "nominal_axial_strain")
        if strain <= 0.0:
            raise ValueError("nominal_axial_strain must be positive")
        if type(self.prescribed_displacement) is not CanonicalQuantity:
            raise TypeError("prescribed_displacement must be a CanonicalQuantity")
        if self.prescribed_displacement.unit is not Unit.MM:
            raise ValueError("prescribed_displacement must use millimetres")
        if type(self.outcome) is not FemPilotRecord:
            raise TypeError("outcome must be a FemPilotRecord")
        if self.case_id != self.outcome.case_id:
            raise ValueError("strain record case identity does not match its outcome")
        expected = prescribed_displacement_mm(self.outcome.parameters[0], strain)
        if not math.isclose(
            self.prescribed_displacement.value,
            expected,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("prescribed displacement does not match nominal strain")
        if self.outcome.success:
            if self.right_reaction_x is None:
                raise ValueError("successful strain record must retain right reaction")
            if self.minimum_deformation_jacobian is None:
                raise ValueError(
                    "successful strain record must retain deformation Jacobian"
                )
            object.__setattr__(
                self,
                "right_reaction_x",
                _finite(self.right_reaction_x, "right_reaction_x"),
            )
            object.__setattr__(
                self,
                "minimum_deformation_jacobian",
                _finite(
                    self.minimum_deformation_jacobian,
                    "minimum_deformation_jacobian",
                ),
            )
        elif (
            self.right_reaction_x is not None
            or self.minimum_deformation_jacobian is not None
        ):
            raise ValueError("failed strain record cannot carry solution diagnostics")
        object.__setattr__(self, "nominal_axial_strain", strain)


@dataclass(frozen=True, slots=True)
class StrainResponseComparison:
    """Reaction scaling between the two proposed nominal-strain endpoints."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    case_id: str
    lower_nominal_strain: float
    upper_nominal_strain: float
    observed_reaction_ratio: float
    linear_reference_ratio: float
    reaction_ratio_relative_to_linear: float
    relative_departure_from_linear: float

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("case_id must be non-empty text")
        for field_name in (
            "lower_nominal_strain",
            "upper_nominal_strain",
            "observed_reaction_ratio",
            "linear_reference_ratio",
            "reaction_ratio_relative_to_linear",
            "relative_departure_from_linear",
        ):
            value = _finite(getattr(self, field_name), field_name)
            if value <= 0.0 and field_name != "relative_departure_from_linear":
                raise ValueError(f"{field_name} must be positive")
            if value < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
            object.__setattr__(self, field_name, value)
        if self.upper_nominal_strain <= self.lower_nominal_strain:
            raise ValueError("strain comparison endpoints must be increasing")


@dataclass(frozen=True, slots=True)
class StrainPilotReport:
    """Canonical public evidence for the unratified strain-load proposal."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    pilot_id: str
    load_policy_id: str
    nominal_strain_levels: tuple[float, ...]
    compiled_family_hash: str
    reference_mesh_id: str
    validity_policy_hash: str
    material: MaterialSpec
    solver_config: FemSolverConfig
    execution_profile: ExecutionProfile
    runtime_environment: Mapping[str, str]
    backend_name: str
    records: tuple[StrainPilotRecord, ...]
    response_comparisons: tuple[StrainResponseComparison, ...]
    outcome_summary: FemPilotSummary
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        levels = tuple(
            _finite(value, "nominal_strain_levels")
            for value in self.nominal_strain_levels
        )
        if levels != NOMINAL_STRAIN_LEVELS:
            raise ValueError("strain pilot levels do not match the fixed selection")
        records = tuple(self.records)
        if any(type(record) is not StrainPilotRecord for record in records):
            raise TypeError("records must contain StrainPilotRecord values")
        expected_order = tuple(
            (case.case_id, strain)
            for case in pilot_geometry_cases()
            for strain in levels
        )
        observed_order = tuple(
            (record.case_id, record.nominal_axial_strain) for record in records
        )
        if observed_order != expected_order:
            raise ValueError("strain pilot record order does not match its policy")
        comparisons = tuple(self.response_comparisons)
        if any(
            type(comparison) is not StrainResponseComparison
            for comparison in comparisons
        ):
            raise TypeError(
                "response_comparisons must contain StrainResponseComparison values"
            )
        if len(records) != self.outcome_summary.attempted_count:
            raise ValueError("strain records and outcome summary do not reconcile")
        runtime_environment = dict(sorted(self.runtime_environment.items()))
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in runtime_environment.items()
        ):
            raise ValueError("runtime environment entries must be non-empty text")
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported strain pilot schema {self.schema_version!r}")
        object.__setattr__(self, "nominal_strain_levels", levels)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "response_comparisons", comparisons)
        object.__setattr__(
            self,
            "runtime_environment",
            MappingProxyType(runtime_environment),
        )


def _right_displacement(
    request: FemRequest,
    displacement_mm: float,
) -> tuple[BoundaryConditionSpec, ...]:
    matches = tuple(
        condition
        for condition in request.boundary_conditions
        if condition.name == _RIGHT_DISPLACEMENT_NAME
    )
    if len(matches) != 1:
        raise ValueError("pilot request must contain one right displacement condition")
    return tuple(
        replace(
            condition,
            values=(CanonicalQuantity(displacement_mm, Unit.MM),),
        )
        if condition.name == _RIGHT_DISPLACEMENT_NAME
        else condition
        for condition in request.boundary_conditions
    )


def build_strain_pilot_requests(
    *, execution_profile: ExecutionProfile
) -> tuple[FemRequest, ...]:
    """Build the fixed two-endpoint strain pilot over all public sentinels."""

    base_requests = build_pilot_requests(execution_profile=execution_profile)
    requests: list[FemRequest] = []
    for case_index, (case, base) in enumerate(
        zip(pilot_geometry_cases(), base_requests, strict=True)
    ):
        for strain_index, strain in enumerate(NOMINAL_STRAIN_LEVELS):
            displacement = prescribed_displacement_mm(case.parameters[0], strain)
            requests.append(
                replace(
                    base,
                    request_id=(
                        f"{STRAIN_PILOT_ID}-{case_index:02d}-{strain_index:02d}-"
                        f"{case.case_id}"
                    ),
                    boundary_conditions=_right_displacement(base, displacement),
                    config=replace(
                        base.config,
                        load_steps=STRAIN_PILOT_LOAD_STEPS,
                    ),
                )
            )
    return tuple(requests)


def _determinant_3x3(matrix: tuple[tuple[float, float, float], ...]) -> float:
    (a, b, c), (d, e, f), (g, h, i) = matrix
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _minimum_deformation_jacobian(result: FemResult) -> float:
    determinants = tuple(
        _determinant_3x3(matrix)
        for cell in result.deformation_gradient
        for matrix in cell
    )
    if not determinants:
        raise ValueError("successful FEM result omitted deformation gradients")
    if any(not math.isfinite(value) for value in determinants):
        raise NonFiniteNumberError("deformation Jacobians must be finite")
    return min(determinants)


def _strain_record(
    case: PilotGeometryCase,
    strain: float,
    request: FemRequest,
    attempt: FemAttempt,
    elapsed_seconds: float,
) -> StrainPilotRecord:
    outcome = record_pilot_attempt(case, request, attempt, elapsed_seconds)
    displacement = CanonicalQuantity(
        prescribed_displacement_mm(case.parameters[0], strain),
        Unit.MM,
    )
    if not outcome.success:
        return StrainPilotRecord(
            case.case_id,
            strain,
            displacement,
            outcome,
            None,
            None,
        )
    result = attempt.result
    if result is None:
        raise ValueError("successful strain attempt omitted its result")
    reaction = result.reactions.get(_RIGHT_DISPLACEMENT_NAME)
    if reaction is None:
        raise ValueError("successful strain result omitted the right reaction")
    return StrainPilotRecord(
        case.case_id,
        strain,
        displacement,
        outcome,
        reaction[0],
        _minimum_deformation_jacobian(result),
    )


def _response_comparisons(
    records: tuple[StrainPilotRecord, ...],
) -> tuple[StrainResponseComparison, ...]:
    by_case: dict[str, dict[float, StrainPilotRecord]] = {}
    for record in records:
        by_case.setdefault(record.case_id, {})[record.nominal_axial_strain] = record
    lower_strain, upper_strain = NOMINAL_STRAIN_LEVELS
    comparisons: list[StrainResponseComparison] = []
    for case in pilot_geometry_cases():
        lower = by_case[case.case_id][lower_strain]
        upper = by_case[case.case_id][upper_strain]
        if not lower.outcome.success or not upper.outcome.success:
            continue
        lower_reaction = abs(lower.right_reaction_x or 0.0)
        upper_reaction = abs(upper.right_reaction_x or 0.0)
        if lower_reaction == 0.0:
            raise ValueError("successful lower-strain result has zero right reaction")
        observed_ratio = upper_reaction / lower_reaction
        linear_ratio = upper_strain / lower_strain
        relative_ratio = observed_ratio / linear_ratio
        comparisons.append(
            StrainResponseComparison(
                case.case_id,
                lower_strain,
                upper_strain,
                observed_ratio,
                linear_ratio,
                relative_ratio,
                abs(relative_ratio - 1.0),
            )
        )
    return tuple(comparisons)


def run_strain_pilot(
    *,
    backend: FemBackend,
    execution_profile: ExecutionProfile,
    clock: Callable[[], float],
    runtime_environment: Mapping[str, str] | None = None,
) -> StrainPilotReport:
    """Execute every geometry at both candidate strain endpoints."""

    requests = build_strain_pilot_requests(execution_profile=execution_profile)
    cases = pilot_geometry_cases()
    records: list[StrainPilotRecord] = []
    for index, request in enumerate(requests):
        case = cases[index // len(NOMINAL_STRAIN_LEVELS)]
        strain = NOMINAL_STRAIN_LEVELS[index % len(NOMINAL_STRAIN_LEVELS)]
        started = clock()
        attempts = backend.solve_many((request,))
        elapsed = clock() - started
        if len(attempts) != 1:
            raise ValueError("FEM backend must return exactly one attempt per request")
        records.append(_strain_record(case, strain, request, attempts[0], elapsed))
    frozen_records = tuple(records)
    outcomes = tuple(record.outcome for record in frozen_records)
    family = compiled_family()
    return StrainPilotReport(
        pilot_id=STRAIN_PILOT_ID,
        load_policy_id=LOAD_POLICY_ID,
        nominal_strain_levels=NOMINAL_STRAIN_LEVELS,
        compiled_family_hash=hash_spec(family, "content"),
        reference_mesh_id=family.reference_mesh.mesh_id,
        validity_policy_hash=family.membership.validity_policy_hash,
        material=requests[0].material,
        solver_config=requests[0].config,
        execution_profile=execution_profile,
        runtime_environment={} if runtime_environment is None else runtime_environment,
        backend_name=backend.backend_name,
        records=frozen_records,
        response_comparisons=_response_comparisons(frozen_records),
        outcome_summary=summarize_pilot_records(outcomes),
    )


def write_strain_pilot_report(report: StrainPilotReport, target: Path) -> None:
    """Create one canonical artifact, refusing to replace different evidence."""

    if type(report) is not StrainPilotReport:
        raise TypeError("report must be a StrainPilotReport")
    if not isinstance(target, Path):
        raise TypeError("target must be a pathlib.Path")
    payload = canonical_json_bytes(report) + b"\n"
    if target.exists():
        if target.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace existing evidence: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def _execution_profile() -> ExecutionProfile:
    return ExecutionProfile(
        python_version=platform.python_version(),
        library_versions={
            package: version(package)
            for package in ("numpy", "piginot", "scikit-fem", "scipy", "torch")
        },
        blas_threads=1,
        device="cpu",
        dtype="float64",
        rng_algorithm="none",
        seed=0,
    )


def _runtime_environment() -> Mapping[str, str]:
    return {
        "logical_cpu_count": str(os.cpu_count() or "unknown"),
        "machine": platform.machine() or "unknown",
        "operating_system": f"{platform.system()} {platform.release()} {platform.version()}",
        "processor": platform.processor() or "unknown",
    }


def main() -> int:
    """Run the fixed local pilot and create its public artifact once."""

    report = run_strain_pilot(
        backend=ScikitFemBackend(),
        execution_profile=_execution_profile(),
        clock=perf_counter,
        runtime_environment=_runtime_environment(),
    )
    repository_root = Path(__file__).resolve().parents[2]
    target = repository_root / _ARTIFACT_RELATIVE_PATH
    write_strain_pilot_report(report, target)
    print(f"wrote {target}")
    print(f"content hash: {hash_spec(report, 'content')}")
    print(
        "outcomes: "
        f"{report.outcome_summary.success_count}/"
        f"{report.outcome_summary.attempted_count} successful"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "LOAD_POLICY_ID",
    "NOMINAL_STRAIN_LEVELS",
    "STRAIN_PILOT_ID",
    "STRAIN_PILOT_LOAD_STEPS",
    "StrainPilotRecord",
    "StrainPilotReport",
    "StrainResponseComparison",
    "build_strain_pilot_requests",
    "prescribed_displacement_mm",
    "run_strain_pilot",
    "write_strain_pilot_report",
)
