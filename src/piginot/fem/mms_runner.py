"""Primary-backend execution of nonlinear manufactured FEM problems."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalQuantity,
    ExecutionProfile,
    LoadSpec,
    MaterialSpec,
    Unit,
)
from piginot.fem.backend import ScikitFemBackend
from piginot.fem.manufactured import (
    BoundaryKind,
    ManufacturedSolution,
    build_manufactured_solution,
)
from piginot.fem.manufactured_providers import (
    ManufacturedBodyForceProvider,
    ManufacturedDisplacementProvider,
    ManufacturedTractionProvider,
)
from piginot.fem.models import (
    FemRequest,
    FemResult,
    FemSolverConfig,
)
from piginot.fem.verification_meshes import (
    VerificationMeshKind,
    unit_square_verification_mesh,
    verification_mesh_metadata,
)
from piginot.fem.verification_norms import (
    FemVerificationNorms,
    compute_verification_norms,
)


@dataclass(frozen=True, slots=True)
class ManufacturedFemProblem:
    """One exact problem and its registered, recipe-addressed field providers."""

    request: FemRequest
    manufactured_solution: ManufacturedSolution
    displacement_provider: ManufacturedDisplacementProvider
    body_force_provider: ManufacturedBodyForceProvider
    traction_provider: ManufacturedTractionProvider


@dataclass(frozen=True, slots=True)
class MmsLevelResult:
    """One successful nonlinear solve with true manufactured-field errors."""

    case_name: str
    level: int
    element_order: int
    mesh_kind: VerificationMeshKind
    mesh_id: str
    h_max_mm: float
    dof_count: int
    result_hash: str
    rigid_mode_multiplier_norm: float
    norms: FemVerificationNorms
    request: FemRequest
    result: FemResult
    manufactured_solution: ManufacturedSolution


def verification_material() -> MaterialSpec:
    """Return the fixed material used by the S8 calibration population."""

    return MaterialSpec(
        "s8_verification_material",
        "compressible_neo_hookean",
        {
            "elastic_modulus": CanonicalQuantity(100.0, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(0.3, Unit.ONE),
        },
    )


def build_mms_problem(
    case_name: str,
    *,
    level: int,
    element_order: int,
    mesh_kind: VerificationMeshKind | str = VerificationMeshKind.STRUCTURED,
) -> ManufacturedFemProblem:
    """Translate one symbolic case into the canonical primary-backend contract."""

    kind = VerificationMeshKind(mesh_kind)
    mesh = unit_square_verification_mesh(level, kind=kind)
    material = verification_material()
    exact = build_manufactured_solution(case_name, material)
    displacement = ManufacturedDisplacementProvider(case_name, material)
    body = ManufacturedBodyForceProvider(case_name, material)
    traction = ManufacturedTractionProvider(case_name, material)

    dirichlet_tags = tuple(
        tag
        for tag, boundary_kind in exact.boundary_types.items()
        if boundary_kind is BoundaryKind.DIRICHLET
    )
    if len(dirichlet_tags) == 4:
        boundary_conditions: tuple[BoundaryConditionSpec, ...] = (
            BoundaryConditionSpec(
                "manufactured_displacement",
                "all",
                "dirichlet",
                ("x", "y"),
            ),
        )
    else:
        boundary_conditions = tuple(
            BoundaryConditionSpec(
                f"manufactured_displacement_{tag}",
                tag,
                "dirichlet",
                ("x", "y"),
            )
            for tag in dirichlet_tags
        )
    boundary_provider_ids = {
        condition.name: displacement.provider_id for condition in boundary_conditions
    }

    traction_tags = tuple(
        tag
        for tag, boundary_kind in exact.boundary_types.items()
        if boundary_kind is BoundaryKind.TRACTION
    )
    traction_loads = tuple(
        LoadSpec(f"manufactured_traction_{tag}", "traction", tag)
        for tag in traction_tags
    )
    body_load = LoadSpec("manufactured_body_force", "body_force")
    loads = (body_load, *traction_loads)
    traction_provider_ids = {load.name: traction.provider_id for load in traction_loads}
    request = FemRequest(
        request_id=f"s8-{case_name}-{kind.value}-l{level}-p{element_order}",
        mesh=mesh,
        material=material,
        boundary_conditions=boundary_conditions,
        loads=loads,
        qois=(),
        config=FemSolverConfig(
            element_order=element_order,
            solver_tolerance=1.0e-11,
            max_newton_iterations=28,
            plane_stress_max_iterations=32,
            load_steps=16 if case_name == "strongly_varying_f33" else 8,
            max_dofs=100_000,
        ),
        execution_profile=verification_execution_profile(),
        body_force_provider_ids={body_load.name: body.provider_id},
        boundary_field_provider_ids=boundary_provider_ids,
        traction_field_provider_ids=traction_provider_ids,
        rigid_mode_constraint=(
            "zero_mean_translation_rotation" if exact.rigid_mode_constraints else None
        ),
    )
    return ManufacturedFemProblem(
        request=request,
        manufactured_solution=exact,
        displacement_provider=displacement,
        body_force_provider=body,
        traction_provider=traction,
    )


@lru_cache(maxsize=128)
def solve_mms_level(
    case_name: str,
    *,
    level: int,
    element_order: int,
    mesh_kind: VerificationMeshKind | str = VerificationMeshKind.STRUCTURED,
) -> MmsLevelResult:
    """Solve and measure one manufactured problem with the S7 primary backend."""

    problem = build_mms_problem(
        case_name,
        level=level,
        element_order=element_order,
        mesh_kind=mesh_kind,
    )
    backend = ScikitFemBackend(
        body_force_providers=(problem.body_force_provider,),
        dirichlet_field_providers=(problem.displacement_provider,),
        traction_field_providers=(problem.traction_provider,),
    )
    attempt = backend.solve_many((problem.request,))[0]
    if not attempt.success or attempt.result is None or attempt.result_hash is None:
        code = "UNKNOWN" if attempt.failure_code is None else attempt.failure_code.value
        raise RuntimeError(
            f"MMS solve {problem.request.request_id!r} failed [{code}]: "
            f"{attempt.message}"
        )
    norms = compute_verification_norms(
        problem.request,
        attempt.result,
        problem.manufactured_solution,
    )
    metadata = verification_mesh_metadata(problem.request.mesh)
    rigid_multiplier_norm = (
        sum(value * value for value in attempt.result.rigid_mode_multipliers) ** 0.5
    )
    return MmsLevelResult(
        case_name=case_name,
        level=level,
        element_order=element_order,
        mesh_kind=metadata.kind,
        mesh_id=problem.request.mesh.mesh_id,
        h_max_mm=metadata.h_max_mm,
        dof_count=len(attempt.result.displacement),
        result_hash=attempt.result_hash,
        rigid_mode_multiplier_norm=rigid_multiplier_norm,
        norms=norms,
        request=problem.request,
        result=attempt.result,
        manufactured_solution=problem.manufactured_solution,
    )


def verification_execution_profile() -> ExecutionProfile:
    libraries = {
        name: _installed_version(name)
        for name in ("numpy", "scikit-fem", "scipy", "sympy", "torch")
    }
    return ExecutionProfile(
        python_version=(
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        library_versions=libraries,
        blas_threads=1,
        device="cpu",
        dtype="float64",
        rng_algorithm="none",
        seed=0,
    )


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"required FEM distribution {distribution!r} is missing"
        ) from exc


__all__ = (
    "ManufacturedFemProblem",
    "MmsLevelResult",
    "build_mms_problem",
    "solve_mms_level",
    "verification_execution_profile",
    "verification_material",
)
