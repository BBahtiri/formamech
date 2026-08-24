"""Executable primary-solver physical checks for the S8 gate."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalQuantity,
    LoadSpec,
    Unit,
    hash_spec,
)
from piginot.contracts.specs import FrozenMap
from piginot.fem.backend import ScikitFemBackend
from piginot.fem.field_providers import FloatArray
from piginot.fem.mms_runner import (
    verification_execution_profile,
    verification_material,
)
from piginot.fem.models import FemRequest, FemResult, FemSolverConfig
from piginot.fem.verification_checks import (
    affine_patch_error,
    cauchy_skew_ratio,
    frame_objectivity_metrics,
    reaction_equilibrium_error,
)
from piginot.fem.verification_meshes import (
    VerificationMeshKind,
    unit_square_verification_mesh,
)
from piginot.physics import plane_stress_closure


@dataclass(frozen=True, slots=True)
class S8PhysicalVerification:
    """Measured physical invariants and primary-backend balance errors."""

    patch_errors: Mapping[str, float]
    patch_max_error: float
    objectivity_invariant_change: float
    angular_skew_ratio: float
    energy_relative_error: float
    reaction_relative_error: float
    small_strain_relative_error: float
    energy_path_quadrature_order: int
    energy_path_solve_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "patch_errors", FrozenMap.copy_of(self.patch_errors))


@dataclass(frozen=True, slots=True)
class _AffineDisplacement:
    gradient: tuple[tuple[float, float], tuple[float, float]]
    offset: tuple[float, float]

    @property
    def provider_id(self) -> str:
        return hash_spec(
            {
                "provider_kind": "s8-affine-displacement-v1",
                "gradient": self.gradient,
                "offset": self.offset,
            },
            "recipe",
        )

    def evaluate(self, reference_points: FloatArray) -> FloatArray:
        points = np.asarray(reference_points, dtype=np.float64)
        gradient = np.asarray(self.gradient, dtype=np.float64)
        offset = np.asarray(self.offset, dtype=np.float64)
        values = np.einsum("ij,j...->i...", gradient, points) + offset.reshape(
            (2, *([1] * (points.ndim - 1)))
        )
        return np.asarray(values, dtype=np.float64)


@lru_cache(maxsize=1)
def run_s8_physical_verification() -> S8PhysicalVerification:
    """Run patch, balance, limit, and invariant checks without surrogate values."""

    patch_errors = _patch_errors()
    objectivity, angular = _constitutive_invariants()
    energy_error, path_solve_count = _backend_energy_balance()
    reaction_error = _backend_reaction_balance()
    small_strain_error = _backend_small_strain_limit()
    return S8PhysicalVerification(
        patch_errors=patch_errors,
        patch_max_error=max(patch_errors.values()),
        objectivity_invariant_change=objectivity,
        angular_skew_ratio=angular,
        energy_relative_error=energy_error,
        reaction_relative_error=reaction_error,
        small_strain_relative_error=small_strain_error,
        energy_path_quadrature_order=16,
        energy_path_solve_count=path_solve_count,
    )


def _patch_errors() -> dict[str, float]:
    gradient = ((0.04, 0.01), (-0.015, 0.025))
    offset = (0.007, -0.004)
    provider = _AffineDisplacement(gradient, offset)
    errors: dict[str, float] = {}
    for kind in VerificationMeshKind:
        for order in (1, 2):
            mesh = unit_square_verification_mesh(2, kind=kind)
            request = FemRequest(
                request_id=f"s8-patch-{kind.value}-p{order}",
                mesh=mesh,
                material=verification_material(),
                boundary_conditions=(
                    BoundaryConditionSpec(
                        "affine_boundary",
                        "all",
                        "dirichlet",
                        ("x", "y"),
                    ),
                ),
                loads=(),
                qois=(),
                config=_solver_config(order, load_steps=4),
                execution_profile=verification_execution_profile(),
                boundary_field_provider_ids={"affine_boundary": provider.provider_id},
            )
            result = _solve(request, dirichlet_provider=provider)
            errors[f"{kind.value}-p{order}"] = affine_patch_error(
                np.asarray(result.displacement_dof_coordinates, dtype=np.float64),
                np.asarray(result.displacement, dtype=np.float64),
                np.asarray(gradient, dtype=np.float64),
                np.asarray(offset, dtype=np.float64),
            )
    return errors


def _constitutive_invariants() -> tuple[float, float]:
    material = verification_material()
    deformation_2d = torch.tensor(((1.12, 0.04), (-0.02, 0.94)), dtype=torch.float64)
    f33 = plane_stress_closure(deformation_2d, material)
    deformation = torch.zeros((3, 3), dtype=torch.float64)
    deformation[:2, :2] = deformation_2d
    deformation[2, 2] = f33
    angle = math.radians(37.0)
    rotation = torch.tensor(
        (
            (math.cos(angle), -math.sin(angle), 0.0),
            (math.sin(angle), math.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=torch.float64,
    )
    objectivity = frame_objectivity_metrics(deformation, rotation, material)
    angular = cauchy_skew_ratio(deformation, material)
    return objectivity.maximum_relative_error, angular


def _backend_energy_balance() -> tuple[float, int]:
    quadrature_order = 16
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    parameters = 0.5 * (nodes + 1.0)
    scaled_weights = 0.5 * weights
    axial_strain = 0.08
    lateral_strain = -0.03
    work_density = 0.0
    solve_count = 0
    for parameter, weight in zip(parameters, scaled_weights, strict=True):
        result = _solve(_diagonal_displacement_request(float(parameter)))
        solve_count += 1
        integrand = (
            result.reactions["right_x"][0] * axial_strain
            + result.reactions["top_y"][1] * lateral_strain
        )
        work_density += float(weight) * integrand
    final = _solve(_diagonal_displacement_request(1.0))
    solve_count += 1
    strain_energy = final.total_strain_energy
    relative_error = abs(work_density - strain_energy) / abs(strain_energy)
    return relative_error, solve_count


def _diagonal_displacement_request(parameter: float) -> FemRequest:
    axial = parameter * 0.08
    lateral = parameter * -0.03
    zero = CanonicalQuantity(0.0, Unit.MM)
    return FemRequest(
        request_id=f"s8-energy-{parameter.hex()}",
        mesh=unit_square_verification_mesh(1),
        material=verification_material(),
        boundary_conditions=(
            BoundaryConditionSpec("left_x", "left", "dirichlet", ("x",), (zero,)),
            BoundaryConditionSpec(
                "right_x",
                "right",
                "dirichlet",
                ("x",),
                (CanonicalQuantity(axial, Unit.MM),),
            ),
            BoundaryConditionSpec("bottom_y", "bottom", "dirichlet", ("y",), (zero,)),
            BoundaryConditionSpec(
                "top_y",
                "top",
                "dirichlet",
                ("y",),
                (CanonicalQuantity(lateral, Unit.MM),),
            ),
        ),
        loads=(),
        qois=(),
        config=_solver_config(2, load_steps=4),
        execution_profile=verification_execution_profile(),
    )


def _backend_reaction_balance() -> float:
    traction = 0.2
    zero = CanonicalQuantity(0.0, Unit.MM)
    request = FemRequest(
        request_id="s8-reaction-balance",
        mesh=unit_square_verification_mesh(2),
        material=verification_material(),
        boundary_conditions=(
            BoundaryConditionSpec("left_x", "left", "dirichlet", ("x",), (zero,)),
            BoundaryConditionSpec("bottom_y", "bottom", "dirichlet", ("y",), (zero,)),
        ),
        loads=(
            LoadSpec(
                "right_traction",
                "traction",
                "right",
                ("x",),
                (CanonicalQuantity(traction, Unit.MPA),),
            ),
        ),
        qois=(),
        config=_solver_config(2, load_steps=4),
        execution_profile=verification_execution_profile(),
    )
    result = _solve(request)
    return reaction_equilibrium_error(result.reactions, (traction, 0.0))


def _backend_small_strain_limit() -> float:
    axial_strain = 1.0e-3
    poisson = verification_material().properties["poisson_ratio"].value
    errors: list[float] = []
    zero = CanonicalQuantity(0.0, Unit.MM)
    for order in (1, 2):
        request = FemRequest(
            request_id=f"s8-small-strain-p{order}",
            mesh=unit_square_verification_mesh(2),
            material=verification_material(),
            boundary_conditions=(
                BoundaryConditionSpec("left_x", "left", "dirichlet", ("x",), (zero,)),
                BoundaryConditionSpec(
                    "right_x",
                    "right",
                    "dirichlet",
                    ("x",),
                    (CanonicalQuantity(axial_strain, Unit.MM),),
                ),
                BoundaryConditionSpec(
                    "bottom_y", "bottom", "dirichlet", ("y",), (zero,)
                ),
            ),
            loads=(),
            qois=(),
            config=_solver_config(order, load_steps=2),
            execution_profile=verification_execution_profile(),
        )
        result = _solve(request)
        coordinates = np.asarray(result.displacement_dof_coordinates, dtype=np.float64)
        expected = np.column_stack(
            (
                axial_strain * coordinates[:, 0],
                -poisson * axial_strain * coordinates[:, 1],
            )
        )
        actual = np.asarray(result.displacement, dtype=np.float64)
        errors.append(
            float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
        )
    return max(errors)


def _solver_config(element_order: int, *, load_steps: int) -> FemSolverConfig:
    return FemSolverConfig(
        element_order=element_order,
        solver_tolerance=1.0e-11,
        max_newton_iterations=24,
        plane_stress_max_iterations=32,
        load_steps=load_steps,
    )


def _solve(
    request: FemRequest,
    *,
    dirichlet_provider: _AffineDisplacement | None = None,
) -> FemResult:
    providers = () if dirichlet_provider is None else (dirichlet_provider,)
    attempt = ScikitFemBackend(dirichlet_field_providers=providers).solve_many(
        (request,)
    )[0]
    if not attempt.success or attempt.result is None:
        code = "UNKNOWN" if attempt.failure_code is None else attempt.failure_code.value
        raise RuntimeError(f"physical check failed [{code}]: {attempt.message}")
    return attempt.result


__all__ = ("S8PhysicalVerification", "run_s8_physical_verification")
