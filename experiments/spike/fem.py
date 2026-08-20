"""Crude scikit-fem plane-stress solver for the quarantined S1 spike."""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Final

import numpy as np
from numpy.typing import NDArray
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP1,
    ElementVector,
    LinearForm,
    MeshTri,
    asm,
    condense,
    solve,
)

from experiments.spike.geometry import SpikeMesh

DEFAULT_E: Final = 760.0
DEFAULT_NU: Final = 0.23
DEFAULT_AXIAL_STRAIN: Final = 0.10
DEFAULT_APPLIED_DISPLACEMENT: Final = 1.0
MINIMUM_JACOBIAN: Final = 0.05

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FEMResult:
    """Finite-element fields and solver diagnostics for one geometry."""

    converged: bool
    failure_reason: str | None
    displacement: FloatArray | None
    first_piola: FloatArray | None
    solve_seconds: float
    newton_iterations: int


@dataclass(frozen=True)
class SanityResult:
    """Measured unit-square agreement with the independent affine solution."""

    converged: bool
    failure_reason: str | None
    lateral_stretch: float
    relative_l2_u: float
    relative_l2_p: float
    max_error: float


def lame_parameters(youngs_modulus: float, poisson_ratio: float) -> tuple[float, float]:
    """Return the three-dimensional Lamé parameters in MPa."""
    if not math.isfinite(youngs_modulus) or youngs_modulus <= 0.0:
        raise ValueError("youngs_modulus must be finite and positive")
    if not math.isfinite(poisson_ratio) or not -1.0 < poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must be finite and in (-1, 0.5)")
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    lam = (
        youngs_modulus
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    return mu, lam


def _as_deformation_batch(f2: FloatArray) -> tuple[FloatArray, tuple[int, ...]]:
    array = np.asarray(f2, dtype=np.float64)
    if array.shape[-2:] != (2, 2):
        raise ValueError("F2 must end in shape (2, 2)")
    original_shape = array.shape[:-2]
    return array.reshape((-1, 2, 2)), original_shape


def plane_stress_f33(
    f2: FloatArray,
    youngs_modulus: float = DEFAULT_E,
    poisson_ratio: float = DEFAULT_NU,
    *,
    tolerance: float = 1e-13,
    max_iterations: int = 40,
) -> FloatArray:
    """Solve the unique positive closure root by damped Newton in log(F33)."""
    batch, original_shape = _as_deformation_batch(f2)
    determinant = np.linalg.det(batch)
    if np.any(~np.isfinite(determinant)) or np.any(determinant <= 0.0):
        raise ValueError("F2 must have a finite positive determinant")

    mu, lam = lame_parameters(youngs_modulus, poisson_ratio)

    log_determinant = np.log(determinant)
    log_value = -lam * log_determinant / (2.0 * mu + lam)
    scale = mu + abs(lam)

    for _ in range(max_iterations):
        exponential = np.exp(2.0 * log_value)
        current = mu * np.expm1(2.0 * log_value) + lam * (log_determinant + log_value)
        if float(np.max(np.abs(current))) <= tolerance * scale:
            return np.exp(log_value).reshape(original_shape)
        derivative = 2.0 * mu * exponential + lam
        newton_step = current / derivative
        damping = np.ones_like(log_value)
        candidate = log_value - newton_step
        for _ in range(16):
            candidate_residual = mu * np.expm1(2.0 * candidate) + lam * (
                log_determinant + candidate
            )
            accepted = np.isfinite(candidate_residual) & (
                np.abs(candidate_residual) < np.abs(current)
            )
            if np.all(accepted):
                break
            damping = np.where(accepted, damping, damping * 0.5)
            candidate = np.where(accepted, candidate, log_value - damping * newton_step)
        log_value = candidate
    raise RuntimeError("plane-stress closure did not converge")


def plane_stress_f33_derivative(
    f2: FloatArray,
    youngs_modulus: float = DEFAULT_E,
    poisson_ratio: float = DEFAULT_NU,
) -> FloatArray:
    """Return dF33/dF2 from the implicit-function theorem."""
    batch, original_shape = _as_deformation_batch(f2)
    mu, lam = lame_parameters(youngs_modulus, poisson_ratio)
    f33 = plane_stress_f33(batch, youngs_modulus, poisson_ratio).reshape(-1)
    inverse_transpose = np.swapaxes(np.linalg.inv(batch), -1, -2)
    denominator = 2.0 * mu * f33 + lam / f33
    derivative = -lam * inverse_transpose / denominator[:, None, None]
    return derivative.reshape((*original_shape, 2, 2))


def first_piola_plane_stress(
    f2: FloatArray,
    youngs_modulus: float = DEFAULT_E,
    poisson_ratio: float = DEFAULT_NU,
) -> FloatArray:
    """Return the reduced in-plane first-Piola stress."""
    batch, original_shape = _as_deformation_batch(f2)
    mu, _ = lame_parameters(youngs_modulus, poisson_ratio)
    f33 = plane_stress_f33(batch, youngs_modulus, poisson_ratio).reshape(-1)
    inverse_transpose = np.swapaxes(np.linalg.inv(batch), -1, -2)
    stress = mu * (batch - f33[:, None, None] ** 2 * inverse_transpose)
    return stress.reshape((*original_shape, 2, 2))


def plane_stress_tangent(
    f2: FloatArray,
    youngs_modulus: float = DEFAULT_E,
    poisson_ratio: float = DEFAULT_NU,
) -> FloatArray:
    """Return the consistent reduced tangent dP_iI/dF_kK."""
    batch, original_shape = _as_deformation_batch(f2)
    mu, lam = lame_parameters(youngs_modulus, poisson_ratio)
    f33 = plane_stress_f33(batch, youngs_modulus, poisson_ratio).reshape(-1)
    inverse_transpose = np.swapaxes(np.linalg.inv(batch), -1, -2)
    identity = np.eye(2)
    material_identity = np.einsum("ik,IK->iIkK", identity, identity)
    inverse_term = np.einsum(
        "niK,nkI->niIkK",
        inverse_transpose,
        inverse_transpose,
    )
    volumetric_term = np.einsum(
        "niI,nkK->niIkK",
        inverse_transpose,
        inverse_transpose,
    )
    gamma = 2.0 * mu * lam * f33 / (2.0 * mu * f33 + lam / f33)
    tangent = (
        mu * material_identity[None]
        + mu * f33[:, None, None, None, None] ** 2 * inverse_term
        + gamma[:, None, None, None, None] * volumetric_term
    )
    return tangent.reshape((*original_shape, 2, 2, 2, 2))


def _matrix_field_to_last(field: FloatArray) -> FloatArray:
    return np.moveaxis(field, (0, 1), (-2, -1))


def _matrix_field_to_first(field: FloatArray) -> FloatArray:
    return np.moveaxis(field, (-2, -1), (0, 1))


def _tangent_field_to_first(field: FloatArray) -> FloatArray:
    return np.moveaxis(field, (-4, -3, -2, -1), (0, 1, 2, 3))


@LinearForm
def _internal_force(test, state):  # type: ignore[no-untyped-def]
    return np.einsum("iI...,iI...->...", state.stress, test.grad)


@BilinearForm
def _consistent_stiffness(trial, test, state):  # type: ignore[no-untyped-def]
    return np.einsum(
        "iIkK...,kK...,iI...->...",
        state.tangent,
        trial.grad,
        test.grad,
    )


def _constitutive_fields(
    basis: Basis,
    displacement_vector: FloatArray,
    *,
    include_tangent: bool,
) -> tuple[FloatArray, FloatArray | None]:
    gradient = basis.interpolate(displacement_vector).grad
    identity = np.eye(2)[:, :, None, None]
    deformation = _matrix_field_to_last(identity + gradient)
    stress = _matrix_field_to_first(first_piola_plane_stress(deformation))
    if not include_tangent:
        return stress, None
    tangent = _tangent_field_to_first(plane_stress_tangent(deformation))
    return stress, tangent


def _residual_and_stiffness(
    basis: Basis,
    displacement_vector: FloatArray,
) -> tuple[FloatArray, object]:
    stress, tangent = _constitutive_fields(
        basis,
        displacement_vector,
        include_tangent=True,
    )
    if tangent is None:
        raise AssertionError("tangent requested but not computed")
    residual = asm(_internal_force, basis, stress=stress)
    stiffness = asm(_consistent_stiffness, basis, tangent=tangent)
    return residual, stiffness


def _minimum_jacobian(basis: Basis, displacement_vector: FloatArray) -> float:
    gradient = basis.interpolate(displacement_vector).grad
    deformation = _matrix_field_to_last(np.eye(2)[:, :, None, None] + gradient)
    return float(np.min(np.linalg.det(deformation)))


def _solve_displacement_control(
    mesh: MeshTri,
    target_displacement: float,
    *,
    load_steps: int,
    tolerance: float,
    max_newton_iterations: int,
) -> tuple[bool, str | None, FloatArray, Basis, int]:
    basis = Basis(mesh, ElementVector(ElementTriP1()), intorder=3)
    scale = max(float(np.ptp(mesh.p[0])), float(np.ptp(mesh.p[1])), 1.0)
    coordinate_tolerance = 1e-10 * scale
    x_max = float(np.max(mesh.p[0]))
    left_x = basis.get_dofs(
        lambda x: np.isclose(x[0], 0.0, atol=coordinate_tolerance)
    ).all("u^1")
    bottom_y = basis.get_dofs(
        lambda x: np.isclose(x[1], 0.0, atol=coordinate_tolerance)
    ).all("u^2")
    right_x = basis.get_dofs(
        lambda x: np.isclose(x[0], x_max, atol=coordinate_tolerance)
    ).all("u^1")
    constrained = np.unique(np.concatenate((left_x, bottom_y, right_x)))
    free = np.setdiff1d(np.arange(basis.N), constrained)
    displacement = np.zeros(basis.N)
    total_iterations = 0

    for step_index in range(1, load_steps + 1):
        prescribed = target_displacement * step_index / load_steps
        displacement[right_x] = prescribed

        for _ in range(max_newton_iterations):
            total_iterations += 1
            try:
                residual, stiffness = _residual_and_stiffness(basis, displacement)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                return (
                    False,
                    f"constitutive failure: {error}",
                    displacement,
                    basis,
                    total_iterations,
                )
            residual_norm = float(np.linalg.norm(residual[free]))
            reference = max(float(np.linalg.norm(residual)), 1.0)
            if residual_norm <= tolerance * reference:
                break

            try:
                increment = solve(*condense(stiffness, -residual, D=constrained))
            except Exception as error:  # scikit-fem delegates to SciPy backends
                return (
                    False,
                    f"linear solve failure: {error}",
                    displacement,
                    basis,
                    total_iterations,
                )

            accepted = False
            damping = 1.0
            for _ in range(16):
                candidate = displacement + damping * increment
                candidate[right_x] = prescribed
                if _minimum_jacobian(basis, candidate) <= MINIMUM_JACOBIAN:
                    damping *= 0.5
                    continue
                try:
                    candidate_stress, _ = _constitutive_fields(
                        basis,
                        candidate,
                        include_tangent=False,
                    )
                    candidate_residual = asm(
                        _internal_force, basis, stress=candidate_stress
                    )
                except (ValueError, RuntimeError, np.linalg.LinAlgError):
                    damping *= 0.5
                    continue
                candidate_norm = float(np.linalg.norm(candidate_residual[free]))
                if candidate_norm < residual_norm or damping <= 2.0**-12:
                    displacement = candidate
                    accepted = True
                    break
                damping *= 0.5
            if not accepted:
                return (
                    False,
                    "Newton line search failed",
                    displacement,
                    basis,
                    total_iterations,
                )
        else:
            return (
                False,
                "Newton iteration limit reached",
                displacement,
                basis,
                total_iterations,
            )

    return True, None, displacement, basis, total_iterations


def _nodal_fields(
    basis: Basis, displacement_vector: FloatArray
) -> tuple[FloatArray, FloatArray]:
    displacement = displacement_vector[basis.nodal_dofs].T.copy()
    gradient = basis.interpolate(displacement_vector).grad
    deformation = _matrix_field_to_last(np.eye(2)[:, :, None, None] + gradient)
    element_stress = first_piola_plane_stress(deformation).mean(axis=1)

    nodal_stress = np.zeros((basis.mesh.nvertices, 2, 2))
    counts = np.zeros(basis.mesh.nvertices)
    for local_vertex in range(3):
        vertices = basis.mesh.t[local_vertex]
        np.add.at(nodal_stress, vertices, element_stress)
        np.add.at(counts, vertices, 1.0)
    nodal_stress = nodal_stress / counts[:, None, None]
    return displacement, nodal_stress


def solve_case(
    spike_mesh: SpikeMesh,
    *,
    applied_displacement: float = DEFAULT_APPLIED_DISPLACEMENT,
    axial_strain: float | None = None,
    load_steps: int = 5,
    tolerance: float = 1e-10,
    max_newton_iterations: int = 30,
) -> FEMResult:
    """Solve with upstream's 1 mm loading, or an explicit strain override."""
    started = perf_counter()
    mesh = MeshTri(spike_mesh.points.T, spike_mesh.cells.T)
    x_max = float(np.max(spike_mesh.points[:, 0]))
    target = (
        axial_strain * x_max
        if axial_strain is not None
        else float(applied_displacement)
    )
    if target <= 0.0 or not math.isfinite(target):
        raise ValueError("target displacement must be finite and positive")
    converged, reason, vector, basis, iterations = _solve_displacement_control(
        mesh,
        target,
        load_steps=load_steps,
        tolerance=tolerance,
        max_newton_iterations=max_newton_iterations,
    )
    elapsed = perf_counter() - started
    if not converged:
        return FEMResult(False, reason, None, None, elapsed, iterations)
    displacement, stress = _nodal_fields(basis, vector)
    return FEMResult(True, None, displacement, stress, elapsed, iterations)


def _uniaxial_lateral_stretch(axial_stretch: float) -> float:
    mu, lam = lame_parameters(DEFAULT_E, DEFAULT_NU)
    lower, upper = 0.2, 2.0
    for _ in range(120):
        midpoint = 0.5 * (lower + upper)
        residual = mu * (midpoint**2 - 1.0) + lam * math.log(
            axial_stretch * midpoint**2
        )
        if residual > 0.0:
            upper = midpoint
        else:
            lower = midpoint
    return 0.5 * (lower + upper)


def solve_uniaxial_square(
    *,
    axial_stretch: float = 1.10,
    nx: int = 6,
    ny: int = 6,
) -> SanityResult:
    """Run the actual FEM solver and compare it with the affine closed form."""
    if axial_stretch <= 0.0:
        raise ValueError("axial_stretch must be positive")
    mesh = MeshTri.init_tensor(
        np.linspace(0.0, 1.0, nx + 1), np.linspace(0.0, 1.0, ny + 1)
    )
    converged, reason, vector, basis, _ = _solve_displacement_control(
        mesh,
        axial_stretch - 1.0,
        load_steps=5,
        tolerance=1e-12,
        max_newton_iterations=40,
    )
    if not converged:
        return SanityResult(False, reason, math.nan, math.inf, math.inf, math.inf)

    displacement, stress = _nodal_fields(basis, vector)
    coordinates = basis.mesh.p.T
    lateral = _uniaxial_lateral_stretch(axial_stretch)
    expected_displacement = np.column_stack(
        (
            (axial_stretch - 1.0) * coordinates[:, 0],
            (lateral - 1.0) * coordinates[:, 1],
        )
    )
    expected_stress_value = first_piola_plane_stress(np.diag([axial_stretch, lateral]))
    expected_stress = np.broadcast_to(expected_stress_value, stress.shape)
    relative_u = float(
        np.linalg.norm(displacement - expected_displacement)
        / np.linalg.norm(expected_displacement)
    )
    relative_p = float(
        np.linalg.norm(stress - expected_stress) / np.linalg.norm(expected_stress)
    )
    top = np.isclose(coordinates[:, 1], 1.0)
    measured_lateral = 1.0 + float(np.mean(displacement[top, 1]))
    lateral_error = abs(measured_lateral - lateral) / abs(lateral)
    return SanityResult(
        True,
        None,
        measured_lateral,
        relative_u,
        relative_p,
        max(relative_u, relative_p, lateral_error),
    )
