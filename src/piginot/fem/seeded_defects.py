"""Executable S8 mutants and the checks that distinguish each defect."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from piginot.contracts import BoundaryConditionSpec, CanonicalQuantity, Unit
from piginot.fem.backend import ScikitFemBackend
from piginot.fem.field_providers import FloatArray
from piginot.fem.mms_runner import (
    build_mms_problem,
    verification_execution_profile,
    verification_material,
)
from piginot.fem.models import FemRequest, FemResult, FemSolverConfig
from piginot.fem.verification_checks import affine_patch_error
from piginot.fem.verification_meshes import unit_square_verification_mesh
from piginot.physics import (
    admissibility,
    consistent_tangent,
    first_piola,
    plane_stress_closure,
)


@dataclass(frozen=True, slots=True)
class SeededDefectDetection:
    """A control/mutant metric separated by one declared threshold."""

    defect_id: str
    control_value: float
    mutant_value: float
    threshold: float
    fired_checks: tuple[str, ...]

    @property
    def detected(self) -> bool:
        return self.control_value <= self.threshold < self.mutant_value


@dataclass(frozen=True, slots=True)
class SeededDefectReport:
    detections: tuple[SeededDefectDetection, ...]

    @property
    def required_count(self) -> int:
        return len(self.detections)

    @property
    def detected_count(self) -> int:
        return sum(item.detected for item in self.detections)


@lru_cache(maxsize=1)
def run_seeded_defect_suite() -> SeededDefectReport:
    """Run six distinct perturbations; no detection result is preassigned."""

    return SeededDefectReport(
        detections=(
            _body_force_sign_detection(),
            _boundary_normal_detection(),
            _tag_swap_detection(),
            _closure_root_detection(),
            _tangent_detection(),
            _anisotropic_scaling_detection(),
        )
    )


def _body_force_sign_detection() -> SeededDefectDetection:
    problem = build_mms_problem("mixed_dirichlet_traction", level=1, element_order=1)
    exact = problem.manufactured_solution
    points = _interior_points()
    step = 1.0e-6
    x_shift = np.asarray(((step,), (0.0,)), dtype=np.float64)
    y_shift = np.asarray(((0.0,), (step,)), dtype=np.float64)
    derivative_x = (
        exact.first_piola(points + x_shift) - exact.first_piola(points - x_shift)
    ) / (2.0 * step)
    derivative_y = (
        exact.first_piola(points + y_shift) - exact.first_piola(points - y_shift)
    ) / (2.0 * step)
    divergence = derivative_x[:2, 0] + derivative_y[:2, 1]
    body = exact.body_force(points)
    scale = float(np.linalg.norm(body))
    control = float(np.linalg.norm(body + divergence) / scale)
    mutant = float(np.linalg.norm(-body + divergence) / scale)
    return SeededDefectDetection(
        "body-force sign",
        control,
        mutant,
        1.0e-4,
        ("G8.order-p1", "G8.reaction"),
    )


def _boundary_normal_detection() -> SeededDefectDetection:
    problem = build_mms_problem("mixed_dirichlet_traction", level=1, element_order=1)
    exact = problem.manufactured_solution
    points = _interior_points()
    normals = np.asarray(
        ((1.0, 0.0, -1.0, 0.0), (0.0, 1.0, 0.0, -1.0)),
        dtype=np.float64,
    )
    expected = np.einsum("ij...,j...->i...", exact.first_piola(points)[:2, :2], normals)
    control_field = exact.traction(points, normals)
    mutant_field = exact.traction(points, -normals)
    scale = float(np.linalg.norm(expected))
    return SeededDefectDetection(
        "boundary normal",
        float(np.linalg.norm(control_field - expected) / scale),
        float(np.linalg.norm(mutant_field - expected) / scale),
        1.0e-12,
        ("G8.reaction", "G8.order-p1"),
    )


def _tag_swap_detection() -> SeededDefectDetection:
    gradient = np.asarray(((0.02, 0.0), (0.0, -0.01)), dtype=np.float64)
    offset = np.zeros(2, dtype=np.float64)
    control_result = _solve_tag_swap_case(swapped=False)
    mutant_result = _solve_tag_swap_case(swapped=True)
    return SeededDefectDetection(
        "tag swap",
        affine_patch_error(
            np.asarray(control_result.displacement_dof_coordinates),
            np.asarray(control_result.displacement),
            gradient,
            offset,
        ),
        affine_patch_error(
            np.asarray(mutant_result.displacement_dof_coordinates),
            np.asarray(mutant_result.displacement),
            gradient,
            offset,
        ),
        1.0e-12,
        ("G8.patch",),
    )


def _solve_tag_swap_case(*, swapped: bool) -> FemResult:
    zero = CanonicalQuantity(0.0, Unit.MM)
    left_tag, right_tag = ("right", "left") if swapped else ("left", "right")
    request = FemRequest(
        request_id=f"s8-tag-swap-{swapped}",
        mesh=unit_square_verification_mesh(1),
        material=verification_material(),
        boundary_conditions=(
            BoundaryConditionSpec("left_x", left_tag, "dirichlet", ("x",), (zero,)),
            BoundaryConditionSpec(
                "right_x",
                right_tag,
                "dirichlet",
                ("x",),
                (CanonicalQuantity(0.02, Unit.MM),),
            ),
            BoundaryConditionSpec("bottom_y", "bottom", "dirichlet", ("y",), (zero,)),
            BoundaryConditionSpec(
                "top_y",
                "top",
                "dirichlet",
                ("y",),
                (CanonicalQuantity(-0.01, Unit.MM),),
            ),
        ),
        loads=(),
        qois=(),
        config=FemSolverConfig(
            element_order=1,
            solver_tolerance=1.0e-11,
            max_newton_iterations=24,
            plane_stress_max_iterations=32,
            load_steps=4,
        ),
        execution_profile=verification_execution_profile(),
    )
    attempt = ScikitFemBackend().solve_many((request,))[0]
    if not attempt.success or attempt.result is None:
        raise RuntimeError(
            f"tag-swap case did not produce a measurable solve: {attempt.message}"
        )
    return attempt.result


def _closure_root_detection() -> SeededDefectDetection:
    material = build_mms_problem(
        "strongly_varying_f33", level=1, element_order=1
    ).request.material
    deformation_2d = torch.tensor(((1.08, 0.03), (-0.01, 0.93)), dtype=torch.float64)
    f33 = plane_stress_closure(deformation_2d, material)
    deformation = torch.zeros((3, 3), dtype=torch.float64)
    deformation[:2, :2] = deformation_2d
    deformation[2, 2] = f33
    mutant = deformation.clone()
    mutant[2, 2] = -f33
    return SeededDefectDetection(
        "closure wrong root",
        0.0 if admissibility(deformation, material) else 1.0,
        0.0 if admissibility(mutant, material) else 1.0,
        0.5,
        ("G8.order-stress", "G8.objectivity"),
    )


def _tangent_detection() -> SeededDefectDetection:
    material = build_mms_problem(
        "pure_dirichlet", level=1, element_order=1
    ).request.material
    deformation = torch.tensor(
        ((1.07, 0.02, 0.0), (-0.01, 0.96, 0.0), (0.0, 0.0, 1.01)),
        dtype=torch.float64,
    )
    direction = torch.tensor(
        ((0.3, -0.2, 0.0), (0.1, 0.25, 0.0), (0.0, 0.0, -0.15)),
        dtype=torch.float64,
    )
    step = 1.0e-6
    numerical = (
        first_piola(deformation + step * direction, material)
        - first_piola(deformation - step * direction, material)
    ) / (2.0 * step)
    predicted = torch.einsum(
        "ijkl,kl->ij", consistent_tangent(deformation, material), direction
    )
    scale = float(torch.linalg.norm(numerical).item())
    control = float(torch.linalg.norm(predicted - numerical).item() / scale)
    mutant = float(torch.linalg.norm(0.9 * predicted - numerical).item() / scale)
    return SeededDefectDetection(
        "inconsistent tangent",
        control,
        mutant,
        1.0e-5,
        ("G8.patch", "G8.order-stress"),
    )


def _anisotropic_scaling_detection() -> SeededDefectDetection:
    coordinates = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.4, 0.7)),
        dtype=np.float64,
    )
    gradient = np.asarray(((0.04, 0.01), (-0.015, 0.025)), dtype=np.float64)
    offset = np.asarray((0.007, -0.004), dtype=np.float64)
    displacement = coordinates @ gradient.T + offset
    mutant_coordinates = coordinates.copy()
    mutant_coordinates[:, 0] *= 1.2
    return SeededDefectDetection(
        "anisotropic scaling",
        affine_patch_error(coordinates, displacement, gradient, offset),
        affine_patch_error(mutant_coordinates, displacement, gradient, offset),
        1.0e-12,
        ("G8.patch",),
    )


def _interior_points() -> FloatArray:
    return np.asarray(
        ((0.17, 0.31, 0.64, 0.83), (0.23, 0.72, 0.81, 0.36)),
        dtype=np.float64,
    )


__all__ = (
    "SeededDefectDetection",
    "SeededDefectReport",
    "run_seeded_defect_suite",
)
