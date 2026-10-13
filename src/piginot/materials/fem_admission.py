"""Admission check 8: optimal FEM convergence on a manufactured solution.

Needs the ``verify`` extra. The manufactured field is the S8
``strongly_varying_f33`` displacement on the unit square (stretches up to
about 1.6, thickness stretch varying across the square), with Dirichlet data
on all four sides. Unlike S8, which derives the body force symbolically for
the closed-form Neo-Hookean model, it is derived here by automatic
differentiation of the material's own kernel, ``b = -Div P(I + grad u)``,
so the check applies to any registered model. The reference solver must
converge at the optimal P2 orders: displacement L2 order 3 and first-Piola
L2 order 2, with the S8 tolerances (0.15 and 0.2, ADR 0002).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from piginot.contracts import BoundaryConditionSpec, LoadSpec, MaterialSpec, hash_spec
from piginot.materials.admission import CheckResult
from piginot.materials.hyperelastic import embed

DISPLACEMENT_ORDER = 3.0
STRESS_ORDER = 2.0
DISPLACEMENT_ORDER_TOLERANCE = 0.15
STRESS_ORDER_TOLERANCE = 0.2
LEVELS = (3, 4)
CASE = "strongly_varying_f33"
BOUNDARY_TYPES = {tag: "dirichlet" for tag in ("left", "right", "top", "bottom")}


def _displacement(points: torch.Tensor) -> torch.Tensor:
    x, y = points[0], points[1]
    return torch.stack(
        (0.2 * x**2 * (1.0 + y / 2.0), -0.15 * y**2 * (1.0 - x / 4.0)), dim=0
    )


def _points(reference_points: Any) -> torch.Tensor:
    array = np.asarray(reference_points, dtype=np.float64)
    if array.ndim < 1 or array.shape[0] != 2 or not np.isfinite(array).all():
        raise ValueError("reference points must be finite with shape (2, ...)")
    return torch.from_numpy(np.ascontiguousarray(array))


@dataclass(frozen=True, slots=True)
class AutogradManufacturedSolution:
    """The S8 field set for any material, by automatic differentiation.

    Evaluators take reference points of shape ``(2, ...)`` and return
    component-leading arrays, as :class:`piginot.fem.manufactured.ManufacturedSolution`.
    """

    material: MaterialSpec
    name: str = CASE

    def displacement(self, reference_points: Any) -> Any:
        return _displacement(_points(reference_points)).numpy()

    def _fields(self, reference_points: Any, *, body: bool) -> dict[str, torch.Tensor]:
        from piginot.physics import first_piola, plane_stress_closure

        points = _points(reference_points)
        shape = points.shape[1:]
        flat = points.reshape(2, -1).T.clone().requires_grad_(True)
        u = _displacement(flat.T)
        rows = [
            torch.autograd.grad(u[i].sum(), flat, create_graph=body, retain_graph=True)[
                0
            ]
            for i in range(2)
        ]
        gradient = torch.stack(rows, dim=-2)
        deformation = torch.eye(2, dtype=torch.float64) + gradient
        f33 = plane_stress_closure(deformation, self.material)
        stress = first_piola(embed(deformation, f33), self.material)
        fields = {
            "gradient": gradient.detach(),
            "f33": f33.detach(),
            "stress": stress.detach(),
        }
        if body:
            force: list[torch.Tensor] = []
            for i in range(2):
                divergence = torch.zeros_like(f33)
                for j in range(2):
                    (derivative,) = torch.autograd.grad(
                        stress[:, i, j].sum(), flat, retain_graph=True
                    )
                    divergence = divergence + derivative[:, j]
                force.append(-divergence)
            fields["body"] = torch.stack(force, dim=-1).detach()
        return {
            key: value.reshape(*shape, *value.shape[1:])
            for key, value in fields.items()
        }

    @staticmethod
    def _leading(values: torch.Tensor, components: int) -> Any:
        array = values.numpy()
        point_axes = array.ndim - components
        order = (*range(point_axes, array.ndim), *range(point_axes))
        return np.ascontiguousarray(np.transpose(array, order))

    def gradient(self, reference_points: Any) -> Any:
        return self._leading(self._fields(reference_points, body=False)["gradient"], 2)

    def first_piola(self, reference_points: Any) -> Any:
        return self._leading(self._fields(reference_points, body=False)["stress"], 2)

    def thickness_stretch(self, reference_points: Any) -> Any:
        return self._fields(reference_points, body=False)["f33"].numpy()

    def body_force(self, reference_points: Any) -> Any:
        return self._leading(self._fields(reference_points, body=True)["body"], 1)


@dataclass(frozen=True, slots=True)
class _Provider:
    kind: str
    solution: AutogradManufacturedSolution

    @property
    def provider_id(self) -> str:
        return hash_spec(
            {
                "provider_kind": f"{self.kind}-autograd-v1",
                "case_name": CASE,
                "material": self.solution.material,
            },
            "recipe",
        )

    def evaluate(self, reference_points: Any) -> Any:
        if self.kind == "displacement":
            return self.solution.displacement(reference_points)
        return self.solution.body_force(reference_points)


def solve_level(material: MaterialSpec, level: int) -> tuple[float, float, float]:
    """``(h_max, displacement L2 error, first-Piola L2 error)`` at one level."""

    from piginot.fem.backend import ScikitFemBackend
    from piginot.fem.mms_runner import verification_execution_profile
    from piginot.fem.models import FemRequest, FemSolverConfig
    from piginot.fem.verification_meshes import (
        unit_square_verification_mesh,
        verification_mesh_metadata,
    )
    from piginot.fem.verification_norms import compute_verification_norms

    solution = AutogradManufacturedSolution(material)
    displacement = _Provider("displacement", solution)
    body = _Provider("body-force", solution)
    mesh = unit_square_verification_mesh(level)
    condition = BoundaryConditionSpec(
        "manufactured_displacement", "all", "dirichlet", ("x", "y")
    )
    load = LoadSpec("manufactured_body_force", "body_force")
    request = FemRequest(
        request_id=f"s25-{CASE}-{material.model}-l{level}-p2",
        mesh=mesh,
        material=material,
        boundary_conditions=(condition,),
        loads=(load,),
        qois=(),
        config=FemSolverConfig(
            element_order=2,
            solver_tolerance=1.0e-11,
            max_newton_iterations=28,
            plane_stress_max_iterations=32,
            load_steps=16,
            max_dofs=100_000,
        ),
        execution_profile=verification_execution_profile(),
        body_force_provider_ids={load.name: body.provider_id},
        boundary_field_provider_ids={condition.name: displacement.provider_id},
    )
    backend = ScikitFemBackend(
        body_force_providers=(body,),
        dirichlet_field_providers=(displacement,),
    )
    attempt = backend.solve_many((request,))[0]
    if not attempt.success or attempt.result is None:
        raise RuntimeError(f"manufactured solve failed: {attempt.message}")
    norms = compute_verification_norms(
        request,
        attempt.result,
        solution,  # type: ignore[arg-type]
    )
    return (
        verification_mesh_metadata(mesh).h_max_mm,
        norms.displacement_l2_error,
        norms.first_piola_l2_error,
    )


def check_fem(material: MaterialSpec, levels: tuple[int, int] = LEVELS) -> CheckResult:
    """Admission check 8 for ``material``."""

    try:
        coarse, fine = (solve_level(material, level) for level in levels)
    except RuntimeError as exc:
        return CheckResult("fem", math.inf, 0.0, False, {"error": str(exc)})
    ratio = math.log(coarse[0] / fine[0])
    displacement = math.log(coarse[1] / fine[1]) / ratio
    stress = math.log(coarse[2] / fine[2]) / ratio
    deviation = max(
        abs(displacement - DISPLACEMENT_ORDER) / DISPLACEMENT_ORDER_TOLERANCE,
        abs(stress - STRESS_ORDER) / STRESS_ORDER_TOLERANCE,
    )
    return CheckResult(
        "fem",
        deviation,
        1.0,
        deviation <= 1.0,
        {
            "case": CASE,
            "levels": list(levels),
            "displacement_l2_order": displacement,
            "first_piola_l2_order": stress,
            "displacement_order_tolerance": DISPLACEMENT_ORDER_TOLERANCE,
            "stress_order_tolerance": STRESS_ORDER_TOLERANCE,
        },
    )


__all__ = (
    "BOUNDARY_TYPES",
    "CASE",
    "LEVELS",
    "AutogradManufacturedSolution",
    "check_fem",
    "solve_level",
)
