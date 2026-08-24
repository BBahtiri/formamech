"""Exact manufactured solutions for nonlinear plane-stress FEM verification.

The fields in this module are derived symbolically on the reference unit square.
They use the same compressible Neo-Hookean energy and three-dimensional Lamé
parameterization as :mod:`piginot.physics`, with the positive plane-stress root
written in closed form using the principal Lambert-W branch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

import numpy as np
import sympy as sp  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy.special import lambertw  # type: ignore[import-untyped]

from piginot.contracts import MaterialSpec
from piginot.contracts.specs import FrozenMap
from piginot.physics._validation import validate_material

FloatArray = NDArray[np.float64]
SymbolicVector = tuple[sp.Expr, sp.Expr]
SymbolicMatrix2 = tuple[tuple[sp.Expr, sp.Expr], tuple[sp.Expr, sp.Expr]]
SymbolicMatrix3 = tuple[
    tuple[sp.Expr, sp.Expr, sp.Expr],
    tuple[sp.Expr, sp.Expr, sp.Expr],
    tuple[sp.Expr, sp.Expr, sp.Expr],
]

REFERENCE_X, REFERENCE_Y = sp.symbols("X Y", real=True)
_BOUNDARY_TAGS = ("left", "right", "top", "bottom")


class BoundaryKind(StrEnum):
    """Physical boundary type prescribed by a manufactured case."""

    DIRICHLET = "dirichlet"
    TRACTION = "traction"


@dataclass(frozen=True, slots=True)
class ManufacturedSolution:
    """Symbolic MMS fields plus deterministic component-leading evaluators.

    Reference points have shape ``(2, ...)``.  Vector and tensor results preserve
    the trailing point axes, e.g. stress has shape ``(3, 3, ...)``.  Stress is in
    MPa and body force is in N/mm³ for coordinates expressed in mm.
    """

    name: str
    material: MaterialSpec
    boundary_types: Mapping[str, BoundaryKind]
    rigid_mode_constraints: tuple[str, ...]
    displacement_symbolic: SymbolicVector
    gradient_symbolic: SymbolicMatrix2
    deformation_gradient_symbolic: SymbolicMatrix2
    thickness_stretch_symbolic: sp.Expr
    first_piola_symbolic: SymbolicMatrix3
    body_force_symbolic: SymbolicVector
    plane_stress_residual_symbolic: sp.Expr

    def __post_init__(self) -> None:
        boundaries: FrozenMap[BoundaryKind] = FrozenMap.copy_of(self.boundary_types)
        if set(boundaries) != set(_BOUNDARY_TAGS):
            raise ValueError("manufactured cases must type all unit-square boundaries")
        if any(not isinstance(kind, BoundaryKind) for kind in boundaries.values()):
            raise ValueError("boundary types must be BoundaryKind values")
        object.__setattr__(self, "boundary_types", boundaries)
        constraints = tuple(self.rigid_mode_constraints)
        if any(
            constraint not in {"mean_x", "mean_y", "mean_rotation"}
            for constraint in constraints
        ):
            raise ValueError("unknown rigid-mode constraint")
        object.__setattr__(self, "rigid_mode_constraints", constraints)

    def displacement(self, reference_points: FloatArray) -> FloatArray:
        return _evaluate(self.displacement_symbolic, (2,), reference_points)

    def gradient(self, reference_points: FloatArray) -> FloatArray:
        return _evaluate(_flatten2(self.gradient_symbolic), (2, 2), reference_points)

    def deformation_gradient(self, reference_points: FloatArray) -> FloatArray:
        return _evaluate(
            _flatten2(self.deformation_gradient_symbolic),
            (2, 2),
            reference_points,
        )

    def thickness_stretch(self, reference_points: FloatArray) -> FloatArray:
        return _evaluate_scalar(self.thickness_stretch_symbolic, reference_points)

    def first_piola(self, reference_points: FloatArray) -> FloatArray:
        return _evaluate(_flatten3(self.first_piola_symbolic), (3, 3), reference_points)

    def body_force(self, reference_points: FloatArray) -> FloatArray:
        return _evaluate(self.body_force_symbolic, (2,), reference_points)

    def plane_stress_residual(self, reference_points: FloatArray) -> FloatArray:
        return _evaluate_scalar(
            self.plane_stress_residual_symbolic,
            reference_points,
        )

    def traction(
        self,
        reference_points: FloatArray,
        outward_normals: FloatArray,
    ) -> FloatArray:
        points = _reference_points(reference_points)
        normals = np.asarray(outward_normals, dtype=np.float64)
        if normals.shape != points.shape:
            raise ValueError(
                "outward normals must have the same shape as reference points"
            )
        if not np.isfinite(normals).all():
            raise ValueError("outward normals must contain only finite values")
        stress = self.first_piola(points)[:2, :2]
        result = np.einsum("ij...,j...->i...", stress, normals, optimize=True)
        return np.asarray(result, dtype=np.float64)


MANUFACTURED_CASE_NAMES = (
    "pure_dirichlet",
    "mixed_dirichlet_traction",
    "pure_traction_body_force",
    "strongly_varying_f33",
)


def _displacement(case_name: str) -> SymbolicVector:
    x_coordinate = REFERENCE_X
    y_coordinate = REFERENCE_Y
    if case_name == "pure_dirichlet":
        bubble = x_coordinate * (1 - x_coordinate) * y_coordinate * (1 - y_coordinate)
        return (
            sp.Rational(1, 50) * bubble * (1 + y_coordinate),
            -sp.Rational(3, 200) * bubble * (1 + x_coordinate),
        )
    if case_name == "mixed_dirichlet_traction":
        return (
            sp.Rational(3, 100) * x_coordinate**2 * (1 + y_coordinate / 2)
            + sp.Rational(1, 100) * x_coordinate * y_coordinate * (1 - y_coordinate),
            -sp.Rational(1, 50) * x_coordinate * y_coordinate * (1 - y_coordinate / 2)
            + sp.Rational(1, 100) * x_coordinate**2 * y_coordinate,
        )
    if case_name == "pure_traction_body_force":
        return _remove_rigid_modes(
            (
                sp.Rational(1, 50) * x_coordinate**2
                + sp.Rational(1, 100) * x_coordinate * y_coordinate
                + sp.Rational(1, 300) * x_coordinate**2 * y_coordinate,
                -sp.Rational(3, 200) * y_coordinate**2
                + sp.Rational(1, 125) * x_coordinate * y_coordinate
                + sp.Rational(1, 280) * x_coordinate * y_coordinate**2,
            )
        )
    if case_name == "strongly_varying_f33":
        return (
            sp.Rational(1, 5) * x_coordinate**2 * (1 + y_coordinate / 2),
            -sp.Rational(3, 20) * y_coordinate**2 * (1 - x_coordinate / 4),
        )
    raise ValueError(
        f"unknown manufactured case {case_name!r}; expected one of "
        f"{', '.join(MANUFACTURED_CASE_NAMES)}"
    )


def _remove_rigid_modes(displacement: SymbolicVector) -> SymbolicVector:
    ux, uy = displacement
    x_coordinate = REFERENCE_X
    y_coordinate = REFERENCE_Y
    mean_x = sp.integrate(ux, (x_coordinate, 0, 1), (y_coordinate, 0, 1))
    mean_y = sp.integrate(uy, (x_coordinate, 0, 1), (y_coordinate, 0, 1))
    moment = sp.integrate(
        x_coordinate * uy - y_coordinate * ux,
        (x_coordinate, 0, 1),
        (y_coordinate, 0, 1),
    )
    rotation = -6 * (moment + (mean_x - mean_y) / 2)
    translation_x = -mean_x + rotation / 2
    translation_y = -mean_y - rotation / 2
    return (
        sp.expand(ux + translation_x - rotation * y_coordinate),
        sp.expand(uy + translation_y + rotation * x_coordinate),
    )


def _case_metadata(
    case_name: str,
) -> tuple[dict[str, BoundaryKind], tuple[str, ...]]:
    if case_name in {"pure_dirichlet", "strongly_varying_f33"}:
        return ({tag: BoundaryKind.DIRICHLET for tag in _BOUNDARY_TAGS}, ())
    if case_name == "mixed_dirichlet_traction":
        return (
            {
                "left": BoundaryKind.DIRICHLET,
                "right": BoundaryKind.TRACTION,
                "top": BoundaryKind.TRACTION,
                "bottom": BoundaryKind.TRACTION,
            },
            (),
        )
    if case_name == "pure_traction_body_force":
        return (
            {tag: BoundaryKind.TRACTION for tag in _BOUNDARY_TAGS},
            ("mean_x", "mean_y", "mean_rotation"),
        )
    raise ValueError(f"unknown manufactured case {case_name!r}")


@lru_cache(maxsize=32)
def build_manufactured_solution(
    case_name: str,
    material: MaterialSpec,
) -> ManufacturedSolution:
    """Build one exact symbolic plane-stress manufactured solution."""

    if case_name not in MANUFACTURED_CASE_NAMES:
        _displacement(case_name)
    validate_material(material)

    young = sp.Rational(str(material.properties["elastic_modulus"].value))
    poisson = sp.Rational(str(material.properties["poisson_ratio"].value))
    shear_modulus = young / (2 * (1 + poisson))
    first_lame = young * poisson / ((1 + poisson) * (1 - 2 * poisson))

    displacement = _displacement(case_name)
    gradient: SymbolicMatrix2 = (
        (
            sp.diff(displacement[0], REFERENCE_X),
            sp.diff(displacement[0], REFERENCE_Y),
        ),
        (
            sp.diff(displacement[1], REFERENCE_X),
            sp.diff(displacement[1], REFERENCE_Y),
        ),
    )
    deformation: SymbolicMatrix2 = (
        (1 + gradient[0][0], gradient[0][1]),
        (gradient[1][0], 1 + gradient[1][1]),
    )
    determinant_2d = sp.expand(
        deformation[0][0] * deformation[1][1] - deformation[0][1] * deformation[1][0]
    )

    if first_lame == 0:
        thickness_stretch: sp.Expr = sp.Integer(1)
    else:
        ratio = 2 * shear_modulus / first_lame
        thickness_stretch = sp.sqrt(
            sp.LambertW(ratio * sp.exp(ratio) / determinant_2d**2) / ratio
        )

    determinant_3d = determinant_2d * thickness_stretch
    coefficient = first_lame * sp.log(determinant_3d) - shear_modulus
    inverse_transpose: SymbolicMatrix2 = (
        (deformation[1][1] / determinant_2d, -deformation[1][0] / determinant_2d),
        (-deformation[0][1] / determinant_2d, deformation[0][0] / determinant_2d),
    )
    in_plane_stress: SymbolicMatrix2 = (
        (
            shear_modulus * deformation[0][0] + coefficient * inverse_transpose[0][0],
            shear_modulus * deformation[0][1] + coefficient * inverse_transpose[0][1],
        ),
        (
            shear_modulus * deformation[1][0] + coefficient * inverse_transpose[1][0],
            shear_modulus * deformation[1][1] + coefficient * inverse_transpose[1][1],
        ),
    )
    zero = sp.Integer(0)
    first_piola: SymbolicMatrix3 = (
        (in_plane_stress[0][0], in_plane_stress[0][1], zero),
        (in_plane_stress[1][0], in_plane_stress[1][1], zero),
        (zero, zero, zero),
    )
    body_force: SymbolicVector = (
        -sp.diff(in_plane_stress[0][0], REFERENCE_X)
        - sp.diff(in_plane_stress[0][1], REFERENCE_Y),
        -sp.diff(in_plane_stress[1][0], REFERENCE_X)
        - sp.diff(in_plane_stress[1][1], REFERENCE_Y),
    )
    closure_residual = shear_modulus * (thickness_stretch**2 - 1) + first_lame * sp.log(
        determinant_3d
    )
    if first_lame == 0:
        closure_residual = zero
    boundaries, rigid_mode_constraints = _case_metadata(case_name)
    return ManufacturedSolution(
        name=case_name,
        material=material,
        boundary_types=boundaries,
        rigid_mode_constraints=rigid_mode_constraints,
        displacement_symbolic=displacement,
        gradient_symbolic=gradient,
        deformation_gradient_symbolic=deformation,
        thickness_stretch_symbolic=thickness_stretch,
        first_piola_symbolic=first_piola,
        body_force_symbolic=body_force,
        plane_stress_residual_symbolic=closure_residual,
    )


def _flatten2(matrix: SymbolicMatrix2) -> tuple[sp.Expr, ...]:
    return tuple(value for row in matrix for value in row)


def _flatten3(matrix: SymbolicMatrix3) -> tuple[sp.Expr, ...]:
    return tuple(value for row in matrix for value in row)


def _reference_points(reference_points: FloatArray) -> FloatArray:
    points = np.asarray(reference_points, dtype=np.float64)
    if points.ndim < 1 or points.shape[0] != 2:
        raise ValueError("reference points must have shape (2, ...)")
    if not np.isfinite(points).all():
        raise ValueError("reference points must contain only finite values")
    return points


ScalarEvaluator = Callable[[FloatArray, FloatArray], object]


def _real_lambertw(argument: object) -> FloatArray:
    return np.asarray(np.real(lambertw(argument, k=0)), dtype=np.float64)


@lru_cache(maxsize=128)
def _compile(expressions: tuple[sp.Expr, ...]) -> tuple[ScalarEvaluator, ...]:
    modules = ({"LambertW": _real_lambertw}, "numpy")
    return tuple(
        sp.lambdify((REFERENCE_X, REFERENCE_Y), expression, modules=modules)
        for expression in expressions
    )


def _evaluate(
    expressions: tuple[sp.Expr, ...],
    component_shape: tuple[int, ...],
    reference_points: FloatArray,
) -> FloatArray:
    points = _reference_points(reference_points)
    point_shape = points.shape[1:]
    values: list[FloatArray] = []
    for evaluator in _compile(expressions):
        value = np.asarray(evaluator(points[0], points[1]), dtype=np.float64)
        values.append(np.broadcast_to(value, point_shape))
    stacked = np.stack(values).reshape((*component_shape, *point_shape))
    if not np.isfinite(stacked).all():
        raise ValueError("manufactured field evaluation produced a non-finite value")
    return np.asarray(stacked, dtype=np.float64)


def _evaluate_scalar(expression: sp.Expr, reference_points: FloatArray) -> FloatArray:
    points = _reference_points(reference_points)
    point_shape = points.shape[1:]
    evaluator = _compile((expression,))[0]
    value = np.asarray(evaluator(points[0], points[1]), dtype=np.float64)
    result = np.asarray(np.broadcast_to(value, point_shape), dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("manufactured field evaluation produced a non-finite value")
    return result


__all__ = (
    "BoundaryKind",
    "FloatArray",
    "MANUFACTURED_CASE_NAMES",
    "ManufacturedSolution",
    "REFERENCE_X",
    "REFERENCE_Y",
    "build_manufactured_solution",
)
