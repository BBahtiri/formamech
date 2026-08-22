"""Deterministic RBF and free-form-deformation morph maps."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from piginot.contracts import CanonicalQuantity, Unit, hash_spec
from piginot.morphing.models import (
    FFDControlPoint,
    Matrix2,
    MorphingError,
    MorphReasonCode,
    Point,
    RBFControlPoint,
)


def _configuration_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MorphingError(
            MorphReasonCode.INVALID_MAP_CONFIGURATION,
            f"{label} must be a real number",
        )
    number = float(value)
    if not math.isfinite(number):
        raise MorphingError(
            MorphReasonCode.INVALID_MAP_CONFIGURATION,
            f"{label} must be finite",
        )
    return number


def _evaluation_point(value: object) -> Point:
    if not isinstance(value, tuple) or len(value) != 2:
        raise MorphingError(
            MorphReasonCode.INVALID_MAP_CONFIGURATION,
            "evaluation point must be a two-coordinate tuple",
        )
    point = tuple(
        _configuration_number(item, "evaluation coordinate") for item in value
    )
    return point[0], point[1]


def _parameter_names(
    control_points: tuple[RBFControlPoint, ...] | tuple[FFDControlPoint, ...],
) -> tuple[str, ...]:
    names = {
        name
        for control in control_points
        for name in (control.x_parameter, control.y_parameter)
        if name is not None
    }
    return tuple(sorted(names))


def _parameter_values(
    expected_names: tuple[str, ...],
    theta: Mapping[str, CanonicalQuantity],
) -> tuple[Mapping[str, float], str]:
    if not isinstance(theta, Mapping):
        raise MorphingError(
            MorphReasonCode.INVALID_PARAMETER_VALUE,
            "theta must be a mapping of parameter names to canonical quantities",
        )

    submitted = dict(theta)
    unknown = tuple(
        sorted(repr(name) for name in submitted if name not in expected_names)
    )
    if unknown:
        raise MorphingError(
            MorphReasonCode.UNKNOWN_PARAMETER,
            f"unknown morph parameters: {', '.join(unknown)}",
        )
    missing = tuple(name for name in expected_names if name not in submitted)
    if missing:
        raise MorphingError(
            MorphReasonCode.MISSING_PARAMETER,
            f"missing morph parameters: {', '.join(missing)}",
        )

    values: dict[str, float] = {}
    for name in expected_names:
        quantity = submitted[name]
        if not isinstance(quantity, CanonicalQuantity):
            raise MorphingError(
                MorphReasonCode.INVALID_PARAMETER_VALUE,
                f"parameter {name!r} must be a CanonicalQuantity",
            )
        if quantity.unit is not Unit.MM:
            raise MorphingError(
                MorphReasonCode.PARAMETER_UNIT_MISMATCH,
                f"parameter {name!r} must use canonical millimetres",
            )
        values[name] = quantity.value
    canonical_theta = {name: submitted[name] for name in expected_names}
    return values, hash_spec(canonical_theta, domain="content")


def _control_displacements(
    control_points: tuple[RBFControlPoint, ...] | tuple[FFDControlPoint, ...],
    values: Mapping[str, float],
) -> tuple[Point, ...]:
    return tuple(
        (
            0.0 if control.x_parameter is None else values[control.x_parameter],
            0.0 if control.y_parameter is None else values[control.y_parameter],
        )
        for control in control_points
    )


@dataclass(frozen=True, slots=True)
class RBFMap:
    """Gaussian interpolatory RBF map with an affine polynomial tail."""

    kind: ClassVar[str] = "rbf"

    control_points: tuple[RBFControlPoint, ...]
    radius_mm: float

    def __post_init__(self) -> None:
        controls = tuple(self.control_points)
        if len(controls) < 3 or not all(
            isinstance(control, RBFControlPoint) for control in controls
        ):
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "RBF maps require at least three typed control points",
            )
        points = tuple(control.point for control in controls)
        if len(set(points)) != len(points):
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "RBF control-point coordinates must be unique",
            )
        if not _contains_noncollinear_triplet(points):
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "RBF control points must contain a non-collinear triplet",
            )

        radius = _configuration_number(self.radius_mm, "RBF radius_mm")
        if radius <= 0.0:
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "RBF radius_mm must be positive",
            )
        object.__setattr__(self, "control_points", controls)
        object.__setattr__(self, "radius_mm", radius)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return _parameter_names(self.control_points)

    def bind(self, theta: Mapping[str, CanonicalQuantity]) -> BoundRBFDeformation:
        values, parameter_hash = _parameter_values(self.parameter_names, theta)
        displacements = _control_displacements(self.control_points, values)
        centers = tuple(control.point for control in self.control_points)
        system = _rbf_system(centers, self.radius_mm)
        right_hand_side = np.vstack(
            (
                np.asarray(displacements, dtype=np.float64),
                np.zeros((3, 2), dtype=np.float64),
            )
        )
        try:
            coefficients = np.linalg.solve(system, right_hand_side)
        except np.linalg.LinAlgError as exc:
            raise MorphingError(
                MorphReasonCode.SINGULAR_RBF_SYSTEM,
                "RBF interpolation system is singular",
            ) from exc
        if not np.isfinite(coefficients).all():
            raise MorphingError(
                MorphReasonCode.SINGULAR_RBF_SYSTEM,
                "RBF interpolation produced non-finite coefficients",
            )

        radial_count = len(centers)
        radial_coefficients = tuple(
            (float(row[0]), float(row[1])) for row in coefficients[:radial_count]
        )
        affine_coefficients = (
            (
                float(coefficients[radial_count][0]),
                float(coefficients[radial_count][1]),
            ),
            (
                float(coefficients[radial_count + 1][0]),
                float(coefficients[radial_count + 1][1]),
            ),
            (
                float(coefficients[radial_count + 2][0]),
                float(coefficients[radial_count + 2][1]),
            ),
        )
        return BoundRBFDeformation(
            centers=centers,
            radius_mm=self.radius_mm,
            radial_coefficients=radial_coefficients,
            affine_coefficients=affine_coefficients,
            parameter_hash=parameter_hash,
        )


def _contains_noncollinear_triplet(points: tuple[Point, ...]) -> bool:
    origin = points[0]
    first_index = next(
        index for index, point in enumerate(points[1:], start=1) if point != origin
    )
    first = points[first_index]
    first_delta = first[0] - origin[0], first[1] - origin[1]
    return any(
        first_delta[0] * (point[1] - origin[1])
        - first_delta[1] * (point[0] - origin[0])
        != 0.0
        for point in points[first_index + 1 :]
    )


def _rbf_system(
    centers: tuple[Point, ...], radius_mm: float
) -> npt.NDArray[np.float64]:
    points = np.asarray(centers, dtype=np.float64)
    deltas = points[:, None, :] - points[None, :, :]
    squared_distance = np.einsum("ijk,ijk->ij", deltas, deltas)
    kernel = np.exp(-squared_distance / (radius_mm * radius_mm))
    polynomial = np.column_stack((np.ones(len(centers), dtype=np.float64), points))
    zeros = np.zeros((3, 3), dtype=np.float64)
    return np.block([[kernel, polynomial], [polynomial.T, zeros]])


@dataclass(frozen=True, slots=True)
class BoundRBFDeformation:
    """A parameter-bound RBF deformation with immutable solved coefficients."""

    kind: ClassVar[str] = "rbf"

    centers: tuple[Point, ...]
    radius_mm: float
    radial_coefficients: tuple[Point, ...]
    affine_coefficients: tuple[Point, Point, Point]
    parameter_hash: str

    def map_point(self, point: Point) -> Point:
        x_value, y_value = _evaluation_point(point)
        displacement_x, displacement_y = _rbf_displacement(
            (x_value, y_value),
            self.centers,
            self.radius_mm,
            self.radial_coefficients,
            self.affine_coefficients,
        )
        return x_value + displacement_x, y_value + displacement_y

    def jacobian(self, point: Point) -> Matrix2:
        x_value, y_value = _evaluation_point(point)
        radius_squared = self.radius_mm * self.radius_mm
        derivative_x = [
            self.affine_coefficients[1][component] for component in range(2)
        ]
        derivative_y = [
            self.affine_coefficients[2][component] for component in range(2)
        ]
        for center, coefficient in zip(
            self.centers, self.radial_coefficients, strict=True
        ):
            delta_x = x_value - center[0]
            delta_y = y_value - center[1]
            basis = math.exp(-(delta_x * delta_x + delta_y * delta_y) / radius_squared)
            basis_x = -2.0 * delta_x * basis / radius_squared
            basis_y = -2.0 * delta_y * basis / radius_squared
            for component in range(2):
                derivative_x[component] += coefficient[component] * basis_x
                derivative_y[component] += coefficient[component] * basis_y
        return (
            (1.0 + derivative_x[0], derivative_y[0]),
            (derivative_x[1], 1.0 + derivative_y[1]),
        )


def _rbf_displacement(
    point: Point,
    centers: tuple[Point, ...],
    radius_mm: float,
    radial_coefficients: tuple[Point, ...],
    affine_coefficients: tuple[Point, Point, Point],
) -> Point:
    x_value, y_value = point
    displacement = [
        affine_coefficients[0][component]
        + affine_coefficients[1][component] * x_value
        + affine_coefficients[2][component] * y_value
        for component in range(2)
    ]
    radius_squared = radius_mm * radius_mm
    for center, coefficient in zip(centers, radial_coefficients, strict=True):
        delta_x = x_value - center[0]
        delta_y = y_value - center[1]
        basis = math.exp(-(delta_x * delta_x + delta_y * delta_y) / radius_squared)
        for component in range(2):
            displacement[component] += coefficient[component] * basis
    return displacement[0], displacement[1]


@dataclass(frozen=True, slots=True)
class FFDMap:
    """Tensor-product Bernstein free-form deformation over a rectangular box."""

    kind: ClassVar[str] = "ffd"

    bounds: tuple[Point, Point]
    control_shape: tuple[int, int]
    control_points: tuple[FFDControlPoint, ...]
    unit: Unit = Unit.MM

    def __post_init__(self) -> None:
        if not isinstance(self.bounds, tuple) or len(self.bounds) != 2:
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "FFD bounds must contain lower and upper points",
            )
        lower = _evaluation_point(self.bounds[0])
        upper = _evaluation_point(self.bounds[1])
        if lower[0] >= upper[0] or lower[1] >= upper[1]:
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "FFD lower bounds must be strictly below upper bounds",
            )
        if self.unit is not Unit.MM:
            raise MorphingError(
                MorphReasonCode.PARAMETER_UNIT_MISMATCH,
                "FFD coordinates must use canonical millimetres",
            )
        if (
            not isinstance(self.control_shape, tuple)
            or len(self.control_shape) != 2
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 2
                for size in self.control_shape
            )
        ):
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "FFD control_shape must contain two integers of at least two",
            )

        controls = tuple(self.control_points)
        expected_count = self.control_shape[0] * self.control_shape[1]
        if len(controls) != expected_count or not all(
            isinstance(control, FFDControlPoint) for control in controls
        ):
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "FFD controls must be a typed x-major lattice matching control_shape",
            )
        object.__setattr__(self, "bounds", (lower, upper))
        object.__setattr__(self, "control_points", controls)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return _parameter_names(self.control_points)

    def bind(self, theta: Mapping[str, CanonicalQuantity]) -> BoundFFDDeformation:
        values, parameter_hash = _parameter_values(self.parameter_names, theta)
        return BoundFFDDeformation(
            bounds=self.bounds,
            control_shape=self.control_shape,
            displacements=_control_displacements(self.control_points, values),
            parameter_hash=parameter_hash,
        )


@dataclass(frozen=True, slots=True)
class BoundFFDDeformation:
    """A parameter-bound tensor-product Bernstein deformation."""

    kind: ClassVar[str] = "ffd"

    bounds: tuple[Point, Point]
    control_shape: tuple[int, int]
    displacements: tuple[Point, ...]
    parameter_hash: str

    def map_point(self, point: Point) -> Point:
        x_value, y_value, x_normalized, y_normalized = self._normalized(point)
        basis_x = _bernstein_basis(self.control_shape[0] - 1, x_normalized)
        basis_y = _bernstein_basis(self.control_shape[1] - 1, y_normalized)
        displacement_x, displacement_y = self._weighted_displacement(basis_x, basis_y)
        return x_value + displacement_x, y_value + displacement_y

    def jacobian(self, point: Point) -> Matrix2:
        _, _, x_normalized, y_normalized = self._normalized(point)
        x_count, y_count = self.control_shape
        basis_x = _bernstein_basis(x_count - 1, x_normalized)
        basis_y = _bernstein_basis(y_count - 1, y_normalized)
        derivative_x = _bernstein_derivative(x_count - 1, x_normalized)
        derivative_y = _bernstein_derivative(y_count - 1, y_normalized)
        width = self.bounds[1][0] - self.bounds[0][0]
        height = self.bounds[1][1] - self.bounds[0][1]
        displacement_dx = self._weighted_displacement(derivative_x, basis_y)
        displacement_dy = self._weighted_displacement(basis_x, derivative_y)
        return (
            (1.0 + displacement_dx[0] / width, displacement_dy[0] / height),
            (displacement_dx[1] / width, 1.0 + displacement_dy[1] / height),
        )

    def _normalized(self, point: Point) -> tuple[float, float, float, float]:
        x_value, y_value = _evaluation_point(point)
        lower, upper = self.bounds
        if not (lower[0] <= x_value <= upper[0] and lower[1] <= y_value <= upper[1]):
            raise MorphingError(
                MorphReasonCode.POINT_OUTSIDE_FFD_BOX,
                f"point {(x_value, y_value)!r} lies outside the FFD box",
            )
        return (
            x_value,
            y_value,
            (x_value - lower[0]) / (upper[0] - lower[0]),
            (y_value - lower[1]) / (upper[1] - lower[1]),
        )

    def _weighted_displacement(
        self, weights_x: tuple[float, ...], weights_y: tuple[float, ...]
    ) -> Point:
        x_total = 0.0
        y_total = 0.0
        y_count = self.control_shape[1]
        for x_index, x_weight in enumerate(weights_x):
            for y_index, y_weight in enumerate(weights_y):
                weight = x_weight * y_weight
                displacement = self.displacements[x_index * y_count + y_index]
                x_total += weight * displacement[0]
                y_total += weight * displacement[1]
        return x_total, y_total


def _bernstein_basis(degree: int, coordinate: float) -> tuple[float, ...]:
    return tuple(
        math.comb(degree, index)
        * coordinate**index
        * (1.0 - coordinate) ** (degree - index)
        for index in range(degree + 1)
    )


def _bernstein_derivative(degree: int, coordinate: float) -> tuple[float, ...]:
    lower_basis = _bernstein_basis(degree - 1, coordinate)
    return tuple(
        degree
        * (
            (lower_basis[index - 1] if index > 0 else 0.0)
            - (lower_basis[index] if index < degree else 0.0)
        )
        for index in range(degree + 1)
    )
