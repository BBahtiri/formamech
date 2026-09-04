"""Conservative S10 field projection and cross-mesh comparison contracts.

Projection errors in this module are deliberately bounded only against the
stored quadrature samples that participate in a projection.  They are not
claims about the continuum field between those samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from piginot.contracts import (
    CanonicalMesh,
    Dimension,
    Unit,
    canonical_json_bytes,
    hash_spec,
)
from piginot.contracts.specs import SpecContract
from piginot.datasets.units import AreaUnit

_ERROR_METHOD = "stored-quadrature-componentwise-linf-v1"
_ERROR_SCOPE = (
    "componentwise L-infinity envelope over the stored quadrature samples "
    "in the recorded source cells; not a continuum error bound"
)
_HASH_LENGTH = 64

Point = tuple[float, float]
Components = tuple[float, ...]


class ProjectionSupport(StrEnum):
    """Support on which projected values are represented."""

    CELL = "CELL"
    QUERY = "QUERY"


class CrossMeshMethod(StrEnum):
    """The sole S10 cross-mesh comparison admitted by this module."""

    PHYSICAL_DOMAIN_INTEGRAL = "physical_domain_integral"


@dataclass(frozen=True, slots=True)
class QuadratureField(SpecContract):
    """One finite field sampled at physical FEM quadrature points."""

    __canonical_namespace__: ClassVar[str] = "piginot.datasets"

    field_name: str
    mesh_id: str
    fem_result_hash: str
    unit: Unit
    component_shape: tuple[int, ...]
    points: tuple[tuple[Point, ...], ...]
    weights: tuple[tuple[float, ...], ...]
    values: tuple[tuple[Components, ...], ...]
    coordinate_unit: Unit = Unit.MM
    quadrature_weight_unit: AreaUnit = AreaUnit.SQUARE_MILLIMETRE

    def __post_init__(self) -> None:
        _nonempty(self.field_name, "field_name")
        _nonempty(self.mesh_id, "mesh_id")
        _content_hash(self.fem_result_hash, "fem_result_hash")
        _canonical_unit(self.unit)
        if self.coordinate_unit is not Unit.MM:
            raise ValueError("quadrature coordinate unit must be canonical mm")
        if self.quadrature_weight_unit is not AreaUnit.SQUARE_MILLIMETRE:
            raise ValueError("quadrature weight unit must be canonical mm^2")
        component_count = _validated_component_count(self.component_shape)

        points = tuple(
            tuple(_point(point, "quadrature point") for point in cell)
            for cell in self.points
        )
        weights = tuple(
            tuple(_finite(weight, "quadrature weight") for weight in cell)
            for cell in self.weights
        )
        values = tuple(
            tuple(
                _components(value, component_count, "quadrature field value")
                for value in cell
            )
            for cell in self.values
        )
        if not points or len(points) != len(weights) or len(points) != len(values):
            raise ValueError(
                "quadrature points, weights, and values must have one common cell layout"
            )
        for cell_points, cell_weights, cell_values in zip(
            points, weights, values, strict=True
        ):
            if (
                not cell_points
                or len(cell_points) != len(cell_weights)
                or len(cell_points) != len(cell_values)
            ):
                raise ValueError(
                    "quadrature points, weights, and values must have one common layout"
                )
            if any(weight <= 0.0 for weight in cell_weights):
                raise ValueError("quadrature weights must be positive")

        object.__setattr__(self, "component_shape", tuple(self.component_shape))
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "values", values)

    @property
    def content_hash(self) -> str:
        return hash_spec(self, "content")


@dataclass(frozen=True, slots=True)
class TransferErrorArtifact(SpecContract):
    """Mandatory sample-scoped error envelope for one projected field."""

    __canonical_namespace__: ClassVar[str] = "piginot.datasets"

    method: str
    scope: str
    source_field_hash: str
    unit: Unit
    absolute_bounds: tuple[Components, ...]
    source_sample_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.method != _ERROR_METHOD:
            raise ValueError(f"transfer-error method must be {_ERROR_METHOD!r}")
        if self.scope != _ERROR_SCOPE:
            raise ValueError("transfer-error scope must state its stored-sample limit")
        _content_hash(self.source_field_hash, "source_field_hash")
        _canonical_unit(self.unit)
        bounds = tuple(
            tuple(_nonnegative(value, "absolute transfer-error bound") for value in row)
            for row in self.absolute_bounds
        )
        counts = tuple(self.source_sample_counts)
        if not bounds or len(bounds) != len(counts):
            raise ValueError(
                "transfer-error bounds and sample counts must have one common layout"
            )
        if not bounds[0] or any(len(row) != len(bounds[0]) for row in bounds):
            raise ValueError(
                "transfer-error component layout must be non-empty and fixed"
            )
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in counts
        ):
            raise ValueError(
                "transfer-error source sample counts must be positive integers"
            )
        object.__setattr__(self, "absolute_bounds", bounds)
        object.__setattr__(self, "source_sample_counts", counts)

    @property
    def content_hash(self) -> str:
        return hash_spec(self, "content")


@dataclass(frozen=True, slots=True)
class ProjectedField(SpecContract):
    """Shape-valid projected values with mandatory transfer uncertainty.

    Construction validates only the artifact's local structure.  Call
    :func:`verify_projected_field` to establish derivation from a source field
    before the artifact crosses a persistence or trust boundary.
    """

    __canonical_namespace__: ClassVar[str] = "piginot.datasets"

    field_name: str
    mesh_id: str
    unit: Unit
    component_shape: tuple[int, ...]
    support: ProjectionSupport
    query_points: tuple[Point, ...]
    values: tuple[Components, ...]
    source_cell_ids: tuple[tuple[int, ...], ...]
    transfer_error: TransferErrorArtifact
    coordinate_unit: Unit = Unit.MM

    def __post_init__(self) -> None:
        _nonempty(self.field_name, "field_name")
        _nonempty(self.mesh_id, "mesh_id")
        _canonical_unit(self.unit)
        if self.coordinate_unit is not Unit.MM:
            raise ValueError("projected coordinate unit must be canonical mm")
        component_count = _validated_component_count(self.component_shape)
        if not isinstance(self.support, ProjectionSupport):
            raise ValueError("support must be a ProjectionSupport")
        if type(self.transfer_error) is not TransferErrorArtifact:
            raise ValueError("transfer_error must be a TransferErrorArtifact")
        if self.transfer_error.unit is not self.unit:
            raise ValueError("transfer_error unit must match the projected field")

        points = tuple(_point(point, "query point") for point in self.query_points)
        values = tuple(
            _components(value, component_count, "projected field value")
            for value in self.values
        )
        source_cells = tuple(tuple(cell_ids) for cell_ids in self.source_cell_ids)
        if not values or len(values) != len(source_cells):
            raise ValueError("projected values and source cells must share one layout")
        if any(
            not cell_ids
            or tuple(sorted(set(cell_ids))) != cell_ids
            or any(
                isinstance(cell_id, bool) or not isinstance(cell_id, int) or cell_id < 0
                for cell_id in cell_ids
            )
            for cell_ids in source_cells
        ):
            raise ValueError("source_cell_ids must be sorted unique non-negative IDs")
        if self.support is ProjectionSupport.CELL and points:
            raise ValueError("cell-supported projections cannot carry query points")
        if self.support is ProjectionSupport.QUERY and len(points) != len(values):
            raise ValueError("query-supported values require one query point per value")
        if len(self.transfer_error.absolute_bounds) != len(values) or any(
            len(row) != component_count for row in self.transfer_error.absolute_bounds
        ):
            raise ValueError(
                "transfer_error component layout must match projected values"
            )

        object.__setattr__(self, "component_shape", tuple(self.component_shape))
        object.__setattr__(self, "query_points", points)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "source_cell_ids", source_cells)

    @property
    def content_hash(self) -> str:
        return hash_spec(self, "content")


@dataclass(frozen=True, slots=True)
class CrossMeshIntegralComparison(SpecContract):
    """Shape-valid stress comparison without node correspondence.

    Construction validates local arithmetic only.  Call
    :func:`verify_cross_mesh_comparison` to establish derivation from both
    source fields before the artifact crosses a persistence or trust boundary.
    """

    __canonical_namespace__: ClassVar[str] = "piginot.datasets"

    field_name: str
    method: CrossMeshMethod
    left_mesh_id: str
    right_mesh_id: str
    left_source_field_hash: str
    right_source_field_hash: str
    integral_unit: Unit
    component_shape: tuple[int, ...]
    left_integral: Components
    right_integral: Components
    absolute_difference: Components
    relative_l2_difference: float
    left_quadrature_sample_count: int
    right_quadrature_sample_count: int

    def __post_init__(self) -> None:
        _nonempty(self.field_name, "field_name")
        if self.method is not CrossMeshMethod.PHYSICAL_DOMAIN_INTEGRAL:
            raise ValueError(
                "cross-mesh artifact method must be a physical-domain integral"
            )
        _nonempty(self.left_mesh_id, "left_mesh_id")
        _nonempty(self.right_mesh_id, "right_mesh_id")
        _content_hash(self.left_source_field_hash, "left source field hash")
        _content_hash(self.right_source_field_hash, "right source field hash")
        if self.integral_unit is not Unit.N:
            raise ValueError("cross-mesh stress integral unit must be canonical N")
        component_count = _validated_component_count(self.component_shape)
        left = _components(self.left_integral, component_count, "left integral")
        right = _components(self.right_integral, component_count, "right integral")
        difference = _components(
            self.absolute_difference,
            component_count,
            "absolute difference",
        )
        expected_difference = tuple(
            abs(left_value - right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
        if difference != expected_difference:
            raise ValueError("absolute difference must match the physical integrals")
        relative = _nonnegative(
            self.relative_l2_difference,
            "relative difference",
        )
        difference_norm = math.sqrt(sum(value * value for value in difference))
        denominator = max(
            math.sqrt(sum(value * value for value in left)),
            math.sqrt(sum(value * value for value in right)),
        )
        expected_relative = 0.0 if denominator == 0.0 else difference_norm / denominator
        if not math.isclose(relative, expected_relative, rel_tol=1.0e-12, abs_tol=0.0):
            raise ValueError("relative difference must match the physical integrals")
        for count, label in (
            (self.left_quadrature_sample_count, "left quadrature sample count"),
            (self.right_quadrature_sample_count, "right quadrature sample count"),
        ):
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"{label} must be a positive integer")
        object.__setattr__(self, "component_shape", tuple(self.component_shape))
        object.__setattr__(self, "left_integral", left)
        object.__setattr__(self, "right_integral", right)
        object.__setattr__(self, "absolute_difference", difference)
        object.__setattr__(self, "relative_l2_difference", relative)

    @property
    def content_hash(self) -> str:
        return hash_spec(self, "content")


def project_quadrature_to_cells(
    mesh: CanonicalMesh, field: QuadratureField
) -> ProjectedField:
    """Project stored quadrature values to weighted cell means."""

    _validate_mesh_field(mesh, field)
    _validate_declared_field_unit(field)
    values: list[Components] = []
    bounds: list[Components] = []
    counts: list[int] = []
    for cell_values, cell_weights in zip(field.values, field.weights, strict=True):
        mean = _weighted_mean(cell_values, cell_weights)
        values.append(mean)
        bounds.append(_sample_envelope(mean, cell_values))
        counts.append(len(cell_values))

    error = _transfer_error(field, tuple(bounds), tuple(counts))
    return ProjectedField(
        field_name=field.field_name,
        mesh_id=mesh.mesh_id,
        unit=field.unit,
        component_shape=field.component_shape,
        support=ProjectionSupport.CELL,
        query_points=(),
        values=tuple(values),
        source_cell_ids=tuple((cell_id,) for cell_id in range(len(mesh.cells))),
        transfer_error=error,
    )


def project_quadrature_to_queries(
    mesh: CanonicalMesh,
    field: QuadratureField,
    query_points: tuple[Point, ...],
) -> ProjectedField:
    """Project to common physical points with sample-scoped error envelopes.

    A query on a shared cell edge takes the mean from the lowest canonical cell
    ID.  Its error envelope includes every stored sample in every incident cell,
    so the deterministic tie break cannot hide a sampled inter-cell jump.
    """

    _validate_mesh_field(mesh, field)
    _validate_declared_field_unit(field)
    queries = tuple(_point(point, "query point") for point in query_points)
    if not queries:
        raise ValueError("query_points must be non-empty")
    cell_means = tuple(
        _weighted_mean(values, weights)
        for values, weights in zip(field.values, field.weights, strict=True)
    )

    projected: list[Components] = []
    source_ids: list[tuple[int, ...]] = []
    bounds: list[Components] = []
    counts: list[int] = []
    for query_id, query in enumerate(queries):
        containing = tuple(
            cell_id
            for cell_id, cell in enumerate(mesh.cells)
            if _point_in_triangle(
                query,
                (mesh.nodes[cell[0]], mesh.nodes[cell[1]], mesh.nodes[cell[2]]),
            )
        )
        if not containing:
            raise ValueError(f"query point {query_id} lies outside the mesh")
        value = cell_means[containing[0]]
        samples = tuple(
            sample for cell_id in containing for sample in field.values[cell_id]
        )
        projected.append(value)
        source_ids.append(containing)
        bounds.append(_sample_envelope(value, samples))
        counts.append(len(samples))

    error = _transfer_error(field, tuple(bounds), tuple(counts))
    return ProjectedField(
        field_name=field.field_name,
        mesh_id=mesh.mesh_id,
        unit=field.unit,
        component_shape=field.component_shape,
        support=ProjectionSupport.QUERY,
        query_points=queries,
        values=tuple(projected),
        source_cell_ids=tuple(source_ids),
        transfer_error=error,
    )


def compare_cross_mesh_stress(
    left_mesh: CanonicalMesh,
    left: QuadratureField,
    right_mesh: CanonicalMesh,
    right: QuadratureField,
    *,
    method: CrossMeshMethod | str = CrossMeshMethod.PHYSICAL_DOMAIN_INTEGRAL,
) -> CrossMeshIntegralComparison:
    """Compare stress fields by physical-domain integrals, never node pairing."""

    if method != CrossMeshMethod.PHYSICAL_DOMAIN_INTEGRAL:
        raise ValueError(
            "cross-mesh raw or nearest node pairing is forbidden; "
            "use physical_domain_integral"
        )
    _validate_mesh_field(left_mesh, left)
    _validate_mesh_field(right_mesh, right)
    _require_stress_unit(left.unit)
    _require_stress_unit(right.unit)
    if left.field_name != right.field_name:
        raise ValueError("cross-mesh comparison requires the same field name")
    if left.unit is not right.unit or left.component_shape != right.component_shape:
        raise ValueError("cross-mesh fields must have identical units and components")

    left_integral = _physical_integral(left)
    right_integral = _physical_integral(right)
    difference = tuple(
        abs(left_value - right_value)
        for left_value, right_value in zip(left_integral, right_integral, strict=True)
    )
    difference_norm = math.sqrt(sum(value * value for value in difference))
    left_norm = math.sqrt(sum(value * value for value in left_integral))
    right_norm = math.sqrt(sum(value * value for value in right_integral))
    denominator = max(left_norm, right_norm)
    relative = 0.0 if denominator == 0.0 else difference_norm / denominator
    return CrossMeshIntegralComparison(
        field_name=left.field_name,
        method=CrossMeshMethod.PHYSICAL_DOMAIN_INTEGRAL,
        left_mesh_id=left_mesh.mesh_id,
        right_mesh_id=right_mesh.mesh_id,
        left_source_field_hash=left.content_hash,
        right_source_field_hash=right.content_hash,
        integral_unit=Unit.N,
        component_shape=left.component_shape,
        left_integral=left_integral,
        right_integral=right_integral,
        absolute_difference=difference,
        relative_l2_difference=relative,
        left_quadrature_sample_count=sum(map(len, left.values)),
        right_quadrature_sample_count=sum(map(len, right.values)),
    )


def verify_projected_field(
    mesh: CanonicalMesh,
    source_field: QuadratureField,
    projected: ProjectedField,
) -> None:
    """Recompute and verify a projected field's complete canonical payload."""

    if type(projected) is not ProjectedField:
        raise ValueError("projected must be a ProjectedField")
    if projected.support is ProjectionSupport.CELL:
        expected = project_quadrature_to_cells(mesh, source_field)
    else:
        expected = project_quadrature_to_queries(
            mesh,
            source_field,
            projected.query_points,
        )
    if canonical_json_bytes(projected) != canonical_json_bytes(expected):
        raise ValueError("derived projection does not match its source field")


def verify_cross_mesh_comparison(
    left_mesh: CanonicalMesh,
    left_field: QuadratureField,
    right_mesh: CanonicalMesh,
    right_field: QuadratureField,
    artifact: CrossMeshIntegralComparison,
) -> None:
    """Recompute and verify a cross-mesh comparison's canonical payload."""

    if type(artifact) is not CrossMeshIntegralComparison:
        raise ValueError("artifact must be a CrossMeshIntegralComparison")
    expected = compare_cross_mesh_stress(
        left_mesh,
        left_field,
        right_mesh,
        right_field,
    )
    if canonical_json_bytes(artifact) != canonical_json_bytes(expected):
        raise ValueError(
            "derived cross-mesh comparison does not match its source fields"
        )


def _validate_mesh_field(mesh: CanonicalMesh, field: QuadratureField) -> None:
    if type(mesh) is not CanonicalMesh:
        raise ValueError("mesh must be a CanonicalMesh")
    if type(field) is not QuadratureField:
        raise ValueError("field must be a QuadratureField")
    if mesh.mesh_id != field.mesh_id:
        raise ValueError("field mesh_id does not match the canonical mesh")
    if mesh.unit is not field.coordinate_unit:
        raise ValueError("field coordinate unit does not match the canonical mesh")
    if len(mesh.cells) != len(field.points):
        raise ValueError("field cell layout does not match the canonical mesh")
    for cell_id, (cell, points) in enumerate(
        zip(mesh.cells, field.points, strict=True)
    ):
        triangle = (mesh.nodes[cell[0]], mesh.nodes[cell[1]], mesh.nodes[cell[2]])
        if any(not _point_in_triangle(point, triangle) for point in points):
            raise ValueError(
                f"quadrature point lies outside its declared mesh cell {cell_id}"
            )
        cell_area = abs(_orientation(*triangle)) / 2.0
        integrated_weight = sum(field.weights[cell_id])
        if not math.isfinite(cell_area) or not math.isclose(
            integrated_weight,
            cell_area,
            rel_tol=1.0e-10,
            abs_tol=64.0 * math.ulp(cell_area),
        ):
            raise ValueError(
                f"quadrature weights do not integrate to physical cell area {cell_id}"
            )


def _validate_declared_field_unit(field: QuadratureField) -> None:
    if "stress" in field.field_name.casefold():
        _require_stress_unit(field.unit)


def _require_stress_unit(unit: Unit) -> None:
    if unit.dimension is not Dimension.STRESS or unit is not Unit.MPA:
        raise ValueError("stress field must use the canonical MPa stress unit")


def _physical_integral(field: QuadratureField) -> Components:
    component_count = math.prod(field.component_shape)
    result = [0.0] * component_count
    for cell_values, cell_weights in zip(field.values, field.weights, strict=True):
        for sample, weight in zip(cell_values, cell_weights, strict=True):
            for component, value in enumerate(sample):
                result[component] += weight * value
    return tuple(result)


def _transfer_error(
    field: QuadratureField,
    bounds: tuple[Components, ...],
    counts: tuple[int, ...],
) -> TransferErrorArtifact:
    return TransferErrorArtifact(
        method=_ERROR_METHOD,
        scope=_ERROR_SCOPE,
        source_field_hash=field.content_hash,
        unit=field.unit,
        absolute_bounds=bounds,
        source_sample_counts=counts,
    )


def _weighted_mean(
    values: tuple[Components, ...], weights: tuple[float, ...]
) -> Components:
    total = sum(weights)
    return tuple(
        sum(
            weight * value[component]
            for value, weight in zip(values, weights, strict=True)
        )
        / total
        for component in range(len(values[0]))
    )


def _sample_envelope(value: Components, samples: tuple[Components, ...]) -> Components:
    return tuple(
        max(abs(sample[component] - value[component]) for sample in samples)
        for component in range(len(value))
    )


def _point_in_triangle(point: Point, triangle: tuple[Point, Point, Point]) -> bool:
    first, second, third = triangle
    area = _orientation(first, second, third)
    scale = max(
        1.0,
        *(abs(coordinate) for vertex in triangle for coordinate in vertex),
        abs(point[0]),
        abs(point[1]),
    )
    tolerance = 64.0 * math.ulp(scale * scale)
    if abs(area) <= tolerance:
        return False
    edges = (
        _orientation(first, second, point),
        _orientation(second, third, point),
        _orientation(third, first, point),
    )
    if area > 0.0:
        return all(edge >= -tolerance for edge in edges)
    return all(edge <= tolerance for edge in edges)


def _orientation(left: Point, right: Point, point: Point) -> float:
    return (right[0] - left[0]) * (point[1] - left[1]) - (right[1] - left[1]) * (
        point[0] - left[0]
    )


def _validated_component_count(shape: tuple[int, ...]) -> int:
    result = tuple(shape)
    if not result or any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in result
    ):
        raise ValueError("component_shape must contain positive integers")
    return math.prod(result)


def _components(value: object, size: int, label: str) -> Components:
    if not isinstance(value, tuple) or len(value) != size:
        raise ValueError(f"{label} does not match component_shape")
    return tuple(_finite(component, label) for component in value)


def _point(value: object, label: str) -> Point:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{label} must be a two-coordinate tuple")
    return _finite(value[0], label), _finite(value[1], label)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must contain real numbers")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must contain finite values")
    return result


def _nonnegative(value: object, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _canonical_unit(unit: object) -> Unit:
    if not isinstance(unit, Unit) or not unit.is_canonical:
        raise ValueError("field unit must be a canonical Unit")
    return unit


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _content_hash(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _HASH_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase 256-bit hexadecimal hash")
    return value


__all__ = (
    "CrossMeshIntegralComparison",
    "CrossMeshMethod",
    "ProjectedField",
    "ProjectionSupport",
    "QuadratureField",
    "TransferErrorArtifact",
    "compare_cross_mesh_stress",
    "project_quadrature_to_cells",
    "project_quadrature_to_queries",
    "verify_cross_mesh_comparison",
    "verify_projected_field",
)
