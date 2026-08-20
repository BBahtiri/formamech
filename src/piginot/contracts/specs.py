"""Immutable problem-family specification contracts."""

from __future__ import annotations

import keyword
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Self, TypeVar

from piginot.contracts.errors import (
    BareDimensionalValueError,
    ContractValidationError,
    UnitMismatchError,
    UnknownFieldError,
)
from piginot.contracts.expressions import DerivedExpression
from piginot.contracts.units import CanonicalQuantity, Dimension, Unit


def _name(value: object, label: str = "name") -> str:
    if (
        not isinstance(value, str)
        or not value.isidentifier()
        or keyword.iskeyword(value)
    ):
        raise ContractValidationError(f"{label} must be a valid identifier")
    return value


def _canonical(value: object, label: str) -> CanonicalQuantity:
    if not isinstance(value, CanonicalQuantity):
        raise BareDimensionalValueError(
            f"{label} must be an explicitly converted CanonicalQuantity"
        )
    return value


class SpecContract:
    """Shared strict mapping boundary for frozen specification dataclasses."""

    __canonical_namespace__: ClassVar[str] = "piginot.contracts"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> Self:
        dataclass_fields = getattr(cls, "__dataclass_fields__", None)
        if not isinstance(dataclass_fields, dict):
            raise ContractValidationError(f"{cls.__name__} is not a dataclass contract")
        expected = set(dataclass_fields)
        unknown = sorted(set(payload) - expected)
        if unknown:
            raise UnknownFieldError(
                f"unknown field(s) for {cls.__name__}: {', '.join(unknown)}"
            )
        try:
            return cls(**dict(payload))
        except TypeError as exc:
            raise ContractValidationError(
                f"invalid fields for {cls.__name__}: {exc}"
            ) from exc


_Value = TypeVar("_Value")


@dataclass(frozen=True, slots=True)
class FrozenMap(Mapping[str, _Value], Generic[_Value]):
    """A small hashable mapping used to make nested contract state immutable."""

    _items: tuple[tuple[str, _Value], ...]

    @classmethod
    def copy_of(cls, values: object) -> FrozenMap[_Value]:
        if not isinstance(values, Mapping):
            raise ContractValidationError("mapping fields must be mappings")
        items = tuple(sorted(values.items()))
        if any(not isinstance(key, str) or not key for key, _ in items):
            raise ContractValidationError("mapping keys must be non-empty strings")
        return cls(items)

    def __getitem__(self, key: str) -> _Value:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{label} must be non-empty text")
    return value


def _spec_tuple(
    values: Sequence[Any], expected: type[Any], label: str
) -> tuple[Any, ...]:
    result = tuple(values)
    if any(not isinstance(value, expected) for value in result):
        raise ContractValidationError(
            f"{label} must contain {expected.__name__} values"
        )
    return result


@dataclass(frozen=True, slots=True)
class ParameterSpec(SpecContract):
    name: str
    lower_bound: CanonicalQuantity
    upper_bound: CanonicalQuantity
    description: str = ""

    def __post_init__(self) -> None:
        _name(self.name)
        lower = _canonical(self.lower_bound, "lower_bound")
        upper = _canonical(self.upper_bound, "upper_bound")
        if lower.dimension is not upper.dimension:
            raise UnitMismatchError("parameter bounds must have matching dimensions")
        if lower.value > upper.value:
            raise ContractValidationError("lower_bound must not exceed upper_bound")
        if not isinstance(self.description, str):
            raise ContractValidationError("description must be text")

    @property
    def dimension(self) -> Dimension:
        dimension = self.lower_bound.dimension
        if not isinstance(dimension, Dimension):
            raise ContractValidationError("parameter dimension is unresolved")
        return dimension


@dataclass(frozen=True, slots=True)
class DerivedQuantitySpec(SpecContract):
    name: str
    expression: DerivedExpression
    lower_bound: CanonicalQuantity
    upper_bound: CanonicalQuantity
    unit: Unit
    description: str = ""

    def __post_init__(self) -> None:
        _name(self.name)
        if not isinstance(self.expression, DerivedExpression):
            raise ContractValidationError("expression must be a DerivedExpression")
        lower = _canonical(self.lower_bound, "lower_bound")
        upper = _canonical(self.upper_bound, "upper_bound")
        if not isinstance(self.unit, Unit) or not self.unit.is_canonical:
            raise ContractValidationError("derived quantity unit must be canonical")
        if lower.unit is not self.unit or upper.unit is not self.unit:
            raise UnitMismatchError(
                "derived quantity bounds must match its explicit unit"
            )
        if lower.dimension is not upper.dimension:
            raise UnitMismatchError(
                "derived quantity bounds must have matching dimensions"
            )
        if lower.value > upper.value:
            raise ContractValidationError("lower_bound must not exceed upper_bound")
        if not isinstance(self.description, str):
            raise ContractValidationError("description must be text")

    def validate_against(self, parameters: Sequence[ParameterSpec]) -> None:
        dimensions: dict[str, Dimension] = {}
        for parameter in parameters:
            if not isinstance(parameter, ParameterSpec):
                raise ContractValidationError(
                    "parameters must contain ParameterSpec values"
                )
            if parameter.name in dimensions:
                raise ContractValidationError(
                    f"duplicate primitive parameter {parameter.name!r}"
                )
            dimensions[parameter.name] = parameter.dimension
        self.expression.validate(
            parameter_dimensions=dimensions,
            expected_dimension=self.dimension,
        )

    @property
    def dimension(self) -> Dimension:
        dimension = self.lower_bound.dimension
        if not isinstance(dimension, Dimension):
            raise ContractValidationError("derived quantity dimension is unresolved")
        return dimension


@dataclass(frozen=True, slots=True)
class GeometryFamilySpec(SpecContract):
    name: str
    parameters: tuple[ParameterSpec, ...] = ()
    derived_quantities: tuple[DerivedQuantitySpec, ...] = ()
    topology_signature: str | None = None
    semantic_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.name, "geometry family name")
        parameters = _spec_tuple(self.parameters, ParameterSpec, "parameters")
        derived = _spec_tuple(
            self.derived_quantities, DerivedQuantitySpec, "derived_quantities"
        )
        tags = tuple(self.semantic_tags)
        if any(not isinstance(tag, str) or not tag for tag in tags):
            raise ContractValidationError("semantic tags must be non-empty strings")
        names = [item.name for item in (*parameters, *derived)]
        if len(names) != len(set(names)):
            raise ContractValidationError(
                "parameter and derived quantity names must be unique"
            )
        for item in derived:
            item.validate_against(parameters)
        if self.topology_signature is not None:
            _nonempty(self.topology_signature, "topology_signature")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "derived_quantities", derived)
        object.__setattr__(self, "semantic_tags", tags)


@dataclass(frozen=True, slots=True)
class MaterialSpec(SpecContract):
    name: str
    model: str
    properties: Mapping[str, CanonicalQuantity]

    def __post_init__(self) -> None:
        _nonempty(self.name, "material name")
        _nonempty(self.model, "material model")
        properties: FrozenMap[CanonicalQuantity] = FrozenMap.copy_of(self.properties)
        for key, value in properties.items():
            _canonical(value, f"material property {key!r}")
        object.__setattr__(self, "properties", properties)


def _components(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise ContractValidationError("components must be non-empty strings")
    return result


def _quantity_tuple(
    values: Sequence[CanonicalQuantity], label: str
) -> tuple[CanonicalQuantity, ...]:
    result = tuple(values)
    for value in result:
        _canonical(value, label)
    return result


@dataclass(frozen=True, slots=True)
class BoundaryConditionSpec(SpecContract):
    name: str
    boundary_tag: str
    kind: str
    components: tuple[str, ...] = ()
    values: tuple[CanonicalQuantity, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.name, "boundary condition name")
        _nonempty(self.boundary_tag, "boundary tag")
        _nonempty(self.kind, "boundary condition kind")
        object.__setattr__(self, "components", _components(self.components))
        object.__setattr__(
            self, "values", _quantity_tuple(self.values, "boundary value")
        )
        if self.values and len(self.components) != len(self.values):
            raise ContractValidationError("boundary components and values must align")


@dataclass(frozen=True, slots=True)
class LoadSpec(SpecContract):
    name: str
    kind: str
    boundary_tag: str | None = None
    components: tuple[str, ...] = ()
    values: tuple[CanonicalQuantity, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.name, "load name")
        _nonempty(self.kind, "load kind")
        if self.boundary_tag is not None:
            _nonempty(self.boundary_tag, "boundary tag")
        object.__setattr__(self, "components", _components(self.components))
        object.__setattr__(self, "values", _quantity_tuple(self.values, "load value"))
        if self.values and len(self.components) != len(self.values):
            raise ContractValidationError("load components and values must align")


@dataclass(frozen=True, slots=True)
class QoISpec(SpecContract):
    name: str
    field: str
    reduction: str
    region_tag: str
    unit: Unit
    dimension: Dimension | None = None

    def __post_init__(self) -> None:
        _nonempty(self.name, "QoI name")
        _nonempty(self.field, "QoI field")
        _nonempty(self.reduction, "QoI reduction")
        _nonempty(self.region_tag, "QoI region tag")
        if not isinstance(self.unit, Unit) or not self.unit.is_canonical:
            raise ContractValidationError("QoI unit must be canonical")
        dimension = self.unit.dimension if self.dimension is None else self.dimension
        if not isinstance(dimension, Dimension):
            raise ContractValidationError("QoI dimension must be a Dimension")
        if not self.unit.supports_dimension(dimension):
            raise UnitMismatchError(
                f"{self.unit.value!r} cannot represent {dimension.value}"
            )
        object.__setattr__(self, "dimension", dimension)


@dataclass(frozen=True, slots=True)
class ToleranceSpec(SpecContract):
    name: str
    metric: str
    limit: CanonicalQuantity

    def __post_init__(self) -> None:
        _nonempty(self.name, "tolerance name")
        _nonempty(self.metric, "tolerance metric")
        limit = _canonical(self.limit, "tolerance limit")
        if limit.value < 0:
            raise ContractValidationError("tolerance limit must be non-negative")


@dataclass(frozen=True, slots=True)
class CampaignPlan(SpecContract):
    campaign_id: str
    sample_count: int
    seed: int

    def __post_init__(self) -> None:
        _nonempty(self.campaign_id, "campaign_id")
        if isinstance(self.sample_count, bool) or not isinstance(
            self.sample_count, int
        ):
            raise ContractValidationError("sample_count must be an integer")
        if self.sample_count <= 0:
            raise ContractValidationError("sample_count must be positive")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ContractValidationError("seed must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ProblemFamilySpec(SpecContract):
    name: str
    geometry: GeometryFamilySpec
    material: MaterialSpec
    boundary_conditions: tuple[BoundaryConditionSpec, ...]
    loads: tuple[LoadSpec, ...]
    qois: tuple[QoISpec, ...]
    tolerances: tuple[ToleranceSpec, ...]
    campaign: CampaignPlan
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        _nonempty(self.name, "problem family name")
        _nonempty(self.schema_version, "schema_version")
        if not isinstance(self.geometry, GeometryFamilySpec):
            raise ContractValidationError("geometry must be a GeometryFamilySpec")
        if not isinstance(self.material, MaterialSpec):
            raise ContractValidationError("material must be a MaterialSpec")
        if not isinstance(self.campaign, CampaignPlan):
            raise ContractValidationError("campaign must be a CampaignPlan")
        object.__setattr__(
            self,
            "boundary_conditions",
            _spec_tuple(
                self.boundary_conditions, BoundaryConditionSpec, "boundary_conditions"
            ),
        )
        object.__setattr__(self, "loads", _spec_tuple(self.loads, LoadSpec, "loads"))
        object.__setattr__(self, "qois", _spec_tuple(self.qois, QoISpec, "qois"))
        object.__setattr__(
            self,
            "tolerances",
            _spec_tuple(self.tolerances, ToleranceSpec, "tolerances"),
        )


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        from piginot.contracts.errors import NonFiniteNumberError

        raise NonFiniteNumberError(f"{label} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class CanonicalMesh(SpecContract):
    """Immutable structural payload only; mesh algorithms intentionally live in S5."""

    mesh_id: str
    nodes: tuple[tuple[float, float], ...]
    cells: tuple[tuple[int, ...], ...]
    cell_type: str
    boundary_facets: tuple[tuple[int, int], ...]
    semantic_tags: Mapping[str, tuple[int, ...]]
    region_ids: tuple[int, ...]
    unit: Unit
    orientation: str
    topology_signature: str
    quality_metrics: Mapping[str, float]
    source_provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        _nonempty(self.mesh_id, "mesh_id")
        _nonempty(self.cell_type, "cell_type")
        _nonempty(self.orientation, "orientation")
        _nonempty(self.topology_signature, "topology_signature")
        if self.unit is not Unit.MM:
            raise ContractValidationError("canonical mesh coordinate unit must be mm")

        nodes = tuple(
            tuple(_finite(value, "node coordinate") for value in node)
            for node in self.nodes
        )
        if not nodes or any(len(node) != 2 for node in nodes):
            raise ContractValidationError("nodes must be non-empty 2D coordinate pairs")
        cells = tuple(tuple(cell) for cell in self.cells)
        if not cells or any(len(cell) < 3 for cell in cells):
            raise ContractValidationError(
                "cells must contain at least three node indices"
            )
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(nodes)
            for cell in cells
            for index in cell
        ):
            raise ContractValidationError(
                "cell connectivity contains an invalid node index"
            )
        facets = tuple(tuple(facet) for facet in self.boundary_facets)
        if any(len(facet) != 2 for facet in facets):
            raise ContractValidationError("boundary facets must be node-index pairs")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(nodes)
            for facet in facets
            for index in facet
        ):
            raise ContractValidationError(
                "boundary facet contains an invalid node index"
            )
        regions = tuple(self.region_ids)
        if len(regions) != len(cells) or any(
            isinstance(region, bool) or not isinstance(region, int)
            for region in regions
        ):
            raise ContractValidationError(
                "region_ids must contain one integer per cell"
            )

        if not isinstance(self.semantic_tags, Mapping):
            raise ContractValidationError("mapping fields must be mappings")
        tags: FrozenMap[tuple[int, ...]] = FrozenMap.copy_of(
            {key: tuple(value) for key, value in self.semantic_tags.items()}
        )
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(facets)
            for indices in tags.values()
            for index in indices
        ):
            raise ContractValidationError(
                "semantic tag contains an invalid facet index"
            )
        if not isinstance(self.quality_metrics, Mapping):
            raise ContractValidationError("mapping fields must be mappings")
        metrics: FrozenMap[float] = FrozenMap.copy_of(
            {
                key: _finite(value, f"quality metric {key!r}")
                for key, value in self.quality_metrics.items()
            }
        )
        provenance: FrozenMap[str] = FrozenMap.copy_of(self.source_provenance)
        if any(not isinstance(value, str) for value in provenance.values()):
            raise ContractValidationError("source provenance values must be strings")

        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "boundary_facets", facets)
        object.__setattr__(self, "semantic_tags", tags)
        object.__setattr__(self, "region_ids", regions)
        object.__setattr__(self, "quality_metrics", metrics)
        object.__setattr__(self, "source_provenance", provenance)
