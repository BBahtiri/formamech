"""Deterministic, coupled-domain S9 design-of-experiments construction."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from types import MappingProxyType
from typing import ClassVar, Protocol, TypeVar

from piginot.contracts import CanonicalQuantity, Dimension, Unit

_CANDIDATE_DOMAIN = "piginot.s9.doe.candidate-lhs.v1"
_VALIDATION_DOMAIN = "piginot.s9.doe.validation-lhs.v1"
_GUARD_DOMAIN = "piginot.s9.doe.guard-band.v1"
_PERSON = b"piginot-s9-doe"
_SEED_PERSON = b"piginot-s9-seed"
_Value = TypeVar("_Value")


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _positive_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _freeze(values: Mapping[str, _Value]) -> Mapping[str, _Value]:
    return MappingProxyType(dict(sorted(values.items())))


class _QuantityLike(Protocol):
    value: float
    unit: Unit
    dimension: Dimension | None


class _BoundLike(Protocol):
    name: str
    lower_bound: _QuantityLike
    upper_bound: _QuantityLike


class _DecisionLike(Protocol):
    accepted: bool
    derived_values: Mapping[str, _QuantityLike]


class FamilyMembershipPredicateLike(Protocol):
    """Public surface consumed from S6; no expression internals are required."""

    @property
    def primitive_bounds(self) -> Sequence[_BoundLike]: ...

    @property
    def derived_bounds(self) -> Sequence[_BoundLike]: ...

    def evaluate(self, theta: Mapping[str, object]) -> _DecisionLike: ...


class AxisSource(StrEnum):
    """Origin of a normalized DOE axis."""

    PRIMITIVE = "primitive"
    DERIVED = "derived"


@dataclass(frozen=True, slots=True)
class DesignPoint:
    """One immutable S9 case with primitive, derived, and fixed-strata coordinates."""

    __canonical_namespace__: ClassVar[str] = "piginot.doe"

    case_id: str
    leakage_group_id: str
    coordinates: Mapping[str, float]
    derived_coordinates: Mapping[str, float]
    strata: Mapping[str, int]
    guard_band: bool = False

    def __post_init__(self) -> None:
        case_id = _text(self.case_id, "case_id")
        if ":" in case_id:
            raise ValueError("case_id must identify a case-level sample")
        primitive = {
            name: _finite(value, f"coordinates[{name!r}]")
            for name, value in self.coordinates.items()
        }
        derived = {
            name: _finite(value, f"derived_coordinates[{name!r}]")
            for name, value in self.derived_coordinates.items()
        }
        if not primitive:
            raise ValueError("coordinates must not be empty")
        if set(primitive) & set(derived):
            raise ValueError("primitive and derived coordinate names must be disjoint")
        strata = dict(self.strata)
        if not strata or any(
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for name, value in strata.items()
        ):
            raise ValueError("strata must map names to non-negative integers")
        if not isinstance(self.guard_band, bool):
            raise ValueError("guard_band must be boolean")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(
            self,
            "leakage_group_id",
            _text(self.leakage_group_id, "leakage_group_id"),
        )
        object.__setattr__(self, "coordinates", _freeze(primitive))
        object.__setattr__(self, "derived_coordinates", _freeze(derived))
        object.__setattr__(self, "strata", _freeze(strata))

    @property
    def all_coordinates(self) -> Mapping[str, float]:
        return _freeze({**self.coordinates, **self.derived_coordinates})


@dataclass(frozen=True, slots=True)
class RejectedProposal:
    """Visible membership rejection from one explicitly separated DOE stream."""

    case_id: str
    stream_domain: str
    coordinates: Mapping[str, float]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _text(self.case_id, "case_id"))
        object.__setattr__(
            self,
            "stream_domain",
            _text(self.stream_domain, "stream_domain"),
        )
        coordinates = {
            name: _finite(value, f"rejected coordinates[{name!r}]")
            for name, value in self.coordinates.items()
        }
        reasons = tuple(self.reason_codes)
        if not coordinates:
            raise ValueError("rejected coordinates must not be empty")
        if not reasons or any(
            not isinstance(reason, str) or not reason for reason in reasons
        ):
            raise ValueError("rejected proposal must expose stable reason codes")
        object.__setattr__(self, "coordinates", _freeze(coordinates))
        object.__setattr__(self, "reason_codes", reasons)


@dataclass(frozen=True, slots=True)
class AxisSpec:
    """One weighted, fixed-strata axis in the coupled design space."""

    name: str
    source: AxisSource
    weight: float = 1.0
    strata: int = 4
    minimum_per_stratum: int = 1
    boundary_width: float = 0.10
    minimum_per_boundary: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "axis name"))
        object.__setattr__(self, "source", AxisSource(self.source))
        weight = _finite(self.weight, "axis weight")
        if weight <= 0.0:
            raise ValueError("axis weights must be positive")
        object.__setattr__(self, "weight", weight)
        _positive_int(self.strata, "axis strata")
        _positive_int(
            self.minimum_per_stratum,
            "minimum_per_stratum",
            allow_zero=True,
        )
        width = _finite(self.boundary_width, "boundary_width")
        if not 0.0 < width <= 0.5:
            raise ValueError("boundary_width must lie in (0, 0.5]")
        object.__setattr__(self, "boundary_width", width)
        _positive_int(
            self.minimum_per_boundary,
            "minimum_per_boundary",
            allow_zero=True,
        )


@dataclass(frozen=True, slots=True)
class InteractionSpec:
    """Declared fixed-strata interaction whose cells carry explicit quotas."""

    interaction_id: str
    axes: tuple[str, ...]
    minimum_per_cell: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "interaction_id",
            _text(self.interaction_id, "interaction_id"),
        )
        axes = tuple(self.axes)
        if len(axes) < 2 or any(not isinstance(axis, str) or not axis for axis in axes):
            raise ValueError("interaction axes require at least two names")
        if len(axes) != len(set(axes)):
            raise ValueError("interaction axes must be unique")
        object.__setattr__(self, "axes", axes)
        _positive_int(
            self.minimum_per_cell,
            "minimum_per_cell",
            allow_zero=True,
        )


@dataclass(frozen=True, slots=True)
class DifficultRegion:
    """Named region in normalized primitive-plus-derived coordinates."""

    region_id: str
    normalized_bounds: Mapping[str, tuple[float, float]]
    minimum_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_id", _text(self.region_id, "region_id"))
        checked: dict[str, tuple[float, float]] = {}
        for axis, interval in self.normalized_bounds.items():
            _text(axis, "difficult-region axis")
            if not isinstance(interval, tuple) or len(interval) != 2:
                raise ValueError("difficult-region bounds must be two-item tuples")
            lower = _finite(interval[0], f"{axis} lower bound")
            upper = _finite(interval[1], f"{axis} upper bound")
            if lower < 0.0 or upper > 1.0 or lower > upper:
                raise ValueError("difficult-region bounds must be normalized")
            checked[axis] = (lower, upper)
        if not checked:
            raise ValueError("difficult region must constrain at least one axis")
        object.__setattr__(self, "normalized_bounds", _freeze(checked))
        _positive_int(self.minimum_count, "minimum_count", allow_zero=True)

    def contains(self, normalized: Mapping[str, float]) -> bool:
        return all(
            axis in normalized and lower <= normalized[axis] <= upper
            for axis, (lower, upper) in self.normalized_bounds.items()
        )


@dataclass(frozen=True, slots=True)
class CoupledDesignSpec:
    """Complete deterministic DOE controls, including independent validation."""

    seed: bytes | str
    sample_count: int
    candidate_count: int
    validation_count: int
    validation_candidate_count: int
    axes: tuple[AxisSpec, ...]
    reserve_count: int = 0
    guard_band_count: int = 0
    guard_band_fraction: float = 0.05
    interactions: tuple[InteractionSpec, ...] = ()
    difficult_regions: tuple[DifficultRegion, ...] = ()

    def __post_init__(self) -> None:
        seed = self.seed.encode("utf-8") if isinstance(self.seed, str) else self.seed
        if not isinstance(seed, bytes) or not seed:
            raise ValueError("DOE seed must be non-empty bytes or text")
        object.__setattr__(self, "seed", seed)
        _positive_int(self.sample_count, "sample_count")
        _positive_int(self.reserve_count, "reserve_count", allow_zero=True)
        _positive_int(self.candidate_count, "candidate_count")
        if self.candidate_count < self.sample_count + self.reserve_count:
            raise ValueError("candidate_count must cover samples and reserves")
        _positive_int(self.validation_count, "validation_count")
        _positive_int(self.validation_candidate_count, "validation_candidate_count")
        if self.validation_candidate_count < self.validation_count:
            raise ValueError(
                "validation candidate count is smaller than validation count"
            )
        _positive_int(self.guard_band_count, "guard_band_count", allow_zero=True)
        fraction = _finite(self.guard_band_fraction, "guard_band_fraction")
        if fraction <= 0.0:
            raise ValueError("guard_band_fraction must be positive")
        object.__setattr__(self, "guard_band_fraction", fraction)

        axes = tuple(self.axes)
        if not axes or any(not isinstance(axis, AxisSpec) for axis in axes):
            raise ValueError("axes must contain AxisSpec values")
        names = tuple(axis.name for axis in axes)
        if len(names) != len(set(names)):
            raise ValueError("declared axis names must be unique")
        object.__setattr__(self, "axes", axes)

        interactions = tuple(self.interactions)
        if any(not isinstance(item, InteractionSpec) for item in interactions):
            raise ValueError("interactions must contain InteractionSpec values")
        interaction_ids = tuple(item.interaction_id for item in interactions)
        if len(interaction_ids) != len(set(interaction_ids)):
            raise ValueError("interaction identifiers must be unique")
        for interaction in interactions:
            if any(axis not in names for axis in interaction.axes):
                raise ValueError("interaction references an undeclared axis")
        object.__setattr__(self, "interactions", interactions)

        difficult = tuple(self.difficult_regions)
        if any(not isinstance(item, DifficultRegion) for item in difficult):
            raise ValueError("difficult_regions must contain DifficultRegion values")
        region_ids = tuple(item.region_id for item in difficult)
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("difficult-region identifiers must be unique")
        for region in difficult:
            if any(axis not in names for axis in region.normalized_bounds):
                raise ValueError("difficult region references an undeclared axis")
        object.__setattr__(self, "difficult_regions", difficult)


@dataclass(frozen=True, slots=True)
class CoupledDesign:
    """Selected DOE, preregistered reserves, guards, and validation fixture."""

    selected_points: tuple[DesignPoint, ...]
    reserve_points: tuple[DesignPoint, ...]
    guard_band_points: tuple[DesignPoint, ...]
    validation_points: tuple[DesignPoint, ...]
    candidate_rejections: tuple[RejectedProposal, ...]
    validation_rejections: tuple[RejectedProposal, ...]
    guard_band_rejections: tuple[RejectedProposal, ...]
    candidate_accepted_count: int
    validation_accepted_count: int
    selection_trace: tuple[str, ...]
    unresolved_quota_ids: tuple[str, ...]
    candidate_stream_id: str
    validation_stream_id: str
    guard_band_policy_id: str = "primitive-bound-negative-v1"

    def __post_init__(self) -> None:
        for field_name, item_type in (
            ("selected_points", DesignPoint),
            ("reserve_points", DesignPoint),
            ("guard_band_points", DesignPoint),
            ("validation_points", DesignPoint),
            ("candidate_rejections", RejectedProposal),
            ("validation_rejections", RejectedProposal),
            ("guard_band_rejections", RejectedProposal),
        ):
            values = tuple(getattr(self, field_name))
            if any(not isinstance(value, item_type) for value in values):
                raise ValueError(f"{field_name} contains invalid evidence")
            object.__setattr__(self, field_name, values)
        _positive_int(
            self.candidate_accepted_count,
            "candidate_accepted_count",
            allow_zero=True,
        )
        _positive_int(
            self.validation_accepted_count,
            "validation_accepted_count",
            allow_zero=True,
        )
        if self.candidate_accepted_count < len(self.selected_points) + len(
            self.reserve_points
        ):
            raise ValueError("candidate acceptance accounting is inconsistent")
        if self.validation_accepted_count < len(self.validation_points):
            raise ValueError("validation acceptance accounting is inconsistent")
        if len(self.guard_band_points) != len(self.guard_band_rejections):
            raise ValueError("every guard-band negative must retain its rejection")
        if tuple(point.case_id for point in self.guard_band_points) != tuple(
            rejection.case_id for rejection in self.guard_band_rejections
        ):
            raise ValueError("guard-band points and rejection records must align")
        if self.guard_band_policy_id != "primitive-bound-negative-v1":
            raise ValueError("unsupported guard-band policy")

    @property
    def points(self) -> tuple[DesignPoint, ...]:
        """Return active and guard-band cases without mixing reserve evidence."""

        return (*self.selected_points, *self.guard_band_points)


@dataclass(frozen=True, slots=True)
class _CompiledAxis:
    spec: AxisSpec
    lower: float
    upper: float

    def normalize(self, value: float) -> float:
        if self.upper == self.lower:
            return 0.5
        return (value - self.lower) / (self.upper - self.lower)


@dataclass(frozen=True, slots=True)
class _Candidate:
    point: DesignPoint
    normalized: Mapping[str, float]


def generate_coupled_design(
    predicate: FamilyMembershipPredicateLike,
    spec: CoupledDesignSpec,
) -> CoupledDesign:
    """Build a constrained maximin DOE using only S6's public predicate surface."""

    if not isinstance(spec, CoupledDesignSpec):
        raise TypeError("spec must be a CoupledDesignSpec")
    compiled = _compile_axes(predicate, spec)
    candidates, candidate_rejections = _evaluate_lhs_proposals(
        predicate,
        spec,
        compiled,
        count=spec.candidate_count,
        domain=_CANDIDATE_DOMAIN,
        case_prefix="candidate",
    )
    required = spec.sample_count + spec.reserve_count
    if len(candidates) < required:
        raise ValueError(
            "coupled membership accepted fewer candidates than samples and reserves"
        )

    requirements = quota_requirements(spec)
    selected, remaining, trace, counts = _select_rarest_quota_first(
        candidates,
        spec.sample_count,
        spec,
        requirements,
    )
    reserves = _select_maximin(
        remaining,
        spec.reserve_count,
        selected,
        spec.axes,
    )
    validation_pool, validation_rejections = _evaluate_lhs_proposals(
        predicate,
        spec,
        compiled,
        count=spec.validation_candidate_count,
        domain=_VALIDATION_DOMAIN,
        case_prefix="validation",
    )
    if len(validation_pool) < spec.validation_count:
        raise ValueError("independent validation stream has too few accepted points")
    validation = validation_pool[: spec.validation_count]
    guards, guard_rejections = _guard_band_points(predicate, spec, compiled)
    unresolved = tuple(
        quota_id
        for quota_id, target in requirements.items()
        if counts.get(quota_id, 0) < target
    )
    return CoupledDesign(
        selected_points=tuple(candidate.point for candidate in selected),
        reserve_points=tuple(candidate.point for candidate in reserves),
        guard_band_points=guards,
        validation_points=tuple(candidate.point for candidate in validation),
        candidate_rejections=candidate_rejections,
        validation_rejections=validation_rejections,
        guard_band_rejections=guard_rejections,
        candidate_accepted_count=len(candidates),
        validation_accepted_count=len(validation_pool),
        selection_trace=trace,
        unresolved_quota_ids=unresolved,
        candidate_stream_id=_stream_id(spec.seed, _CANDIDATE_DOMAIN),
        validation_stream_id=_stream_id(spec.seed, _VALIDATION_DOMAIN),
    )


def normalized_coordinates(
    predicate: FamilyMembershipPredicateLike,
    axes: Sequence[AxisSpec],
    point: DesignPoint,
) -> Mapping[str, float]:
    """Normalize one accepted point against the predicate's declared bounds."""

    temporary = CoupledDesignSpec(
        seed=b"normalization-contract",
        sample_count=1,
        candidate_count=1,
        validation_count=1,
        validation_candidate_count=1,
        axes=tuple(axes),
    )
    compiled = _compile_axes(predicate, temporary)
    combined = point.all_coordinates
    values = {
        axis.spec.name: axis.normalize(combined[axis.spec.name]) for axis in compiled
    }
    return _freeze(values)


def quota_requirements(spec: CoupledDesignSpec) -> Mapping[str, int]:
    """Return the finite, preregistered quota table for a design."""

    axis_by_name = {axis.name: axis for axis in spec.axes}
    requirements: dict[str, int] = {}
    for axis in spec.axes:
        for index in range(axis.strata):
            if axis.minimum_per_stratum:
                requirements[f"stratum:{axis.name}:{index}"] = axis.minimum_per_stratum
        if axis.minimum_per_boundary:
            requirements[f"boundary:{axis.name}:lower"] = axis.minimum_per_boundary
            requirements[f"boundary:{axis.name}:upper"] = axis.minimum_per_boundary
    for interaction in spec.interactions:
        ranges = tuple(range(axis_by_name[name].strata) for name in interaction.axes)
        for cell in product(*ranges):
            if interaction.minimum_per_cell:
                suffix = ",".join(
                    f"{axis}={index}"
                    for axis, index in zip(interaction.axes, cell, strict=True)
                )
                requirements[f"interaction:{interaction.interaction_id}:{suffix}"] = (
                    interaction.minimum_per_cell
                )
    for region in spec.difficult_regions:
        if region.minimum_count:
            requirements[f"difficult:{region.region_id}"] = region.minimum_count
    return _freeze(requirements)


def requirement_ids_for_point(
    normalized: Mapping[str, float],
    spec: CoupledDesignSpec,
) -> frozenset[str]:
    """Identify every declared quota covered by normalized coordinates."""

    strata = {
        axis.name: _stratum(normalized[axis.name], axis.strata) for axis in spec.axes
    }
    covered: set[str] = {
        f"stratum:{axis.name}:{strata[axis.name]}"
        for axis in spec.axes
        if axis.minimum_per_stratum
    }
    for axis in spec.axes:
        if axis.minimum_per_boundary and normalized[axis.name] <= axis.boundary_width:
            covered.add(f"boundary:{axis.name}:lower")
        if (
            axis.minimum_per_boundary
            and normalized[axis.name] >= 1.0 - axis.boundary_width
        ):
            covered.add(f"boundary:{axis.name}:upper")
    for interaction in spec.interactions:
        if interaction.minimum_per_cell:
            suffix = ",".join(f"{name}={strata[name]}" for name in interaction.axes)
            covered.add(f"interaction:{interaction.interaction_id}:{suffix}")
    for region in spec.difficult_regions:
        if region.minimum_count and region.contains(normalized):
            covered.add(f"difficult:{region.region_id}")
    return frozenset(covered)


def weighted_distance(
    left: Mapping[str, float],
    right: Mapping[str, float],
    axes: Sequence[AxisSpec],
) -> float:
    """Weighted Euclidean distance in normalized primitive-plus-derived space."""

    declared = tuple(axes)
    total_weight = sum(axis.weight for axis in declared)
    if total_weight <= 0.0:
        raise ValueError("distance requires positive declared axis weights")
    return math.sqrt(
        sum(
            axis.weight * (left[axis.name] - right[axis.name]) ** 2 for axis in declared
        )
        / total_weight
    )


def _compile_axes(
    predicate: FamilyMembershipPredicateLike,
    spec: CoupledDesignSpec,
) -> tuple[_CompiledAxis, ...]:
    primitive = _bound_table(predicate.primitive_bounds, "primitive")
    derived = _bound_table(predicate.derived_bounds, "derived")
    declarations = {axis.name: axis for axis in spec.axes}
    variable_primitive = {
        name for name, (lower, upper, _bound) in primitive.items() if upper > lower
    }
    missing_primitive = variable_primitive - set(declarations)
    if missing_primitive:
        raise ValueError(
            "DOE is missing variable primitive axes: "
            + ", ".join(sorted(missing_primitive))
        )
    missing_derived = set(derived) - set(declarations)
    if missing_derived:
        raise ValueError(
            "DOE is missing derived governing axes: "
            + ", ".join(sorted(missing_derived))
        )
    known = set(primitive) | set(derived)
    unknown = set(declarations) - known
    if unknown:
        raise ValueError("DOE contains axes absent from the membership predicate")

    compiled = []
    for axis in spec.axes:
        table = primitive if axis.source is AxisSource.PRIMITIVE else derived
        if axis.name not in table:
            raise ValueError(
                f"axis {axis.name!r} has the wrong primitive/derived source"
            )
        lower, upper, _bound = table[axis.name]
        if lower == upper and axis.strata != 1:
            raise ValueError("fixed predicate axes must declare exactly one stratum")
        compiled.append(_CompiledAxis(axis, lower, upper))
    return tuple(compiled)


def _bound_table(
    bounds: Sequence[_BoundLike],
    label: str,
) -> dict[str, tuple[float, float, _BoundLike]]:
    table: dict[str, tuple[float, float, _BoundLike]] = {}
    for bound in tuple(bounds):
        name = _text(getattr(bound, "name", None), f"{label} bound name")
        if name in table:
            raise ValueError(f"duplicate {label} membership bound")
        lower = _finite(bound.lower_bound.value, f"{name} lower bound")
        upper = _finite(bound.upper_bound.value, f"{name} upper bound")
        if lower > upper:
            raise ValueError(f"{name} membership bounds are reversed")
        table[name] = (lower, upper, bound)
    if label == "primitive" and not table:
        raise ValueError("membership predicate has no primitive bounds")
    return table


def _evaluate_lhs_proposals(
    predicate: FamilyMembershipPredicateLike,
    spec: CoupledDesignSpec,
    compiled: Sequence[_CompiledAxis],
    *,
    count: int,
    domain: str,
    case_prefix: str,
) -> tuple[tuple[_Candidate, ...], tuple[RejectedProposal, ...]]:
    primitive = _bound_table(predicate.primitive_bounds, "primitive")
    permutations = {
        name: _permutation(spec.seed, f"{domain}/permutation/{name}", count)
        for name in primitive
    }
    accepted: tuple[_Candidate, ...] = ()
    rejected: tuple[RejectedProposal, ...] = ()
    for row in range(count):
        raw = {
            name: lower
            + (upper - lower)
            * (
                permutations[name][row]
                + _uniform(spec.seed, f"{domain}/jitter/{name}", row)
            )
            / count
            for name, (lower, upper, _bound) in primitive.items()
        }
        evaluated = _evaluate_candidate(
            predicate,
            compiled,
            raw,
            case_id=f"{case_prefix}-{row:06d}",
            stream_domain=domain,
        )
        if isinstance(evaluated, _Candidate):
            accepted = (*accepted, evaluated)
        else:
            rejected = (*rejected, evaluated)
    return accepted, rejected


def _evaluate_candidate(
    predicate: FamilyMembershipPredicateLike,
    compiled: Sequence[_CompiledAxis],
    raw: Mapping[str, float],
    *,
    case_id: str,
    stream_domain: str,
) -> _Candidate | RejectedProposal:
    primitive = _bound_table(predicate.primitive_bounds, "primitive")
    theta: dict[str, object] = {}
    for name, value in raw.items():
        bound = primitive[name][2]
        dimension = bound.lower_bound.dimension
        if not isinstance(bound.lower_bound.unit, Unit) or not isinstance(
            dimension, Dimension
        ):
            raise ValueError("membership bounds must expose canonical units")
        theta[name] = CanonicalQuantity(value, bound.lower_bound.unit, dimension)
    decision = predicate.evaluate(theta)
    if not isinstance(decision.accepted, bool):
        raise ValueError("membership decision accepted flag must be boolean")
    if not decision.accepted:
        return RejectedProposal(
            case_id=case_id,
            stream_domain=stream_domain,
            coordinates=raw,
            reason_codes=_stable_reason_codes(decision),
        )
    expected_derived = {bound.name for bound in predicate.derived_bounds}
    missing = expected_derived - set(decision.derived_values)
    if missing:
        raise ValueError("accepted membership decision omitted derived coordinates")
    derived = {
        name: _finite(value.value, f"derived coordinate {name}")
        for name, value in decision.derived_values.items()
    }
    combined = {**raw, **derived}
    normalized = {
        axis.spec.name: axis.normalize(combined[axis.spec.name]) for axis in compiled
    }
    if any(value < 0.0 or value > 1.0 for value in normalized.values()):
        raise ValueError("accepted membership point lies outside declared axis bounds")
    strata = {
        axis.spec.name: _stratum(normalized[axis.spec.name], axis.spec.strata)
        for axis in compiled
    }
    return _Candidate(
        point=DesignPoint(
            case_id=case_id,
            leakage_group_id=f"leakage-{_coordinate_id(raw)}",
            coordinates=raw,
            derived_coordinates=derived,
            strata=strata,
        ),
        normalized=_freeze(normalized),
    )


def _select_rarest_quota_first(
    candidates: Sequence[_Candidate],
    count: int,
    spec: CoupledDesignSpec,
    requirements: Mapping[str, int],
) -> tuple[
    tuple[_Candidate, ...],
    tuple[_Candidate, ...],
    tuple[str, ...],
    Mapping[str, int],
]:
    selected: tuple[_Candidate, ...] = ()
    remaining = tuple(candidates)
    trace: tuple[str, ...] = ()
    counts: Mapping[str, int] = _freeze({key: 0 for key in requirements})
    coverage = {
        candidate.point.case_id: requirement_ids_for_point(candidate.normalized, spec)
        for candidate in candidates
    }
    for _ in range(count):
        unmet = tuple(
            quota_id
            for quota_id, target in requirements.items()
            if counts[quota_id] < target
        )
        reachable = tuple(
            (
                sum(
                    quota_id in coverage[candidate.point.case_id]
                    for candidate in remaining
                ),
                quota_id,
            )
            for quota_id in unmet
        )
        reachable = tuple(item for item in reachable if item[0] > 0)
        target = min(reachable)[1] if reachable else None
        eligible = (
            tuple(
                candidate
                for candidate in remaining
                if target in coverage[candidate.point.case_id]
            )
            if target is not None
            else remaining
        )
        chosen = _maximin_choice(eligible, selected, spec.axes, coverage, unmet)
        chosen_ids = coverage[chosen.point.case_id]
        counts = _freeze(
            {
                quota_id: value + int(quota_id in chosen_ids)
                for quota_id, value in counts.items()
            }
        )
        selected = (*selected, chosen)
        remaining = tuple(
            candidate
            for candidate in remaining
            if candidate.point.case_id != chosen.point.case_id
        )
        trace = (*trace, f"quota:{target}" if target is not None else "maximin")
    return selected, remaining, trace, counts


def _select_maximin(
    candidates: Sequence[_Candidate],
    count: int,
    anchors: Sequence[_Candidate],
    axes: Sequence[AxisSpec],
) -> tuple[_Candidate, ...]:
    selected: tuple[_Candidate, ...] = ()
    remaining = tuple(candidates)
    for _ in range(count):
        chosen = _maximin_choice(
            remaining,
            (*tuple(anchors), *selected),
            axes,
            {},
            (),
        )
        selected = (*selected, chosen)
        remaining = tuple(
            candidate
            for candidate in remaining
            if candidate.point.case_id != chosen.point.case_id
        )
    return selected


def _maximin_choice(
    candidates: Sequence[_Candidate],
    selected: Sequence[_Candidate],
    axes: Sequence[AxisSpec],
    coverage: Mapping[str, frozenset[str]],
    unmet: Sequence[str],
) -> _Candidate:
    if not candidates:
        raise ValueError("maximin selection exhausted its candidate population")
    unmet_set = frozenset(unmet)

    def key(candidate: _Candidate) -> tuple[float, int, str]:
        distance = (
            min(
                weighted_distance(candidate.normalized, other.normalized, axes)
                for other in selected
            )
            if selected
            else 0.0
        )
        extra = len(coverage.get(candidate.point.case_id, frozenset()) & unmet_set)
        return (-distance, -extra, candidate.point.case_id)

    return min(candidates, key=key)


def _guard_band_points(
    predicate: FamilyMembershipPredicateLike,
    spec: CoupledDesignSpec,
    compiled: Sequence[_CompiledAxis],
) -> tuple[tuple[DesignPoint, ...], tuple[RejectedProposal, ...]]:
    """Create primitive-bound negative cases; derived OOD guards are not implied."""

    primitive = _bound_table(predicate.primitive_bounds, "primitive")
    names = tuple(primitive)
    guards: tuple[DesignPoint, ...] = ()
    rejections: tuple[RejectedProposal, ...] = ()
    for index in range(spec.guard_band_count):
        target = names[index % len(names)]
        upper_side = (index // len(names)) % 2 == 1
        raw = {
            name: lower
            + (upper - lower)
            * _uniform(spec.seed, f"{_GUARD_DOMAIN}/interior/{name}", index)
            for name, (lower, upper, _bound) in primitive.items()
        }
        lower, upper, _bound = primitive[target]
        span = upper - lower
        offset = spec.guard_band_fraction * (span if span > 0.0 else 1.0)
        raw[target] = upper + offset if upper_side else lower - offset

        theta: dict[str, object] = {}
        for name, value in raw.items():
            bound = primitive[name][2]
            dimension = bound.lower_bound.dimension
            if not isinstance(bound.lower_bound.unit, Unit) or not isinstance(
                dimension, Dimension
            ):
                raise ValueError("membership bounds must expose canonical units")
            theta[name] = CanonicalQuantity(value, bound.lower_bound.unit, dimension)
        decision = predicate.evaluate(theta)
        if decision.accepted:
            raise ValueError(
                "guard-band OOD point was accepted by membership predicate"
            )
        rejection = RejectedProposal(
            case_id=f"guard-{index:06d}",
            stream_domain=_GUARD_DOMAIN,
            coordinates=raw,
            reason_codes=_stable_reason_codes(decision),
        )
        derived = {
            name: _finite(value.value, f"guard derived coordinate {name}")
            for name, value in decision.derived_values.items()
        }
        guards = (
            *guards,
            DesignPoint(
                case_id=f"guard-{index:06d}",
                leakage_group_id=f"guard-{index:06d}",
                coordinates=raw,
                derived_coordinates=derived,
                strata={axis.spec.name: axis.spec.strata for axis in compiled},
                guard_band=True,
            ),
        )
        rejections = (*rejections, rejection)
    return guards, rejections


def _stable_reason_codes(decision: _DecisionLike) -> tuple[str, ...]:
    raw_reasons = getattr(decision, "reason_codes", ())
    if not isinstance(raw_reasons, Sequence) or isinstance(
        raw_reasons, (str, bytes, bytearray)
    ):
        raise ValueError("membership rejection reasons must be a sequence")
    stable = tuple(_stable_reason_value(reason) for reason in raw_reasons)
    return stable or ("UNSPECIFIED_MEMBERSHIP_REJECTION",)


def _stable_reason_value(reason: object) -> str:
    value = getattr(reason, "value", reason)
    if not isinstance(value, str) or not value:
        raise ValueError("membership rejection reasons must have stable text values")
    return value


def _stratum(normalized: float, strata: int) -> int:
    return min(strata - 1, max(0, int(normalized * strata)))


def _seed_key(seed: bytes | str) -> bytes:
    raw = seed.encode("utf-8") if isinstance(seed, str) else seed
    return hashlib.blake2b(raw, digest_size=32, person=_SEED_PERSON).digest()


def _digest(seed: bytes | str, domain: str, index: int) -> bytes:
    payload = domain.encode("utf-8") + b"\0" + index.to_bytes(8, "big")
    return hashlib.blake2b(
        payload,
        key=_seed_key(seed),
        digest_size=16,
        person=_PERSON,
    ).digest()


def _uniform(seed: bytes | str, domain: str, index: int) -> float:
    integer = int.from_bytes(_digest(seed, domain, index)[:8], "big")
    return (integer + 0.5) / 2**64


def _permutation(seed: bytes | str, domain: str, count: int) -> tuple[int, ...]:
    return tuple(
        sorted(range(count), key=lambda index: (_digest(seed, domain, index), index))
    )


def _stream_id(seed: bytes | str, domain: str) -> str:
    return hashlib.blake2b(
        domain.encode("utf-8"),
        key=_seed_key(seed),
        digest_size=16,
        person=_PERSON,
    ).hexdigest()


def _coordinate_id(coordinates: Mapping[str, float]) -> str:
    payload = "\0".join(
        f"{name}={value.hex()}" for name, value in sorted(coordinates.items())
    ).encode("ascii")
    return hashlib.blake2b(payload, digest_size=12, person=_PERSON).hexdigest()
