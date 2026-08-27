"""Deterministic, coupled-domain S9 design-of-experiments construction."""

from __future__ import annotations

import base64
import binascii
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import product
from types import MappingProxyType
from typing import ClassVar, Protocol, TypeVar

from piginot.contracts import Dimension, Unit
from piginot.doe.leakage import LeakageDescriptor, LeakagePolicy, bind_leakage_groups

_BASE64URL_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
)
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


def _seed_payload(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _stratum(normalized: float, strata: int) -> int:
    return min(strata - 1, max(0, int(normalized * strata)))


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


class SeedEncoding(StrEnum):
    """Unambiguous canonical origin and encoding of DOE seed material."""

    TEXT_UTF8_BASE64URL_V1 = "text-utf8-base64url-v1"
    BYTES_BASE64URL_V1 = "bytes-base64url-v1"


@dataclass(frozen=True, slots=True)
class DoeSeed:
    """Canonical seed preimage; payload is hidden from ordinary representations."""

    __canonical_namespace__: ClassVar[str] = "piginot.doe"

    encoding: SeedEncoding
    payload: str = field(repr=False)

    def __post_init__(self) -> None:
        encoding = SeedEncoding(self.encoding)
        payload = _text(self.payload, "canonical seed payload")
        if any(character not in _BASE64URL_CHARACTERS for character in payload):
            raise ValueError("canonical seed payload must be unpadded base64url")
        padding = "=" * (-len(payload) % 4)
        try:
            raw = base64.urlsafe_b64decode(payload + padding)
        except binascii.Error as exc:
            raise ValueError("canonical seed payload is not valid base64url") from exc
        if not raw or _seed_payload(raw) != payload:
            raise ValueError("canonical seed payload is not minimal base64url")
        if encoding is SeedEncoding.TEXT_UTF8_BASE64URL_V1:
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("text seed payload must decode as UTF-8") from exc
        object.__setattr__(self, "encoding", encoding)
        object.__setattr__(self, "payload", payload)

    @classmethod
    def from_input(cls, value: DoeSeed | bytes | str) -> DoeSeed:
        """Tag text and bytes separately before applying unpadded base64url."""

        if isinstance(value, DoeSeed):
            return value
        if isinstance(value, str):
            encoding = SeedEncoding.TEXT_UTF8_BASE64URL_V1
            raw = value.encode("utf-8")
        elif isinstance(value, bytes):
            encoding = SeedEncoding.BYTES_BASE64URL_V1
            raw = value
        else:
            raise ValueError("DOE seed must be non-empty bytes or text")
        if not raw:
            raise ValueError("DOE seed must be non-empty bytes or text")
        return cls(encoding=encoding, payload=_seed_payload(raw))

    def key_material(self) -> bytes:
        """Return a tagged ASCII preimage for deterministic BLAKE2b derivation."""

        return f"{self.encoding.value}\0{self.payload}".encode("ascii")


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
    """Canonical DOE controls with tagged seed identity and independent validation."""

    seed: DoeSeed | bytes | str = field(repr=False)
    sample_count: int
    candidate_count: int
    validation_count: int
    validation_candidate_count: int
    axes: tuple[AxisSpec, ...]
    leakage_policy: LeakagePolicy
    reserve_count: int = 0
    guard_band_count: int = 0
    guard_band_fraction: float = 0.05
    interactions: tuple[InteractionSpec, ...] = ()
    difficult_regions: tuple[DifficultRegion, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", DoeSeed.from_input(self.seed))
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
        if type(self.leakage_policy) is not LeakagePolicy:
            raise ValueError("leakage_policy must be a LeakagePolicy")

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
    leakage_policy: LeakagePolicy
    leakage_descriptors: tuple[LeakageDescriptor, ...]
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
        if type(self.leakage_policy) is not LeakagePolicy:
            raise ValueError("leakage_policy must be a LeakagePolicy")
        descriptors = tuple(self.leakage_descriptors)
        if any(type(item) is not LeakageDescriptor for item in descriptors):
            raise ValueError(
                "leakage_descriptors must contain LeakageDescriptor values"
            )
        campaign_points = (*self.selected_points, *self.reserve_points)
        campaign_case_ids = tuple(point.case_id for point in campaign_points)
        descriptor_case_ids = tuple(item.case_id for item in descriptors)
        if set(descriptor_case_ids) != set(campaign_case_ids):
            raise ValueError(
                "leakage descriptors must match selected and reserve cases"
            )
        if len(descriptor_case_ids) != len(set(descriptor_case_ids)):
            raise ValueError("leakage descriptors must have unique case IDs")
        canonical_descriptors = tuple(
            sorted(descriptors, key=lambda item: item.case_id)
        )
        groups = bind_leakage_groups(
            canonical_descriptors,
            policy=self.leakage_policy,
        )
        for point in campaign_points:
            if point.leakage_group_id != groups[point.case_id]:
                raise ValueError(
                    f"case {point.case_id!r} does not match leakage grouping policy"
                )
        object.__setattr__(self, "leakage_descriptors", canonical_descriptors)

    @property
    def points(self) -> tuple[DesignPoint, ...]:
        """Return active and guard-band cases without mixing reserve evidence."""

        return (*self.selected_points, *self.guard_band_points)


def generate_coupled_design(
    predicate: FamilyMembershipPredicateLike,
    spec: CoupledDesignSpec,
) -> CoupledDesign:
    """Build a constrained maximin DOE using only S6's public predicate surface."""

    from piginot.doe._design_sampling import generate_coupled_design_impl

    return generate_coupled_design_impl(predicate, spec)


def normalized_coordinates(
    predicate: FamilyMembershipPredicateLike,
    axes: Sequence[AxisSpec],
    point: DesignPoint,
) -> Mapping[str, float]:
    """Normalize one accepted point against the predicate's declared bounds."""

    from piginot.doe._design_sampling import normalized_coordinates_impl

    return normalized_coordinates_impl(predicate, axes, point)


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
