"""Independent, non-aggregate S9 coupled-design coverage measurements."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from piginot.doe.design import (
    CoupledDesign,
    CoupledDesignSpec,
    FamilyMembershipPredicateLike,
    normalized_coordinates,
    quota_requirements,
    requirement_ids_for_point,
    weighted_distance,
)


def _fraction(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must lie in [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class DesignCoverage:
    """Separate DOE coverage measures; deliberately no composite percentage."""

    axes: tuple[str, ...]
    empirical_fill_distance: float
    minimum_stratum_count: int
    stratum_coverage: float
    boundary_coverage: float
    interaction_coverage: float
    difficult_region_coverage: float
    uncovered_ids: tuple[str, ...]
    unresolved_ids: tuple[str, ...]
    validation_count: int
    validation_stream_id: str
    requirement_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        if not axes or any(not isinstance(axis, str) or not axis for axis in axes):
            raise ValueError("coverage axes must be non-empty text")
        if len(axes) != len(set(axes)):
            raise ValueError("coverage axes must be unique")
        object.__setattr__(self, "axes", axes)
        fill = float(self.empirical_fill_distance)
        if not math.isfinite(fill) or fill < 0.0:
            raise ValueError("empirical_fill_distance must be finite and non-negative")
        object.__setattr__(self, "empirical_fill_distance", fill)
        if (
            isinstance(self.minimum_stratum_count, bool)
            or not isinstance(self.minimum_stratum_count, int)
            or self.minimum_stratum_count < 0
        ):
            raise ValueError("minimum_stratum_count must be a non-negative integer")
        for field_name in (
            "stratum_coverage",
            "boundary_coverage",
            "interaction_coverage",
            "difficult_region_coverage",
        ):
            object.__setattr__(
                self,
                field_name,
                _fraction(getattr(self, field_name), field_name),
            )
        uncovered = tuple(self.uncovered_ids)
        unresolved = tuple(self.unresolved_ids)
        if any(
            not isinstance(item, str) or not item for item in (*uncovered, *unresolved)
        ):
            raise ValueError("coverage requirement identifiers must be non-empty text")
        if set(uncovered) - set(unresolved):
            raise ValueError("every uncovered requirement must also be unresolved")
        object.__setattr__(self, "uncovered_ids", tuple(sorted(uncovered)))
        object.__setattr__(self, "unresolved_ids", tuple(sorted(unresolved)))
        if (
            isinstance(self.validation_count, bool)
            or not isinstance(self.validation_count, int)
            or self.validation_count <= 0
        ):
            raise ValueError("validation_count must be positive")
        if (
            not isinstance(self.validation_stream_id, str)
            or not self.validation_stream_id
        ):
            raise ValueError("validation_stream_id must be non-empty text")
        counts = dict(sorted(self.requirement_counts.items()))
        if any(
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in counts.items()
        ):
            raise ValueError("requirement counts must map identifiers to counts")
        object.__setattr__(self, "requirement_counts", MappingProxyType(counts))


def measure_design_coverage(
    predicate: FamilyMembershipPredicateLike,
    spec: CoupledDesignSpec,
    design: CoupledDesign,
) -> DesignCoverage:
    """Measure selected points against an independently generated validation stream."""

    if not design.selected_points:
        raise ValueError("coverage requires selected DOE points")
    if len(design.validation_points) != spec.validation_count:
        raise ValueError("coverage validation population differs from its declaration")

    selected = tuple(
        normalized_coordinates(predicate, spec.axes, point)
        for point in design.selected_points
    )
    validation = tuple(
        normalized_coordinates(predicate, spec.axes, point)
        for point in design.validation_points
    )
    fill_distance = max(
        min(weighted_distance(probe, point, spec.axes) for point in selected)
        for probe in validation
    )

    requirements = quota_requirements(spec)
    counts: Mapping[str, int] = MappingProxyType(
        {quota_id: 0 for quota_id in requirements}
    )
    for point in selected:
        covered = requirement_ids_for_point(point, spec)
        counts = MappingProxyType(
            {
                quota_id: value + int(quota_id in covered)
                for quota_id, value in counts.items()
            }
        )

    stratum_counts = _all_stratum_counts(spec, selected)
    minimum_stratum_count = min(stratum_counts.values())
    unresolved = tuple(
        quota_id
        for quota_id, target in requirements.items()
        if counts[quota_id] < target
    )
    uncovered = tuple(quota_id for quota_id in unresolved if counts[quota_id] == 0)

    return DesignCoverage(
        axes=tuple(axis.name for axis in spec.axes),
        empirical_fill_distance=fill_distance,
        minimum_stratum_count=minimum_stratum_count,
        stratum_coverage=_quota_fraction("stratum:", requirements, counts),
        boundary_coverage=_quota_fraction("boundary:", requirements, counts),
        interaction_coverage=_quota_fraction("interaction:", requirements, counts),
        difficult_region_coverage=_quota_fraction("difficult:", requirements, counts),
        uncovered_ids=uncovered,
        unresolved_ids=unresolved,
        validation_count=len(validation),
        validation_stream_id=design.validation_stream_id,
        requirement_counts=counts,
    )


def coverage_gate_checks(
    report: DesignCoverage,
    *,
    minimum_fraction: float,
    maximum_fill_distance: float,
    minimum_stratum_count: int,
) -> Mapping[str, bool]:
    """Apply each declared threshold independently, never via a geometric score."""

    fraction = _fraction(minimum_fraction, "minimum_fraction")
    if (
        isinstance(maximum_fill_distance, bool)
        or not isinstance(maximum_fill_distance, (int, float))
        or not math.isfinite(float(maximum_fill_distance))
        or maximum_fill_distance < 0.0
    ):
        raise ValueError("maximum_fill_distance must be finite and non-negative")
    if (
        isinstance(minimum_stratum_count, bool)
        or not isinstance(minimum_stratum_count, int)
        or minimum_stratum_count < 0
    ):
        raise ValueError("minimum_stratum_count must be non-negative")
    return MappingProxyType(
        {
            "empirical_fill_distance": report.empirical_fill_distance
            <= maximum_fill_distance,
            "minimum_stratum_count": report.minimum_stratum_count
            >= minimum_stratum_count,
            "stratum_coverage": report.stratum_coverage >= fraction,
            "boundary_coverage": report.boundary_coverage >= fraction,
            "interaction_coverage": report.interaction_coverage >= fraction,
            "difficult_region_coverage": report.difficult_region_coverage >= fraction,
            "unresolved_requirements": not report.unresolved_ids,
        }
    )


def _all_stratum_counts(
    spec: CoupledDesignSpec,
    points: tuple[Mapping[str, float], ...],
) -> Mapping[str, int]:
    counts = {
        f"stratum:{axis.name}:{index}": 0
        for axis in spec.axes
        for index in range(axis.strata)
    }
    for point in points:
        for axis in spec.axes:
            normalized = point[axis.name]
            index = min(axis.strata - 1, max(0, int(normalized * axis.strata)))
            identifier = f"stratum:{axis.name}:{index}"
            counts = {**counts, identifier: counts[identifier] + 1}
    return MappingProxyType(dict(sorted(counts.items())))


def _quota_fraction(
    prefix: str,
    requirements: Mapping[str, int],
    counts: Mapping[str, int],
) -> float:
    relevant = tuple(
        quota_id for quota_id in requirements if quota_id.startswith(prefix)
    )
    if not relevant:
        return 1.0
    met = sum(counts[quota_id] >= requirements[quota_id] for quota_id in relevant)
    return met / len(relevant)
