"""Deterministic leakage groups for correlated and near-duplicate DOE cases."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

from piginot.contracts import canonical_json_bytes

_LEAKAGE_POLICY_ID = "geometry-lineage-morph-seed-chebyshev-v1"
_JCS_SAFE_INTEGER = 2**53 - 1


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _coordinate(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite normalized number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must lie in [0, 1]")
    return number


def _tolerance(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("near-duplicate tolerance must be numeric")
    tolerance = float(value)
    if not math.isfinite(tolerance) or not 0.0 <= tolerance <= 1.0:
        raise ValueError("near-duplicate tolerance must lie in [0, 1]")
    return tolerance


@dataclass(frozen=True, slots=True)
class LeakagePolicy:
    """Approved S9 rule for turning correlation evidence into leakage groups."""

    __canonical_namespace__: ClassVar[str] = "piginot.doe"

    near_duplicate_tolerance: float
    policy_id: str = _LEAKAGE_POLICY_ID

    def __post_init__(self) -> None:
        if self.policy_id != _LEAKAGE_POLICY_ID:
            raise ValueError("unsupported leakage policy")
        object.__setattr__(
            self,
            "near_duplicate_tolerance",
            _tolerance(self.near_duplicate_tolerance),
        )


@dataclass(frozen=True, slots=True)
class LeakageDescriptor:
    """Correlation evidence available before role assignment."""

    __canonical_namespace__: ClassVar[str] = "piginot.doe"

    case_id: str
    geometry_lineage_id: str
    morph_seed: int
    normalized_coordinates: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _identifier(self.case_id, "case_id"))
        object.__setattr__(
            self,
            "geometry_lineage_id",
            _identifier(self.geometry_lineage_id, "geometry_lineage_id"),
        )
        if (
            isinstance(self.morph_seed, bool)
            or not isinstance(self.morph_seed, int)
            or self.morph_seed < 0
            or self.morph_seed > _JCS_SAFE_INTEGER
        ):
            raise ValueError("morph_seed must be a non-negative JSON-safe integer")
        if not isinstance(self.normalized_coordinates, Mapping):
            raise ValueError("normalized_coordinates must be a mapping")
        coordinates = {
            _identifier(name, "coordinate name"): _coordinate(
                value, f"normalized coordinate {name!r}"
            )
            for name, value in self.normalized_coordinates.items()
        }
        if not coordinates:
            raise ValueError("normalized_coordinates must not be empty")
        object.__setattr__(
            self,
            "normalized_coordinates",
            MappingProxyType(dict(sorted(coordinates.items()))),
        )


def bind_leakage_groups(
    descriptors: Sequence[LeakageDescriptor],
    *,
    policy: LeakagePolicy,
) -> Mapping[str, str]:
    """Bind the transitive closure of lineage, seed, and proximity relations."""

    population = tuple(descriptors)
    if not population:
        raise ValueError("leakage grouping requires at least one descriptor")
    if any(type(item) is not LeakageDescriptor for item in population):
        raise ValueError("descriptors must contain LeakageDescriptor values")
    case_ids = tuple(item.case_id for item in population)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case_id values must be unique")
    if type(policy) is not LeakagePolicy:
        raise ValueError("policy must be a LeakagePolicy")
    tolerance = policy.near_duplicate_tolerance
    schemas = {tuple(item.normalized_coordinates) for item in population}
    if len(schemas) != 1:
        raise ValueError("normalized coordinate schema must match across cases")

    parents = list(range(len(population)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        lower, upper = sorted((left_root, right_root))
        parents[upper] = lower

    for left_index, left in enumerate(population):
        for right_index in range(left_index + 1, len(population)):
            right = population[right_index]
            correlated = (
                left.geometry_lineage_id == right.geometry_lineage_id
                or left.morph_seed == right.morph_seed
                or _maximum_distance(left, right) <= tolerance
            )
            if correlated:
                union(left_index, right_index)

    components: dict[int, list[LeakageDescriptor]] = {}
    for index, descriptor in enumerate(population):
        components.setdefault(find(index), []).append(descriptor)

    result: dict[str, str] = {}
    for members in components.values():
        group_id = _group_id(members)
        result.update({member.case_id: group_id for member in members})
    return MappingProxyType(dict(sorted(result.items())))


def _maximum_distance(
    left: LeakageDescriptor,
    right: LeakageDescriptor,
) -> float:
    return max(
        abs(left.normalized_coordinates[name] - right.normalized_coordinates[name])
        for name in left.normalized_coordinates
    )


def _group_id(members: Sequence[LeakageDescriptor]) -> str:
    payload = [
        {
            "case_id": member.case_id,
            "geometry_lineage_id": member.geometry_lineage_id,
            "morph_seed": member.morph_seed,
            "normalized_coordinates": member.normalized_coordinates,
        }
        for member in sorted(members, key=lambda item: item.case_id)
    ]
    digest = hashlib.blake2b(canonical_json_bytes(payload), digest_size=32).hexdigest()
    return f"leakage-{digest}"


__all__ = ("LeakageDescriptor", "LeakagePolicy", "bind_leakage_groups")
