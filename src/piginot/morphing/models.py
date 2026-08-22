"""Immutable contracts for S6 morphing, validity, membership, and ledgers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TypeVar

from piginot.contracts import CanonicalMesh, CanonicalQuantity, canonical_json_bytes

Point = tuple[float, float]
Matrix2 = tuple[tuple[float, float], tuple[float, float]]


class MorphReasonCode(StrEnum):
    """Stable reasons emitted by the S6 fail-closed boundary."""

    DERIVED_EVALUATION_ERROR = "DERIVED_EVALUATION_ERROR"
    DERIVED_QUANTITY_OUT_OF_BOUNDS = "DERIVED_QUANTITY_OUT_OF_BOUNDS"
    FEATURE_SIZE_BELOW_MARGIN = "FEATURE_SIZE_BELOW_MARGIN"
    INVALID_MAP_CONFIGURATION = "INVALID_MAP_CONFIGURATION"
    INVALID_MORPHED_MESH = "INVALID_MORPHED_MESH"
    INVALID_PARAMETER_VALUE = "INVALID_PARAMETER_VALUE"
    INVALID_SERIALIZED_PREDICATE = "INVALID_SERIALIZED_PREDICATE"
    MESH_QUALITY_OUT_OF_BOUNDS = "MESH_QUALITY_OUT_OF_BOUNDS"
    MISSING_PARAMETER = "MISSING_PARAMETER"
    NONINJECTIVE_MORPH = "NONINJECTIVE_MORPH"
    PARAMETER_OUT_OF_BOUNDS = "PARAMETER_OUT_OF_BOUNDS"
    PARAMETER_UNIT_MISMATCH = "PARAMETER_UNIT_MISMATCH"
    POINT_OUTSIDE_FFD_BOX = "POINT_OUTSIDE_FFD_BOX"
    SINGULAR_RBF_SYSTEM = "SINGULAR_RBF_SYSTEM"
    TAG_IDENTITY_CHANGED = "TAG_IDENTITY_CHANGED"
    TOPOLOGY_CHANGED = "TOPOLOGY_CHANGED"
    UNKNOWN_PARAMETER = "UNKNOWN_PARAMETER"
    VALIDATION_CONTEXT_MISMATCH = "VALIDATION_CONTEXT_MISMATCH"


class MorphStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class MorphingError(ValueError):
    """Expected typed failure while configuring or evaluating a morph."""

    def __init__(self, reason_code: MorphReasonCode, message: str) -> None:
        super().__init__(f"{reason_code.value}: {message}")
        self.reason_code = reason_code


def _parameter_name(value: str | None, label: str) -> str | None:
    if value is not None and (not isinstance(value, str) or not value.isidentifier()):
        raise MorphingError(
            MorphReasonCode.INVALID_MAP_CONFIGURATION,
            f"{label} must be a valid identifier or None",
        )
    return value


def _point(value: object, label: str) -> Point:
    if not isinstance(value, tuple) or len(value) != 2:
        raise MorphingError(
            MorphReasonCode.INVALID_MAP_CONFIGURATION,
            f"{label} must be a two-coordinate tuple",
        )
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise MorphingError(
            MorphReasonCode.INVALID_MAP_CONFIGURATION,
            f"{label} coordinates must be real numbers",
        )
    coordinates = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in coordinates):
        raise MorphingError(
            MorphReasonCode.INVALID_MAP_CONFIGURATION,
            f"{label} coordinates must be finite",
        )
    return coordinates[0], coordinates[1]


@dataclass(frozen=True, slots=True)
class RBFControlPoint:
    """One RBF interpolation point and its named displacement parameters."""

    point: Point
    x_parameter: str | None
    y_parameter: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", _point(self.point, "RBF control point"))
        object.__setattr__(
            self,
            "x_parameter",
            _parameter_name(self.x_parameter, "x_parameter"),
        )
        object.__setattr__(
            self,
            "y_parameter",
            _parameter_name(self.y_parameter, "y_parameter"),
        )


@dataclass(frozen=True, slots=True)
class FFDControlPoint:
    """Named x/y displacement parameters for one FFD lattice point."""

    x_parameter: str | None
    y_parameter: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "x_parameter",
            _parameter_name(self.x_parameter, "x_parameter"),
        )
        object.__setattr__(
            self,
            "y_parameter",
            _parameter_name(self.y_parameter, "y_parameter"),
        )


@dataclass(frozen=True, slots=True)
class MorphValidityCriteria:
    """Fixed numerical policy used to judge one morphed reference mesh."""

    jacobian_margin: float = 0.0
    minimum_feature_size_mm: float = 0.0
    minimum_angle_degrees: float = 0.0
    maximum_aspect_ratio: float = 1.0e308

    def __post_init__(self) -> None:
        values = (
            self.jacobian_margin,
            self.minimum_feature_size_mm,
            self.minimum_angle_degrees,
            self.maximum_aspect_ratio,
        )
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in values
        ):
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "validity criteria must be finite real numbers",
            )
        if self.jacobian_margin < 0.0 or self.minimum_feature_size_mm < 0.0:
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "Jacobian and feature-size margins must be non-negative",
            )
        if not 0.0 <= self.minimum_angle_degrees <= 60.0:
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "minimum triangle angle must lie in [0, 60] degrees",
            )
        if self.maximum_aspect_ratio < 1.0:
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "maximum aspect ratio must be at least one",
            )


_Value = TypeVar("_Value")


def _frozen_mapping(values: Mapping[str, _Value]) -> Mapping[str, _Value]:
    return MappingProxyType(dict(sorted(values.items())))


@dataclass(frozen=True, slots=True)
class MembershipDecision:
    accepted: bool
    reason_codes: tuple[MorphReasonCode, ...] = ()
    derived_values: Mapping[str, CanonicalQuantity] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reasons = tuple(MorphReasonCode(reason) for reason in self.reason_codes)
        if self.accepted == bool(reasons):
            raise ValueError("accepted decisions have no reasons; rejected ones do")
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(
            self,
            "derived_values",
            _frozen_mapping(self.derived_values),
        )

    @property
    def reason_code(self) -> MorphReasonCode | None:
        return self.reason_codes[0] if self.reason_codes else None


@dataclass(frozen=True, slots=True)
class MorphValidityReport:
    reference_mesh_id: str
    validity_policy_hash: str
    quadrature_rule: str
    quadrature_points_checked: int
    minimum_jacobian_determinant: float | None
    topology_signature: str | None
    semantic_tag_counts: Mapping[str, int]
    tags_preserved: bool
    minimum_feature_size_mm: float | None
    minimum_angle_degrees: float | None
    maximum_aspect_ratio: float | None
    reason_codes: tuple[MorphReasonCode, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "semantic_tag_counts",
            _frozen_mapping(self.semantic_tag_counts),
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(MorphReasonCode(reason) for reason in self.reason_codes),
        )

    @property
    def accepted(self) -> bool:
        return not self.reason_codes


@dataclass(frozen=True, slots=True)
class MorphResult:
    mesh: CanonicalMesh | None
    report: MorphValidityReport
    membership: MembershipDecision
    reason_codes: tuple[MorphReasonCode, ...]

    @property
    def accepted(self) -> bool:
        return self.mesh is not None and not self.reason_codes


@dataclass(frozen=True, slots=True)
class MorphAttempt:
    attempt_id: str
    theta: Mapping[str, CanonicalQuantity]

    def __post_init__(self) -> None:
        if not isinstance(self.attempt_id, str) or not self.attempt_id:
            raise ValueError("attempt_id must be non-empty text")
        object.__setattr__(self, "theta", _frozen_mapping(self.theta))


@dataclass(frozen=True, slots=True)
class MorphRecord:
    attempt_id: str
    status: MorphStatus
    reason_codes: tuple[MorphReasonCode, ...]
    mesh: CanonicalMesh | None
    report: MorphValidityReport

    def __post_init__(self) -> None:
        reasons = tuple(MorphReasonCode(reason) for reason in self.reason_codes)
        accepted = self.status is MorphStatus.ACCEPTED
        if accepted and (reasons or self.mesh is None):
            raise ValueError("accepted records require a mesh and no rejection reason")
        if not accepted and not reasons:
            raise ValueError("rejected records require at least one reason")
        object.__setattr__(self, "reason_codes", reasons)

    @property
    def accepted(self) -> bool:
        return self.status is MorphStatus.ACCEPTED

    @property
    def reason_code(self) -> MorphReasonCode | None:
        return self.reason_codes[0] if self.reason_codes else None

    @property
    def mesh_id(self) -> str | None:
        return None if self.mesh is None else self.mesh.mesh_id


@dataclass(frozen=True, slots=True)
class MorphLedger:
    records: tuple[MorphRecord, ...]

    def __post_init__(self) -> None:
        records = tuple(self.records)
        attempt_ids = tuple(record.attempt_id for record in records)
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("ledger attempt IDs must be unique")
        object.__setattr__(self, "records", records)

    @property
    def attempted_count(self) -> int:
        return len(self.records)

    @property
    def accepted(self) -> tuple[MorphRecord, ...]:
        return tuple(record for record in self.records if record.accepted)

    @property
    def rejected(self) -> tuple[MorphRecord, ...]:
        return tuple(record for record in self.records if not record.accepted)

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self)
