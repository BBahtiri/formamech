"""Typed executable S9 envelopes spanning geometry, material, and loading."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import ClassVar, cast

from piginot.contracts import (
    CanonicalQuantity,
    canonical_json_bytes,
    hash_spec,
)
from piginot.morphing.membership import (
    DerivedBound,
    FamilyMembershipPredicate,
    PrimitiveBound,
)

_CANONICAL_NAMESPACE = "piginot.doe"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_SCHEMA_VERSION = "1.0"
_REFUSAL_POLICY_ID = "reject-any-bound-or-derived-violation-v1"


class EnvelopeReasonCode(StrEnum):
    """Stable non-geometry rejection reasons emitted by executable envelopes."""

    INVALID_PARAMETER_VALUE = "INVALID_PARAMETER_VALUE"
    MISSING_PARAMETER = "MISSING_PARAMETER"
    UNKNOWN_PARAMETER = "UNKNOWN_PARAMETER"
    PARAMETER_UNIT_MISMATCH = "PARAMETER_UNIT_MISMATCH"
    PARAMETER_OUT_OF_BOUNDS = "PARAMETER_OUT_OF_BOUNDS"


@dataclass(frozen=True, slots=True)
class EnvelopeDecision:
    """One executable-envelope membership decision."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    accepted: bool
    reason_codes: tuple[object, ...] = ()
    derived_values: Mapping[str, CanonicalQuantity] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise ValueError("accepted must be boolean")
        reasons = tuple(self.reason_codes)
        if self.accepted and reasons:
            raise ValueError("accepted decisions must not carry rejection reasons")
        if not self.accepted and not reasons:
            raise ValueError("rejected decisions require at least one stable reason")
        if not self.accepted and any(
            not isinstance(getattr(reason, "value", reason), str)
            or not getattr(reason, "value", reason)
            for reason in reasons
        ):
            raise ValueError("rejection reasons must expose stable text values")
        derived = dict(sorted(self.derived_values.items()))
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, CanonicalQuantity)
            for name, value in derived.items()
        ):
            raise ValueError("derived_values must map names to CanonicalQuantity")
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "derived_values", MappingProxyType(derived))


@dataclass(frozen=True, slots=True)
class ExecutableEnvelope:
    """Executable, canonical S9 envelope used by DOE and preregistration."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    envelope_id: str
    compiled_family_hash: str
    geometry_membership: FamilyMembershipPredicate
    geometry_membership_hash: str
    material_bounds: tuple[PrimitiveBound, ...]
    load_bounds: tuple[PrimitiveBound, ...]
    refusal_policy_id: str = _REFUSAL_POLICY_ID
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.envelope_id, str) or not self.envelope_id:
            raise ValueError("envelope_id must be non-empty text")
        object.__setattr__(
            self,
            "compiled_family_hash",
            _hash(self.compiled_family_hash, "compiled_family_hash"),
        )
        if type(self.geometry_membership) is not FamilyMembershipPredicate:
            raise ValueError("geometry_membership must be a FamilyMembershipPredicate")
        object.__setattr__(
            self,
            "geometry_membership_hash",
            _hash(self.geometry_membership_hash, "geometry_membership_hash"),
        )
        expected_membership_hash = hash_spec(self.geometry_membership, "content")
        if self.geometry_membership_hash != expected_membership_hash:
            raise ValueError("geometry_membership_hash does not match membership")
        material = _checked_bounds(self.material_bounds, "material_bounds")
        load = _checked_bounds(self.load_bounds, "load_bounds")
        geometry_names = {
            bound.name for bound in self.geometry_membership.primitive_bounds
        } | {bound.name for bound in self.geometry_membership.derived_bounds}
        additional_names = tuple(bound.name for bound in (*material, *load))
        if len(additional_names) != len(set(additional_names)):
            raise ValueError("envelope bound names must be unique")
        if geometry_names & set(additional_names):
            raise ValueError("envelope bound names must be unique")
        if self.refusal_policy_id != _REFUSAL_POLICY_ID:
            raise ValueError(
                f"unsupported refusal_policy_id {self.refusal_policy_id!r}"
            )
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported executable envelope {self.schema_version!r}")
        object.__setattr__(self, "material_bounds", material)
        object.__setattr__(self, "load_bounds", load)

    @property
    def primitive_bounds(self) -> tuple[PrimitiveBound, ...]:
        """Return geometry and non-geometry primitive bounds in DOE order."""

        return (
            *self.geometry_membership.primitive_bounds,
            *self.material_bounds,
            *self.load_bounds,
        )

    @property
    def derived_bounds(self) -> tuple[DerivedBound, ...]:
        """Return derived governing bounds inherited from geometry membership."""

        return self.geometry_membership.derived_bounds

    def evaluate(self, theta: Mapping[str, object]) -> EnvelopeDecision:
        """Evaluate one point across the complete executable envelope."""

        if not isinstance(theta, Mapping):
            return _reject(EnvelopeReasonCode.INVALID_PARAMETER_VALUE)
        supplied = tuple(theta)
        if any(not isinstance(name, str) for name in supplied):
            return _reject(EnvelopeReasonCode.INVALID_PARAMETER_VALUE)
        expected = frozenset(bound.name for bound in self.primitive_bounds)
        supplied_set = frozenset(supplied)
        if expected - supplied_set:
            return _reject(EnvelopeReasonCode.MISSING_PARAMETER)
        if supplied_set - expected:
            return _reject(EnvelopeReasonCode.UNKNOWN_PARAMETER)

        for bound in (*self.material_bounds, *self.load_bounds):
            value = theta[bound.name]
            if not isinstance(value, CanonicalQuantity) or not math.isfinite(
                value.value
            ):
                return _reject(EnvelopeReasonCode.INVALID_PARAMETER_VALUE)
            if value.unit is not bound.unit or value.dimension is not bound.dimension:
                return _reject(EnvelopeReasonCode.PARAMETER_UNIT_MISMATCH)
            if not (bound.lower_bound.value <= value.value <= bound.upper_bound.value):
                return _reject(EnvelopeReasonCode.PARAMETER_OUT_OF_BOUNDS)

        geometry_theta = {
            bound.name: theta[bound.name]
            for bound in self.geometry_membership.primitive_bounds
        }
        geometry_decision = self.geometry_membership.evaluate(geometry_theta)
        if not geometry_decision.accepted:
            return EnvelopeDecision(
                accepted=False,
                reason_codes=tuple(geometry_decision.reason_codes),
                derived_values=geometry_decision.derived_values,
            )
        return EnvelopeDecision(
            accepted=True,
            derived_values=geometry_decision.derived_values,
        )

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self)


def _hash(value: object, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 64-character hex hash")
    return value


def _checked_bounds(
    values: object,
    label: str,
) -> tuple[PrimitiveBound, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{label} must contain at least one PrimitiveBound")
    if any(type(bound) is not PrimitiveBound for bound in values):
        raise ValueError(f"{label} must contain PrimitiveBound values")
    return cast(tuple[PrimitiveBound, ...], values)


def _reject(reason: EnvelopeReasonCode) -> EnvelopeDecision:
    return EnvelopeDecision(accepted=False, reason_codes=(reason,))


__all__ = (
    "EnvelopeDecision",
    "EnvelopeReasonCode",
    "ExecutableEnvelope",
)
