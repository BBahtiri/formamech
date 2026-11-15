"""Safe, canonical publication boundary for non-holdout S9 DOE cases."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import ClassVar, cast

from piginot.contracts import canonical_json_bytes, hash_spec, strict_json_loads
from piginot.doe.roles import EvidenceRole

_SCHEMA_VERSION = "1.0"
_CANONICAL_NAMESPACE = "piginot.doe"
_CASE_TYPE = f"{_CANONICAL_NAMESPACE}.PublishedCase"
_MANIFEST_TYPE = f"{_CANONICAL_NAMESPACE}.PublishedCaseManifest"
_CASE_FIELDS = frozenset(
    {
        "$type",
        "case_id",
        "leakage_group_id",
        "role",
        "primitive_coordinates",
        "derived_coordinates",
        "strata",
        "status",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "$type",
        "manifest_id",
        "primitive_axes",
        "derived_axes",
        "stratum_axes",
        "cases",
        "schema_version",
    }
)
_PUBLIC_ROLES = frozenset(
    {
        EvidenceRole.PILOT,
        EvidenceRole.TRAIN,
        EvidenceRole.DEVELOPMENT,
        EvidenceRole.LEGACY,
    }
)


class PublishedCaseStatus(StrEnum):
    """Whether a public DOE case is selected now or reserved for replacement."""

    ACTIVE = "ACTIVE"
    RESERVE = "RESERVE"


def _require_exact_keys(
    value: object,
    expected: frozenset[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    keys = set(value)
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing:
        raise ValueError(f"{label} is missing field(s): {', '.join(missing)}")
    if unknown:
        raise ValueError(
            f"{label} has unknown field(s): {', '.join(cast(list[str], unknown))}"
        )
    return cast(Mapping[str, object], value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _case_id(value: object) -> str:
    case_id = _text(value, "case_id")
    if ":" in case_id:
        raise ValueError("case_id must identify a case-level sample")
    return case_id


def _axis_name(value: object, label: str) -> str:
    name = _text(value, label)
    if not name.isidentifier():
        raise ValueError(f"{label} must contain valid identifiers")
    return name


def _axis_tuple(
    values: object, label: str, *, nonempty: bool = True
) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be an array")
    axes = tuple(_axis_name(value, label) for value in values)
    if nonempty and not axes:
        raise ValueError(f"{label} must not be empty")
    if len(axes) != len(set(axes)):
        raise ValueError(f"{label} must contain unique identifiers")
    return tuple(sorted(axes))


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _coordinates(
    value: object, label: str, *, nonempty: bool = True
) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if nonempty and not value:
        raise ValueError(f"{label} must not be empty")
    result: dict[str, float] = {}
    for name, coordinate in value.items():
        axis = _axis_name(name, f"{label} names")
        result[axis] = _finite_number(coordinate, f"{label}.{axis}")
    return MappingProxyType(dict(sorted(result.items())))


def _strata(
    value: object, coordinate_axes: frozenset[str], *, nonempty: bool = True
) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError("strata must be an object")
    if nonempty and not value:
        raise ValueError("strata must not be empty")
    result: dict[str, int] = {}
    for name, stratum in value.items():
        axis = _axis_name(name, "strata names")
        if isinstance(stratum, bool) or not isinstance(stratum, int) or stratum < 0:
            raise ValueError("strata must map names to non-negative integers")
        result[axis] = stratum
    if not set(result) <= coordinate_axes:
        raise ValueError(
            "strata must refer only to coordinate axes "
            "(primitive_coordinates and derived_coordinates)"
        )
    return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True, slots=True)
class PublishedCase:
    """One public non-holdout case, including its immutable evidence role."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    case_id: str
    leakage_group_id: str
    role: EvidenceRole
    primitive_coordinates: Mapping[str, float]
    derived_coordinates: Mapping[str, float]
    strata: Mapping[str, int]
    status: PublishedCaseStatus

    def __post_init__(self) -> None:
        case_id = _case_id(self.case_id)
        group_id = _text(self.leakage_group_id, "leakage_group_id")
        role = EvidenceRole(self.role)
        if role not in _PUBLIC_ROLES:
            raise ValueError("HOLDOUT cases must never enter a public case manifest")
        status = PublishedCaseStatus(self.status)
        primitive = _coordinates(
            self.primitive_coordinates,
            "primitive_coordinates",
        )
        derived = _coordinates(
            self.derived_coordinates, "derived_coordinates", nonempty=False
        )
        primitive_axes = frozenset(primitive)
        derived_axes = frozenset(derived)
        if primitive_axes & derived_axes:
            raise ValueError("primitive and derived coordinate names must be disjoint")
        strata = _strata(self.strata, primitive_axes | derived_axes, nonempty=False)

        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "leakage_group_id", group_id)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "primitive_coordinates", primitive)
        object.__setattr__(self, "derived_coordinates", derived)
        object.__setattr__(self, "strata", strata)
        object.__setattr__(self, "status", status)


@dataclass(frozen=True, slots=True)
class PublishedCaseManifest:
    """Canonical public manifest that structurally excludes holdout coordinates."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    manifest_id: str
    primitive_axes: tuple[str, ...]
    derived_axes: tuple[str, ...]
    stratum_axes: tuple[str, ...]
    cases: tuple[PublishedCase, ...]
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        manifest_id = _text(self.manifest_id, "manifest_id")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(
                f"unsupported public manifest schema version {self.schema_version!r}"
            )
        primitive_axes = _axis_tuple(self.primitive_axes, "primitive_axes")
        derived_axes = _axis_tuple(self.derived_axes, "derived_axes", nonempty=False)
        stratum_axes = _axis_tuple(self.stratum_axes, "stratum_axes", nonempty=False)
        coordinate_axes = frozenset((*primitive_axes, *derived_axes))
        if set(primitive_axes) & set(derived_axes):
            raise ValueError("primitive_axes and derived_axes must be disjoint")
        if not set(stratum_axes) <= coordinate_axes:
            raise ValueError(
                "stratum_axes must refer only to coordinate axes "
                "declared in primitive_axes and derived_axes"
            )

        cases = tuple(self.cases)
        if not cases:
            raise ValueError("cases must not be empty")
        if any(not isinstance(case, PublishedCase) for case in cases):
            raise ValueError("cases must contain PublishedCase values")
        case_ids = tuple(case.case_id for case in cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case IDs must be unique")

        group_roles: dict[str, EvidenceRole] = {}
        for case in cases:
            if tuple(case.primitive_coordinates) != tuple(sorted(primitive_axes)):
                raise ValueError("case primitive coordinate schema is incoherent")
            if tuple(case.derived_coordinates) != tuple(sorted(derived_axes)):
                raise ValueError(
                    "case derived coordinate schema is incoherent with derived_axes"
                )
            if tuple(case.strata) != tuple(sorted(stratum_axes)):
                raise ValueError("case strata schema is incoherent with stratum_axes")
            prior_role = group_roles.setdefault(case.leakage_group_id, case.role)
            if prior_role is not case.role:
                raise ValueError("each leakage group must have exactly one role")

        object.__setattr__(self, "manifest_id", manifest_id)
        object.__setattr__(self, "primitive_axes", primitive_axes)
        object.__setattr__(self, "derived_axes", derived_axes)
        object.__setattr__(self, "stratum_axes", stratum_axes)
        object.__setattr__(
            self,
            "cases",
            tuple(sorted(cases, key=lambda case: case.case_id)),
        )

    def to_bytes(self) -> bytes:
        """Return the RFC 8785 canonical public representation."""

        return canonical_json_bytes(self)

    @property
    def content_hash(self) -> str:
        """Return the BLAKE2b-256 identity in the standard content domain."""

        return hash_spec(self, "content")

    @classmethod
    def from_bytes(cls, document: str | bytes) -> PublishedCaseManifest:
        """Strictly parse a canonical-schema public manifest."""

        if isinstance(document, str):
            encoded = document.encode("utf-8")
        elif isinstance(document, bytes):
            encoded = document
        else:
            raise ValueError("public manifest input must be str or bytes")
        root = _require_exact_keys(
            strict_json_loads(encoded),
            _MANIFEST_FIELDS,
            "manifest",
        )
        if root["$type"] != _MANIFEST_TYPE:
            raise ValueError("manifest has an invalid type discriminator")
        if root["schema_version"] != _SCHEMA_VERSION:
            raise ValueError(
                f"unsupported public manifest schema {root['schema_version']!r}"
            )
        cases_payload = root["cases"]
        if not isinstance(cases_payload, list):
            raise ValueError("cases must be an array")
        cases = tuple(_case_from_payload(item) for item in cases_payload)
        manifest = cls(
            manifest_id=_text(root["manifest_id"], "manifest_id"),
            primitive_axes=_json_axes(root["primitive_axes"], "primitive_axes"),
            derived_axes=_json_axes(root["derived_axes"], "derived_axes"),
            stratum_axes=_json_axes(root["stratum_axes"], "stratum_axes"),
            cases=cases,
            schema_version=_SCHEMA_VERSION,
        )
        if manifest.to_bytes() != encoded:
            raise ValueError("public manifest input must use canonical JSON")
        return manifest


def _json_axes(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return tuple(_axis_name(item, label) for item in value)


def _json_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


def _case_from_payload(value: object) -> PublishedCase:
    payload = _require_exact_keys(value, _CASE_FIELDS, "published case")
    if payload["$type"] != _CASE_TYPE:
        raise ValueError("published case has an invalid type discriminator")
    return PublishedCase(
        case_id=_text(payload["case_id"], "case_id"),
        leakage_group_id=_text(
            payload["leakage_group_id"],
            "leakage_group_id",
        ),
        role=EvidenceRole(_text(payload["role"], "role")),
        primitive_coordinates=cast(
            Mapping[str, float],
            _json_mapping(
                payload["primitive_coordinates"],
                "primitive_coordinates",
            ),
        ),
        derived_coordinates=cast(
            Mapping[str, float],
            _json_mapping(
                payload["derived_coordinates"],
                "derived_coordinates",
            ),
        ),
        strata=cast(
            Mapping[str, int],
            _json_mapping(payload["strata"], "strata"),
        ),
        status=PublishedCaseStatus(_text(payload["status"], "status")),
    )


__all__ = (
    "PublishedCase",
    "PublishedCaseManifest",
    "PublishedCaseStatus",
)
