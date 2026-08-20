"""Strict JSON parsing and RFC 8785 canonical serialization."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, TypeAlias, cast

import rfc8785

from piginot.contracts.errors import (
    ContractValidationError,
    DuplicateKeyError,
    NonFiniteNumberError,
)

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON object name: {key!r}")
        result[key] = value
    return result


def _reject_non_finite(token: str) -> None:
    raise NonFiniteNumberError(f"JSON number must be finite, got {token}")


def _parse_finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        _reject_non_finite(token)
    return value


def strict_json_loads(document: str | bytes) -> JsonValue:
    """Parse JSON without Python's duplicate-key or non-finite extensions."""

    try:
        value = json.loads(
            document,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_non_finite,
            parse_float=_parse_finite_float,
        )
    except UnicodeDecodeError as exc:
        raise ContractValidationError("JSON input must be valid UTF-8") from exc
    return cast(JsonValue, value)


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NonFiniteNumberError("canonical JSON numbers must be finite")
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractValidationError(
                    "canonical JSON object names must be strings"
                )
            result[key] = _json_value(item)
        return result
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
            if field.repr or field.compare
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    raise ContractValidationError(
        f"unsupported canonical JSON value: {type(value).__name__}"
    )


def canonical_json_bytes(payload: object) -> bytes:
    """Return RFC 8785 canonical UTF-8 bytes for a contract payload."""

    normalized = _json_value(payload)
    try:
        return rfc8785.dumps(normalized)
    except rfc8785.FloatDomainError as exc:
        raise NonFiniteNumberError("canonical JSON numbers must be finite") from exc
    except rfc8785.CanonicalizationError as exc:
        raise ContractValidationError(str(exc)) from exc
