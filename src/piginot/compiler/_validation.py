"""Pure scalar and object validation for campaign configuration."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any, cast

from piginot.contracts.canonical import JsonValue

MAX_SAFE_INTEGER = 2**53 - 1
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class CompilerConfigError(ValueError):
    """An invalid campaign input, with a stable refusal category."""

    def __init__(self, rule: str, message: str) -> None:
        super().__init__(f"[{rule}] {message}")
        self.rule = rule


def object_fields(
    value: object, label: str, required: Sequence[str], optional: Sequence[str] = ()
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise CompilerConfigError("type", f"{label} must be a JSON object")
    unknown = set(value) - set(required) - set(optional)
    if unknown:
        raise CompilerConfigError(
            "unknown-key", f"{label}: unknown keys {sorted(unknown)}"
        )
    missing = set(required) - set(value)
    if missing:
        raise CompilerConfigError(
            "missing-key", f"{label}: missing keys {sorted(missing)}"
        )
    return dict(value)


def text(value: object, label: str, *, identifier: bool = False) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or (identifier and _IDENTIFIER.fullmatch(value) is None)
    ):
        raise CompilerConfigError("value", f"{label} must be canonical nonempty text")
    return value


def digest(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise CompilerConfigError(
            "value", f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def integer(value: object, label: str, *, minimum: int = 1) -> int:
    if type(value) is not int or not minimum <= value <= MAX_SAFE_INTEGER:
        raise CompilerConfigError(
            "value", f"{label} must be a safe integer >= {minimum}"
        )
    return value


def positive(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise CompilerConfigError("type", f"{label} must be a finite positive number")
    try:
        result = float(cast(int | float, value))
    except OverflowError as error:
        raise CompilerConfigError("value", f"{label} must be finite") from error
    if not math.isfinite(result) or result <= 0:
        raise CompilerConfigError("value", f"{label} must be finite and positive")
    return result


def names(value: object, label: str, *, nonempty: bool = True) -> tuple[str, ...]:
    if type(value) is not tuple or (nonempty and not value):
        raise CompilerConfigError("type", f"{label} must be a tuple of names")
    items = tuple(text(item, label) for item in value)
    if len(items) != len(set(items)):
        raise CompilerConfigError("value", f"{label} must not contain duplicates")
    return items


def array(value: object, label: str) -> tuple[Any, ...]:
    if type(value) is not list:
        raise CompilerConfigError("type", f"{label} must be a JSON array")
    return tuple(value)


def primitive(value: object) -> JsonValue:
    """Copy immutable contract values into plain JSON, without contract tags."""

    if value is None or type(value) in (str, int, float, bool):
        return cast(JsonValue, value)
    if isinstance(value, Mapping):
        return {str(key): primitive(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: primitive(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, tuple):
        return [primitive(item) for item in value]
    raise TypeError(f"unsupported campaign value: {type(value).__name__}")
