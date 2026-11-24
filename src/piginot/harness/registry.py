"""Progressive-discovery capability registry & JSON Schema validator (V4-0)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Final

from piginot.harness.contracts import CapabilityDomain, CapabilitySpec, CostClass

if TYPE_CHECKING:
    from piginot.harness.session import HarnessContext

HARNESS_PLUGIN_GROUP: Final = "formamech.harness.capabilities"
CapabilityHandler = Callable[["HarnessContext", Mapping[str, Any]], Mapping[str, Any]]


def validate_json_schema_args(
    capability_name: str,
    schema: Mapping[str, Any],
    args: Mapping[str, Any],
) -> None:
    """Validate capability arguments against declarative JSON Schema."""
    props: Mapping[str, Any] = schema.get("properties", {})
    required: Sequence[str] = schema.get("required", ())
    for key in required:
        if key not in args or args[key] is None:
            raise ValueError(
                f"capability {capability_name!r} missing required argument {key!r} "
                f"(required: {list(required)})"
            )
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(args) - set(props))
        if unknown:
            raise ValueError(
                f"capability {capability_name!r} received unexpected argument(s) "
                f"{unknown}; allowed: {sorted(props)}"
            )
    for key, value in args.items():
        if value is None or key not in props:
            continue
        ps = props[key]
        etype = ps.get("type")
        if etype == "string":
            if not isinstance(value, str):
                raise TypeError(f"argument {key!r} must be a string")
        elif etype == "object":
            if not isinstance(value, Mapping):
                raise TypeError(f"argument {key!r} must be a JSON object")
        elif etype == "array":
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"argument {key!r} must be a JSON array")
        elif etype == "boolean":
            if not isinstance(value, bool):
                raise TypeError(f"argument {key!r} must be a boolean")
        elif etype == "integer":
            if type(value) is not int:
                raise TypeError(f"argument {key!r} must be an integer")
            if "minimum" in ps and value < ps["minimum"]:
                raise ValueError(f"argument {key!r} must be >= {ps['minimum']}")
            if "maximum" in ps and value > ps["maximum"]:
                raise ValueError(f"argument {key!r} must be <= {ps['maximum']}")
        elif etype == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"argument {key!r} must be a number")
            if not math.isfinite(float(value)):
                raise ValueError(f"argument {key!r} must be finite")
            if "minimum" in ps and float(value) < float(ps["minimum"]):
                raise ValueError(f"argument {key!r} must be >= {ps['minimum']}")
            if "maximum" in ps and float(value) > float(ps["maximum"]):
                raise ValueError(f"argument {key!r} must be <= {ps['maximum']}")
        if "enum" in ps and value not in ps["enum"]:
            raise ValueError(
                f"argument {key!r} must be one of {list(ps['enum'])}, got {value!r}"
            )


class CapabilityRegistry:
    """Central registry powering progressive AI-agent discovery and dispatch."""

    def __init__(self) -> None:
        self._specs: dict[str, CapabilitySpec] = {}
        self._handlers: dict[str, CapabilityHandler] = {}

    def register(
        self,
        spec: CapabilitySpec,
        handler: CapabilityHandler,
        *,
        overwrite: bool = False,
    ) -> None:
        if not spec.name or not isinstance(spec.name, str):
            raise ValueError("capability name must be non-empty")
        if spec.name in self._specs and not overwrite:
            raise ValueError(f"capability {spec.name!r} is already registered")
        if not callable(handler):
            raise TypeError("capability handler must be callable")
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def has(self, name: str) -> bool:
        return name in self._specs

    def get(self, name: str) -> tuple[CapabilitySpec, CapabilityHandler]:
        if name not in self._specs:
            raise KeyError(
                f"unknown capability {name!r}; call discover.capabilities to list valid names"
            )
        return self._specs[name], self._handlers[name]

    def describe(self, name: str) -> dict[str, Any]:
        spec, _ = self.get(name)
        return spec.full_descriptor()

    def list_capabilities(
        self,
        *,
        domain: CapabilityDomain | str | None = None,
        cost_class: CostClass | str | None = None,
        search: str | None = None,
    ) -> list[dict[str, Any]]:
        dom_filter = None if domain is None else CapabilityDomain(domain)
        cost_filter = None if cost_class is None else CostClass(cost_class)
        q = None if not search else search.strip().lower()
        out: list[dict[str, Any]] = []
        for name in sorted(self._specs):
            spec = self._specs[name]
            if dom_filter is not None and spec.domain is not dom_filter:
                continue
            if cost_filter is not None and spec.cost_class is not cost_filter:
                continue
            if q is not None:
                haystack = " ".join(
                    (spec.name, spec.summary, spec.description, " ".join(spec.tags))
                ).lower()
                if q not in haystack:
                    continue
            out.append(spec.compact_descriptor())
        return out

    def domains_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for spec in self._specs.values():
            counts[spec.domain.value] = counts.get(spec.domain.value, 0) + 1
        return dict(sorted(counts.items()))

    def load_entry_point_plugins(self) -> tuple[str, ...]:
        loaded: list[str] = []
        for ep in entry_points(group=HARNESS_PLUGIN_GROUP):
            registrar = ep.load()
            if callable(registrar):
                registrar(self)
                loaded.append(ep.name)
        return tuple(sorted(loaded))

    def __len__(self) -> int:
        return len(self._specs)


__all__ = (
    "HARNESS_PLUGIN_GROUP",
    "CapabilityHandler",
    "CapabilityRegistry",
    "validate_json_schema_args",
)
