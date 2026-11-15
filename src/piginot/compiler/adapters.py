"""Pure metadata boundary for explicitly installed compiler adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Protocol, cast

from piginot.compiler.config_types import AdapterRef

ADAPTER_GROUP = "piginot.compiler.adapters"
ENERGY_ARCHITECTURE = "pi-ginot-energy-parametric-film-v1"
_ID = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class AdapterError(ValueError):
    """A registered adapter or its declared capability cannot be used."""

    def __init__(self, rule: str, message: str) -> None:
        self.rule = rule
        super().__init__(f"[{rule}] {message}")


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise AdapterError("identity", f"{label} must be a canonical identifier")
    return value


def _digest(value: object, label: str) -> None:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise AdapterError("identity", f"{label} must be a lowercase SHA256 digest")


def _capabilities(values: object, label: str) -> tuple[str, ...]:
    if type(values) is not tuple or not values:
        raise AdapterError("capability", f"{label} must be a nonempty tuple")
    checked = tuple(_identifier(value, label) for value in values)
    if len(set(checked)) != len(checked):
        raise AdapterError("capability", f"{label} must be unique")
    return checked


@dataclass(frozen=True, slots=True)
class GeometryIdentity:
    reference_mesh_id: str
    morph_map_sha256: str

    def __post_init__(self) -> None:
        _digest(self.reference_mesh_id, "reference_mesh_id")
        _digest(self.morph_map_sha256, "morph_map_sha256")


@dataclass(frozen=True, slots=True)
class AdapterDescriptor:
    adapter_id: str
    version: str
    implementation_sha256: str
    geometries: tuple[GeometryIdentity, ...]
    materials: tuple[str, ...]
    liftings: tuple[str, ...]
    architectures: tuple[str, ...]
    reference_initializations: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.adapter_id, "adapter_id")
        if not isinstance(self.version, str) or not self.version.strip():
            raise AdapterError("identity", "adapter version must be nonempty")
        _digest(self.implementation_sha256, "implementation_sha256")
        if (
            type(self.geometries) is not tuple
            or not self.geometries
            or any(type(item) is not GeometryIdentity for item in self.geometries)
            or len(set(self.geometries)) != len(self.geometries)
        ):
            raise AdapterError("identity", "geometries must be unique exact identities")
        for name in (
            "materials",
            "liftings",
            "architectures",
            "reference_initializations",
        ):
            _capabilities(getattr(self, name), name)
        if not set(self.liftings) <= {"quarter", "adf"}:
            raise AdapterError("capability", "unsupported lifting capability")
        if set(self.architectures) != {ENERGY_ARCHITECTURE}:
            raise AdapterError("capability", "unsupported release architecture")
        if not set(self.reference_initializations) <= {
            "cold",
            "neo-hookean-full-load-hgo",
        }:
            raise AdapterError("capability", "unsupported reference initialization")

    def document(self) -> dict[str, Any]:
        return {
            "id": self.adapter_id,
            "version": self.version,
            "implementation_sha256": self.implementation_sha256,
            "geometries": [
                {
                    "reference_mesh_id": item.reference_mesh_id,
                    "morph_map_sha256": item.morph_map_sha256,
                }
                for item in self.geometries
            ],
            **{
                name: list(getattr(self, name))
                for name in (
                    "materials",
                    "liftings",
                    "architectures",
                    "reference_initializations",
                )
            },
        }


class InstalledAdapter(Protocol):
    @property
    def descriptor(self) -> AdapterDescriptor: ...


def resolve_adapter(
    adapter_id: str,
    *,
    geometry_resource: Path | None = None,
    geometry_sha256: str | None = None,
) -> InstalledAdapter:
    """Load the one matching registered adapter, without import-string fallback."""

    _identifier(adapter_id, "adapter_id")
    candidates = tuple(
        entry for entry in entry_points(group=ADAPTER_GROUP) if entry.name == adapter_id
    )
    if len(candidates) != 1:
        raise AdapterError(
            "availability", f"expected one registered adapter for {adapter_id!r}"
        )
    try:
        adapter = candidates[0].load()
    except ImportError as error:
        raise AdapterError("availability", f"adapter unavailable: {error}") from error
    binder = getattr(adapter, "bind_resource", None)
    if geometry_resource is None:
        if geometry_sha256 is not None or callable(binder):
            raise AdapterError(
                "resource", "this adapter requires a bound geometry resource"
            )
    else:
        if geometry_sha256 is None or not callable(binder):
            raise AdapterError(
                "resource", "adapter cannot bind the requested geometry resource"
            )
        _digest(geometry_sha256, "geometry resource SHA-256")
        try:
            adapter = binder(geometry_resource, geometry_sha256)
        except ValueError as error:
            raise AdapterError("resource", str(error)) from error
    descriptor = getattr(adapter, "descriptor", None)
    if type(descriptor) is not AdapterDescriptor:
        raise AdapterError(
            "capability", "adapter descriptor must be an exact AdapterDescriptor"
        )
    if descriptor.adapter_id != adapter_id:
        raise AdapterError(
            "identity", "registered and declared adapter identities differ"
        )
    return cast(InstalledAdapter, adapter)


def resolve_configured_adapter(
    reference: AdapterRef, config_path: Path
) -> InstalledAdapter:
    """Bind an explicit resource relative to the configuration, never the CWD."""
    resource = reference.geometry_resource
    if resource is None:
        return resolve_adapter(reference.id)
    path = Path(resource.path)
    if not path.is_absolute():
        path = config_path.resolve().parent / path
    return resolve_adapter(
        reference.id, geometry_resource=path, geometry_sha256=resource.sha256
    )
