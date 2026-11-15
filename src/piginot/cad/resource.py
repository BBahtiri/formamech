"""Pure, hash-bound metadata for one prepared fixed CAD geometry."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from piginot.contracts import (
    CanonicalMesh,
    Unit,
    canonical_json_bytes,
    hash_spec,
    strict_json_loads,
)

SCHEMA = "piginot.cad-geometry-v1"
LEVELS = (1, 2, 4)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MESH_FIELDS = (
    "$type",
    "mesh_id",
    "nodes",
    "cells",
    "cell_type",
    "boundary_facets",
    "semantic_tags",
    "region_ids",
    "unit",
    "orientation",
    "topology_signature",
    "quality_metrics",
    "source_provenance",
)


class CadResourceError(ValueError):
    """A prepared geometry cannot be used with its declared binding."""


def _fields(value: object, expected: tuple[str, ...], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise CadResourceError(f"{label} has invalid fields")
    return value


def _digest(value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CadResourceError("expected a lowercase SHA-256 digest")
    return value


def _payload(path: Path, sha256: object) -> bytes:
    expected = _digest(sha256)
    if path.is_symlink() or not path.is_file():
        raise CadResourceError(f"geometry input is not a regular file: {path}")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise CadResourceError(f"geometry input bytes differ: {path}")
    return payload


def _member(root: Path, name: object) -> Path:
    if not isinstance(name, str):
        raise CadResourceError("geometry member path must be text")
    relative = PurePosixPath(name)
    if (
        not name
        or relative.is_absolute()
        or relative.as_posix() != name
        or "\\" in name
        or ":" in name
        or any(part in (".", "..") for part in relative.parts)
    ):
        raise CadResourceError("geometry member path must be canonical and relative")
    target = root.joinpath(*relative.parts)
    if any(
        (root.joinpath(*relative.parts[:i])).is_symlink()
        for i in range(1, len(relative.parts) + 1)
    ):
        raise CadResourceError("geometry member path cannot contain symlinks")
    if not target.resolve().is_relative_to(root):
        raise CadResourceError("geometry member escapes its resource directory")
    return target


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _decode_mesh(payload: bytes) -> CanonicalMesh:
    document = _fields(strict_json_loads(payload), _MESH_FIELDS, "canonical mesh")
    if document["$type"] != "piginot.contracts.CanonicalMesh":
        raise CadResourceError("geometry member is not a canonical mesh")
    mesh = CanonicalMesh(
        **{
            **{
                name: value
                for name, value in document.items()
                if name not in ("$type", "unit")
            },
            "unit": Unit(document["unit"]),
        }
    )
    if mesh.cell_type != "triangle" or any(len(cell) != 3 for cell in mesh.cells):
        raise CadResourceError("prepared CAD geometry requires linear triangles")
    identity = hash_spec(
        {
            "schema": "canonical-triangle-mesh-v1",
            "nodes": mesh.nodes,
            "cells": mesh.cells,
            "cell_type": mesh.cell_type,
            "boundary_facets": mesh.boundary_facets,
            "semantic_tags": mesh.semantic_tags,
            "region_ids": mesh.region_ids,
            "unit": mesh.unit,
            "orientation": mesh.orientation,
        },
        "content",
    )
    if canonical_json_bytes(mesh) != payload or identity != mesh.mesh_id:
        raise CadResourceError("mesh bytes or structural identity are not canonical")
    return mesh


@dataclass(frozen=True, slots=True)
class GeometryMesh:
    path: Path
    sha256: str
    mesh_id: str


@dataclass(frozen=True, slots=True)
class GeometryResource:
    path: Path
    sha256: str
    source_path: Path
    source_sha256: str
    meshes: Mapping[int, GeometryMesh]
    boundary_kinds: Mapping[str, str]
    provenance: Mapping[str, Any]
    _mesh_values: Mapping[int, CanonicalMesh] = field(repr=False)

    @property
    def geometry_inputs(self) -> tuple[tuple[str, Path], ...]:
        return (
            ("geometry-resource", self.path),
            ("geometry-source", self.source_path),
            *((f"geometry-mesh-{level}", self.meshes[level].path) for level in LEVELS),
        )

    def mesh(self, level: int = 1) -> CanonicalMesh:
        if type(level) is not int or level not in LEVELS:
            raise CadResourceError("geometry level must be one of 1, 2, 4")
        return load_geometry_resource(self.path, self.sha256)._mesh_values[level]


def load_geometry_resource(path: Path, sha256: str) -> GeometryResource:
    """Validate every declared member before exposing any geometry capability.

    Mesh files contain exact canonical JSON with no trailing newline. The
    manifest itself may use any strict JSON formatting; its raw bytes are bound.
    """
    try:
        payload = _payload(path, sha256)
        path = path.resolve()
        document = _fields(
            strict_json_loads(payload),
            ("schema", "source", "meshes", "boundary_kinds", "provenance"),
            "CAD geometry resource",
        )
        if document["schema"] != SCHEMA:
            raise CadResourceError("unsupported CAD geometry resource schema")
        source = _fields(document["source"], ("path", "sha256"), "CAD source")
        source_path = _member(path.parent, source["path"])
        _payload(source_path, source["sha256"])
        entries = _fields(document["meshes"], ("1", "2", "4"), "geometry mesh levels")
        mesh_fields = {
            level: _fields(
                entries[str(level)], ("path", "sha256", "mesh_id"), "mesh resource"
            )
            for level in LEVELS
        }
        meshes = {
            level: GeometryMesh(
                _member(path.parent, entry["path"]),
                _digest(entry["sha256"]),
                _digest(entry["mesh_id"]),
            )
            for level, entry in mesh_fields.items()
        }
        paths = (path, source_path, *(entry.path for entry in meshes.values()))
        if len(set(paths)) != len(paths):
            raise CadResourceError("geometry resource member paths must be distinct")
        values = {
            level: _decode_mesh(_payload(entry.path, entry.sha256))
            for level, entry in meshes.items()
        }
        kinds = document["boundary_kinds"]
        if (
            not isinstance(kinds, dict)
            or not kinds
            or any(kind not in ("line", "arc", "other") for kind in kinds.values())
        ):
            raise CadResourceError("boundary kinds must identify line, arc or other")
        if any(
            set(mesh.semantic_tags) != set(kinds)
            or mesh.mesh_id != meshes[level].mesh_id
            for level, mesh in values.items()
        ):
            raise CadResourceError("mesh identity or boundary vocabulary differs")
        provenance = document["provenance"]
        if not isinstance(provenance, dict):
            raise CadResourceError("CAD provenance must be a data object")
        return GeometryResource(
            path,
            sha256,
            source_path,
            source["sha256"],
            MappingProxyType(meshes),
            MappingProxyType(dict(kinds)),
            _freeze(provenance),
            MappingProxyType(values),
        )
    except CadResourceError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise CadResourceError(f"invalid prepared CAD resource: {error}") from error
