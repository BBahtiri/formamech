"""Canonical 3D volumetric mesh contract with volumetric-locking refusal (Phase 6).

Supports ``TET10`` (quadratic 10-node tetrahedra), ``HEX8`` (8-node hexahedra),
and ``TET4`` (linear 4-node tetrahedra), while explicitly refusing ``TET4`` for
nearly incompressible hyperelasticity (``poisson_ratio >= 0.45``) where linear
tetrahedra suffer severe volumetric locking.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from piginot.contracts import Unit, canonical_json_bytes

MESH3D_SCHEMA: Final = "formamech.canonical-mesh-3d-v1"
TET4_LOCKING_POISSON_LIMIT: Final = 0.45


class CellType3D(StrEnum):
    TET4 = "tet4"
    TET10 = "tet10"
    HEX8 = "hex8"


class VolumetricLockingError(ValueError):
    """Raised when linear TET4 elements are requested for nearly incompressible hyperelasticity."""


_NODES_PER_CELL = {
    CellType3D.TET4: 4,
    CellType3D.TET10: 10,
    CellType3D.HEX8: 8,
}


def _tet_signed_volume(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    p3: tuple[float, float, float],
) -> float:
    ax, ay, az = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    bx, by, bz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
    cx, cy, cz = p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]
    cross_x = ay * bz - az * by
    cross_y = az * bx - ax * bz
    cross_z = ax * by - ay * bx
    return (cross_x * cx + cross_y * cy + cross_z * cz) / 6.0


@dataclass(frozen=True, slots=True)
class CanonicalMesh3D:
    """Content-addressed 3D volumetric mesh with oriented boundary surface triangles."""

    nodes: tuple[tuple[float, float, float], ...]
    cells: tuple[tuple[int, ...], ...]
    cell_type: CellType3D
    surface_triangles: tuple[tuple[int, int, int], ...]
    semantic_face_tags: Mapping[str, tuple[int, ...]]
    unit: Unit = Unit.MM
    metadata: Mapping[str, Any] = ()  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("3D mesh must have at least one node")
        if not self.cells:
            raise ValueError("3D mesh must have at least one cell")
        ctype = CellType3D(self.cell_type)
        object.__setattr__(self, "cell_type", ctype)
        expected_n = _NODES_PER_CELL[ctype]
        n_nodes = len(self.nodes)
        for pt in self.nodes:
            if len(pt) != 3 or not all(math.isfinite(c) for c in pt):
                raise ValueError("each 3D node must have 3 finite coordinates")
        for cell in self.cells:
            if len(cell) != expected_n:
                raise ValueError(
                    f"{ctype.value} cell must have {expected_n} nodes, got {len(cell)}"
                )
            if any(idx < 0 or idx >= n_nodes for idx in cell):
                raise ValueError("cell node index out of bounds")
            if ctype in (CellType3D.TET4, CellType3D.TET10):
                vol = _tet_signed_volume(
                    self.nodes[cell[0]],
                    self.nodes[cell[1]],
                    self.nodes[cell[2]],
                    self.nodes[cell[3]],
                )
                if vol <= 0.0:
                    raise ValueError(
                        f"{ctype.value} element has non-positive signed volume {vol:.3e}"
                    )
        n_tris = len(self.surface_triangles)
        for tri in self.surface_triangles:
            if len(tri) != 3 or any(idx < 0 or idx >= n_nodes for idx in tri):
                raise ValueError("surface triangle indices out of bounds")
        cleaned_tags: dict[str, tuple[int, ...]] = {}
        for tag, indices in sorted(self.semantic_face_tags.items()):
            idx_tuple = tuple(int(i) for i in indices)
            if any(i < 0 or i >= n_tris for i in idx_tuple):
                raise ValueError(
                    f"surface tag {tag!r} contains out-of-range triangle index"
                )
            cleaned_tags[str(tag)] = idx_tuple
        object.__setattr__(self, "semantic_face_tags", cleaned_tags)
        meta_dict = dict(self.metadata) if isinstance(self.metadata, Mapping) else {}
        object.__setattr__(self, "metadata", meta_dict)

    def validate_incompressibility_locking(self, poisson_ratio: float) -> None:
        """Refuse linear ``TET4`` meshes for nearly incompressible hyperelasticity."""
        if not math.isfinite(poisson_ratio) or not (-1.0 < poisson_ratio < 0.5):
            raise ValueError("poisson_ratio must lie strictly in (-1, 0.5)")
        if (
            self.cell_type is CellType3D.TET4
            and poisson_ratio >= TET4_LOCKING_POISSON_LIMIT
        ):
            raise VolumetricLockingError(
                f"TET4 linear tetrahedra are refused for nearly incompressible "
                f"hyperelasticity (poisson_ratio={poisson_ratio:.3f} >= "
                f"{TET4_LOCKING_POISSON_LIMIT}); use TET10 or HEX8."
            )

    def surface_geometry(
        self,
    ) -> tuple[
        tuple[tuple[float, float, float], ...],
        tuple[tuple[float, float, float], ...],
        tuple[float, ...],
    ]:
        """Return ``(centroids, unit_normals, areas)`` for every surface triangle."""
        centroids: list[tuple[float, float, float]] = []
        normals: list[tuple[float, float, float]] = []
        areas: list[float] = []
        for a, b, c in self.surface_triangles:
            pa, pb, pc = self.nodes[a], self.nodes[b], self.nodes[c]
            centroids.append(
                (
                    (pa[0] + pb[0] + pc[0]) / 3.0,
                    (pa[1] + pb[1] + pc[1]) / 3.0,
                    (pa[2] + pb[2] + pc[2]) / 3.0,
                )
            )
            e1 = (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2])
            e2 = (pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2])
            cx = e1[1] * e2[2] - e1[2] * e2[1]
            cy = e1[2] * e2[0] - e1[0] * e2[2]
            cz = e1[0] * e2[1] - e1[1] * e2[0]
            norm = math.sqrt(cx * cx + cy * cy + cz * cz)
            area = 0.5 * norm
            areas.append(area)
            if norm > 0.0:
                normals.append((cx / norm, cy / norm, cz / norm))
            else:
                normals.append((0.0, 0.0, 1.0))
        return tuple(centroids), tuple(normals), tuple(areas)

    @property
    def mesh_id(self) -> str:
        payload = {
            "schema": MESH3D_SCHEMA,
            "cell_type": self.cell_type.value,
            "unit": self.unit.value,
            "nodes": [list(p) for p in self.nodes],
            "cells": [list(c) for c in self.cells],
            "surface_triangles": [list(t) for t in self.surface_triangles],
            "semantic_face_tags": {
                k: list(v) for k, v in self.semantic_face_tags.items()
            },
        }
        return hashlib.blake2b(
            canonical_json_bytes(payload), digest_size=32
        ).hexdigest()


def _brick_grid(
    lx: float, ly: float, lz: float, nx: int, ny: int, nz: int
) -> tuple[
    list[tuple[float, float, float]],
    list[tuple[int, int, int, int, int, int, int, int]],
    list[tuple[int, int, int]],
    dict[str, list[int]],
]:
    nodes: list[tuple[float, float, float]] = []
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                nodes.append((lx * i / nx, ly * j / ny, lz * k / nz))

    def _idx(i: int, j: int, k: int) -> int:
        return (k * (ny + 1) + j) * (nx + 1) + i

    hexes: list[tuple[int, int, int, int, int, int, int, int]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                hexes.append(
                    (
                        _idx(i, j, k),
                        _idx(i + 1, j, k),
                        _idx(i + 1, j + 1, k),
                        _idx(i, j + 1, k),
                        _idx(i, j, k + 1),
                        _idx(i + 1, j, k + 1),
                        _idx(i + 1, j + 1, k + 1),
                        _idx(i, j + 1, k + 1),
                    )
                )

    tris: list[tuple[int, int, int]] = []
    tags: dict[str, list[int]] = {
        "x_min": [],
        "x_max": [],
        "y_min": [],
        "y_max": [],
        "z_min": [],
        "z_max": [],
    }

    def _add_quad(tag: str, a: int, b: int, c: int, d: int) -> None:
        tags[tag].append(len(tris))
        tris.append((a, b, c))
        tags[tag].append(len(tris))
        tris.append((a, c, d))

    # Outwardly oriented quads on all 6 faces
    for j in range(ny):
        for k in range(nz):
            _add_quad(
                "x_min",
                _idx(0, j, k),
                _idx(0, j, k + 1),
                _idx(0, j + 1, k + 1),
                _idx(0, j + 1, k),
            )
            _add_quad(
                "x_max",
                _idx(nx, j, k),
                _idx(nx, j + 1, k),
                _idx(nx, j + 1, k + 1),
                _idx(nx, j, k + 1),
            )
    for i in range(nx):
        for k in range(nz):
            _add_quad(
                "y_min",
                _idx(i, 0, k),
                _idx(i + 1, 0, k),
                _idx(i + 1, 0, k + 1),
                _idx(i, 0, k + 1),
            )
            _add_quad(
                "y_max",
                _idx(i, ny, k),
                _idx(i, ny, k + 1),
                _idx(i + 1, ny, k + 1),
                _idx(i + 1, ny, k),
            )
    for i in range(nx):
        for j in range(ny):
            _add_quad(
                "z_min",
                _idx(i, j, 0),
                _idx(i, j + 1, 0),
                _idx(i + 1, j + 1, 0),
                _idx(i + 1, j, 0),
            )
            _add_quad(
                "z_max",
                _idx(i, j, nz),
                _idx(i + 1, j, nz),
                _idx(i + 1, j + 1, nz),
                _idx(i, j + 1, nz),
            )
    return nodes, hexes, tris, tags


def _split_hex_to_6_tets(
    h: tuple[int, int, int, int, int, int, int, int],
    nodes: list[tuple[float, float, float]],
) -> list[tuple[int, int, int, int]]:
    """Freudenthal subdivision of one hexahedron into 6 positively oriented tetrahedra."""
    n0, n1, n2, n3, n4, n5, n6, n7 = h
    raw = (
        (n0, n1, n2, n6),
        (n0, n2, n3, n6),
        (n0, n3, n7, n6),
        (n0, n7, n4, n6),
        (n0, n4, n5, n6),
        (n0, n5, n1, n6),
    )
    oriented: list[tuple[int, int, int, int]] = []
    for a, b, c, d in raw:
        if _tet_signed_volume(nodes[a], nodes[b], nodes[c], nodes[d]) < 0.0:
            oriented.append((a, c, b, d))
        else:
            oriented.append((a, b, c, d))
    return oriented


def make_brick_hex8_mesh(
    lx: float, ly: float, lz: float, nx: int = 2, ny: int = 2, nz: int = 2
) -> CanonicalMesh3D:
    """Build a canonical 3D ``HEX8`` brick mesh."""
    nodes, hexes, tris, tags = _brick_grid(lx, ly, lz, nx, ny, nz)
    return CanonicalMesh3D(
        nodes=tuple(nodes),
        cells=tuple(hexes),
        cell_type=CellType3D.HEX8,
        surface_triangles=tuple(tris),
        semantic_face_tags={k: tuple(v) for k, v in tags.items()},
        metadata={"kind": "brick-hex8", "dims": [lx, ly, lz]},
    )


def make_brick_tet4_mesh(
    lx: float, ly: float, lz: float, nx: int = 2, ny: int = 2, nz: int = 2
) -> CanonicalMesh3D:
    """Build a canonical 3D ``TET4`` brick mesh (compressible materials only)."""
    nodes, hexes, tris, tags = _brick_grid(lx, ly, lz, nx, ny, nz)
    tets: list[tuple[int, int, int, int]] = []
    for h in hexes:
        tets.extend(_split_hex_to_6_tets(h, nodes))
    return CanonicalMesh3D(
        nodes=tuple(nodes),
        cells=tuple(tets),
        cell_type=CellType3D.TET4,
        surface_triangles=tuple(tris),
        semantic_face_tags={k: tuple(v) for k, v in tags.items()},
        metadata={"kind": "brick-tet4", "dims": [lx, ly, lz]},
    )


def make_brick_tet10_mesh(
    lx: float, ly: float, lz: float, nx: int = 2, ny: int = 2, nz: int = 2
) -> CanonicalMesh3D:
    """Build a canonical 3D ``TET10`` quadratic tetrahedral brick mesh."""
    nodes, hexes, tris, tags = _brick_grid(lx, ly, lz, nx, ny, nz)
    tets4: list[tuple[int, int, int, int]] = []
    for h in hexes:
        tets4.extend(_split_hex_to_6_tets(h, nodes))
    edge_mid: dict[tuple[int, int], int] = {}
    tet10s: list[tuple[int, ...]] = []

    def _mid(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        existing = edge_mid.get(key)
        if existing is not None:
            return existing
        pa, pb = nodes[a], nodes[b]
        idx = len(nodes)
        nodes.append(
            (0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1]), 0.5 * (pa[2] + pb[2]))
        )
        edge_mid[key] = idx
        return idx

    for v0, v1, v2, v3 in tets4:
        tet10s.append(
            (
                v0,
                v1,
                v2,
                v3,
                _mid(v0, v1),
                _mid(v1, v2),
                _mid(v2, v0),
                _mid(v0, v3),
                _mid(v1, v3),
                _mid(v2, v3),
            )
        )
    return CanonicalMesh3D(
        nodes=tuple(nodes),
        cells=tuple(tet10s),
        cell_type=CellType3D.TET10,
        surface_triangles=tuple(tris),
        semantic_face_tags={k: tuple(v) for k, v in tags.items()},
        metadata={"kind": "brick-tet10", "dims": [lx, ly, lz]},
    )


__all__ = (
    "MESH3D_SCHEMA",
    "TET4_LOCKING_POISSON_LIMIT",
    "CanonicalMesh3D",
    "CellType3D",
    "VolumetricLockingError",
    "make_brick_hex8_mesh",
    "make_brick_tet4_mesh",
    "make_brick_tet10_mesh",
)
