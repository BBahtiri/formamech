"""ParaView VTU / PVD time-series and glTF 2.0 (GLB) field exporters (V3-3).

Exports 2D ``CanonicalMesh`` and 3D ``CanonicalMesh3D`` predictions, FEM
reference solutions, DWR nodal error indicators, and conformal upper bounds
into industry-standard visualization archives with RFC 8785 / SHA-256 receipts.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from piginot.contracts import CanonicalMesh, canonical_json_bytes
from piginot.geometry.mesh3d import CanonicalMesh3D, CellType3D

FloatArray = NDArray[np.float64]
VTU_EXPORT_SCHEMA: Final = "formamech.vtu-export-v1"
GLB_EXPORT_SCHEMA: Final = "formamech.glb-export-v1"


@dataclass(frozen=True, slots=True)
class VtuExportResult:
    """Auditable metadata record for one exported ParaView ``.vtu`` file."""

    output_path: str
    file_sha256: str
    spatial_dim: int
    node_count: int
    cell_count: int
    cell_type: str
    point_field_names: tuple[str, ...]
    cell_field_names: tuple[str, ...]

    def document(self) -> dict[str, Any]:
        return {
            "schema": VTU_EXPORT_SCHEMA,
            "output_path": self.output_path,
            "file_sha256": self.file_sha256,
            "spatial_dim": self.spatial_dim,
            "node_count": self.node_count,
            "cell_count": self.cell_count,
            "cell_type": self.cell_type,
            "point_field_names": list(self.point_field_names),
            "cell_field_names": list(self.cell_field_names),
        }


def _mesh_to_vtk_arrays(
    mesh: CanonicalMesh | CanonicalMesh3D,
) -> tuple[FloatArray, str, NDArray[np.int64], int]:
    if isinstance(mesh, CanonicalMesh3D):
        pts = np.asarray(mesh.nodes, dtype=np.float64)
        cells = np.asarray(mesh.cells, dtype=np.int64)
        if mesh.cell_type is CellType3D.TET10:
            # CanonicalMesh3D TET10 already matches VTK_QUADRATIC_TETRA node ordering
            return pts, "tetra10", cells, 3
        if mesh.cell_type is CellType3D.HEX8:
            return pts, "hexahedron", cells, 3
        return pts, "tetra", cells, 3
    pts_2d = np.asarray(mesh.nodes, dtype=np.float64)
    pts_3d = np.column_stack((pts_2d, np.zeros(len(pts_2d), dtype=np.float64)))
    cells_2d = np.asarray(mesh.cells, dtype=np.int64)
    return pts_3d, "triangle", cells_2d, 2


def export_fields_to_vtu(
    mesh: CanonicalMesh | CanonicalMesh3D,
    output_path: Path | str,
    *,
    point_fields: Mapping[str, FloatArray] | None = None,
    cell_fields: Mapping[str, FloatArray] | None = None,
    write_manifest: bool = False,
) -> VtuExportResult:
    """Write a ParaView ``.vtu`` unstructured grid with nodal/cell physics & trust fields."""
    import meshio  # type: ignore[import-untyped]

    dest = Path(output_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    pts, vtk_cell_type, cells, dim = _mesh_to_vtk_arrays(mesh)
    n_nodes = int(pts.shape[0])
    n_cells = int(cells.shape[0])

    p_data: dict[str, FloatArray] = {}
    if point_fields is not None:
        for name, arr in sorted(point_fields.items()):
            a = np.asarray(arr, dtype=np.float64)
            if a.shape[0] != n_nodes:
                raise ValueError(
                    f"point field {name!r} has {a.shape[0]} rows, expected {n_nodes}"
                )
            # Pad 2D vector fields to 3D vectors so ParaView WarpByVector works natively
            if a.ndim == 2 and a.shape[1] == 2:
                a = np.column_stack((a, np.zeros(n_nodes, dtype=np.float64)))
            p_data[str(name)] = a

    c_data: dict[str, list[FloatArray]] = {}
    if cell_fields is not None:
        for name, arr in sorted(cell_fields.items()):
            a = np.asarray(arr, dtype=np.float64)
            if a.shape[0] != n_cells:
                raise ValueError(
                    f"cell field {name!r} has {a.shape[0]} rows, expected {n_cells}"
                )
            c_data[str(name)] = [a]

    m_out = meshio.Mesh(
        points=pts,
        cells=[(vtk_cell_type, cells)],
        point_data=p_data,
        cell_data=c_data,
    )
    meshio.write(str(dest), m_out, file_format="vtu")
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    result = VtuExportResult(
        output_path=str(dest),
        file_sha256=sha,
        spatial_dim=dim,
        node_count=n_nodes,
        cell_count=n_cells,
        cell_type=vtk_cell_type,
        point_field_names=tuple(sorted(p_data.keys())),
        cell_field_names=tuple(sorted(c_data.keys())),
    )
    if write_manifest:
        dest.with_suffix(".vtu.manifest.json").write_bytes(
            canonical_json_bytes(result.document()) + b"\n"
        )
    return result


def export_trajectory_to_pvd(
    mesh: CanonicalMesh | CanonicalMesh3D,
    output_dir: Path | str,
    *,
    times: Sequence[float],
    point_fields_per_step: Sequence[Mapping[str, FloatArray]],
    stem: str = "trajectory",
) -> dict[str, Any]:
    """Export a stateful viscoelastic/plasticity rollout as a ParaView ``.pvd`` series."""
    if len(times) != len(point_fields_per_step) or not times:
        raise ValueError(
            "times and point_fields_per_step must have equal non-empty length"
        )
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    step_records: list[dict[str, Any]] = []
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for idx, (t_val, p_fields) in enumerate(
        zip(times, point_fields_per_step, strict=True)
    ):
        fname = f"{stem}_{idx:04d}.vtu"
        rec = export_fields_to_vtu(mesh, out / fname, point_fields=p_fields)
        step_records.append(rec.document())
        lines.append(
            f'    <DataSet timestep="{float(t_val):.9g}" part="0" file="{fname}"/>'
        )
    lines.extend(["  </Collection>", "</VTKFile>", ""])
    pvd_path = out / f"{stem}.pvd"
    pvd_bytes = "\n".join(lines).encode("utf-8")
    pvd_path.write_bytes(pvd_bytes)
    return {
        "schema": "formamech.pvd-trajectory-v1",
        "pvd_path": str(pvd_path),
        "pvd_sha256": hashlib.sha256(pvd_bytes).hexdigest(),
        "step_count": len(times),
        "steps": step_records,
    }


def _scalar_to_viridis_rgb(values: FloatArray) -> NDArray[np.float32]:
    """Lightweight deterministic blue-teal-yellow colormap (no Matplotlib dependency)."""
    v = np.asarray(values, dtype=np.float64)
    lo = float(np.min(v))
    hi = float(np.max(v))
    t = (
        np.zeros_like(v)
        if hi - lo <= 1.0e-15
        else np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    )
    r = np.clip(0.26 + 0.74 * t**1.3, 0.0, 1.0)
    g = np.clip(0.05 + 0.85 * np.sin(0.5 * np.pi * t), 0.0, 1.0)
    b = np.clip(0.55 * (1.0 - t) + 0.15, 0.0, 1.0)
    return np.stack((r, g, b), axis=-1).astype(np.float32)


def export_deformed_glb(
    mesh: CanonicalMesh3D,
    output_path: Path | str,
    *,
    displacement: FloatArray | None = None,
    scalar_field: FloatArray | None = None,
    displacement_scale: float = 1.0,
) -> dict[str, Any]:
    """Export a 3D deformed surface mesh with RGB vertex colors as binary glTF 2.0 (``.glb``)."""
    dest = Path(output_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(mesh.nodes, dtype=np.float64)
    if displacement is not None:
        pts = pts + float(displacement_scale) * np.asarray(
            displacement, dtype=np.float64
        )
    positions = pts.astype(np.float32)
    indices = np.asarray(mesh.surface_triangles, dtype=np.uint32).reshape(-1)
    if scalar_field is not None:
        colors = _scalar_to_viridis_rgb(scalar_field)
    else:
        colors = np.full((len(positions), 3), 0.65, dtype=np.float32)

    pos_bytes = positions.tobytes()
    col_bytes = colors.tobytes()
    idx_bytes = indices.tobytes()

    def _pad4(data: bytes) -> bytes:
        rem = len(data) % 4
        return data if rem == 0 else data + b"\x00" * (4 - rem)

    pos_off = 0
    col_off = len(_pad4(pos_bytes))
    idx_off = col_off + len(_pad4(col_bytes))
    bin_blob = _pad4(pos_bytes) + _pad4(col_bytes) + _pad4(idx_bytes)

    pos_min = positions.min(axis=0).tolist()
    pos_max = positions.max(axis=0).tolist()
    gltf_json = {
        "asset": {"version": "2.0", "generator": "FormaMech"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "COLOR_0": 1},
                        "indices": 2,
                        "mode": 4,
                    }
                ]
            }
        ],
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": pos_off,
                "byteLength": len(pos_bytes),
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": col_off,
                "byteLength": len(col_bytes),
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": idx_off,
                "byteLength": len(idx_bytes),
                "target": 34963,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(positions),
                "type": "VEC3",
                "min": pos_min,
                "max": pos_max,
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": len(colors),
                "type": "VEC3",
            },
            {
                "bufferView": 2,
                "componentType": 5125,
                "count": len(indices),
                "type": "SCALAR",
            },
        ],
    }
    json_raw = json.dumps(gltf_json, separators=(",", ":")).encode("utf-8")
    json_pad = len(json_raw) % 4
    if json_pad:
        json_raw += b" " * (4 - json_pad)

    total_len = 12 + 8 + len(json_raw) + 8 + len(bin_blob)
    header = struct.pack("<III", 0x46546C67, 2, total_len)
    chunk0 = struct.pack("<II", len(json_raw), 0x4E4F534A) + json_raw
    chunk1 = struct.pack("<II", len(bin_blob), 0x004E4942) + bin_blob
    glb_bytes = header + chunk0 + chunk1
    dest.write_bytes(glb_bytes)

    return {
        "schema": GLB_EXPORT_SCHEMA,
        "output_path": str(dest),
        "file_sha256": hashlib.sha256(glb_bytes).hexdigest(),
        "vertex_count": int(len(positions)),
        "triangle_count": int(len(mesh.surface_triangles)),
        "byte_length": len(glb_bytes),
    }


__all__ = (
    "GLB_EXPORT_SCHEMA",
    "VTU_EXPORT_SCHEMA",
    "VtuExportResult",
    "export_deformed_glb",
    "export_fields_to_vtu",
    "export_trajectory_to_pvd",
)
