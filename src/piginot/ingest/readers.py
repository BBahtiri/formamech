"""Solver-agnostic CAE archive readers (VTU/XDMF, CalculiX, Abaqus, Nastran) (V3-3)."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from piginot.ingest.contracts import CaeSourceFormat

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
READER_ENTRY_POINT_GROUP: Final = "formamech.ingest.readers"


@dataclass(frozen=True, slots=True)
class ParsedCaeSample:
    """In-memory canonical representation of one external CAE simulation run."""

    sample_id: str
    source_format: CaeSourceFormat
    source_sha256: str
    nodes: FloatArray  # (N, 2) or (N, 3)
    cells: IntArray  # (E, K)
    cell_type: str  # "triangle" | "tet4" | "tet10" | "hex8"
    point_fields: Mapping[str, FloatArray]
    cell_fields: Mapping[str, FloatArray]
    parameters: Mapping[str, float]
    metadata: Mapping[str, Any]

    @property
    def spatial_dim(self) -> int:
        return int(self.nodes.shape[1])


def _hash_files(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode("utf-8"))
        h.update(b"\x00")
        h.update(p.read_bytes())
    return h.hexdigest()


_MESHIO_TO_CANON_CELL = {
    "triangle": "triangle",
    "tetra": "tet4",
    "tetra10": "tet10",
    "hexahedron": "hex8",
}


def read_vtu_or_xdmf(
    path: Path | str,
    *,
    sample_id: str | None = None,
    parameters: Mapping[str, float] | None = None,
) -> ParsedCaeSample:
    """Read a ParaView ``.vtu`` or ``.xdmf`` CAE result via ``meshio``."""
    import meshio  # type: ignore[import-untyped]

    src = Path(path).resolve()
    raw = meshio.read(str(src))
    pts = np.asarray(raw.points, dtype=np.float64)
    chosen_type: str | None = None
    chosen_cells: IntArray | None = None
    for block in raw.cells:
        mapped = _MESHIO_TO_CANON_CELL.get(block.type)
        if mapped is not None:
            chosen_type = mapped
            chosen_cells = np.asarray(block.data, dtype=np.int64)
            break
    if chosen_type is None or chosen_cells is None:
        raise ValueError(
            f"unsupported meshio cell types in {src.name}: {[b.type for b in raw.cells]}"
        )
    if chosen_type == "triangle" and pts.shape[1] == 3 and np.allclose(pts[:, 2], 0.0):
        pts = pts[:, :2].copy()

    p_fields: dict[str, FloatArray] = {}
    for k, v in raw.point_data.items():
        p_fields[str(k)] = np.asarray(v, dtype=np.float64)
    c_fields: dict[str, FloatArray] = {}
    for k, blocks in raw.cell_data.items():
        if blocks:
            c_fields[str(k)] = np.asarray(blocks[0], dtype=np.float64)

    fmt = CaeSourceFormat.XDMF if src.suffix.lower() == ".xdmf" else CaeSourceFormat.VTU
    return ParsedCaeSample(
        sample_id=src.stem if sample_id is None else sample_id,
        source_format=fmt,
        source_sha256=_hash_files(src),
        nodes=pts,
        cells=chosen_cells,
        cell_type=chosen_type,
        point_fields=p_fields,
        cell_fields=c_fields,
        parameters={}
        if parameters is None
        else {str(k): float(v) for k, v in parameters.items()},
        metadata={"source_path": str(src)},
    )


def _parse_abaqus_like_inp(
    inp_path: Path,
) -> tuple[FloatArray, IntArray, str, dict[int, int]]:
    """Parse Abaqus / CalculiX ``*NODE`` and ``*ELEMENT`` sections."""
    node_ids: list[int] = []
    coords: list[list[float]] = []
    raw_elems: list[list[int]] = []
    elem_type_raw = ""
    section = ""
    for raw_line in inp_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            upper = line.upper()
            if upper.startswith("*NODE"):
                section = "NODE"
            elif upper.startswith("*ELEMENT"):
                section = "ELEMENT"
                for tok in upper.split(","):
                    if "TYPE=" in tok:
                        elem_type_raw = tok.split("=", 1)[1].strip()
            else:
                section = ""
            continue
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if section == "NODE" and len(parts) >= 3:
            node_ids.append(int(parts[0]))
            coords.append([float(v) for v in parts[1:4]])
        elif section == "ELEMENT" and len(parts) >= 4:
            raw_elems.append([int(v) for v in parts[1:]])

    if not coords or not raw_elems:
        raise ValueError(f"incomplete *NODE / *ELEMENT blocks in {inp_path.name}")
    id_map = {nid: idx for idx, nid in enumerate(node_ids)}
    nodes = np.asarray(coords, dtype=np.float64)
    cells = np.asarray([[id_map[n] for n in el] for el in raw_elems], dtype=np.int64)
    k = cells.shape[1]
    if "C3D10" in elem_type_raw or k == 10:
        ctype = "tet10"
    elif "C3D4" in elem_type_raw or k == 4:
        ctype = "tet4"
    elif "C3D8" in elem_type_raw or k == 8:
        ctype = "hex8"
    else:
        ctype = "triangle"
    if ctype == "triangle" and nodes.shape[1] == 3 and np.allclose(nodes[:, 2], 0.0):
        nodes = nodes[:, :2].copy()
    return nodes, cells, ctype, id_map


def _parse_nodal_fields_file(
    fields_path: Path, id_map: Mapping[int, int], n_nodes: int
) -> dict[str, FloatArray]:
    """Read exported Abaqus/Nastran nodal displacement/stress tables from JSON or CSV."""
    suffix = fields_path.suffix.lower()
    if suffix == ".json":
        doc = json.loads(fields_path.read_text(encoding="utf-8"))
        return {
            str(k): np.asarray(v, dtype=np.float64)
            for k, v in doc.items()
            if isinstance(v, list)
        }
    # CSV format: node_id,ux,uy[,uz][,von_mises]
    u_rows = np.zeros((n_nodes, 3), dtype=np.float64)
    has_uz = False
    vm_rows: FloatArray | None = None
    with fields_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        cols = set(reader.fieldnames or ())
        has_uz = "uz" in cols
        if "von_mises" in cols or "von_mises_mpa" in cols:
            vm_rows = np.zeros(n_nodes, dtype=np.float64)
        for row in reader:
            nid = int(row["node_id"])
            idx = id_map.get(nid, nid - 1)
            u_rows[idx, 0] = float(row["ux"])
            u_rows[idx, 1] = float(row["uy"])
            if has_uz:
                u_rows[idx, 2] = float(row["uz"])
            if vm_rows is not None:
                vm_rows[idx] = float(
                    row.get("von_mises", row.get("von_mises_mpa", 0.0))
                )
    fields: dict[str, FloatArray] = {
        "displacement": u_rows if has_uz else u_rows[:, :2].copy()
    }
    if vm_rows is not None:
        fields["von_mises_mpa"] = vm_rows
    return fields


def read_abaqus_inp_and_fields(
    inp_path: Path | str,
    fields_path: Path | str,
    *,
    sample_id: str | None = None,
    parameters: Mapping[str, float] | None = None,
) -> ParsedCaeSample:
    """Read an Abaqus ``.inp`` mesh paired with exported nodal CSV/JSON fields."""
    inp = Path(inp_path).resolve()
    fld = Path(fields_path).resolve()
    nodes, cells, ctype, id_map = _parse_abaqus_like_inp(inp)
    p_fields = _parse_nodal_fields_file(fld, id_map, len(nodes))
    return ParsedCaeSample(
        sample_id=inp.stem if sample_id is None else sample_id,
        source_format=CaeSourceFormat.ABAQUS_INP,
        source_sha256=_hash_files(inp, fld),
        nodes=nodes,
        cells=cells,
        cell_type=ctype,
        point_fields=p_fields,
        cell_fields={},
        parameters={}
        if parameters is None
        else {str(k): float(v) for k, v in parameters.items()},
        metadata={"inp_path": str(inp), "fields_path": str(fld)},
    )


def read_calculix_inp_frd(
    inp_path: Path | str,
    frd_path: Path | str,
    *,
    sample_id: str | None = None,
    parameters: Mapping[str, float] | None = None,
) -> ParsedCaeSample:
    """Read a CalculiX ``.inp`` mesh paired with an ASCII ``.frd`` nodal result file."""
    inp = Path(inp_path).resolve()
    frd = Path(frd_path).resolve()
    nodes, cells, ctype, id_map = _parse_abaqus_like_inp(inp)
    n_nodes = len(nodes)
    disp = np.zeros((n_nodes, 3), dtype=np.float64)
    stress_6 = np.zeros((n_nodes, 6), dtype=np.float64)
    has_disp = False
    has_stress = False
    block_kind = ""

    for line in frd.read_text(encoding="utf-8").splitlines():
        if line.startswith(" -4"):
            upper = line.upper()
            if "DISP" in upper:
                block_kind = "DISP"
            elif "STRESS" in upper:
                block_kind = "STRESS"
            else:
                block_kind = ""
        elif line.startswith(" -3"):
            block_kind = ""
        elif line.startswith(" -1") and block_kind:
            # Handle both whitespace-separated and Fortran fixed-width glued negatives (e.g. 0.0E+00-3.0E-02)
            tokens = re.findall(
                r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?", line[3:]
            )
            if not tokens:
                continue
            nid = int(float(tokens[0]))
            idx = id_map.get(nid)
            if idx is None:
                continue
            vals = [float(v) for v in tokens[1:]]
            if block_kind == "DISP" and len(vals) >= 3:
                disp[idx, :3] = vals[:3]
                has_disp = True
            elif block_kind == "STRESS" and len(vals) >= 6:
                stress_6[idx, :6] = vals[:6]
                has_stress = True

    p_fields: dict[str, FloatArray] = {}
    if has_disp:
        p_fields["displacement"] = disp[:, :2].copy() if nodes.shape[1] == 2 else disp
    if has_stress:
        sxx, syy, szz, sxy, syz, szx = (stress_6[:, k] for k in range(6))
        vm = np.sqrt(
            0.5
            * (
                (sxx - syy) ** 2
                + (syy - szz) ** 2
                + (szz - sxx) ** 2
                + 6.0 * (sxy**2 + syz**2 + szx**2)
            )
        )
        p_fields["cauchy_stress_voigt"] = stress_6
        p_fields["von_mises_mpa"] = vm

    return ParsedCaeSample(
        sample_id=inp.stem if sample_id is None else sample_id,
        source_format=CaeSourceFormat.CALCULIX_FRD,
        source_sha256=_hash_files(inp, frd),
        nodes=nodes,
        cells=cells,
        cell_type=ctype,
        point_fields=p_fields,
        cell_fields={},
        parameters={}
        if parameters is None
        else {str(k): float(v) for k, v in parameters.items()},
        metadata={"inp_path": str(inp), "frd_path": str(frd)},
    )


def read_nastran_bdf_and_fields(
    bdf_path: Path | str,
    fields_path: Path | str,
    *,
    sample_id: str | None = None,
    parameters: Mapping[str, float] | None = None,
) -> ParsedCaeSample:
    """Read a free-field Nastran ``.bdf`` (GRID/CTETRA/CTRIA3) + nodal CSV/JSON fields."""
    bdf = Path(bdf_path).resolve()
    fld = Path(fields_path).resolve()
    node_ids: list[int] = []
    coords: list[list[float]] = []
    elems: list[list[int]] = []
    ctype = "tet4"
    for raw in bdf.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("$"):
            continue
        parts = [p.strip() for p in line.split(",")]
        card = parts[0].upper()
        if card == "GRID" and len(parts) >= 6:
            node_ids.append(int(parts[1]))
            coords.append([float(parts[3]), float(parts[4]), float(parts[5])])
        elif card == "CTETRA" and len(parts) >= 7:
            elems.append([int(v) for v in parts[3:] if v])
            ctype = "tet10" if len(elems[-1]) == 10 else "tet4"
        elif card == "CTRIA3" and len(parts) >= 6:
            elems.append([int(v) for v in parts[3:6]])
            ctype = "triangle"
    if not coords or not elems:
        raise ValueError(f"incomplete GRID / element cards in {bdf.name}")
    id_map = {nid: idx for idx, nid in enumerate(node_ids)}
    nodes = np.asarray(coords, dtype=np.float64)
    if ctype == "triangle" and np.allclose(nodes[:, 2], 0.0):
        nodes = nodes[:, :2].copy()
    cells = np.asarray([[id_map[n] for n in el] for el in elems], dtype=np.int64)
    p_fields = _parse_nodal_fields_file(fld, id_map, len(nodes))
    return ParsedCaeSample(
        sample_id=bdf.stem if sample_id is None else sample_id,
        source_format=CaeSourceFormat.NASTRAN_BDF,
        source_sha256=_hash_files(bdf, fld),
        nodes=nodes,
        cells=cells,
        cell_type=ctype,
        point_fields=p_fields,
        cell_fields={},
        parameters={}
        if parameters is None
        else {str(k): float(v) for k, v in parameters.items()},
        metadata={"bdf_path": str(bdf), "fields_path": str(fld)},
    )


ReaderPluginFn = Callable[[Path], ParsedCaeSample]
_CUSTOM_READERS: dict[str, ReaderPluginFn] = {}


def register_cae_reader(format_name: str, fn: ReaderPluginFn) -> None:
    _CUSTOM_READERS[format_name] = fn


def resolve_cae_reader(format_name: str) -> ReaderPluginFn:
    if format_name in _CUSTOM_READERS:
        return _CUSTOM_READERS[format_name]
    for ep in metadata.entry_points(group=READER_ENTRY_POINT_GROUP):
        if ep.name == format_name:
            loaded: ReaderPluginFn = ep.load()
            return loaded
    raise LookupError(f"unknown CAE reader plugin {format_name!r}")


__all__ = (
    "READER_ENTRY_POINT_GROUP",
    "ParsedCaeSample",
    "read_abaqus_inp_and_fields",
    "read_calculix_inp_frd",
    "read_nastran_bdf_and_fields",
    "read_vtu_or_xdmf",
    "register_cae_reader",
    "resolve_cae_reader",
)
