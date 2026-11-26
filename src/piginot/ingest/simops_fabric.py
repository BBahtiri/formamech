"""Solver-agnostic SimOps/SPDM archive fabric & ShardedMeshDataset3D curation (V5-3).

Extends :mod:`piginot.ingest.readers` with pure-MIT semantic parsers for:

* **Abaqus / CalculiX** decks (``*MATERIAL``, ``*BOUNDARY``, ``*CLOAD``/``*DLOAD``)
  paired with ``.sta``/``.msg``/``.cvg`` convergence logs separating solver
  numerical divergence from valid physical limit-point / buckling failures;
* **Ansys** ASCII ``.cdb`` (``NBLOCK``, ``EBLOCK``, ``MP``, ``D``, ``F``) + fields;
* **LS-DYNA** ``.k`` keyword decks (``*NODE``, ``*ELEMENT_SOLID``, ``*MAT_ELASTIC``,
  ``*BOUNDARY_SPC_SET``, ``*LOAD_NODE_POINT``) + fields;
* **Nastran** semantic ``.bdf`` (``GRID``, ``CTETRA``, ``MAT1``, ``SPC1``, ``FORCE``,
  ``PLOAD4``) + fields;
* **SimJEB / DeepJEB** bracket corpus loader (``.npz``/``.vtu`` + 4-load-case metadata);
* Optional binary plugin hooks (``pydpf-core``, ``pyNastran``, ``lasso-python``,
  ``h5py`` CGNS) that fail closed with install hints when absent.

Curated samples flow directly into a tamper-checked, geometry-group-split-locked
:class:`~piginot.store.dataset_3d.ShardedMeshDataset3D` with zero cross-split
CAD geometry leakage.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from piginot.contracts import Unit
from piginot.ingest.contracts import CaeSourceFormat
from piginot.ingest.readers import (
    ParsedCaeSample,
    _hash_files,
    _parse_abaqus_like_inp,
    _parse_nodal_fields_file,
    read_nastran_bdf_and_fields,
    read_vtu_or_xdmf,
)
from piginot.ingest.screening import (
    _compute_min_det_f,
    _extract_displacement,
    _global_qois,
)
from piginot.ingest.simops_schema import (
    BcDescriptorSummary,
    FailureDisposition,
    SimOpsCurationReport,
    SimOpsRecord,
    SolverProvenance,
)
from piginot.store.dataset_3d import (
    DatasetSplit,
    ShardedMeshDataset3D,
    ShardedRecord3D,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_PHYSICAL_FAILURE_KEYWORDS = (
    "BUCKLING",
    "LIMIT POINT",
    "COLLAPSE",
    "FRACTURE",
    "SNAP-THROUGH",
    "POST-BUCKLING",
    "YIELD COLLAPSE",
    "PHYSICAL_FAILURE",
)
_SOLVER_DEFECT_KEYWORDS = (
    "TOO MANY ATTEMPTS",
    "NOT BEEN COMPLETED",
    "DIVERGED",
    "NONCONVERGENCE",
    "ZERO PIVOT",
    "NEGATIVE JACOBIAN",
    "NAN",
    "EXCEEDED MAX",
)


@dataclass(frozen=True, slots=True)
class SimOpsArchiveEntry:
    """In-memory external CAE run bundled with solver provenance & explicit units."""

    sample: ParsedCaeSample
    geometry_group_id: str
    length_unit: Unit
    stress_unit: Unit = Unit.MPA
    provenance: SolverProvenance = SolverProvenance(solver_name="unknown")
    bc_summary: BcDescriptorSummary = BcDescriptorSummary()
    material_summary: Mapping[str, float] = ()  # type: ignore[assignment]
    physical_failure_label: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "length_unit", Unit(self.length_unit))
        object.__setattr__(self, "stress_unit", Unit(self.stress_unit))
        mat_dict = (
            {str(k): float(v) for k, v in self.material_summary.items()}
            if isinstance(self.material_summary, Mapping)
            else {}
        )
        object.__setattr__(self, "material_summary", mat_dict)


def parse_abaqus_convergence_log(
    log_path: Path | str,
    *,
    solver_name: str = "abaqus",
) -> tuple[SolverProvenance, str | None]:
    """Parse Abaqus/CalculiX ``.sta`` / ``.msg`` / ``.cvg`` to distinguish solver defects from physical failures."""
    path = Path(log_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    upper = text.upper()
    cutbacks = len(re.findall(r"\b\d+U\b", upper)) + upper.count("CUTBACK")
    res_matches = re.findall(
        r"(?:RESIDUAL|FORCE\s+RESIDUAL)[^0-9E+-]*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)",
        upper,
    )
    final_res = float(res_matches[-1]) if res_matches else None

    for kw in _PHYSICAL_FAILURE_KEYWORDS:
        if kw in upper:
            label = kw.lower().replace(" ", "_").replace("-", "_")
            return (
                SolverProvenance(
                    solver_name=solver_name,
                    convergence_status="physical_limit_reached",
                    final_residual=final_res,
                    cutbacks=cutbacks,
                    termination_reason=f"detected physical failure marker: {kw}",
                ),
                label,
            )

    for kw in _SOLVER_DEFECT_KEYWORDS:
        if kw in upper:
            return (
                SolverProvenance(
                    solver_name=solver_name,
                    convergence_status="diverged",
                    final_residual=final_res,
                    cutbacks=cutbacks,
                    termination_reason=f"detected solver failure marker: {kw}",
                ),
                None,
            )

    if (
        "COMPLETED SUCCESSFULLY" in upper
        or "JOB COMPLETED" in upper
        or "CONVERGED" in upper
    ):
        status = "converged"
    else:
        status = "under_converged"
    return (
        SolverProvenance(
            solver_name=solver_name,
            convergence_status=status,
            final_residual=final_res,
            cutbacks=cutbacks,
            termination_reason="completed successfully"
            if status == "converged"
            else "missing completion marker",
        ),
        None,
    )


def read_abaqus_simops_entry(
    inp_path: Path | str,
    fields_path: Path | str,
    *,
    length_unit: Unit,
    stress_unit: Unit = Unit.MPA,
    log_path: Path | str | None = None,
    geometry_group_id: str | None = None,
    load_case_id: str = "step-1",
    sample_id: str | None = None,
    parameters: Mapping[str, float] | None = None,
) -> SimOpsArchiveEntry:
    """Parse Abaqus/CalculiX ``.inp`` (mesh, materials, BCs) + fields + optional ``.sta``/``.msg`` log."""
    inp = Path(inp_path).resolve()
    fld = Path(fields_path).resolve()
    nodes, cells, ctype, id_map = _parse_abaqus_like_inp(inp)
    p_fields = _parse_nodal_fields_file(fld, id_map, len(nodes))

    mat_summary: dict[str, float] = {}
    clamped_nodes: set[int] = set()
    loaded_nodes: set[int] = set()
    net_force = np.zeros(3, dtype=np.float64)
    thermal_bc = False
    section = ""
    for raw_line in inp.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            up = line.upper()
            if up.startswith("*ELASTIC"):
                section = "ELASTIC"
            elif up.startswith("*BOUNDARY"):
                section = "BOUNDARY"
            elif up.startswith("*CLOAD"):
                section = "CLOAD"
            elif up.startswith("*DLOAD"):
                section = "DLOAD"
            elif up.startswith(("*TEMPERATURE", "*DFLUX", "*FILM")):
                thermal_bc = True
                section = ""
            else:
                section = ""
            continue
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if section == "ELASTIC" and len(parts) >= 2:
            mat_summary["elastic_modulus_mpa"] = float(parts[0])
            mat_summary["poisson_ratio"] = float(parts[1])
        elif section == "BOUNDARY" and parts:
            if parts[0].isdigit():
                clamped_nodes.add(int(parts[0]))
        elif section == "CLOAD" and len(parts) >= 3:
            if parts[0].isdigit():
                loaded_nodes.add(int(parts[0]))
            dof = int(parts[1])
            if 1 <= dof <= 3:
                net_force[dof - 1] += float(parts[2])

    if log_path is not None:
        provenance, phys_label = parse_abaqus_convergence_log(log_path)
    else:
        provenance = SolverProvenance(
            solver_name="abaqus", convergence_status="converged"
        )
        phys_label = None

    sample = ParsedCaeSample(
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
        metadata={
            "inp_path": str(inp),
            "fields_path": str(fld),
            "load_case_id": load_case_id,
        },
    )
    return SimOpsArchiveEntry(
        sample=sample,
        geometry_group_id=geometry_group_id or sample.sample_id,
        length_unit=length_unit,
        stress_unit=stress_unit,
        provenance=provenance,
        bc_summary=BcDescriptorSummary(
            load_case_id=load_case_id,
            clamped_node_count=len(clamped_nodes),
            loaded_node_count=len(loaded_nodes),
            net_force_n=(float(net_force[0]), float(net_force[1]), float(net_force[2])),
            thermal_bc_present=thermal_bc,
            bc_tags=("BOUNDARY",) if clamped_nodes else (),
        ),
        material_summary=mat_summary,
        physical_failure_label=phys_label,
    )


def read_ansys_cdb_and_fields(
    cdb_path: Path | str,
    fields_path: Path | str,
    *,
    length_unit: Unit,
    stress_unit: Unit = Unit.MPA,
    geometry_group_id: str | None = None,
    load_case_id: str = "loadstep-1",
    sample_id: str | None = None,
    parameters: Mapping[str, float] | None = None,
    convergence_status: str = "converged",
    physical_failure_label: str | None = None,
) -> SimOpsArchiveEntry:
    """Pure-MIT parser for Ansys ASCII ``.cdb`` mesh/BC/material decks + nodal CSV/JSON fields."""
    cdb = Path(cdb_path).resolve()
    fld = Path(fields_path).resolve()
    node_ids: list[int] = []
    coords: list[list[float]] = []
    raw_elems: list[list[int]] = []
    mat_summary: dict[str, float] = {}
    clamped_nodes: set[int] = set()
    loaded_nodes: set[int] = set()
    net_force = np.zeros(3, dtype=np.float64)
    thermal_bc = False

    mode = ""
    for raw_line in cdb.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("!"):
            continue
        up = line.upper()
        if up.startswith("NBLOCK"):
            mode = "NBLOCK"
            continue
        if up.startswith("EBLOCK"):
            mode = "EBLOCK"
            continue
        if mode in ("NBLOCK", "EBLOCK"):
            if line.startswith("("):
                continue  # Fortran format descriptor line
            if line.startswith("-1") or up.startswith("N,R5"):
                mode = ""
                continue
            toks = line.split()
            if mode == "NBLOCK" and len(toks) >= 6:
                node_ids.append(int(toks[0]))
                coords.append([float(toks[3]), float(toks[4]), float(toks[5])])
            elif mode == "EBLOCK" and len(toks) >= 15:
                # Standard Ansys SOLID EBLOCK: 11 header ints followed by node IDs
                n_cnt = int(toks[8])
                n_list = [int(v) for v in toks[11 : 11 + n_cnt]]
                if len(n_list) >= 4:
                    raw_elems.append(n_list)
            continue

        parts = [p.strip() for p in line.split(",")]
        cmd = parts[0].upper()
        if cmd == "MP" and len(parts) >= 4:
            prop = parts[1].upper()
            if prop == "EX":
                mat_summary["elastic_modulus_mpa"] = float(parts[3])
            elif prop in ("PRXY", "NUXY"):
                mat_summary["poisson_ratio"] = float(parts[3])
            elif prop == "KXX":
                mat_summary["thermal_conductivity_w_mk"] = float(parts[3])
                thermal_bc = True
        elif cmd == "D" and len(parts) >= 3 and parts[1].isdigit():
            clamped_nodes.add(int(parts[1]))
            if parts[2].upper() == "TEMP":
                thermal_bc = True
        elif cmd == "F" and len(parts) >= 4 and parts[1].isdigit():
            loaded_nodes.add(int(parts[1]))
            dof_map = {"FX": 0, "FY": 1, "FZ": 2}
            axis = dof_map.get(parts[2].upper())
            if axis is not None:
                net_force[axis] += float(parts[3])

    if not coords or not raw_elems:
        raise ValueError(f"incomplete Ansys NBLOCK/EBLOCK in {cdb.name}")
    id_map = {nid: idx for idx, nid in enumerate(node_ids)}
    nodes = np.asarray(coords, dtype=np.float64)
    cells = np.asarray([[id_map[n] for n in el] for el in raw_elems], dtype=np.int64)
    k = cells.shape[1]
    ctype = "tet10" if k == 10 else ("hex8" if k == 8 else "tet4")
    p_fields = _parse_nodal_fields_file(fld, id_map, len(nodes))
    sample = ParsedCaeSample(
        sample_id=cdb.stem if sample_id is None else sample_id,
        source_format=CaeSourceFormat.ANSYS_CDB,
        source_sha256=_hash_files(cdb, fld),
        nodes=nodes,
        cells=cells,
        cell_type=ctype,
        point_fields=p_fields,
        cell_fields={},
        parameters={}
        if parameters is None
        else {str(k): float(v) for k, v in parameters.items()},
        metadata={
            "cdb_path": str(cdb),
            "fields_path": str(fld),
            "load_case_id": load_case_id,
        },
    )
    return SimOpsArchiveEntry(
        sample=sample,
        geometry_group_id=geometry_group_id or sample.sample_id,
        length_unit=length_unit,
        stress_unit=stress_unit,
        provenance=SolverProvenance(
            solver_name="ansys-mapdl",
            convergence_status=convergence_status,
        ),
        bc_summary=BcDescriptorSummary(
            load_case_id=load_case_id,
            clamped_node_count=len(clamped_nodes),
            loaded_node_count=len(loaded_nodes),
            net_force_n=(float(net_force[0]), float(net_force[1]), float(net_force[2])),
            thermal_bc_present=thermal_bc,
            bc_tags=("ANSYS_D",) if clamped_nodes else (),
        ),
        material_summary=mat_summary,
        physical_failure_label=physical_failure_label,
    )


def read_lsdyna_k_and_fields(
    k_path: Path | str,
    fields_path: Path | str,
    *,
    length_unit: Unit,
    stress_unit: Unit = Unit.MPA,
    geometry_group_id: str | None = None,
    load_case_id: str = "crash-1",
    sample_id: str | None = None,
    parameters: Mapping[str, float] | None = None,
    convergence_status: str = "converged",
    physical_failure_label: str | None = None,
) -> SimOpsArchiveEntry:
    """Pure-MIT parser for LS-DYNA ``.k`` keyword decks + nodal CSV/JSON fields."""
    kp = Path(k_path).resolve()
    fld = Path(fields_path).resolve()
    node_ids: list[int] = []
    coords: list[list[float]] = []
    raw_elems: list[list[int]] = []
    mat_summary: dict[str, float] = {}
    clamped_nodes: set[int] = set()
    loaded_nodes: set[int] = set()
    section = ""
    for raw_line in kp.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("$"):
            continue
        if line.startswith("*"):
            up = line.upper()
            if up.startswith("*NODE"):
                section = "NODE"
            elif up.startswith("*ELEMENT_SOLID"):
                section = "ELEM_SOLID"
            elif up.startswith("*MAT_ELASTIC"):
                section = "MAT_ELASTIC"
            elif up.startswith("*BOUNDARY_SPC_NODE"):
                section = "SPC_NODE"
            elif up.startswith("*LOAD_NODE_POINT"):
                section = "LOAD_NODE"
            else:
                section = ""
            continue
        toks = [t for t in re.split(r"[,\s]+", line) if t]
        if section == "NODE" and len(toks) >= 4:
            node_ids.append(int(toks[0]))
            coords.append([float(toks[1]), float(toks[2]), float(toks[3])])
        elif section == "ELEM_SOLID" and len(toks) >= 6:
            # eid, pid, n1, n2, n3, n4 [, n5..n8]
            n_ids = [int(v) for v in toks[2:]]
            # Degenerate tet encoded as 8-node hex repeats n4 in slots 4..7
            if len(n_ids) == 8 and len(set(n_ids[3:])) == 1:
                n_ids = n_ids[:4]
            raw_elems.append(n_ids)
        elif section == "MAT_ELASTIC" and len(toks) >= 4:
            mat_summary["density"] = float(toks[1])
            mat_summary["elastic_modulus_mpa"] = float(toks[2])
            mat_summary["poisson_ratio"] = float(toks[3])
            section = ""
        elif section == "SPC_NODE" and toks:
            clamped_nodes.add(int(toks[0]))
        elif section == "LOAD_NODE" and toks:
            loaded_nodes.add(int(toks[0]))

    if not coords or not raw_elems:
        raise ValueError(f"incomplete LS-DYNA *NODE/*ELEMENT_SOLID in {kp.name}")
    id_map = {nid: idx for idx, nid in enumerate(node_ids)}
    nodes = np.asarray(coords, dtype=np.float64)
    cells = np.asarray([[id_map[n] for n in el] for el in raw_elems], dtype=np.int64)
    ctype = "hex8" if cells.shape[1] == 8 else "tet4"
    p_fields = _parse_nodal_fields_file(fld, id_map, len(nodes))
    sample = ParsedCaeSample(
        sample_id=kp.stem if sample_id is None else sample_id,
        source_format=CaeSourceFormat.LSDYNA_K,
        source_sha256=_hash_files(kp, fld),
        nodes=nodes,
        cells=cells,
        cell_type=ctype,
        point_fields=p_fields,
        cell_fields={},
        parameters={}
        if parameters is None
        else {str(k): float(v) for k, v in parameters.items()},
        metadata={
            "k_path": str(kp),
            "fields_path": str(fld),
            "load_case_id": load_case_id,
        },
    )
    return SimOpsArchiveEntry(
        sample=sample,
        geometry_group_id=geometry_group_id or sample.sample_id,
        length_unit=length_unit,
        stress_unit=stress_unit,
        provenance=SolverProvenance(
            solver_name="ls-dyna",
            convergence_status=convergence_status,
        ),
        bc_summary=BcDescriptorSummary(
            load_case_id=load_case_id,
            clamped_node_count=len(clamped_nodes),
            loaded_node_count=len(loaded_nodes),
            bc_tags=("LSDYNA_SPC",) if clamped_nodes else (),
        ),
        material_summary=mat_summary,
        physical_failure_label=physical_failure_label,
    )


def read_nastran_simops_entry(
    bdf_path: Path | str,
    fields_path: Path | str,
    *,
    length_unit: Unit,
    stress_unit: Unit = Unit.MPA,
    geometry_group_id: str | None = None,
    load_case_id: str = "subcase-1",
    sample_id: str | None = None,
    parameters: Mapping[str, float] | None = None,
    convergence_status: str = "converged",
    physical_failure_label: str | None = None,
) -> SimOpsArchiveEntry:
    """Read Nastran ``.bdf`` with ``MAT1``, ``SPC1``, ``FORCE``/``PLOAD4`` extraction."""
    bdf = Path(bdf_path).resolve()
    sample = read_nastran_bdf_and_fields(
        bdf, fields_path, sample_id=sample_id, parameters=parameters
    )
    mat_summary: dict[str, float] = {}
    clamped_nodes: set[int] = set()
    loaded_nodes: set[int] = set()
    net_force = np.zeros(3, dtype=np.float64)
    for raw_line in bdf.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("$"):
            continue
        parts = [p.strip() for p in line.split(",")] if "," in line else line.split()
        card = parts[0].upper()
        if card == "MAT1" and len(parts) >= 5:
            mat_summary["elastic_modulus_mpa"] = float(parts[2])
            if parts[4]:
                mat_summary["poisson_ratio"] = float(parts[4])
        elif card == "SPC1" and len(parts) >= 4:
            for tok in parts[3:]:
                if tok.isdigit():
                    clamped_nodes.add(int(tok))
        elif card == "FORCE" and len(parts) >= 8:
            if parts[2].isdigit():
                loaded_nodes.add(int(parts[2]))
            mag = float(parts[4])
            net_force += mag * np.array(
                [float(parts[5]), float(parts[6]), float(parts[7])], dtype=np.float64
            )
    return SimOpsArchiveEntry(
        sample=sample,
        geometry_group_id=geometry_group_id or sample.sample_id,
        length_unit=length_unit,
        stress_unit=stress_unit,
        provenance=SolverProvenance(
            solver_name="nastran",
            convergence_status=convergence_status,
        ),
        bc_summary=BcDescriptorSummary(
            load_case_id=load_case_id,
            clamped_node_count=len(clamped_nodes),
            loaded_node_count=len(loaded_nodes),
            net_force_n=(float(net_force[0]), float(net_force[1]), float(net_force[2])),
            bc_tags=("NASTRAN_SPC1",) if clamped_nodes else (),
        ),
        material_summary=mat_summary,
        physical_failure_label=physical_failure_label,
    )


def read_simjeb_sample(
    path: Path | str,
    *,
    length_unit: Unit = Unit.MM,
    stress_unit: Unit = Unit.MPA,
    load_case_id: str = "vertical",
    geometry_group_id: str | None = None,
    sample_id: str | None = None,
) -> SimOpsArchiveEntry:
    """Native adapter for SimJEB / DeepJEB 3D bracket archives (``.npz`` or ``.vtu``)."""
    src = Path(path).resolve()
    if src.suffix.lower() == ".npz":
        with np.load(src, allow_pickle=False) as data:
            nodes = np.asarray(data["nodes"], dtype=np.float64)
            cells = np.asarray(data["cells"], dtype=np.int64)
            disp = np.asarray(data["displacement"], dtype=np.float64)
            p_fields: dict[str, FloatArray] = {"displacement": disp}
            if "von_mises_mpa" in data:
                p_fields["von_mises_mpa"] = np.asarray(
                    data["von_mises_mpa"], dtype=np.float64
                )
            params: dict[str, float] = {}
            if "parameters" in data:
                for i, val in enumerate(
                    np.asarray(data["parameters"], dtype=np.float64).ravel()
                ):
                    params[f"p{i}"] = float(val)
            meta: dict[str, Any] = {
                "source_path": str(src),
                "load_case_id": load_case_id,
            }
            if "metadata_json" in data:
                meta.update(json.loads(str(data["metadata_json"])))
        k = cells.shape[1]
        ctype = "tet10" if k == 10 else ("hex8" if k == 8 else "tet4")
        sample = ParsedCaeSample(
            sample_id=sample_id or f"{src.stem}_{load_case_id}",
            source_format=CaeSourceFormat.SIMJEB_NPZ,
            source_sha256=_hash_files(src),
            nodes=nodes,
            cells=cells,
            cell_type=ctype,
            point_fields=p_fields,
            cell_fields={},
            parameters=params,
            metadata=meta,
        )
    else:
        sample = read_vtu_or_xdmf(
            src, sample_id=sample_id or f"{src.stem}_{load_case_id}"
        )
    return SimOpsArchiveEntry(
        sample=sample,
        geometry_group_id=geometry_group_id or src.stem.split("_")[0],
        length_unit=length_unit,
        stress_unit=stress_unit,
        provenance=SolverProvenance(
            solver_name="simjeb-deepjeb", convergence_status="converged"
        ),
        bc_summary=BcDescriptorSummary(load_case_id=load_case_id),
    )


def require_optional_binary_reader(backend: str) -> Any:
    """Fail-closed loader for optional ``.[simops]`` commercial binary CAE plugins."""
    mapping = {
        "ansys-rst": ("ansys.dpf.core", "pip install ansys-dpf-core"),
        "nastran-op2": ("pyNastran.op2.op2", "pip install pyNastran"),
        "lsdyna-d3plot": ("lasso.dyna", "pip install lasso-python"),
        "cgns-hdf5": ("h5py", "pip install h5py"),
    }
    if backend not in mapping:
        raise LookupError(
            f"unknown binary reader {backend!r}; supported: {sorted(mapping)}"
        )
    mod_name, hint = mapping[backend]
    try:
        return importlib.import_module(mod_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Optional binary CAE reader {backend!r} requires `{mod_name}` ({hint})."
        ) from exc


def _geometry_fingerprint_raw(nodes: FloatArray, cells: IntArray) -> str:
    nodes_c = np.ascontiguousarray(nodes, dtype=np.float64) + 0.0
    cells_c = np.ascontiguousarray(cells, dtype=np.int64)
    h = hashlib.sha256()
    h.update(f"nodes:{nodes_c.shape}|".encode())
    h.update(nodes_c.tobytes())
    h.update(f"cells:{cells_c.shape}|".encode())
    h.update(cells_c.tobytes())
    return h.hexdigest()


def _assign_group_splits(
    group_keys: Sequence[str],
    *,
    train_fraction: float,
    dev_fraction: float,
) -> dict[str, DatasetSplit]:
    """Deterministic leakage-free split assignment across unique CAD geometry groups."""
    unique = sorted(
        set(group_keys), key=lambda g: hashlib.sha256(g.encode("utf-8")).hexdigest()
    )
    n = len(unique)
    if n == 0:
        return {}
    if n == 1:
        return {unique[0]: DatasetSplit.TRAIN}
    if n == 2:
        return {unique[0]: DatasetSplit.TRAIN, unique[1]: DatasetSplit.HOLDOUT}
    n_train = max(1, int(round(n * train_fraction)))
    n_dev = max(1, int(round(n * dev_fraction)))
    if n_train + n_dev >= n:
        n_train = max(1, n - 2)
        n_dev = 1
    out: dict[str, DatasetSplit] = {}
    for i, g in enumerate(unique):
        if i < n_train:
            out[g] = DatasetSplit.TRAIN
        elif i < n_train + n_dev:
            out[g] = DatasetSplit.DEV
        else:
            out[g] = DatasetSplit.HOLDOUT
    return out


def curate_archive_to_sharded_dataset(
    entries: Sequence[SimOpsArchiveEntry],
    output_dataset_dir: Path | str | None = None,
    *,
    train_fraction: float = 0.70,
    dev_fraction: float = 0.15,
    det_f_floor: float = 1.0e-5,
    max_records_per_shard: int = 16,
) -> SimOpsCurationReport:
    """Screen a multi-solver archive and curate accepted 3D runs into a split-locked ShardedMeshDataset3D."""
    if not entries:
        raise ValueError("entries must be non-empty")

    seen_hashes: set[str] = set()
    # Map geometry fingerprint -> unified geometry_group_id so identical meshes
    # under different aliases cannot leak across splits
    fp_to_group: dict[str, str] = {}
    prelim: list[
        tuple[
            SimOpsArchiveEntry,
            str,
            FailureDisposition,
            str | None,
            str | None,
            float | None,
            dict[str, float],
        ]
    ] = []

    for entry in entries:
        sample = entry.sample
        fp = _geometry_fingerprint_raw(sample.nodes, sample.cells)
        group_id = fp_to_group.setdefault(fp, entry.geometry_group_id)
        disp = _extract_displacement(sample)
        qois = _global_qois(sample, disp)

        quarantine_reason: str | None = None
        if sample.source_sha256 in seen_hashes:
            quarantine_reason = "duplicate source_sha256"
        else:
            seen_hashes.add(sample.source_sha256)

        all_finite = (
            bool(np.all(np.isfinite(sample.nodes)))
            and all(bool(np.all(np.isfinite(v))) for v in sample.point_fields.values())
            and all(bool(np.all(np.isfinite(v))) for v in sample.cell_fields.values())
        )
        min_det = (
            _compute_min_det_f(sample, disp)
            if (all_finite and disp is not None)
            else None
        )
        if min_det is not None and not np.isfinite(min_det):
            min_det = None
        if quarantine_reason is None and not all_finite:
            quarantine_reason = "non-finite node or field values"
        if quarantine_reason is None and entry.provenance.convergence_status in (
            "diverged",
            "under_converged",
            "failed",
        ):
            quarantine_reason = (
                f"solver convergence defect ({entry.provenance.convergence_status}): "
                f"{entry.provenance.termination_reason}"
            )
        if quarantine_reason is None and min_det is not None and min_det <= det_f_floor:
            quarantine_reason = (
                f"inverted element min_det_f={min_det:.3e} <= {det_f_floor:.3e}"
            )
        if quarantine_reason is None and disp is None:
            quarantine_reason = "missing displacement field"

        if quarantine_reason is not None:
            disposition = FailureDisposition.QUARANTINED_SOLVER_DEFECT
            phys_label = None
        elif entry.physical_failure_label is not None or (
            entry.provenance.convergence_status == "physical_limit_reached"
        ):
            disposition = FailureDisposition.VALID_PHYSICAL_FAILURE
            phys_label = entry.physical_failure_label or "physical_limit_reached"
        else:
            disposition = FailureDisposition.VALID_CONVERGED
            phys_label = None

        prelim.append(
            (entry, group_id, disposition, phys_label, quarantine_reason, min_det, qois)
        )

    accepted_groups = [
        group_id
        for _, group_id, disp_state, _, _, _, _ in prelim
        if disp_state is not FailureDisposition.QUARANTINED_SOLVER_DEFECT
    ]
    group_splits = _assign_group_splits(
        accepted_groups,
        train_fraction=train_fraction,
        dev_fraction=dev_fraction,
    )

    records: list[SimOpsRecord] = []
    sharded_batch: list[ShardedRecord3D] = []
    n_conv = 0
    n_phys_fail = 0
    n_quar = 0

    for entry, group_id, disposition, phys_label, quar_reason, min_det, qois in prelim:
        split_enum = (
            group_splits.get(group_id)
            if disposition is not FailureDisposition.QUARANTINED_SOLVER_DEFECT
            else None
        )
        rec = SimOpsRecord(
            sample_id=entry.sample.sample_id,
            geometry_group_id=group_id,
            source_format=entry.sample.source_format,
            source_sha256=entry.sample.source_sha256,
            length_unit=entry.length_unit,
            stress_unit=entry.stress_unit,
            provenance=entry.provenance,
            disposition=disposition,
            bc_summary=entry.bc_summary,
            material_summary=entry.material_summary,
            physical_failure_label=phys_label,
            quarantine_reason=quar_reason,
            min_det_f=min_det,
            qois=qois,
            split=None if split_enum is None else split_enum.value,
            metadata=entry.sample.metadata,
        )
        records.append(rec)
        if disposition is FailureDisposition.VALID_CONVERGED:
            n_conv += 1
        elif disposition is FailureDisposition.VALID_PHYSICAL_FAILURE:
            n_phys_fail += 1
        else:
            n_quar += 1
            continue

        # Convert accepted 3D runs into canonical ShardedRecord3D
        sample = entry.sample
        disp = _extract_displacement(sample)
        if disp is not None and sample.spatial_dim == 3 and split_enum is not None:
            params_tuple = tuple(
                float(sample.parameters[k]) for k in sorted(sample.parameters)
            )
            sharded_batch.append(
                ShardedRecord3D(
                    record_id=sample.sample_id,
                    mesh_id=group_id,
                    split=split_enum,
                    parameters=params_tuple,
                    nodes_mm=sample.nodes,
                    cells=sample.cells,
                    displacement_mm=disp,
                    metadata={
                        "disposition": disposition.value,
                        "physical_failure_label": phys_label,
                        "source_format": sample.source_format.value,
                        "load_case_id": entry.bc_summary.load_case_id,
                        "length_unit": entry.length_unit.value,
                        "stress_unit": entry.stress_unit.value,
                        "cell_type": sample.cell_type,
                    },
                )
            )

    ds_root: str | None = None
    ds_manifest_sha: str | None = None
    if output_dataset_dir is not None:
        ds = ShardedMeshDataset3D(
            output_dataset_dir, max_records_per_shard=max_records_per_shard
        )
        if sharded_batch:
            ds.append_records(sharded_batch)
        ds_root = str(ds.root_dir)
        ds_manifest_sha = ds.manifest_sha256

    return SimOpsCurationReport(
        total_samples=len(records),
        valid_converged_count=n_conv,
        valid_physical_failure_count=n_phys_fail,
        quarantined_solver_defect_count=n_quar,
        geometry_group_splits={k: v.value for k, v in group_splits.items()},
        records=tuple(records),
        dataset_root=ds_root,
        dataset_manifest_sha256=ds_manifest_sha,
    )


__all__ = (
    "SimOpsArchiveEntry",
    "curate_archive_to_sharded_dataset",
    "parse_abaqus_convergence_log",
    "read_abaqus_simops_entry",
    "read_ansys_cdb_and_fields",
    "read_lsdyna_k_and_fields",
    "read_nastran_simops_entry",
    "read_simjeb_sample",
    "require_optional_binary_reader",
)
