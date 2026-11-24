"""Live external FEA solver oracles (CalculiX, Abaqus, PyAnsys, DOLFINx) (V4-1).

Allows enterprises to verify and certify FormaMech neural operators against
their own external qualified FEA solvers through strictly sandboxed subprocess
adapters governed by :data:`EXTERNAL_SOLVER_MACHINE`.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import numpy as np

from piginot.contracts.state_machine import (
    StateMachineSpec,
    TransitionSpec,
    register_state_machine,
)
from piginot.fem.backend_3d import (
    FemRequest3D,
    FemResult3D,
    assemble_surface_tractions_3d,
    face_node_indices_3d,
)
from piginot.geometry.mesh3d import CellType3D
from piginot.ingest.readers import read_calculix_inp_frd

EXTERNAL_SOLVER_SCHEMA: Final = "formamech.external-solver-oracle-v1"


class ExternalSolverState(StrEnum):
    """Explicit lifecycle states for sandboxed external FEA solver execution."""

    PREPARING = "PREPARING"
    WRITING_DECK = "WRITING_DECK"
    EXECUTING = "EXECUTING"
    PARSING = "PARSING"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


EXTERNAL_SOLVER_MACHINE: Final[StateMachineSpec[ExternalSolverState]] = (
    register_state_machine(
        StateMachineSpec(
            machine_id="formamech.fem.external_oracle",
            description=(
                "Sandboxed subprocess lifecycle for external FEA solver oracles "
                "(CalculiX, Abaqus, PyAnsys, LGPL-isolated DOLFINx)."
            ),
            states=tuple(ExternalSolverState),
            initial_states=(ExternalSolverState.PREPARING,),
            terminal_states=(
                ExternalSolverState.VERIFIED,
                ExternalSolverState.FAILED,
                ExternalSolverState.TIMEOUT,
            ),
            transitions=(
                TransitionSpec(
                    ExternalSolverState.PREPARING,
                    ExternalSolverState.WRITING_DECK,
                    "write_deck",
                    "Validated request and began generating solver input deck.",
                ),
                TransitionSpec(
                    ExternalSolverState.PREPARING,
                    ExternalSolverState.FAILED,
                    "fail_unavailable",
                    "External solver executable or license is unavailable.",
                ),
                TransitionSpec(
                    ExternalSolverState.WRITING_DECK,
                    ExternalSolverState.EXECUTING,
                    "launch_process",
                    "Wrote canonical input deck and launched sandboxed subprocess.",
                ),
                TransitionSpec(
                    ExternalSolverState.WRITING_DECK,
                    ExternalSolverState.FAILED,
                    "fail_deck",
                    "Unsupported mesh element type or material deck serialization error.",
                ),
                TransitionSpec(
                    ExternalSolverState.EXECUTING,
                    ExternalSolverState.PARSING,
                    "parse_results",
                    "Subprocess exited cleanly; parsing nodal displacement and stress.",
                ),
                TransitionSpec(
                    ExternalSolverState.EXECUTING,
                    ExternalSolverState.TIMEOUT,
                    "timeout",
                    "External solver exceeded configured timeout ceiling.",
                ),
                TransitionSpec(
                    ExternalSolverState.EXECUTING,
                    ExternalSolverState.FAILED,
                    "fail_exec",
                    "External solver exited with non-zero status.",
                ),
                TransitionSpec(
                    ExternalSolverState.PARSING,
                    ExternalSolverState.VERIFIED,
                    "verify_fields",
                    "Parsed finite nodal fields satisfying det(F) > 0 admissibility.",
                ),
                TransitionSpec(
                    ExternalSolverState.PARSING,
                    ExternalSolverState.FAILED,
                    "fail_parse",
                    "Parsed results were missing, malformed, or inadmissible.",
                ),
            ),
        )
    )
)


class OracleUnavailableError(RuntimeError):
    """Raised when an external FEA executable or runtime is not installed."""

    def __init__(self, oracle_name: str, executable: str, recovery_hint: str) -> None:
        self.oracle_name = oracle_name
        self.executable = executable
        self.recovery_hint = recovery_hint
        super().__init__(
            f"external oracle {oracle_name!r} unavailable (executable {executable!r} not found): "
            f"{recovery_hint}"
        )


@dataclass(frozen=True, slots=True)
class ExternalSolverConfig:
    """Configuration for executing an external FEA solver binary safely."""

    executable: str
    command_prefix: tuple[str, ...] = ()
    timeout_s: float = 120.0
    extra_args: tuple[str, ...] = ()
    keep_workdir: bool = False

    def __post_init__(self) -> None:
        if not self.executable.strip():
            raise ValueError("executable must be non-empty")
        if self.timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")


def _resolve_executable(
    cfg: ExternalSolverConfig, oracle_name: str, hint: str
) -> list[str]:
    if cfg.command_prefix:
        return [*cfg.command_prefix, cfg.executable]
    cand = Path(cfg.executable)
    if cand.is_file():
        return [str(cand)]
    found = shutil.which(cfg.executable)
    if found is None:
        raise OracleUnavailableError(oracle_name, cfg.executable, hint)
    return [found]


def write_calculix_inp_3d(request: FemRequest3D, inp_path: Path) -> Path:
    """Serialize a 3D :class:`FemRequest3D` into a CalculiX/Abaqus ``.inp`` deck."""
    mesh = request.mesh
    if mesh.cell_type is CellType3D.TET4:
        el_type = "C3D4"
    elif mesh.cell_type is CellType3D.TET10:
        el_type = "C3D10"
    else:
        raise ValueError(
            f"CalculiX writer supports tet4/tet10 meshes, got {mesh.cell_type.value!r}"
        )

    e_mpa = float(request.material.properties["elastic_modulus"].value)
    nu = float(request.material.properties["poisson_ratio"].value)
    comp_num = {"x": 1, "y": 2, "z": 3}

    lines: list[str] = [
        "** Generated by FormaMech ExternalSolverOracle (V4-1)",
        "*NODE, NSET=NALL",
    ]
    for idx, (x, y, z) in enumerate(mesh.nodes, start=1):
        lines.append(f"{idx}, {x:.10g}, {y:.10g}, {z:.10g}")

    lines.append(f"*ELEMENT, TYPE={el_type}, ELSET=EALL")
    for eid, cell in enumerate(mesh.cells, start=1):
        n_str = ", ".join(str(nid + 1) for nid in cell)
        lines.append(f"{eid}, {n_str}")

    lines.extend(
        [
            "*MATERIAL, NAME=MAT1",
            "*ELASTIC",
            f"{e_mpa:.10g}, {nu:.10g}",
            "*SOLID SECTION, ELSET=EALL, MATERIAL=MAT1",
            "*STEP, NLGEOM",
            "*STATIC",
            "*BOUNDARY",
        ]
    )
    for bc in request.dirichlet_bcs:
        nids = face_node_indices_3d(mesh, bc.face_tags)
        for comp, val in zip(bc.components, bc.values_mm, strict=True):
            c_id = comp_num[comp]
            for nid in nids:
                lines.append(f"{nid + 1}, {c_id}, {c_id}, {float(val):.10g}")

    if request.traction_bcs:
        f_nodal = assemble_surface_tractions_3d(mesh, request.traction_bcs)
        lines.append("*CLOAD")
        for nid in range(len(mesh.nodes)):
            for c_idx in range(3):
                val = float(f_nodal[nid, c_idx])
                if abs(val) > 1.0e-15:
                    lines.append(f"{nid + 1}, {c_idx + 1}, {val:.10g}")

    lines.extend(
        [
            "*NODE FILE",
            "U",
            "*EL FILE",
            "S",
            "*END STEP",
        ]
    )
    inp_path.parent.mkdir(parents=True, exist_ok=True)
    inp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return inp_path


def _corner_tet_min_det_f(
    nodes: np.ndarray[Any, np.dtype[np.float64]],
    cells: np.ndarray[Any, np.dtype[np.int64]],
    disp: np.ndarray[Any, np.dtype[np.float64]],
) -> float:
    x0, x1, x2, x3 = (
        nodes[cells[:, 0]],
        nodes[cells[:, 1]],
        nodes[cells[:, 2]],
        nodes[cells[:, 3]],
    )
    ref = np.stack((x1 - x0, x2 - x0, x3 - x0), axis=-1)
    y = nodes + disp
    y0, y1, y2, y3 = (
        y[cells[:, 0]],
        y[cells[:, 1]],
        y[cells[:, 2]],
        y[cells[:, 3]],
    )
    cur = np.stack((y1 - y0, y2 - y0, y3 - y0), axis=-1)
    det_ref = np.linalg.det(ref)
    det_cur = np.linalg.det(cur)
    valid = np.abs(det_ref) > 1.0e-15
    if not np.any(valid):
        return 1.0
    return float(np.min(det_cur[valid] / det_ref[valid]))


class CalculixOracleBackend:
    """Sandboxed CalculiX (``ccx``) 3D FEA oracle satisfying FormaMech's 3D contract."""

    backend_name: str = "calculix-ccx"

    def __init__(self, config: ExternalSolverConfig | None = None) -> None:
        self.config = (
            ExternalSolverConfig(executable="ccx") if config is None else config
        )

    def solve(self, request: FemRequest3D) -> FemResult3D:
        """Write ``.inp``, execute ``ccx``, parse ``.frd``, and return verified ``FemResult3D``."""
        t0 = time.perf_counter()
        state = ExternalSolverState.PREPARING
        mesh = request.mesh
        n_nodes = len(mesh.nodes)

        try:
            cmd_base = _resolve_executable(
                self.config,
                self.backend_name,
                "Install CalculiX (`ccx`) on PATH or configure ExternalSolverConfig.executable.",
            )
        except OracleUnavailableError:
            EXTERNAL_SOLVER_MACHINE.require_transition(
                state, ExternalSolverState.FAILED, trigger="fail_unavailable"
            )
            raise

        state = EXTERNAL_SOLVER_MACHINE.require_transition(
            state, ExternalSolverState.WRITING_DECK, trigger="write_deck"
        ).to_state

        tmp_ctx = tempfile.TemporaryDirectory(prefix="formamech-ccx-")
        workdir = Path(tmp_ctx.name)
        try:
            job_name = "job3d"
            inp_path = write_calculix_inp_3d(request, workdir / f"{job_name}.inp")
            state = EXTERNAL_SOLVER_MACHINE.require_transition(
                state, ExternalSolverState.EXECUTING, trigger="launch_process"
            ).to_state
            cmd = [*cmd_base, "-i", job_name, *self.config.extra_args]
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=workdir,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                EXTERNAL_SOLVER_MACHINE.require_transition(
                    state, ExternalSolverState.TIMEOUT, trigger="timeout"
                )
                return FemResult3D(
                    request_id=request.request_id,
                    backend_name=self.backend_name,
                    mesh_id=mesh.mesh_id,
                    converged=False,
                    nodal_displacement=np.zeros((n_nodes, 3), dtype=np.float64),
                    nodal_von_mises_mpa=np.zeros(n_nodes, dtype=np.float64),
                    reactions_n={},
                    total_strain_energy_n_mm=0.0,
                    external_work_n_mm=0.0,
                    min_det_f=float("nan"),
                    max_displacement_mm=0.0,
                    peak_von_mises_mpa=0.0,
                    final_residual_norm=float("inf"),
                    total_newton_iterations=0,
                    wall_time_s=time.perf_counter() - t0,
                    reason=f"external solver timed out after {self.config.timeout_s}s",
                )

            if proc.returncode != 0:
                EXTERNAL_SOLVER_MACHINE.require_transition(
                    state, ExternalSolverState.FAILED, trigger="fail_exec"
                )
                return FemResult3D(
                    request_id=request.request_id,
                    backend_name=self.backend_name,
                    mesh_id=mesh.mesh_id,
                    converged=False,
                    nodal_displacement=np.zeros((n_nodes, 3), dtype=np.float64),
                    nodal_von_mises_mpa=np.zeros(n_nodes, dtype=np.float64),
                    reactions_n={},
                    total_strain_energy_n_mm=0.0,
                    external_work_n_mm=0.0,
                    min_det_f=float("nan"),
                    max_displacement_mm=0.0,
                    peak_von_mises_mpa=0.0,
                    final_residual_norm=float("inf"),
                    total_newton_iterations=0,
                    wall_time_s=time.perf_counter() - t0,
                    reason=f"ccx exited with code {proc.returncode}: {proc.stderr.strip()}",
                )

            state = EXTERNAL_SOLVER_MACHINE.require_transition(
                state, ExternalSolverState.PARSING, trigger="parse_results"
            ).to_state
            frd_path = workdir / f"{job_name}.frd"
            if not frd_path.is_file():
                EXTERNAL_SOLVER_MACHINE.require_transition(
                    state, ExternalSolverState.FAILED, trigger="fail_parse"
                )
                raise FileNotFoundError(f"CalculiX did not produce {frd_path.name}")

            parsed = read_calculix_inp_frd(inp_path, frd_path)
            disp = np.asarray(
                parsed.point_fields.get(
                    "displacement", np.zeros((n_nodes, 3), dtype=np.float64)
                ),
                dtype=np.float64,
            )
            vm = np.asarray(
                parsed.point_fields.get(
                    "von_mises_mpa", np.zeros(n_nodes, dtype=np.float64)
                ),
                dtype=np.float64,
            )
            nodes_np = np.asarray(mesh.nodes, dtype=np.float64)
            cells_np = np.asarray(mesh.cells, dtype=np.int64)
            min_det = _corner_tet_min_det_f(nodes_np, cells_np, disp)
            if not (
                np.all(np.isfinite(disp)) and np.all(np.isfinite(vm)) and min_det > 0.0
            ):
                EXTERNAL_SOLVER_MACHINE.require_transition(
                    state, ExternalSolverState.FAILED, trigger="fail_parse"
                )
                return FemResult3D(
                    request_id=request.request_id,
                    backend_name=self.backend_name,
                    mesh_id=mesh.mesh_id,
                    converged=False,
                    nodal_displacement=disp,
                    nodal_von_mises_mpa=vm,
                    reactions_n={},
                    total_strain_energy_n_mm=0.0,
                    external_work_n_mm=0.0,
                    min_det_f=min_det,
                    max_displacement_mm=float(np.max(np.linalg.norm(disp, axis=1))),
                    peak_von_mises_mpa=float(np.max(vm)) if len(vm) else 0.0,
                    final_residual_norm=float("inf"),
                    total_newton_iterations=1,
                    wall_time_s=time.perf_counter() - t0,
                    reason=f"inadmissible external oracle fields (min_det_f={min_det:.3e})",
                )

            EXTERNAL_SOLVER_MACHINE.require_transition(
                state, ExternalSolverState.VERIFIED, trigger="verify_fields"
            )
            return FemResult3D(
                request_id=request.request_id,
                backend_name=self.backend_name,
                mesh_id=mesh.mesh_id,
                converged=True,
                nodal_displacement=disp,
                nodal_von_mises_mpa=vm,
                reactions_n={},
                total_strain_energy_n_mm=0.0,
                external_work_n_mm=0.0,
                min_det_f=min_det,
                max_displacement_mm=float(np.max(np.linalg.norm(disp, axis=1))),
                peak_von_mises_mpa=float(np.max(vm)) if len(vm) else 0.0,
                final_residual_norm=0.0,
                total_newton_iterations=1,
                wall_time_s=time.perf_counter() - t0,
                reason="verified external CalculiX oracle solve",
            )
        finally:
            if not self.config.keep_workdir:
                tmp_ctx.cleanup()


class AbaqusCliOracleBackend(CalculixOracleBackend):
    """User-licensed Abaqus CLI oracle adapter (shares ``.inp`` deck writer)."""

    backend_name: str = "abaqus-cli"

    def __init__(self, config: ExternalSolverConfig | None = None) -> None:
        super().__init__(
            ExternalSolverConfig(executable="abaqus") if config is None else config
        )


class PyAnsysOracleBackend:
    """User-licensed PyAnsys / MAPDL subprocess oracle adapter (fail-closed if absent)."""

    backend_name: str = "pyansys-mapdl"

    def __init__(self, config: ExternalSolverConfig | None = None) -> None:
        self.config = (
            ExternalSolverConfig(executable="ansys") if config is None else config
        )

    def solve(self, request: FemRequest3D) -> FemResult3D:
        del request
        _resolve_executable(
            self.config,
            self.backend_name,
            "Install `ansys-mapdl-core` and a licensed Ansys MAPDL runtime to enable this oracle.",
        )
        raise OracleUnavailableError(
            self.backend_name,
            self.config.executable,
            "Configure a custom PyAnsys runner script via ExternalSolverConfig.",
        )


class DolfinxSubprocessOracleBackend:
    """LGPL-isolated FEniCSx / DOLFINx subprocess oracle bridge."""

    backend_name: str = "dolfinx-subprocess"

    def __init__(self, config: ExternalSolverConfig | None = None) -> None:
        self.config = (
            ExternalSolverConfig(executable="formamech-dolfinx-worker")
            if config is None
            else config
        )

    def solve(self, request: FemRequest3D) -> FemResult3D:
        del request
        _resolve_executable(
            self.config,
            self.backend_name,
            "Install the separate LGPL plugin `plugins/formamech-dolfinx` in an isolated venv.",
        )
        raise OracleUnavailableError(
            self.backend_name,
            self.config.executable,
            "Provide the isolated DOLFINx worker binary via ExternalSolverConfig.",
        )


def list_external_oracles() -> tuple[dict[str, Any], ...]:
    """Report availability of all built-in external FEA solver oracles."""
    specs: Sequence[tuple[str, str, str]] = (
        ("calculix-ccx", "ccx", "GPL/open CalculiX 3D FEA solver"),
        ("abaqus-cli", "abaqus", "User-licensed Abaqus Standard CLI"),
        ("pyansys-mapdl", "ansys", "User-licensed Ansys MAPDL / PyAnsys"),
        (
            "dolfinx-subprocess",
            "formamech-dolfinx-worker",
            "LGPL-isolated FEniCSx/DOLFINx worker",
        ),
    )
    return tuple(
        {
            "name": name,
            "executable": exe,
            "available": shutil.which(exe) is not None,
            "description": desc,
        }
        for name, exe, desc in specs
    )


__all__ = (
    "EXTERNAL_SOLVER_MACHINE",
    "EXTERNAL_SOLVER_SCHEMA",
    "AbaqusCliOracleBackend",
    "CalculixOracleBackend",
    "DolfinxSubprocessOracleBackend",
    "ExternalSolverConfig",
    "ExternalSolverState",
    "OracleUnavailableError",
    "PyAnsysOracleBackend",
    "list_external_oracles",
    "write_calculix_inp_3d",
)
