"""Stateful 3D Cyclic Elastoplasticity Load-Path Editor & Runner (V3-5)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from piginot.cad.importing_3d import (
    classify_faces_by_bbox,
    generate_3d_box_step,
    import_cad_3d,
)
from piginot.fem.backend_3d import DirichletBC3D
from piginot.geometry.mesh3d import CellType3D
from piginot.stateful.fem_stepper_3d import (
    LoadStep3D,
    Stateful3DTrajectory,
    solve_incremental_fem_3d_trajectory,
)
from piginot.stateful.j2_advanced import (
    FiniteStrainJ2Config,
    FiniteStrainJ2Plasticity,
    J2CombinedHardeningConfig,
    J2CombinedHardeningPlasticity,
)


@dataclass(frozen=True, slots=True)
class CyclicPathSummary:
    """Headless summary of a 3D stateful cyclic elastoplastic rollout."""

    model_kind: str
    all_converged: bool
    total_dissipation: float
    min_dissipation: float
    step_table: tuple[dict[str, Any], ...]


def run_stateful_3d_cyclic_preview(
    work_dir: Path,
    *,
    ux_sequence_mm: Sequence[float] = (0.05, 0.25, 0.40, 0.10),
    model_kind: str = "j2_combined",
    youngs_modulus_mpa: float = 20_000.0,
    yield_stress_mpa: float = 250.0,
) -> tuple[Stateful3DTrajectory, CyclicPathSummary]:
    """Run a deterministic 3D cyclic J2 trajectory for workbench visualization."""
    work_dir.mkdir(parents=True, exist_ok=True)
    step = generate_3d_box_step(work_dir / "cyclic_box.step", lx=10.0, ly=5.0, lz=5.0)
    imp = import_cad_3d(step, mesh_size_mm=7.0, element_type=CellType3D.TET10)
    faces = classify_faces_by_bbox(imp)

    model: J2CombinedHardeningPlasticity | FiniteStrainJ2Plasticity
    if model_kind == "finite_strain_j2":
        model = FiniteStrainJ2Plasticity(
            FiniteStrainJ2Config(
                youngs_modulus_mpa=youngs_modulus_mpa,
                poisson_ratio=0.30,
                yield_stress_mpa=yield_stress_mpa,
                linear_hardening_mpa=800.0,
            )
        )
    else:
        model = J2CombinedHardeningPlasticity(
            J2CombinedHardeningConfig(
                youngs_modulus_mpa=youngs_modulus_mpa,
                poisson_ratio=0.30,
                yield_stress_mpa=yield_stress_mpa,
            )
        )

    loads = [
        LoadStep3D(
            step_index=i,
            time_s=0.1 * (i + 1),
            dt_s=0.1,
            dirichlet_bcs=(
                DirichletBC3D(
                    "fix", tuple(faces["x_min"]), ("x", "y", "z"), (0.0, 0.0, 0.0)
                ),
                DirichletBC3D("pull", tuple(faces["x_max"]), ("x",), (float(ux),)),
            ),
        )
        for i, ux in enumerate(ux_sequence_mm)
    ]
    traj = solve_incremental_fem_3d_trajectory(
        imp.mesh, model, loads, solver_tolerance=1.0e-6
    )
    rows = tuple(
        {
            "step": s.step_index,
            "ux_mm": float(ux_sequence_mm[s.step_index]),
            "converged": s.converged,
            "newton_iters": s.newton_iterations,
            "dissipation": round(s.dissipation_total, 6),
            "min_det_f": round(s.min_det_f, 5),
            "state_blake2b": s.state_blake2b[:12],
        }
        for s in traj.steps
    )
    summary = CyclicPathSummary(
        model_kind=model_kind,
        all_converged=all(s.converged for s in traj.steps),
        total_dissipation=float(sum(s.dissipation_total for s in traj.steps)),
        min_dissipation=float(
            min((s.dissipation_total for s in traj.steps), default=0.0)
        ),
        step_table=rows,
    )
    return traj, summary


def render_stateful_cyclic_mode(st: Any, workspace_root: Path) -> None:
    """Render the Streamlit Stateful Cyclic elastoplasticity workflow mode."""
    st.title("Stateful 3D Cyclic Plasticity Studio")
    c1, c2, c3 = st.columns(3)
    model_kind = c1.selectbox(
        "Constitutive Model",
        options=["j2_combined", "finite_strain_j2"],
        index=0,
        key="wb3d_cyc_model",
    )
    e_mpa = float(
        c2.number_input("Young's Modulus (MPa)", value=20_000.0, min_value=10.0)
    )
    sig_y = float(c3.number_input("Yield Stress (MPa)", value=250.0, min_value=1.0))
    seq_str = st.text_input(
        "Displacement Load Path ux (mm, comma-separated)",
        value="0.05, 0.25, 0.40, 0.10",
    )
    if st.button("Run Incremental 3D Stateful FEM", type="primary"):
        try:
            seq = [float(x.strip()) for x in seq_str.split(",") if x.strip()]
            _, summary = run_stateful_3d_cyclic_preview(
                workspace_root / "_cyclic_preview",
                ux_sequence_mm=seq,
                model_kind=model_kind,
                youngs_modulus_mpa=e_mpa,
                yield_stress_mpa=sig_y,
            )
            st.metric("All Steps Converged", str(summary.all_converged))
            st.metric("Total Plastic Dissipation", f"{summary.total_dissipation:.4f}")
            st.table(list(summary.step_table))
        except Exception as exc:
            st.error(f"Stateful rollout failed: {exc}")


__all__ = (
    "CyclicPathSummary",
    "render_stateful_cyclic_mode",
    "run_stateful_3d_cyclic_preview",
)
