"""Reproducible 3D scaling, foundation transfer, OOD & parity benchmark suite (V4-2)."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import torch

from piginot.cad.importing_3d import (
    classify_faces_by_bbox,
    import_cad_3d,
)
from piginot.cad.parametric_3d import generate_3d_holed_block_step
from piginot.contracts import MaterialSpec, Quantity, Unit, canonical_json_bytes
from piginot.fem.backend_3d import (
    DirichletBC3D,
    FemRequest3D,
    ScikitFemBackend3D,
    TorchFemBackend3D,
    TractionBC3D,
    analytical_3d_uniaxial_hyperelastic,
)
from piginot.fem.linear_solvers import (
    LinearSolverConfig,
    LinearSolverMethod,
    is_pyamg_available,
)
from piginot.geometry.mesh3d import CellType3D, make_brick_tet4_mesh
from piginot.operators.geometry_3d import (
    GeometryOODDetector3D,
    compute_geometry_descriptor_3d,
)
from piginot.operators.low_rank_adapter_3d import LowRankAdapter3DConfig
from piginot.training.foundation_pretrain import (
    FoundationPretrainConfig,
    adapt_foundation_to_3d_part,
    default_foundation_checkpoint_dir,
    pretrain_struct3d_foundation,
)

BENCHMARK_REPORT_SCHEMA: Final = "formamech.benchmark-report-v1"
BenchmarkProfile = Literal["quick", "full"]


@dataclass(frozen=True, slots=True)
class BenchmarkReport3D:
    """Canonical reproducible benchmark report with JSON & Markdown rendering."""

    profile: str
    solver_scaling: tuple[dict[str, Any], ...]
    foundation_transfer: dict[str, Any]
    ood_stress: dict[str, Any]
    cross_backend_parity: dict[str, Any]
    total_wall_time_s: float

    def document(self) -> dict[str, Any]:
        return {
            "schema": BENCHMARK_REPORT_SCHEMA,
            "profile": self.profile,
            "solver_scaling": list(self.solver_scaling),
            "foundation_transfer": dict(self.foundation_transfer),
            "ood_stress": dict(self.ood_stress),
            "cross_backend_parity": dict(self.cross_backend_parity),
            "total_wall_time_s": round(float(self.total_wall_time_s), 4),
        }

    @property
    def report_hash(self) -> str:
        """Deterministic content hash excluding volatile wall-clock timings."""
        stable_scaling = [
            {k: v for k, v in row.items() if k != "wall_time_s"}
            for row in self.solver_scaling
        ]
        stable_ft = {
            k: v for k, v in self.foundation_transfer.items() if k != "wall_time_s"
        }
        payload = {
            "schema": BENCHMARK_REPORT_SCHEMA,
            "profile": self.profile,
            "solver_scaling": stable_scaling,
            "foundation_transfer": stable_ft,
            "ood_stress": dict(self.ood_stress),
            "cross_backend_parity": dict(self.cross_backend_parity),
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    def to_markdown(self) -> str:
        lines = [
            f"# FormaMech 3D Benchmark Report (`profile={self.profile}`)",
            "",
            f"- **Report SHA-256:** `{self.report_hash}`",
            f"- **Total Wall Time:** `{self.total_wall_time_s:.2f} s`",
            "",
            "## B1 — 3D Linear Solver Scaling",
            "",
            "| Method | DOFs | Converged | Newton Iters | Krylov Iters | Fallbacks | Wall Time (s) |",
            "| :--- | ---: | :---: | ---: | ---: | ---: | ---: |",
        ]
        for row in self.solver_scaling:
            lines.append(
                f"| `{row['method']}` | {row['dofs']} | {row['converged']} | "
                f"{row['newton_iterations']} | {row.get('total_krylov_iterations', 0)} | "
                f"{row.get('fallback_count', 0)} | {row['wall_time_s']:.4f} |"
            )
        ft = self.foundation_transfer
        init_l = (
            "N/A"
            if ft.get("initial_loss") is None
            else f"{float(ft['initial_loss']):.4f}"
        )
        final_l = (
            "N/A" if ft.get("final_loss") is None else f"{float(ft['final_loss']):.4f}"
        )
        step_frac = float(ft.get("step_budget_fraction", 0.0))
        lines.extend(
            [
                "",
                f"## B2 — 3D Foundation LoRA Transfer (`{ft.get('checkpoint_dir', 'unknown')}`)",
                "",
                f"- **Adapter State:** `{ft.get('state', 'UNKNOWN')}`",
                f"- **Step Budget Fraction:** `{step_frac:.4f}` (`<= 0.10` budget)",
                f"- **Trainable Parameter Fraction:** `{float(ft['trainable_fraction']):.4f}`",
                f"- **Initial -> Final Loss:** `{init_l}` -> `{final_l}`",
                f"- **Adaptation Wall Time:** `{float(ft['wall_time_s']):.3f} s`",
                "",
                "## B3 — Geometry OOD Refusal Stress Check",
                "",
                f"- **In-Envelope Accept Rate:** `{self.ood_stress['in_envelope_accept_rate']:.2%}`",
                f"- **Out-of-Envelope Refusal Rate:** `{self.ood_stress['ood_refusal_rate']:.2%}`",
                "",
                "## B4 — Cross-Backend 3D Parity (`torch-fem-3d` vs `scikit-fem-3d` vs Analytical)",
                "",
                f"- **Max Rel Error vs Analytical (`torch-fem-3d`):** `{self.cross_backend_parity['torch_vs_analytical_rel']:.3e}`",
                f"- **Max Rel Error vs Analytical (`scikit-fem-3d`):** `{self.cross_backend_parity.get('skfem_vs_analytical_rel', 0.0):.3e}`",
                f"- **Max Rel Error (`torch-fem-3d` vs `scikit-fem-3d`):** `{self.cross_backend_parity['torch_vs_skfem_rel']:.3e}`",
            ]
        )
        return "\n".join(lines) + "\n"


def run_benchmark_suite_3d(
    work_dir: Path | str,
    *,
    profile: BenchmarkProfile = "quick",
    foundation_dir: Path | str | None = None,
) -> BenchmarkReport3D:
    """Execute reproducible B1–B4 benchmarks and return a canonical report."""
    if profile not in ("quick", "full"):
        raise ValueError(
            f"unsupported benchmark profile {profile!r}; expected 'quick' or 'full'"
        )
    t_start = time.perf_counter()
    root = Path(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    # B1: 3D Solver Scaling
    nx, ny, nz = (3, 2, 2) if profile == "quick" else (5, 3, 3)
    mesh_b1 = make_brick_tet4_mesh(20.0, 8.0, 8.0, nx=nx, ny=ny, nz=nz)
    mat = MaterialSpec(
        name="bench-nh",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(100.0, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(0.30, Unit.ONE).to_canonical(),
        },
    )
    methods: list[LinearSolverMethod] = [
        LinearSolverMethod.DIRECT,
        LinearSolverMethod.PCG_SSOR,
        LinearSolverMethod.PCG_ILU,
    ]
    if is_pyamg_available():
        methods.append(LinearSolverMethod.PCG_AMG)

    backend_t = TorchFemBackend3D()
    b1_rows: list[dict[str, Any]] = []
    for method in methods:
        req = FemRequest3D(
            request_id=f"bench-b1-{method.value}",
            mesh=mesh_b1,
            material=mat,
            dirichlet_bcs=(
                DirichletBC3D("fix", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
            ),
            traction_bcs=(TractionBC3D("pull", ("x_max",), (1.0, 0.0, 0.0)),),
            load_steps=1,
            linear_solver=LinearSolverConfig(method=method, rtol=1.0e-9),
        )
        res = backend_t.solve(req)
        lstats = res.linear_solver_stats or {}
        b1_rows.append(
            {
                "method": method.value,
                "dofs": len(mesh_b1.nodes) * 3,
                "converged": res.converged,
                "newton_iterations": res.total_newton_iterations,
                "total_krylov_iterations": int(
                    lstats.get("total_krylov_iterations", 0)
                ),
                "fallback_count": int(lstats.get("fallback_count", 0)),
                "methods_used": list(lstats.get("methods_used", [method.value])),
                "wall_time_s": round(float(res.wall_time_s), 5),
                "max_displacement_mm": round(float(res.max_displacement_mm), 6),
            }
        )

    # B2: Foundation 3D LoRA transfer
    if foundation_dir is not None:
        f_dir = Path(foundation_dir).resolve()
        if not (f_dir / "foundation.json").is_file():
            raise FileNotFoundError(
                f"explicit foundation_dir does not contain foundation.json: {f_dir}"
            )
    else:
        f_dir = default_foundation_checkpoint_dir()
        if not (f_dir / "foundation.json").is_file():
            f_dir = root / "tmp_foundation"
            pretrain_struct3d_foundation(
                f_dir,
                config=FoundationPretrainConfig(
                    samples_per_family=2, mesh_size_mm=10.0, epochs=4, hidden_dim=24
                ),
            )
    step_path = generate_3d_holed_block_step(
        root / "bench_target.step", lx=28.0, ly=14.0, lz=10.0, hole_radius=3.0
    )
    imp = import_cad_3d(step_path, mesh_size_mm=9.0, element_type=CellType3D.TET4)
    faces = classify_faces_by_bbox(imp)
    max_adapt_steps = 6 if profile == "quick" else 12
    ref_steps = 120
    _, ad_res = adapt_foundation_to_3d_part(
        f_dir,
        mesh=imp.mesh,
        material=mat,
        clamped_face_tags=tuple(faces["x_min"]),
        loaded_face_tags=tuple(faces["x_max"]),
        traction_mpa=(0.35, 0.0, 0.0),
        adapter_config=LowRankAdapter3DConfig(
            rank=4,
            max_steps=max_adapt_steps,
            from_scratch_reference_steps=ref_steps,
            allow_ood_adaptation=True,
        ),
    )
    b2_doc = {
        "checkpoint_dir": str(f_dir.name),
        "state": ad_res.state.value,
        "step_budget_fraction": round(max_adapt_steps / float(ref_steps), 6),
        "trainable_fraction": round(
            ad_res.adapter_parameter_count / max(ad_res.base_parameter_count, 1), 6
        ),
        "initial_loss": (
            float(ad_res.initial_loss) if math.isfinite(ad_res.initial_loss) else None
        ),
        "final_loss": (
            float(ad_res.final_loss) if math.isfinite(ad_res.final_loss) else None
        ),
        "wall_time_s": round(float(ad_res.wall_time_s), 5),
    }

    # B3: Geometry OOD stress test
    train_descs = [
        compute_geometry_descriptor_3d(
            make_brick_tet4_mesh(lx, 8.0, 8.0, nx=2, ny=1, nz=1)
        )
        for lx in (18.0, 20.0, 22.0, 24.0)
    ]
    det = GeometryOODDetector3D.fit(train_descs)
    in_probes = [
        compute_geometry_descriptor_3d(
            make_brick_tet4_mesh(lx, 8.0, 8.0, nx=2, ny=1, nz=1)
        )
        for lx in (19.0, 21.0, 23.0)
    ]
    ood_probes = [
        compute_geometry_descriptor_3d(
            make_brick_tet4_mesh(160.0, 2.0, 2.0, nx=2, ny=1, nz=1)
        ),
        compute_geometry_descriptor_3d(
            make_brick_tet4_mesh(2.0, 80.0, 80.0, nx=2, ny=1, nz=1)
        ),
    ]
    in_acc = sum(1 for d in in_probes if det.score(d).accepted) / len(in_probes)
    ood_ref = sum(1 for d in ood_probes if not det.score(d).accepted) / len(ood_probes)
    b3_doc = {
        "in_envelope_accept_rate": float(in_acc),
        "ood_refusal_rate": float(ood_ref),
    }

    # B4: Cross-backend parity vs analytical uniaxial finite-strain patch test
    mesh_p = make_brick_tet4_mesh(10.0, 5.0, 5.0, nx=2, ny=1, nz=1)
    lam_x = 1.05
    an = analytical_3d_uniaxial_hyperelastic(mat, lam_x)
    p11_an = an["axial_first_piola_mpa"]
    lam_t_an = an["transverse_stretch"]
    req_p = FemRequest3D(
        request_id="bench-b4-parity",
        mesh=mesh_p,
        material=mat,
        dirichlet_bcs=(
            DirichletBC3D("x0", ("x_min",), ("x",), (0.0,)),
            DirichletBC3D("y0", ("y_min",), ("y",), (0.0,)),
            DirichletBC3D("z0", ("z_min",), ("z",), (0.0,)),
            DirichletBC3D("x1", ("x_max",), ("x",), ((lam_x - 1.0) * 10.0,)),
        ),
        load_steps=2,
    )
    res_t = backend_t.solve(req_p)
    res_s = ScikitFemBackend3D().solve(req_p)
    if not (res_t.converged and res_s.converged):
        raise RuntimeError(
            f"B4 cross-backend parity solve failed "
            f"(torch_converged={res_t.converged}, skfem_converged={res_s.converged})"
        )
    nodes_p = np.asarray(mesh_p.nodes, dtype=np.float64)
    u_exact = np.column_stack(
        (
            (lam_x - 1.0) * nodes_p[:, 0],
            (lam_t_an - 1.0) * nodes_p[:, 1],
            (lam_t_an - 1.0) * nodes_p[:, 2],
        )
    )
    denom = max(float(np.linalg.norm(u_exact)), 1.0e-12)
    b4_doc = {
        "analytical_p11_mpa": round(float(p11_an), 6),
        "torch_converged": bool(res_t.converged),
        "skfem_converged": bool(res_s.converged),
        "torch_vs_analytical_rel": float(
            np.linalg.norm(res_t.nodal_displacement - u_exact) / denom
        ),
        "skfem_vs_analytical_rel": float(
            np.linalg.norm(res_s.nodal_displacement - u_exact) / denom
        ),
        "torch_vs_skfem_rel": float(
            np.linalg.norm(res_t.nodal_displacement - res_s.nodal_displacement) / denom
        ),
    }

    # Silence unused torch warning if dtype not referenced directly
    _ = torch.float32
    return BenchmarkReport3D(
        profile=profile,
        solver_scaling=tuple(b1_rows),
        foundation_transfer=b2_doc,
        ood_stress=b3_doc,
        cross_backend_parity=b4_doc,
        total_wall_time_s=time.perf_counter() - t_start,
    )


__all__ = (
    "BENCHMARK_REPORT_SCHEMA",
    "BenchmarkProfile",
    "BenchmarkReport3D",
    "run_benchmark_suite_3d",
)
