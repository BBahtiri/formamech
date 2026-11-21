"""Publication-grade contribution plots for the FormaMech viscoelastic CAD showcase."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore[import-untyped]

PALETTE = {
    "navy": "#1B365D",
    "teal": "#008080",
    "amber": "#D97706",
    "crimson": "#DC2626",
    "emerald": "#059669",
    "slate": "#475569",
    "purple": "#7C3AED",
    "light_bg": "#F8FAFC",
}


def _save_fig(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor="white")
    return path


def _plot_boundary_edges(
    ax: Any,
    nodes: np.ndarray[Any, Any],
    facets: np.ndarray[Any, Any],
    semantic_tags: Mapping[str, tuple[int, ...]],
    *,
    lw: float = 2.4,
) -> None:
    colors = {
        "left_clamp": PALETTE["crimson"],
        "right_load": PALETTE["amber"],
        "notch_free": PALETTE["purple"],
        "left_hole": PALETTE["teal"],
        "right_hole": PALETTE["emerald"],
        "top_free": PALETTE["navy"],
        "bottom_free": PALETTE["slate"],
    }
    labels_done: set[str] = set()
    for tag, f_ids in sorted(semantic_tags.items()):
        c = colors.get(tag, "#333333")
        for f_idx in f_ids:
            a, b = int(facets[f_idx, 0]), int(facets[f_idx, 1])
            lbl = tag.replace("_", " ") if tag not in labels_done else None
            labels_done.add(tag)
            ax.plot(
                [nodes[a, 0], nodes[b, 0]],
                [nodes[a, 1], nodes[b, 1]],
                color=c,
                linewidth=lw,
                label=lbl,
            )


def plot_stage1_cad_and_hierarchy(data: Mapping[str, Any], out_path: Path) -> Path:
    """Plot 1: CAD STEP geometry, M0 vs M1 MeshPolicy meshes, & geometric convergence."""
    fig = Figure(figsize=(15.5, 4.8), layout="constrained", facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)

    # Panel A: CAD Geometry & Semantic Boundaries
    m0_nodes = np.asarray(data["m0_nodes"], dtype=float)
    m0_cells = np.asarray(data["m0_cells"], dtype=int)
    m0_facets = np.asarray(data["m0_facets"], dtype=int)
    tri0 = Triangulation(m0_nodes[:, 0], m0_nodes[:, 1], triangles=m0_cells)
    ax0 = axes[0]
    ax0.triplot(tri0, color="#CBD5E1", linewidth=0.55)
    _plot_boundary_edges(ax0, m0_nodes, m0_facets, data["semantic_tags"], lw=2.6)
    ax0.set(
        title="A. OpenCASCADE STEP Import & Boundaries\n(Genus-2: Relief Notch + 2 Holes)",
        xlabel="x (mm)",
        ylabel="y (mm)",
        aspect="equal",
    )
    ax0.legend(loc="lower center", ncol=2, fontsize=7.5, framealpha=0.92)

    # Panel B: M0 vs M1 Hierarchical Refinement with Local MeshPolicy Sizing
    m1_nodes = np.asarray(data["m1_nodes"], dtype=float)
    m1_cells = np.asarray(data["m1_cells"], dtype=int)
    tri1 = Triangulation(m1_nodes[:, 0], m1_nodes[:, 1], triangles=m1_cells)
    ax1 = axes[1]
    ax1.triplot(tri1, color=PALETTE["teal"], linewidth=0.45, alpha=0.85)
    ax1.set(
        title=(
            f"B. Verified M1 Mesh Hierarchy\n"
            f"(M0: {len(m0_cells)} -> M1: {len(m1_cells)} -> M2: {data['m2_cell_count']} tris)"
        ),
        xlabel="x (mm)",
        ylabel="y (mm)",
        aspect="equal",
    )

    # Panel C: Distinct-Remesh Geometric Convergence
    ax2 = axes[2]
    conv = data["geometry_convergence"]
    sizes = [float(r["mesh_size_mm"]) for r in conv]
    area_err_pct = [float(r["area_rel_error_pct"]) for r in conv]
    chord_err = [float(r["max_chord_error_mm"]) for r in conv]
    ax2.plot(
        sizes,
        area_err_pct,
        "o-",
        color=PALETTE["navy"],
        linewidth=2.2,
        markersize=7,
        label="Area error vs analytical CAD (%)",
    )
    ax2_r = ax2.twinx()
    ax2_r.plot(
        sizes,
        chord_err,
        "s--",
        color=PALETTE["amber"],
        linewidth=2.2,
        markersize=7,
        label="Max curve chord error (mm)",
    )
    ax2.invert_xaxis()
    ax2.set_xlabel("Target Mesh Size h (mm)  [finer ->]")
    ax2.set_ylabel("Area Error (%)", color=PALETTE["navy"])
    ax2_r.set_ylabel("Max Chord Error (mm)", color=PALETTE["amber"])
    ax2.set_title(
        "C. Distinct-Remesh Geometric Convergence\n(Contribution: Quantified CAD Fidelity)"
    )
    ax2.grid(True, linestyle=":", alpha=0.5)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)

    fig.suptitle(
        "Stage 1 — CAD Generation (STEP), Local MeshPolicy & Verified M0/M1/M2 Hierarchy",
        fontsize=13.5,
        fontweight="bold",
    )
    return _save_fig(fig, out_path)


def plot_stage2_viscoelastic_p6(data: Mapping[str, Any], out_path: Path) -> Path:
    """Plot 2: P6 Viscoelastic hysteresis, relaxation, dissipation & deformed stress."""
    fig = Figure(figsize=(14.5, 9.2), layout="constrained", facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)

    times = np.asarray(data["times_s"], dtype=float)
    ux = np.asarray(data["applied_ux_mm"], dtype=float)
    fem_rxn = np.asarray(data["fem_reactions_n"], dtype=float)
    op_rxn = np.asarray(data["operator_reactions_n"], dtype=float)
    guard_rxn = np.asarray(data["guarded_reactions_n"], dtype=float)
    eq_rxn = np.asarray(data["equilibrium_elastic_reactions_n"], dtype=float)
    statuses = list(data["guarded_statuses"])

    # Panel A: Hysteresis Loop (Force vs Displacement)
    ax = axes[0, 0]
    ax.plot(
        [0.0, *ux],
        [0.0, *fem_rxn],
        "o-",
        color=PALETTE["navy"],
        linewidth=2.4,
        markersize=5.5,
        label="P2 Incremental Viscoelastic FEM (Ref)",
    )
    ax.plot(
        [0.0, *ux],
        [0.0, *op_rxn],
        "s--",
        color=PALETTE["amber"],
        linewidth=1.9,
        markersize=5,
        label="Pure History-Aware Operator (P6)",
    )
    ax.plot(
        [0.0, *ux],
        [0.0, *guard_rxn],
        "^:",
        color=PALETTE["emerald"],
        linewidth=2.2,
        markersize=6,
        label="GuardedStatefulStepper (Hybrid)",
    )
    ax.plot(
        [0.0, max(ux)],
        [0.0, float(eq_rxn[int(np.argmax(ux))])],
        "-.",
        color=PALETTE["slate"],
        linewidth=1.3,
        alpha=0.75,
        label="Equilibrium Elastic Limit ($E_\\infty$, no hysteresis)",
    )
    ax.set(
        title="A. Cyclic Viscoelastic Hysteresis Loop\n(Ramp -> Hold -> Unload on Unseen Amplitude)",
        xlabel="Applied Right-Edge Displacement $u_x$ (mm)",
        ylabel="Axial Reaction Force (N/mm)",
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8)

    # Panel B: Time-Domain Stress Relaxation
    ax = axes[0, 1]
    t_full = np.concatenate(([0.0], times))
    ax.axvspan(
        0.5, 1.5, color="#FEF3C7", alpha=0.45, label="Constant-Strain Hold (Relaxation)"
    )
    ax.plot(
        t_full, [0.0, *fem_rxn], "o-", color=PALETTE["navy"], lw=2.4, label="P2 FEM Ref"
    )
    ax.plot(
        t_full,
        [0.0, *op_rxn],
        "s--",
        color=PALETTE["amber"],
        lw=1.9,
        label="Pure Operator",
    )
    ax.plot(
        t_full,
        [0.0, *guard_rxn],
        "^:",
        color=PALETTE["emerald"],
        lw=2.2,
        label="Guarded Stepper",
    )
    ax.plot(
        t_full,
        [0.0, *eq_rxn],
        "-.",
        color=PALETTE["slate"],
        lw=1.3,
        label="Elastic $E_\\infty$ Baseline",
    )
    # Highlight fallback steps
    for idx_s, st in enumerate(statuses):
        if st == "ACCEPTED_FEM_FALLBACK":
            ax.scatter(
                [times[idx_s]],
                [guard_rxn[idx_s]],
                s=110,
                facecolors="none",
                edgecolors=PALETTE["crimson"],
                linewidths=2.0,
                zorder=6,
                label="FEM Fallback Triggered"
                if "FEM Fallback Triggered" not in ax.get_legend_handles_labels()[1]
                else None,
            )
    ax.set(
        title=(
            f"B. Time-Domain Stress Relaxation\n"
            f"(Hold decay: {data['relaxation_drop_pct']:.1f}% | "
            f"Tangent FD err: {data['tangent_fd_rel_err']:.1e})"
        ),
        xlabel="Time $t$ (s)",
        ylabel="Axial Reaction Force (N/mm)",
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower center", ncol=2, fontsize=7.4, framealpha=0.92)

    # Panel C: Residual & Thermodynamic Dissipation per Step
    ax = axes[1, 0]
    op_res = np.asarray(data["operator_residuals_n"], dtype=float)
    guard_res = np.asarray(data["guarded_residuals_n"], dtype=float)
    diss = np.asarray(data["fem_step_dissipation_n_mm"], dtype=float)
    x_pos = np.arange(len(times))
    w = 0.35
    ax.bar(
        x_pos - w / 2,
        op_res,
        width=w,
        color=PALETTE["amber"],
        alpha=0.85,
        label="Pure Operator Residual (N/mm)",
    )
    ax.bar(
        x_pos + w / 2,
        guard_res,
        width=w,
        color=PALETTE["emerald"],
        alpha=0.9,
        label="Guarded Stepper Residual (N/mm)",
    )
    ax.axhline(
        float(data["hybrid_residual_tol"]),
        color=PALETTE["crimson"],
        linestyle="--",
        linewidth=1.6,
        label=f"Guard Threshold ({data['hybrid_residual_tol']:.2f})",
    )
    ax_r = ax.twinx()
    ax_r.plot(
        x_pos,
        diss,
        "D-",
        color=PALETTE["purple"],
        linewidth=1.8,
        markersize=5,
        label="Step Viscous Dissipation $\\mathcal{D} \\geq 0$",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{t:.2f}s" for t in times], fontsize=8)
    ax.set_ylabel("Discrete Equilibrium Residual")
    ax_r.set_ylabel("Viscous Dissipation (N·mm/mm)", color=PALETTE["purple"])
    ax.set_title(
        "C. Guarded Stepper Contribution & Dissipation\n"
        f"(Pure op max err: {data['pure_op_max_rxn_err_pct']:.1f}% -> "
        f"Guarded: {data['guarded_max_rxn_err_pct']:.1f}%)"
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax_r.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=7.5)

    # Panel D: Deformed Stress Concentration Field at Peak Load (t = 0.5 s)
    ax = axes[1, 1]
    nodes = np.asarray(data["peak_nodes_mm"], dtype=float)
    u_peak = np.asarray(data["peak_fem_u_mm"], dtype=float)
    cells = np.asarray(data["peak_cells"], dtype=int)
    p_norm = np.asarray(data["peak_stress_norm_mpa"], dtype=float)
    def_xy = nodes + u_peak
    tri_def = Triangulation(def_xy[:, 0], def_xy[:, 1], triangles=cells)
    img = ax.tripcolor(tri_def, p_norm, shading="gouraud", cmap="magma")
    ax.triplot(
        Triangulation(nodes[:, 0], nodes[:, 1], triangles=cells),
        color="#94A3B8",
        linewidth=0.25,
        alpha=0.35,
    )
    fig.colorbar(img, ax=ax, label="||P|| (MPa)", shrink=0.88)
    ax.set(
        title=(
            f"D. Deformed Peak-Load Stress Field ($t=0.5$ s, 1x scale)\n"
            f"(Stress concentration at top relief notch & holes; min det F={data['min_det_f']:.3f})"
        ),
        xlabel="x (mm)",
        ylabel="y (mm)",
        aspect="equal",
    )

    fig.suptitle(
        "Stage 2 — P6 Stateful Finite-Strain Viscoelasticity (2-Branch Generalized Maxwell on CAD Bracket)",
        fontsize=13.5,
        fontweight="bold",
    )
    return _save_fig(fig, out_path)


def plot_stage3_p1_p2_hierarchical_and_lora(
    data: Mapping[str, Any], out_path: Path
) -> Path:
    """Plot 3: P1 hierarchical energy gain, P2 12-ch tokens, OOD gate, & LoRA adaptation."""
    fig = Figure(figsize=(14.5, 9.0), layout="constrained", facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)

    # Panel A: P1 Hierarchical M0->M1 vs Single-Level Baseline
    ax = axes[0, 0]
    metrics = ["u Rel L2 (%)", "P Rel L2 (%)", "Energy Gap (%)"]
    base_vals = [
        data["p1_baseline"]["u_rel_l2_pct"],
        data["p1_baseline"]["p_rel_l2_pct"],
        data["p1_baseline"]["energy_err_pct"],
    ]
    hier_vals = [
        data["p1_hierarchical"]["u_rel_l2_pct"],
        data["p1_hierarchical"]["p_rel_l2_pct"],
        data["p1_hierarchical"]["energy_err_pct"],
    ]
    x = np.arange(len(metrics))
    w = 0.34
    ax.bar(
        x - w / 2,
        base_vals,
        w,
        color=PALETTE["slate"],
        label="Baseline Single-Level Energy",
    )
    ax.bar(
        x + w / 2,
        hier_vals,
        w,
        color=PALETTE["teal"],
        label="P1 Hierarchical M0->M1 Energy",
    )
    for i, (bv, hv) in enumerate(zip(base_vals, hier_vals, strict=True)):
        ax.text(i - w / 2, bv + 0.3, f"{bv:.1f}%", ha="center", fontsize=8)
        ax.text(
            i + w / 2,
            hv + 0.3,
            f"{hv:.1f}%",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Error vs P2 FEM (%)")
    ax.set_title(
        "A. P1 Hierarchical M0->M1 Energy Contribution\n(Same Optimizer Step Budget on CAD Bracket)"
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8)

    # Panel B: P2 12-Channel Boundary & Load Tokenization Map
    ax = axes[0, 1]
    b_xy = np.asarray(data["token_xy"], dtype=float)
    b_n = np.asarray(data["token_normals"], dtype=float)
    b_roles = np.asarray(data["token_role_code"], dtype=int)
    role_colors = {
        0: PALETTE["slate"],
        1: PALETTE["crimson"],
        2: PALETTE["amber"],
        3: PALETTE["purple"],
    }
    role_names = {
        0: "Traction-free",
        1: "Dirichlet clamp",
        2: "Dead traction / load",
        3: "Follower pressure",
    }
    for code in (0, 1, 2, 3):
        mask = b_roles == code
        if np.any(mask):
            ax.scatter(
                b_xy[mask, 0],
                b_xy[mask, 1],
                c=role_colors[code],
                s=22,
                label=role_names[code],
                zorder=4,
            )
    stride = max(1, len(b_xy) // 45)
    ax.quiver(
        b_xy[::stride, 0],
        b_xy[::stride, 1],
        b_n[::stride, 0],
        b_n[::stride, 1],
        color="#64748B",
        scale=18,
        width=0.004,
        alpha=0.7,
    )
    ax.set(
        title=(
            f"B. P2 12-Channel Boundary/Load Tokenization\n"
            f"({len(b_xy)} weighted tokens across 3 closed loops)"
        ),
        xlabel="x (mm)",
        ylabel="y (mm)",
        aspect="equal",
    )
    ax.legend(loc="lower center", ncol=2, fontsize=7.5)

    # Panel C: P2 Geometry OOD Detector Score Separation
    ax = axes[1, 0]
    ood_labels = list(data["ood_labels"])
    ood_scores = [float(s) for s in data["ood_scores"]]
    ood_thr = float(data["ood_threshold"])
    bar_cols = [
        PALETTE["crimson"] if s > ood_thr else PALETTE["emerald"] for s in ood_scores
    ]
    y_pos = np.arange(len(ood_labels))
    ax.barh(y_pos, ood_scores, color=bar_cols, alpha=0.88)
    ax.axvline(
        ood_thr,
        color=PALETTE["crimson"],
        linestyle="--",
        lw=1.8,
        label=f"OOD Threshold ({ood_thr:.2f})",
    )
    for i, sc in enumerate(ood_scores):
        ax.text(sc + max(ood_scores) * 0.02, i, f"{sc:.2f}", va="center", fontsize=8.5)
    ax.set_xlim(0.0, max(ood_scores) * 1.18)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ood_labels, fontsize=8.5)
    ax.set_xlabel("Geometry Descriptor Mahalanobis Distance")
    ax.set_title(
        "C. P2 Geometry OOD Gate Contribution\n(Rejects Extrapolated Topologies Before Serving)"
    )
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8)

    # Panel D: P2 Low-Rank Adapter (<=10% budget) vs Zero-Shot vs From-Scratch
    ax = axes[1, 1]
    arms = [
        "Zero-Shot\n(0 steps)",
        "P2 LoRA Adapt\n(<=10% budget)",
        "From-Scratch\n(100% budget)",
    ]
    u_errs = [
        data["p2_zero_shot"]["u_rel_l2_pct"],
        data["p2_lora"]["u_rel_l2_pct"],
        data["p2_scratch"]["u_rel_l2_pct"],
    ]
    walls = [
        data["p2_zero_shot"]["wall_s"],
        data["p2_lora"]["wall_s"],
        data["p2_scratch"]["wall_s"],
    ]
    xp = np.arange(len(arms))
    bars = ax.bar(
        xp,
        u_errs,
        width=0.48,
        color=[PALETTE["amber"], PALETTE["emerald"], PALETTE["navy"]],
        alpha=0.9,
    )
    for bar, err_v, w_s in zip(bars, u_errs, walls, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            err_v + max(u_errs) * 0.04,
            f"{err_v:.1f}%\n({w_s:.2f}s)",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.set_ylim(0.0, max(u_errs) * 1.28)
    ax.set_xticks(xp)
    ax.set_xticklabels(arms)
    ax.set_ylabel("Displacement Rel. L2 Error vs P2 FEM (%)")
    ax.set_title(
        "D. P2 Low-Rank CAD Adaptation Contribution\n"
        f"(Unseen CAD Variant; Adapter params: {data['p2_lora']['adapter_params']})"
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(
        "Stage 3 — P1 Hierarchical Energy & P2 Multi-CAD Tokenization, OOD Gate & LoRA Adaptation",
        fontsize=13.5,
        fontweight="bold",
    )
    return _save_fig(fig, out_path)


def plot_stage4_verification_funnel(data: Mapping[str, Any], out_path: Path) -> Path:
    """Plot 4: Dual-backend DWR QoI bounds, Tier 1.5 TTA, Tier 2 repair, & N* break-even."""
    fig = Figure(figsize=(14.5, 9.0), layout="constrained", facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)

    # Panel A: P4 Live DWR Adjoint Error Estimates vs True FEM QoI Errors
    ax = axes[0, 0]
    qoi_rows = data["dwr_qois"]
    q_names = [r["label"] for r in qoi_rows]
    dwr_est = [float(r["dwr_abs_err"]) for r in qoi_rows]
    true_err = [float(r["true_abs_err"]) for r in qoi_rows]
    x = np.arange(len(q_names))
    w = 0.35
    ax.bar(
        x - w / 2, true_err, w, color=PALETTE["navy"], label="True |QoI_FEM - QoI_surr|"
    )
    ax.bar(
        x + w / 2, dwr_est, w, color=PALETTE["teal"], label="P4 DWR Adjoint Estimate"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(q_names, fontsize=8.2)
    ax.set_ylabel("Absolute QoI Error (Native Units, log scale)")
    ax.set_yscale("log")
    min_q = min(min(true_err), min(dwr_est))
    max_q = max(max(true_err), max(dwr_est))
    ax.set_ylim(max(min_q * 0.5, 1e-8), max_q * 2.4)
    ax.set_title(
        "A. P4 Live DWR Adjoint QoI Attribution\n"
        f"(TorchFem vs ScikitFem Parity: {data['dual_backend_rel_diff']:.1e})"
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8)

    # Panel B: Tier 1.5 Test-Time Physics LoRA Energy & Residual Drop
    ax = axes[0, 1]
    tta_steps = list(range(len(data["tta_energy_trace"])))
    tta_e = [float(v) for v in data["tta_energy_trace"]]
    ax.plot(tta_steps, tta_e, "o-", color=PALETTE["purple"], lw=2.2, markersize=4.5)
    ax.set_xlabel("Tier 1.5 Test-Time LoRA Step (Label-Free)")
    ax.set_ylabel("Nondimensional Potential Energy")
    ax.set_title(
        "B. Tier 1.5 Test-Time Physics LoRA Contribution\n"
        f"(Energy drop: {data['tta_energy_drop_pct']:.2f}% | "
        f"FEM residual: {data['tta_res_before']:.2f} -> {data['tta_res_after']:.2f})"
    )
    ax.grid(True, linestyle=":", alpha=0.5)

    # Panel C: Tier 2 Spectrally Safe Warm-Start Repair & Iteration Reduction
    ax = axes[1, 0]
    ws_labels = [
        "Cold FEM\n(Zero Start)",
        "Warm Newton\n(Surrogate Start)",
        "Raw Folded\n(No Repair)",
        "Repaired Warm\n(Spectral Repair)",
    ]
    ws_iters = [
        int(data["cold_fem_iters"]),
        int(data["warm_newton_iters"]),
        0 if not data["raw_folded_converged"] else int(data["raw_folded_iters"]),
        int(data["repaired_warm_iters"]),
    ]
    ws_cols = [
        PALETTE["slate"],
        PALETTE["emerald"],
        PALETTE["crimson"],
        PALETTE["teal"],
    ]
    bars = ax.bar(np.arange(len(ws_labels)), ws_iters, width=0.5, color=ws_cols)
    for idx_b, (bar, it_cnt) in enumerate(zip(bars, ws_iters, strict=True)):
        txt = (
            "FAILED\n(det F < 0)"
            if (idx_b == 2 and not data["raw_folded_converged"])
            else f"{it_cnt} iters"
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(it_cnt, 0.2) + 0.15,
            txt,
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color=PALETTE["crimson"]
            if (idx_b == 2 and not data["raw_folded_converged"])
            else "#111827",
        )
    ax.set_ylim(0.0, max(ws_iters) * 1.16)
    ax.set_xticks(np.arange(len(ws_labels)))
    ax.set_xticklabels(ws_labels, fontsize=8.2)
    ax.set_ylabel("Converged Newton Iterations")
    ax.set_title(
        "C. Tier 2 Spectrally Safe Warm-Start Contribution\n"
        f"(Repair restores det F: {data['folded_init_det']:.2f} -> {data['repaired_det']:.2f})"
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    # Panel D: Amortized Break-Even Query Curve N*
    ax = axes[1, 1]
    am = data["amortization"]
    n_star = float(am["break_even_queries"])
    q_max = max(int(math.ceil(n_star * 2.2)), 25)
    qs = np.linspace(1, q_max, 100)
    fem_cost = qs * float(am["fem_seconds"])
    surr_cost = (
        float(am["training_wall_seconds"])
        + int(am["calibration_cases"]) * float(am["fem_seconds"])
        + qs * float(am["inference_seconds"])
    )
    ax.plot(
        qs,
        fem_cost,
        "-",
        color=PALETTE["slate"],
        lw=2.3,
        label="Pure FEM Cumulative Time",
    )
    ax.plot(
        qs,
        surr_cost,
        "-",
        color=PALETTE["teal"],
        lw=2.3,
        label="FormaMech (Train + Cal + Serve)",
    )
    ax.axvline(
        n_star,
        color=PALETTE["amber"],
        linestyle="--",
        lw=1.8,
        label=f"Break-Even N* = {n_star:.1f} queries",
    )
    ax.scatter(
        [n_star],
        [n_star * float(am["fem_seconds"])],
        color=PALETTE["amber"],
        s=70,
        zorder=5,
    )
    ax.set_xlabel("Production Query Count N")
    ax.set_ylabel("Cumulative Wall Time (s)")
    ax.set_title(
        "D. Amortized Break-Even Analysis\n"
        f"(Online speedup: {am['online_speedup']:.1f}x | Tier 0 gate: {data['tier0_ms']:.2f} ms)"
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Stage 4 — Three-Tier Verification Funnel, Dual-Backend Parity, Spectral Repair & Break-Even N*",
        fontsize=13.5,
        fontweight="bold",
    )
    return _save_fig(fig, out_path)


def plot_stage5_p3_active_and_p5_design(
    data: Mapping[str, Any], out_path: Path
) -> Path:
    """Plot 5: P3 DWR active learning arm comparison & P5 FEM-guarded design optimization."""
    fig = Figure(figsize=(15.5, 4.9), layout="constrained", facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)

    # Panel A: P3 Active Learning Arms
    ax = axes[0]
    arms = ["Random DOE", "Physics\nIndicator", "Physics DWR +\nGeom Diversity"]
    mean_err = [
        float(data["p3_arms"]["random_doe"]["mean_true_err_pct"]),
        float(data["p3_arms"]["physics_indicator"]["mean_true_err_pct"]),
        float(data["p3_arms"]["physics_dwr_diversity"]["mean_true_err_pct"]),
    ]
    div_score = [
        float(data["p3_arms"]["random_doe"]["mean_geom_spread"]),
        float(data["p3_arms"]["physics_indicator"]["mean_geom_spread"]),
        float(data["p3_arms"]["physics_dwr_diversity"]["mean_geom_spread"]),
    ]
    x = np.arange(len(arms))
    w = 0.36
    ax.bar(
        x - w / 2,
        mean_err,
        w,
        color=PALETTE["navy"],
        label="Mean True FEM Error of Selected (%)",
    )
    ax_r = ax.twinx()
    ax_r.bar(
        x + w / 2,
        div_score,
        w,
        color=PALETTE["teal"],
        alpha=0.85,
        label="Geometry Token Diversity Spread",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=8.5)
    ax.set_ylim(0.0, max(mean_err) * 1.32)
    ax_r.set_ylim(0.0, max(div_score) * 1.32)
    ax.set_ylabel("Mean Selected True Error (%)", color=PALETTE["navy"])
    ax_r.set_ylabel("Geometry Diversity Spread", color=PALETTE["teal"])
    ax.set_title(
        "A. P3 DWR Active Learning Contribution\n"
        f"(TRAIN-only + {data['fallback_records']} Fallback Ledger returns)"
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax_r.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, loc="upper left", fontsize=7.5)

    # Panel B: P5 Design Optimization Trajectory (Area vs FEM-Verified Stiffness Constraint)
    ax = axes[1]
    hist = data["p5_history"]
    iters = [int(h["iteration"]) for h in hist]
    areas = [float(h["area_mm2"]) for h in hist]
    rxns = [float(h["reaction_n"]) for h in hist]
    init_area = float(data.get("p5_init_area_mm2", areas[0]))
    opt_area = float(data.get("p5_opt_area_mm2", areas[-1]))
    ax.plot(
        iters,
        areas,
        "o-",
        color=PALETTE["navy"],
        lw=2.2,
        label="Bracket Area (mm$^2$, objective)",
    )
    ax_r2 = ax.twinx()
    ax_r2.plot(
        iters,
        rxns,
        "s--",
        color=PALETTE["emerald"],
        lw=2.0,
        label="Surrogate Reaction (N/mm)",
    )
    ax_r2.axhline(
        float(data["p5_min_reaction_n"]),
        color=PALETTE["crimson"],
        linestyle=":",
        lw=1.7,
        label=f"Min Stiffness Limit ({data['p5_min_reaction_n']:.1f})",
    )
    ax_r2.scatter(
        [iters[-1]],
        [float(data["p5_final_fem_reaction_n"])],
        color=PALETTE["amber"],
        s=115,
        marker="*",
        zorder=6,
        label=f"Final P2 FEM Verified ({data['p5_final_fem_reaction_n']:.1f})",
    )
    ax.set_xlabel("Trust-Region Iteration")
    ax.set_ylabel("Material Area (mm$^2$)", color=PALETTE["navy"])
    ax_r2.set_ylabel("Reaction Force (N/mm)", color=PALETTE["emerald"])
    ax.set_title(
        "B. P5 FEM-Guarded Lightweighting Trajectory\n"
        f"(Area: {init_area:.1f} -> {opt_area:.1f} mm$^2$, -{data['p5_area_reduction_pct']:.1f}%)"
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax_r2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, loc="lower left", fontsize=7.3)

    # Panel C: Initial vs P5-Optimized CAD Bracket Geometry Overlay
    ax = axes[2]
    init_n = np.asarray(data["p5_init_nodes"], dtype=float)
    init_f = np.asarray(data["p5_init_facets"], dtype=int)
    opt_n = np.asarray(data["p5_opt_nodes"], dtype=float)
    opt_f = np.asarray(data["p5_opt_facets"], dtype=int)
    for a, b in init_f:
        ax.plot(
            [init_n[a, 0], init_n[b, 0]],
            [init_n[a, 1], init_n[b, 1]],
            color=PALETTE["slate"],
            lw=1.4,
            alpha=0.65,
        )
    for a, b in opt_f:
        ax.plot(
            [opt_n[a, 0], opt_n[b, 0]],
            [opt_n[a, 1], opt_n[b, 1]],
            color=PALETTE["emerald"],
            lw=2.2,
        )
    ax.plot([], [], color=PALETTE["slate"], lw=1.5, label="Initial CAD Bracket")
    ax.plot(
        [], [], color=PALETTE["emerald"], lw=2.2, label="P5 Optimized (P2-FEM Verified)"
    )
    ax.set(
        title=(
            f"C. Initial vs Optimized CAD Shape\n"
            f"(Hole R: {data['p5_init_r']:.2f}->{data['p5_opt_r']:.2f} mm, "
            f"Notch R: {data['p5_init_notch']:.2f}->{data['p5_opt_notch']:.2f} mm)"
        ),
        xlabel="x (mm)",
        ylabel="y (mm)",
        aspect="equal",
    )
    ax.legend(loc="lower center", fontsize=8)

    fig.suptitle(
        "Stage 5 — P3 DWR Active Learning + Fallback Loop & P5 FEM-Guarded CAD Design Optimization",
        fontsize=13.5,
        fontweight="bold",
    )
    return _save_fig(fig, out_path)


def plot_stage6_extended_physics_and_deployment(
    data: Mapping[str, Any], out_path: Path
) -> Path:
    """Plot 6: Follower pressure, 2.5D elastomer contact, 3D TET10 operator, & deployment."""
    fig = Figure(figsize=(14.5, 9.2), layout="constrained", facecolor="white")
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, 2)

    # Panel A: Non-Conservative Follower Pressure Deformation
    ax0 = fig.add_subplot(gs[0, 0])
    fn = np.asarray(data["follower_nodes"], dtype=float)
    fu = np.asarray(data["follower_u"], dtype=float)
    fc = np.asarray(data["follower_cells"], dtype=int)
    tri_ref = Triangulation(fn[:, 0], fn[:, 1], triangles=fc)
    tri_def = Triangulation(fn[:, 0] + fu[:, 0], fn[:, 1] + fu[:, 1], triangles=fc)
    ax0.triplot(tri_ref, color="#CBD5E1", lw=0.5, label="Undeformed")
    img0 = ax0.tripcolor(
        tri_def, np.linalg.norm(fu, axis=1), shading="gouraud", cmap="viridis"
    )
    fig.colorbar(img0, ax=ax0, label="||u|| (mm)", shrink=0.85)
    ax0.set(
        title=(
            "A. Non-Conservative Follower Pressure\n"
            f"(Exact Unsymmetric Tangent FD err: {data['follower_tangent_fd_err']:.1e})"
        ),
        xlabel="x (mm)",
        ylabel="y (mm)",
        aspect="equal",
    )

    # Panel B: 2.5D Plane-Strain Elastomer Rigid-Indenter Contact
    ax1 = fig.add_subplot(gs[0, 1])
    cn = np.asarray(data["contact_nodes"], dtype=float)
    cu = np.asarray(data["contact_u"], dtype=float)
    cc = np.asarray(data["contact_cells"], dtype=int)
    tri_c = Triangulation(cn[:, 0] + cu[:, 0], cn[:, 1] + cu[:, 1], triangles=cc)
    img1 = ax1.tripcolor(tri_c, -cu[:, 1], shading="gouraud", cmap="plasma")
    cax1 = ax1.inset_axes([1.03, 0.04, 0.045, 0.92])
    fig.colorbar(img1, cax=cax1, label="-u_y Indentation (mm)")
    # Draw circular rigid indenter
    theta = np.linspace(0, 2 * np.pi, 120)
    cx, cy = data["indenter_center"]
    cr = float(data["indenter_radius"])
    ax1.plot(
        cx + cr * np.cos(theta),
        cy + cr * np.sin(theta),
        color=PALETTE["crimson"],
        lw=2.2,
        label="Rigid Punch",
    )
    ax1.set(
        title=(
            "B. 2.5D Plane-Strain Rigid-Indenter Contact\n"
            f"(Active nodes: {data['contact_active_nodes']} | "
            f"Max pen: {data['contact_max_pen_mm']:.1e} mm | "
            f"F_y={data['contact_fy_n']:.1f} N/mm)"
        ),
        xlabel="x (mm)",
        ylabel="y (mm)",
        aspect="equal",
    )
    ax1.legend(loc="lower right", fontsize=8)

    # Panel C: 3D TET10 Volumetric Bracket & O(N*S) Surface3DEnergyOperator
    ax2 = fig.add_subplot(gs[1, 0], projection="3d")
    n3 = np.asarray(data["mesh3d_nodes"], dtype=float)
    s3 = np.asarray(data["mesh3d_surface_tris"], dtype=int)
    polys = [n3[tri] for tri in s3]
    pc = Poly3DCollection(
        polys, facecolor="#38BDF8", edgecolor="#0F172A", linewidths=0.35, alpha=0.55
    )
    ax2.add_collection3d(pc)
    ax2.set_xlim(0, float(n3[:, 0].max()))
    ax2.set_ylim(0, float(n3[:, 1].max()))
    ax2.set_zlim(0, float(n3[:, 2].max()))
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("y (mm)")
    ax2.set_zlabel("z (mm)")
    ax2.set_title(
        "C. 3D TET10 & O(N*S) Surface3DEnergyOperator\n"
        f"(TET4 locked refused at nu=0.49 | grad_u FD err: {data['op3d_grad_fd_err']:.1e})"
    )

    # Panel D: Production Deployment & Provenance Verification Scoreboard
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    checks = data["deployment_checks"]
    y = 0.92
    ax3.text(
        0.02,
        0.99,
        "D. Production Deployment & Epistemic Verification Gate",
        fontsize=11,
        fontweight="bold",
        color=PALETTE["navy"],
        va="top",
    )
    for row in checks:
        badge_col = PALETTE["emerald"] if row["passed"] else PALETTE["crimson"]
        badge_txt = "PASS" if row["passed"] else "FAIL"
        ax3.text(
            0.03,
            y,
            f"[{badge_txt}]",
            fontsize=9.5,
            fontweight="bold",
            color=badge_col,
            family="monospace",
            va="top",
        )
        ax3.text(
            0.16,
            y,
            f"{row['name']}: {row['detail']}",
            fontsize=9.0,
            color="#1E293B",
            va="top",
        )
        y -= 0.125

    fig.suptitle(
        "Stage 6 — Extended Nonlinear Physics (Follower, Contact, 3D TET10) & Deployment (Inspector, MCP, FMI 3.0)",
        fontsize=13.5,
        fontweight="bold",
    )
    return _save_fig(fig, out_path)
