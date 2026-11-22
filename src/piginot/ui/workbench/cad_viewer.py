"""Interactive Plotly & static Matplotlib CAD/mesh viewers for the Workbench.

Uses ``importlib.import_module`` for optional UI/plotting libraries so strict
type-checking stays clean whether ``formamech[ui]`` is installed or not.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from piginot.ui.workbench.physics_forms import BoundaryAssignment

ROLE_COLORS: Final[Mapping[str, str]] = {
    "dirichlet_fixed": "#DC2626",  # Crimson: clamped zero displacement
    "dirichlet_loaded": "#EA580C",  # Burnt orange: prescribed nonzero load
    "symmetry": "#7C3AED",  # Violet: symmetry plane
    "traction_free": "#0D9488",  # Teal: traction-free edge / hole
    "unassigned": "#64748B",  # Slate: unassigned boundary
}


def _classify_assignment(assignment: BoundaryAssignment | None) -> tuple[str, str]:
    if assignment is None:
        return "unassigned", "Unassigned"
    if assignment.kind == "traction_free":
        return "traction_free", f"{assignment.name} (traction-free)"
    if assignment.kind == "symmetry":
        comps = ",".join(assignment.components)
        return "symmetry", f"{assignment.name} (symmetry {comps})"
    is_zero = all(
        expr.strip() in ("0", "0.0", "-0", "-0.0") for expr, _ in assignment.values
    )
    comps_vals = ", ".join(
        f"u_{c}={e}"
        for c, (e, _) in zip(assignment.components, assignment.values, strict=False)
    )
    if is_zero:
        return "dirichlet_fixed", f"{assignment.name} (clamp: {comps_vals})"
    return "dirichlet_loaded", f"{assignment.name} (load: {comps_vals})"


def _boundary_midpoint(
    segments: Sequence[Sequence[Sequence[float]]],
) -> tuple[float, float]:
    pts: list[tuple[float, float]] = []
    for seg in segments:
        for pt in seg:
            pts.append((float(pt[0]), float(pt[1])))
    if not pts:
        return (0.0, 0.0)
    mid_seg = segments[len(segments) // 2]
    p0, p1 = mid_seg[0], mid_seg[1]
    return (0.5 * (float(p0[0]) + float(p1[0])), 0.5 * (float(p0[1]) + float(p1[1])))


def build_cad_boundary_figure(
    preview: Mapping[str, Any],
    *,
    assignments: Sequence[BoundaryAssignment] | None = None,
    selected_region: str | None = None,
    show_mesh: bool = True,
) -> Any:
    """Build an interactive Plotly figure with clickable semantic boundary traces."""
    try:
        go = importlib.import_module("plotly.graph_objects")
    except ImportError as exc:
        from piginot.serving.hybrid import MissingExtraError

        raise MissingExtraError(
            "Workbench CAD viewer requires formamech[ui] "
            "(install with `pip install 'formamech[ui]'`)."
        ) from exc

    assign_map: dict[str, BoundaryAssignment] = (
        {a.region: a for a in assignments} if assignments is not None else {}
    )
    fig = go.Figure()

    # 1. Optional faint triangular mesh wireframe
    mesh = preview.get("mesh", {})
    nodes = mesh.get("nodes_mm", ())
    tris = mesh.get("triangles", ())
    if show_mesh and nodes and tris:
        mx: list[float | None] = []
        my: list[float | None] = []
        for a, b, c in tris:
            pa, pb, pc = nodes[a], nodes[b], nodes[c]
            mx.extend([pa[0], pb[0], pc[0], pa[0], None])
            my.extend([pa[1], pb[1], pc[1], pa[1], None])
        fig.add_trace(
            go.Scatter(
                x=mx,
                y=my,
                mode="lines",
                line={"color": "rgba(148, 163, 184, 0.40)", "width": 0.8},
                name=f"Mesh ({len(tris)} tris)",
                hoverinfo="skip",
            )
        )

    # 2. One clickable scatter trace per semantic boundary group
    mid_x: list[float] = []
    mid_y: list[float] = []
    mid_text: list[str] = []
    mid_custom: list[str] = []
    mid_colors: list[str] = []
    mid_sizes: list[int] = []

    for entry in preview.get("boundaries", ()):
        region = str(entry["name"])
        bkind = str(entry.get("kind", "other"))
        segments = entry.get("segments_mm", ())
        assignment = assign_map.get(region)
        role_key, role_label = _classify_assignment(assignment)
        color = ROLE_COLORS[role_key]
        is_sel = region == selected_region

        sx: list[float | None] = []
        sy: list[float | None] = []
        cdata: list[str | None] = []
        for seg in segments:
            p0, p1 = seg[0], seg[1]
            sx.extend([float(p0[0]), float(p1[0]), None])
            sy.extend([float(p0[1]), float(p1[1]), None])
            cdata.extend([region, region, None])

        if is_sel:
            fig.add_trace(
                go.Scatter(
                    x=sx,
                    y=sy,
                    mode="lines",
                    line={"color": "rgba(37, 99, 235, 0.30)", "width": 11},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=sx,
                y=sy,
                mode="lines",
                line={"color": color, "width": 5 if is_sel else 3},
                name=f"{region} [{bkind}] — {role_label}",
                customdata=cdata,
                hovertemplate=(
                    f"<b>{region}</b> ({bkind})<br>{role_label}<extra></extra>"
                ),
            )
        )

        mx_pt, my_pt = _boundary_midpoint(segments)
        mid_x.append(mx_pt)
        mid_y.append(my_pt)
        mid_text.append(region)
        mid_custom.append(region)
        mid_colors.append("#1D4ED8" if is_sel else color)
        mid_sizes.append(13 if is_sel else 9)

    # 3. Clickable midpoint handles for effortless Streamlit on_select picking
    if mid_x:
        fig.add_trace(
            go.Scatter(
                x=mid_x,
                y=mid_y,
                mode="markers+text",
                marker={
                    "size": mid_sizes,
                    "color": mid_colors,
                    "line": {"color": "#FFFFFF", "width": 1.5},
                },
                text=mid_text,
                textposition="top center",
                textfont={"size": 10},
                customdata=mid_custom,
                name="Boundary handles (click to select)",
                hovertemplate="Click to select <b>%{customdata}</b><extra></extra>",
            )
        )

    fig.update_layout(
        title={"text": "2D CAD Mesh & Boundary Groups", "x": 0.01},
        xaxis={"title": "x (mm)", "scaleanchor": "y", "scaleratio": 1},
        yaxis={"title": "y (mm)"},
        clickmode="event+select",
        legend={"orientation": "h", "yanchor": "top", "y": -0.15},
        margin={"l": 45, "r": 20, "t": 55, "b": 85},
        height=470,
    )
    return fig


def extract_selected_region(selection_event: Any) -> str | None:
    """Extract clicked ``edge-NNNNNN`` region from a Streamlit ``on_select`` state."""
    if selection_event is None:
        return None
    sel = (
        selection_event.get("selection")
        if isinstance(selection_event, Mapping)
        else getattr(selection_event, "selection", None)
    )
    if not isinstance(sel, Mapping):
        return None
    points = sel.get("points")
    if not isinstance(points, Sequence) or not points:
        return None
    first = points[0]
    if not isinstance(first, Mapping):
        return None
    custom = first.get("customdata")
    if isinstance(custom, str) and custom.strip():
        return custom.strip()
    if isinstance(custom, Sequence) and custom and isinstance(custom[0], str):
        return custom[0].strip()
    return None


def render_workbench_cad_png(
    preview: Mapping[str, Any],
    output_path: Path | str,
    *,
    assignments: Sequence[BoundaryAssignment] | None = None,
    selected_region: str | None = None,
    title: str = "FormaMech Workbench — 2D CAD Mesh & Semantic Boundaries",
) -> Path:
    """Render a crisp static PNG verification plot using Matplotlib."""
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    tri_mod = importlib.import_module("matplotlib.tri")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    assign_map = {a.region: a for a in assignments} if assignments is not None else {}

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=160)
    mesh = preview.get("mesh", {})
    nodes = mesh.get("nodes_mm", ())
    tris = mesh.get("triangles", ())
    if nodes and tris:
        xs = [float(p[0]) for p in nodes]
        ys = [float(p[1]) for p in nodes]
        triang = tri_mod.Triangulation(xs, ys, tris)
        ax.triplot(triang, color="#CBD5E1", linewidth=0.55, zorder=1)

    for entry in preview.get("boundaries", ()):
        region = str(entry["name"])
        bkind = str(entry.get("kind", "other"))
        segments = entry.get("segments_mm", ())
        role_key, role_label = _classify_assignment(assign_map.get(region))
        color = ROLE_COLORS[role_key]
        lw = 3.8 if region == selected_region else 2.4
        first_seg = True
        for seg in segments:
            p0, p1 = seg[0], seg[1]
            ax.plot(
                [float(p0[0]), float(p1[0])],
                [float(p0[1]), float(p1[1])],
                color=color,
                linewidth=lw,
                label=f"{region} ({bkind}): {role_label}" if first_seg else None,
                zorder=3,
            )
            first_seg = False
        mx, my = _boundary_midpoint(segments)
        ax.scatter(
            [mx], [my], s=42, color=color, edgecolors="white", linewidth=1.0, zorder=4
        )
        ax.annotate(
            region,
            (mx, my),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=7.5,
            fontweight="bold",
            color="#0F172A",
            bbox={
                "boxstyle": "round,pad=0.18",
                "fc": "white",
                "ec": color,
                "alpha": 0.85,
            },
            zorder=5,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    n_nodes = mesh.get("node_count", len(nodes))
    n_tris = mesh.get("triangle_count", len(tris))
    ax.set_title(f"{title}\n({n_nodes} nodes, {n_tris} triangles)", fontsize=11, pad=10)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        fontsize=8,
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


__all__ = (
    "ROLE_COLORS",
    "build_cad_boundary_figure",
    "extract_selected_region",
    "render_workbench_cad_png",
)
