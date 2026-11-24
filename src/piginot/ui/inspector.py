"""Data/view-model layer for the FormaMech Verification & Trust Inspector.

Separates numerical inspection and dossier generation from UI rendering so
headless tests, CLI exports, notebooks, and Streamlit/Plotly apps share one
single authoritative source of truth backed by ``serve(..., emit_receipt=True)``.
"""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from piginot.certify import verify_receipt_document
from piginot.certify.admissibility import AdmissibilityThresholds

INSPECTOR_SCHEMA: Final = "formamech.trust-inspector-snapshot-v1"
INSPECTOR_THRESHOLDS: Final = AdmissibilityThresholds()


@dataclass(frozen=True, slots=True)
class InspectorSnapshot:
    """Complete verified field, admissibility, diff, and receipt snapshot."""

    bundle_name: str
    mode: str
    serving_path: str
    verification_tier: str
    claim_level: str
    verdict: str
    nodes: tuple[tuple[float, float], ...]
    cells: tuple[tuple[int, int, int], ...]
    u_surrogate: tuple[tuple[float, float], ...]
    u_magnitude_mm: tuple[float, ...]
    det_f_2d: tuple[float, ...]
    j_3d: tuple[float, ...]
    von_mises_mpa: tuple[float, ...]
    corrected_diff_mm: tuple[float, ...] | None
    fem_diff_mm: tuple[float, ...] | None
    u_fem: tuple[tuple[float, float], ...] | None
    qois: tuple[dict[str, Any], ...]
    tier0_admissibility: dict[str, Any] | None
    certificate: dict[str, Any] | None
    bounds: dict[str, Any] | None
    trust_report: dict[str, Any] | None
    trust_receipt: dict[str, Any] | None
    trust_receipt_sha256: str | None
    receipt_verified: bool
    receipt_errors: tuple[str, ...]
    limitation_paragraph: str
    reason: str = ""
    determinant_floor: float = INSPECTOR_THRESHOLDS.min_det_f_2d

    @property
    def receipt_integrity(self) -> str:
        if self.trust_receipt is None:
            return "Absent"
        return "Verified" if self.receipt_verified else "Invalid"

    def summary(self) -> dict[str, Any]:
        """Compact JSON-safe summary suitable for APIs, UI headers, and dossiers."""
        return {
            "schema": INSPECTOR_SCHEMA,
            "bundle_name": self.bundle_name,
            "mode": self.mode,
            "serving_path": self.serving_path,
            "verification_tier": self.verification_tier,
            "claim_level": self.claim_level,
            "verdict": self.verdict,
            "node_count": len(self.nodes),
            "cell_count": len(self.cells),
            "max_u_mm": max(self.u_magnitude_mm) if self.u_magnitude_mm else None,
            "min_det_f_2d": min(self.det_f_2d) if self.det_f_2d else None,
            "determinant_floor": self.determinant_floor,
            "max_von_mises_mpa": (
                max(self.von_mises_mpa) if self.von_mises_mpa else None
            ),
            "max_corrected_diff_mm": (
                max(self.corrected_diff_mm) if self.corrected_diff_mm else None
            ),
            "max_fem_diff_mm": max(self.fem_diff_mm) if self.fem_diff_mm else None,
            "qois": list(self.qois),
            "tier0_passed": (
                None
                if self.tier0_admissibility is None
                else bool(self.tier0_admissibility.get("passed"))
            ),
            "receipt_sha256": self.trust_receipt_sha256,
            "receipt_verified": self.receipt_verified,
            "receipt_errors": list(self.receipt_errors),
            "reason": self.reason,
        }


def _von_mises_2d(sigma: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Plane-stress Cauchy von Mises equivalent stress at each node."""
    sxx = sigma[:, 0, 0]
    syy = sigma[:, 1, 1]
    sxy = 0.5 * (sigma[:, 0, 1] + sigma[:, 1, 0])
    val = np.maximum(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0)
    return np.asarray(np.sqrt(val), dtype=np.float64)


def build_inspector_snapshot(
    bundle_path: Path | str,
    inputs: Mapping[str, Any],
    *,
    mode: str = "fast",
    recipe_path: Path | str | None = None,
    cad_calibration_path: Path | str | None = None,
    backend: Any = None,
    spectral_repair: bool = False,
    tta_steps: int = 0,
    escalate: bool = False,
    tolerance: float | None = None,
    queries: Sequence[Sequence[float]] | None = None,
) -> InspectorSnapshot:
    """Evaluate one request and package full fields + verification artifacts."""
    from piginot.bounds import CadCalibrationRecord
    from piginot.bundle import load_bundle
    from piginot.bundle.problem import PortableProblem
    from piginot.cli import parse_request
    from piginot.serving import (
        ServingMode,
        load_recipe,
        serve,
    )
    from piginot.serving.test_time_lora import TestTimeLoraConfig

    bpath = Path(bundle_path)
    bundle = load_bundle(bpath)
    parsed_inputs, parsed_queries = parse_request(
        {
            "inputs": dict(inputs),
            **({"queries": [list(q) for q in queries]} if queries is not None else {}),
        }
    )

    serving_mode = ServingMode(mode)
    is_portable = isinstance(getattr(bundle.model, "problem", None), PortableProblem)
    recipe = None
    if serving_mode is not ServingMode.FAST:
        if recipe_path is not None:
            recipe = load_recipe(bundle, bpath, Path(recipe_path))
        elif not is_portable:
            recipe = load_recipe(bundle, bpath)

    cad_cal = (
        CadCalibrationRecord.load(
            Path(cad_calibration_path),
            expected_manifest_sha256=bundle.manifest.manifest_sha256,
        )
        if cad_calibration_path is not None
        else None
    )
    tta_cfg = TestTimeLoraConfig(steps=tta_steps) if tta_steps > 0 else None

    served = serve(
        bundle,
        parsed_inputs,
        cast("Sequence[Sequence[float]] | None", parsed_queries),
        mode=serving_mode,
        recipe=recipe,
        corrected=serving_mode is not ServingMode.FAST,
        escalate=escalate,
        tolerance=tolerance,
        backend=backend,
        cad_calibration=cad_cal,
        trust_report=serving_mode is not ServingMode.FAST,
        tier0_check=True,
        tier0_thresholds=INSPECTOR_THRESHOLDS,
        spectral_repair=spectral_repair,
        tta_config=tta_cfg,
        emit_receipt=True,
    )

    pred = served.prediction
    mesh = pred.mesh
    nodes: tuple[tuple[float, float], ...] = ()
    cells: tuple[tuple[int, int, int], ...] = ()
    u_surr: tuple[tuple[float, float], ...] = ()
    u_mag: tuple[float, ...] = ()
    det_f: tuple[float, ...] = ()
    j_3d: tuple[float, ...] = ()
    vm: tuple[float, ...] = ()
    qois: tuple[dict[str, Any], ...] = ()

    if mesh is not None and pred.vertices is not None:
        nodes = tuple((float(p[0]), float(p[1])) for p in mesh.nodes)
        cells = tuple((int(c[0]), int(c[1]), int(c[2])) for c in mesh.cells)
        u_arr = pred.vertices.u.detach().cpu().numpy().astype(np.float64)
        f2d = pred.vertices.F_2D.detach().cpu().numpy().astype(np.float64)
        j_arr = pred.vertices.J_3D.detach().cpu().numpy().astype(np.float64)
        sig = pred.vertices.sigma.detach().cpu().numpy().astype(np.float64)
        det_arr = f2d[:, 0, 0] * f2d[:, 1, 1] - f2d[:, 0, 1] * f2d[:, 1, 0]
        u_surr = tuple((float(row[0]), float(row[1])) for row in u_arr)
        u_mag = tuple(float(v) for v in np.linalg.norm(u_arr, axis=1))
        det_f = tuple(float(v) for v in det_arr)
        j_3d = tuple(float(v) for v in j_arr)
        vm = tuple(float(v) for v in _von_mises_2d(sig))
        qois = tuple(
            {
                "name": q.name,
                "value": float(q.value),
                "unit": q.unit,
                "vertex_index": int(q.vertex),
                "location_mm": [float(q.location[0]), float(q.location[1])],
            }
            for q in pred.qois
        )

    corrected_diff: tuple[float, ...] | None = None
    if (
        served.corrected_displacement is not None
        and pred.vertices is not None
        and served.corrected_displacement.shape[0] >= len(u_surr)
    ):
        u_arr = pred.vertices.u.detach().cpu().numpy().astype(np.float64)
        corr = np.asarray(
            served.corrected_displacement[: len(u_surr)], dtype=np.float64
        )
        corrected_diff = tuple(float(v) for v in np.linalg.norm(corr - u_arr, axis=1))

    fem_diff: tuple[float, ...] | None = None
    u_fem_rows: tuple[tuple[float, float], ...] | None = None
    if (
        served.exact is not None
        and pred.vertices is not None
        and len(served.exact.displacement) >= len(u_surr)
    ):
        u_arr = pred.vertices.u.detach().cpu().numpy().astype(np.float64)
        fem_arr = np.asarray(served.exact.displacement[: len(u_surr)], dtype=np.float64)
        u_fem_rows = tuple((float(r[0]), float(r[1])) for r in fem_arr)
        fem_diff = tuple(float(v) for v in np.linalg.norm(fem_arr - u_arr, axis=1))

    rc_doc = None if served.trust_receipt is None else dict(served.trust_receipt)
    rc_ok, rc_errs = (
        verify_receipt_document(rc_doc)
        if rc_doc is not None
        else (False, ("no receipt emitted",))
    )
    tier = (
        str(rc_doc.get("verification_tier", "SURROGATE_UNCHECKED"))
        if rc_doc is not None
        else "SURROGATE_UNCHECKED"
    )
    return InspectorSnapshot(
        bundle_name=bundle.manifest.name,
        mode=serving_mode.value,
        serving_path=served.path.value,
        verification_tier=tier,
        claim_level=bundle.claim_level.value,
        verdict=pred.verdict.value,
        nodes=nodes,
        cells=cells,
        u_surrogate=u_surr,
        u_magnitude_mm=u_mag,
        det_f_2d=det_f,
        j_3d=j_3d,
        von_mises_mpa=vm,
        corrected_diff_mm=corrected_diff,
        fem_diff_mm=fem_diff,
        u_fem=u_fem_rows,
        qois=qois,
        tier0_admissibility=(
            None
            if served.tier0_admissibility is None
            else dict(served.tier0_admissibility)
        ),
        certificate=None if served.certificate is None else dict(served.certificate),
        bounds=None if served.bounds is None else dict(served.bounds),
        trust_report=None if served.trust_report is None else dict(served.trust_report),
        trust_receipt=rc_doc,
        trust_receipt_sha256=served.trust_receipt_sha256,
        receipt_verified=rc_ok,
        receipt_errors=rc_errs,
        limitation_paragraph=str(pred.limitations),
        reason=served.reason,
    )


def render_trust_dossier_markdown(snapshot: InspectorSnapshot) -> str:
    """Render a self-contained, auditable Markdown verification dossier."""
    summary = snapshot.summary()
    lines = [
        f"# FormaMech Verification & Trust Dossier — `{snapshot.bundle_name}`",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Mode | `{snapshot.mode}` |",
        f"| Serving Path | `{snapshot.serving_path}` |",
        f"| Verdict | `{snapshot.verdict}` |",
        f"| Verification Tier | `{snapshot.verification_tier}` |",
        f"| Claim Level | `{snapshot.claim_level}` |",
        f"| Nodes / Triangles | {summary['node_count']} / {summary['cell_count']} |",
        f"| Min `det(F_2D)` | `{summary['min_det_f_2d']}` |",
        f"| Max `||u||` (mm) | `{summary['max_u_mm']}` |",
        f"| Max von Mises (MPa) | `{summary['max_von_mises_mpa']}` |",
        f"| Tier 0 Passed | `{summary['tier0_passed']}` |",
        f"| Receipt SHA-256 | `{snapshot.trust_receipt_sha256}` |",
        f"| Receipt Integrity | `{snapshot.receipt_integrity}` |",
        "",
    ]
    if snapshot.qois:
        lines.extend(
            [
                "## Quantities of Interest (QoIs)",
                "",
                "| Name | Value | Unit | Location (mm) |",
                "| --- | --- | --- | --- |",
            ]
        )
        for q in snapshot.qois:
            loc = q["location_mm"]
            lines.append(
                f"| `{q['name']}` | `{q['value']:.6g}` | `{q['unit']}` | "
                f"({loc[0]:.3f}, {loc[1]:.3f}) |"
            )
        lines.append("")
    if snapshot.trust_receipt is not None:
        scope = snapshot.trust_receipt.get("epistemic_scope", {})
        lines.extend(
            [
                "## Epistemic Scope (ASME V&V 10)",
                "",
                "```json",
                json.dumps(scope, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    if snapshot.limitation_paragraph:
        lines.extend(
            [
                "## Declared Scope & Limitations",
                "",
                snapshot.limitation_paragraph,
                "",
            ]
        )
    return "\n".join(lines)


def finite_vertex_values(
    values: Sequence[float], vertex_count: int
) -> list[tuple[int, float]]:
    return [
        (i, value)
        for i, value in enumerate(values[:vertex_count])
        if math.isfinite(value)
    ]


def flagged_vertices(snapshot: InspectorSnapshot) -> list[int]:
    return [
        i
        for i, value in finite_vertex_values(snapshot.det_f_2d, len(snapshot.nodes))
        if value <= snapshot.determinant_floor
    ]


def _determinant_map(go: Any, snapshot: InspectorSnapshot, probe: int | None) -> Any:
    xs, ys = zip(*snapshot.nodes, strict=True)
    mesh_x: list[float | None] = []
    mesh_y: list[float | None] = []
    for i, j, k in snapshot.cells:
        mesh_x.extend((xs[i], xs[j], xs[k], xs[i], None))
        mesh_y.extend((ys[i], ys[j], ys[k], ys[i], None))
    figure = go.Figure(
        go.Scatter(
            x=mesh_x,
            y=mesh_y,
            mode="lines",
            line={"color": "rgba(132,146,164,0.33)", "width": 1},
            hoverinfo="skip",
            name="Mesh",
        )
    )
    finite = finite_vertex_values(snapshot.det_f_2d, len(snapshot.nodes))

    def add_vertices(ids: Sequence[int], name: str, marker: dict[str, Any]) -> None:
        if ids:
            figure.add_scatter(
                x=[xs[i] for i in ids],
                y=[ys[i] for i in ids],
                customdata=list(ids),
                name=name,
                mode="markers",
                marker=marker,
                hovertemplate="Vertex %{customdata}<br>(%{x:.6g}, %{y:.6g}) mm<extra>%{fullData.name}</extra>",
            )

    add_vertices(
        [i for i, _ in finite],
        "Determinant",
        {
            "color": [v for _, v in finite],
            "colorscale": "Viridis",
            "size": 7,
            "showscale": True,
            "colorbar": {"title": "det(F₂D)", "thickness": 12},
        },
    )
    measured = {i for i, _ in finite}
    add_vertices(
        [i for i in range(len(xs)) if i not in measured],
        "Missing or non-finite",
        {
            "symbol": "diamond-open",
            "color": "#64748b",
            "size": 8,
        },
    )
    add_vertices(
        flagged_vertices(snapshot),
        "At or below floor",
        {
            "symbol": "x",
            "color": "#b91c1c",
            "size": 11,
        },
    )
    if probe is not None:
        add_vertices(
            [probe],
            "Probe",
            {
                "symbol": "circle-open",
                "color": "#1d4ed8",
                "size": 17,
                "line": {"width": 3},
            },
        )
    figure.update_layout(
        title="Surrogate determinant map",
        xaxis_title="x (mm)",
        yaxis_title="y (mm)",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        clickmode="event+select",
        dragmode="pan",
        showlegend=False,
        height=380,
        margin={"l": 36, "r": 16, "t": 48, "b": 36},
    )
    return figure


def build_plotly_figures(
    snapshot: InspectorSnapshot, *, probe_vertex: int | None = None
) -> dict[str, Any]:
    """Build interactive Plotly field, admissibility, and FEM-diff figures."""
    try:
        go = importlib.import_module("plotly.graph_objects")
    except ImportError as exc:
        from piginot.serving.hybrid import MissingExtraError

        raise MissingExtraError(
            "Verification & Trust Inspector plots require formamech[ui] "
            "(install with `pip install 'formamech[ui]'`)."
        ) from exc

    if not snapshot.nodes:
        return {}
    xs = [p[0] for p in snapshot.nodes]
    ys = [p[1] for p in snapshot.nodes]
    i_idx = [c[0] for c in snapshot.cells]
    j_idx = [c[1] for c in snapshot.cells]
    k_idx = [c[2] for c in snapshot.cells]

    def _mesh_fig(values: Sequence[float], title: str, colorscale: str) -> Any:
        fig = go.Figure(
            data=[
                go.Mesh3d(
                    x=xs,
                    y=ys,
                    z=[0.0] * len(xs),
                    i=i_idx,
                    j=j_idx,
                    k=k_idx,
                    intensity=list(values),
                    colorscale=colorscale,
                    showscale=True,
                )
            ]
        )
        fig.update_layout(
            title={"text": title, "x": 0.01},
            scene={
                "aspectmode": "data",
                "xaxis": {"title": "x (mm)"},
                "yaxis": {"title": "y (mm)"},
                "zaxis": {"visible": False},
                "camera": {
                    "projection": {"type": "orthographic"},
                    "eye": {"x": 0.0, "y": 0.0, "z": 2.6},
                    "up": {"x": 0.0, "y": 1.0, "z": 0.0},
                },
            },
            margin={"l": 10, "r": 10, "t": 48, "b": 10},
            height=380,
        )
        return fig

    figs: dict[str, Any] = {"det_f_2d": _determinant_map(go, snapshot, probe_vertex)}
    if finite_vertex_values(snapshot.u_magnitude_mm, len(snapshot.nodes)):
        figs["displacement_mm"] = _mesh_fig(
            snapshot.u_magnitude_mm, "Displacement Magnitude ||u|| (mm)", "Viridis"
        )
    if finite_vertex_values(snapshot.von_mises_mpa, len(snapshot.nodes)):
        figs["von_mises_mpa"] = _mesh_fig(
            snapshot.von_mises_mpa, "Cauchy von Mises Stress (MPa)", "Plasma"
        )
    if snapshot.fem_diff_mm is not None:
        figs["fem_diff_mm"] = _mesh_fig(
            snapshot.fem_diff_mm,
            "Surrogate vs Exact FEM ||u_fem - u_surr|| (mm)",
            "Hot",
        )
    elif snapshot.corrected_diff_mm is not None:
        figs["corrected_diff_mm"] = _mesh_fig(
            snapshot.corrected_diff_mm,
            "One-Step Newton Correction ||du|| (mm)",
            "Hot",
        )
    return figs
