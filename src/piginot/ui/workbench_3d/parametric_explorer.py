"""Live 3D Parametric Design Explorer & FEM Warm-Start Verifier (V3-5)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from piginot.training.parametric_3d_engine import (
    load_parametric_3d_bundle,
    predict_parametric_3d,
)


@dataclass(frozen=True, slots=True)
class Parametric3DExplorerSnapshot:
    """Structured snapshot from live 3D parametric inference + optional FEM check."""

    bundle_name: str
    params: dict[str, float]
    ood_status: str
    mahalanobis_distance: float
    refused: bool
    max_displacement_mm: float | None
    peak_von_mises_mpa: float | None
    fem_verified: bool
    fem_max_displacement_mm: float | None
    fem_rel_l2_error: float | None
    reason: str

    def document(self) -> dict[str, Any]:
        return {
            "bundle_name": self.bundle_name,
            "params": dict(self.params),
            "ood_status": self.ood_status,
            "mahalanobis_distance": float(self.mahalanobis_distance),
            "refused": self.refused,
            "max_displacement_mm": self.max_displacement_mm,
            "peak_von_mises_mpa": self.peak_von_mises_mpa,
            "fem_verified": self.fem_verified,
            "fem_max_displacement_mm": self.fem_max_displacement_mm,
            "fem_rel_l2_error": self.fem_rel_l2_error,
            "reason": self.reason,
        }


def evaluate_parametric_3d_candidate(
    bundle_dir: Path,
    params: Mapping[str, float],
    *,
    verify_with_fem: bool = False,
) -> Parametric3DExplorerSnapshot:
    """Evaluate a parametric 3D design point with OOD guard and optional warm-start FEM."""
    bundle = load_parametric_3d_bundle(bundle_dir)
    param_dict = {str(k): float(v) for k, v in params.items()}
    pred = predict_parametric_3d(
        bundle,
        param_dict,
        mode="exact" if verify_with_fem else "fast",
    )
    ood_status = (
        pred.ood_report.status.value
        if pred.ood_report is not None
        else "TOPOLOGY_REFUSED"
    )
    maha = (
        float(pred.ood_report.mahalanobis_distance)
        if pred.ood_report is not None
        else float("inf")
    )
    if not pred.accepted:
        return Parametric3DExplorerSnapshot(
            bundle_name=bundle.spec.template_name,
            params=param_dict,
            ood_status=ood_status,
            mahalanobis_distance=maha,
            refused=True,
            max_displacement_mm=None,
            peak_von_mises_mpa=None,
            fem_verified=False,
            fem_max_displacement_mm=None,
            fem_rel_l2_error=None,
            reason=pred.reason,
        )

    fem_ok = pred.fem_result is not None and pred.fem_result.converged
    return Parametric3DExplorerSnapshot(
        bundle_name=bundle.spec.template_name,
        params=param_dict,
        ood_status=ood_status,
        mahalanobis_distance=maha,
        refused=False,
        max_displacement_mm=pred.max_displacement_mm,
        peak_von_mises_mpa=pred.peak_von_mises_mpa,
        fem_verified=fem_ok,
        fem_max_displacement_mm=(
            float(pred.fem_result.max_displacement_mm)
            if pred.fem_result is not None and pred.fem_result.converged
            else None
        ),
        fem_rel_l2_error=pred.displacement_rel_l2_error,
        reason=pred.reason,
    )


def render_parametric_explorer_mode(st: Any, workspace_root: Path) -> None:
    """Render the Streamlit Parametric Explore & Optimize studio mode."""
    st.title("Parametric 3D Explorer & Design Optimization")
    bundle_str = st.text_input(
        "Parametric 3D Bundle Directory",
        value=str(workspace_root / "parametric_3d_bundle"),
        key="wb3d_param_bundle_path",
    )
    bundle_dir = Path(bundle_str.strip())
    if not (bundle_dir / "manifest.json").is_file():
        st.info(
            "Point to a trained parametric 3D bundle directory (`formamech cad parametric-3d train`) "
            "to explore live geometry variations with OOD and conformal guards."
        )
        return

    bundle = load_parametric_3d_bundle(bundle_dir)
    st.caption(f"Template `{bundle.spec.template_name}` · Claim `{bundle.claim_level}`")
    params: dict[str, float] = {}
    cols = st.columns(max(1, min(len(bundle.spec.parameters), 4)))
    for i, p in enumerate(bundle.spec.parameters):
        mid = 0.5 * (float(p.lower) + float(p.upper))
        params[p.name] = float(
            cols[i % len(cols)].slider(
                f"{p.name} ({p.unit})",
                min_value=float(p.lower),
                max_value=float(p.upper),
                value=mid,
                key=f"wb3d_p_{p.name}",
            )
        )
    verify_fem = st.button("Verify with Warm-Start 3D FEM", type="primary")
    snap = evaluate_parametric_3d_candidate(
        bundle_dir,
        params,
        verify_with_fem=verify_fem,
    )
    if snap.refused:
        st.error(
            f"OOD REFUSED (status={snap.ood_status}, d_M={snap.mahalanobis_distance:.2f}): "
            f"{snap.reason}"
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Geometry OOD Status", snap.ood_status)
        c2.metric(
            "Predicted Max |u| (mm)",
            f"{snap.max_displacement_mm:.4f}"
            if snap.max_displacement_mm is not None
            else "n/a",
        )
        c3.metric(
            "Peak von Mises (MPa)",
            f"{snap.peak_von_mises_mpa:.3f}"
            if snap.peak_von_mises_mpa is not None
            else "n/a",
        )
        if snap.fem_verified and snap.fem_rel_l2_error is not None:
            st.success(
                f"Warm-start 3D FEM verified: max |u|={snap.fem_max_displacement_mm:.4f} mm, "
                f"surrogate rel-L2 error={snap.fem_rel_l2_error:.3e}."
            )


__all__ = (
    "Parametric3DExplorerSnapshot",
    "evaluate_parametric_3d_candidate",
    "render_parametric_explorer_mode",
)
