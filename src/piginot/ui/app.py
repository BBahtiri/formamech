"""Streamlit entrypoint for the FormaMech Verification & Trust Inspector (`formamech ui`)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from piginot.ui.inspector import (
    build_inspector_snapshot,
    build_plotly_figures,
    render_trust_dossier_markdown,
)


def run_streamlit_app(default_bundle: str = "") -> None:
    """Render the interactive Verification & Trust Inspector in Streamlit."""
    import streamlit as st  # type: ignore[import-not-found]

    from piginot.bundle import load_bundle

    st.set_page_config(
        page_title="FormaMech Verification & Trust Inspector",
        layout="wide",
    )
    st.title("FormaMech — Verification & Trust Inspector")
    st.caption(
        "Envelope-bounded surrogate serving, Tier 0/1/2 diagnostics, warm-start FEM "
        "comparison, and RFC 8785 Trust Receipts."
    )

    with st.sidebar:
        bundle_str = st.text_input(
            "Verified Bundle Directory",
            value=default_bundle,
            help="Path to a FormaMech verified bundle directory.",
        )
        mode = st.selectbox(
            "Verification Mode",
            options=["fast", "certified", "exact"],
            index=1,
        )
        spectral_repair = st.checkbox("Spectrally Safe Warm-Start Repair", value=True)
        escalate = st.checkbox("Auto-Escalate to Warm-Start FEM", value=False)
        tta_steps = st.slider(
            "Tier 1.5 Test-Time Physics LoRA Steps",
            min_value=0,
            max_value=50,
            value=0,
            step=5,
        )
        cad_cal_str = st.text_input("Optional CAD Calibration JSON", value="")

    if not bundle_str.strip():
        st.info("Enter a verified bundle path in the sidebar to begin inspection.")
        return

    bundle_path = Path(bundle_str)
    if not bundle_path.exists():
        st.error(f"Bundle path does not exist: {bundle_path}")
        return

    bundle = load_bundle(bundle_path)
    st.sidebar.subheader("Envelope Parameters")
    inputs: dict[str, dict[str, object]] = {}
    for bound in bundle.envelope.primitive_bounds:
        lo = float(bound.lower_bound.value)
        hi = float(bound.upper_bound.value)
        mid = 0.5 * (lo + hi)
        val = st.sidebar.slider(
            f"{bound.name} ({bound.lower_bound.unit.value})",
            min_value=lo,
            max_value=hi,
            value=mid,
        )
        inputs[bound.name] = {"value": val, "unit": bound.lower_bound.unit.value}

    one_click_exact = st.sidebar.button("Run Warm-Start FEM & Compare")
    effective_mode = "exact" if one_click_exact else mode

    snapshot = build_inspector_snapshot(
        bundle_path,
        inputs,
        mode=effective_mode,
        cad_calibration_path=cad_cal_str.strip() or None,
        spectral_repair=spectral_repair,
        tta_steps=tta_steps,
        escalate=escalate,
    )
    summary = snapshot.summary()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verification Tier", snapshot.verification_tier)
    c2.metric("Serving Path", snapshot.serving_path)
    c3.metric(
        "Min det(F_2D)",
        f"{summary['min_det_f_2d']:.4f}"
        if summary["min_det_f_2d"] is not None
        else "N/A",
    )
    c4.metric(
        "Receipt Integrity",
        "VERIFIED" if snapshot.receipt_verified else "FAILED",
    )

    tab_fields, tab_verify, tab_receipt = st.tabs(
        [
            "Field & Admissibility Maps",
            "Verification & FEM Diff",
            "Trust Receipt & Dossier",
        ]
    )
    figs = build_plotly_figures(snapshot)
    with tab_fields:
        col_a, col_b = st.columns(2)
        if "displacement_mm" in figs:
            col_a.plotly_chart(figs["displacement_mm"], use_container_width=True)
        if "von_mises_mpa" in figs:
            col_b.plotly_chart(figs["von_mises_mpa"], use_container_width=True)
        if "det_f_2d" in figs:
            st.plotly_chart(figs["det_f_2d"], use_container_width=True)

    with tab_verify:
        if "fem_diff_mm" in figs:
            st.plotly_chart(figs["fem_diff_mm"], use_container_width=True)
        elif "corrected_diff_mm" in figs:
            st.plotly_chart(figs["corrected_diff_mm"], use_container_width=True)
        if snapshot.qois:
            st.subheader("Quantities of Interest")
            st.table(list(snapshot.qois))
        if snapshot.certificate is not None:
            st.subheader("Physics Certificate")
            st.json(snapshot.certificate)
        if snapshot.trust_report is not None:
            st.subheader("CAD Trust Report")
            st.json(snapshot.trust_report)

    with tab_receipt:
        dossier_md = render_trust_dossier_markdown(snapshot)
        st.download_button(
            "Download Verification Dossier (Markdown)",
            data=dossier_md.encode("utf-8"),
            file_name=f"{snapshot.bundle_name}-trust-dossier.md",
            mime="text/markdown",
        )
        if snapshot.trust_receipt is not None:
            receipt_json = json.dumps(snapshot.trust_receipt, indent=2, sort_keys=True)
            st.download_button(
                "Download RFC 8785 Trust Receipt (JSON)",
                data=receipt_json.encode("utf-8"),
                file_name=f"{snapshot.bundle_name}-trust-receipt.json",
                mime="application/json",
            )
            st.json(snapshot.trust_receipt)


if __name__ == "__main__":
    arg_bundle = sys.argv[1] if len(sys.argv) > 1 else ""
    run_streamlit_app(default_bundle=arg_bundle)
