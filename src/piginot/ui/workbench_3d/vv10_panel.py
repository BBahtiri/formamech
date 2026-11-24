"""CAE Archive Ingestion & Experimental ASME V&V 10 Workbench Panel (V3-5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from piginot.ingest.readers import (
    ParsedCaeSample,
    read_calculix_inp_frd,
    read_vtu_or_xdmf,
)
from piginot.ingest.screening import screen_cae_archive
from piginot.validation.contracts import (
    CouponRole,
    ExperimentalCouponDataset,
    ExperimentalTestMode,
)
from piginot.validation.inverse import (
    CalibrationModelFamily,
    InverseCalibrationConfig,
    calibrate_material_from_experiments,
)


def discover_cae_samples(archive_dir: Path) -> list[ParsedCaeSample]:
    """Discover and parse supported CAE archive files in ``archive_dir``."""
    root = Path(archive_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"CAE archive directory not found: {root}")
    samples: list[ParsedCaeSample] = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        suf = path.suffix.lower()
        if suf in {".vtu", ".xdmf"}:
            samples.append(read_vtu_or_xdmf(path))
        elif suf == ".frd":
            inp = path.with_suffix(".inp")
            if inp.is_file():
                samples.append(read_calculix_inp_frd(inp, path))
    if not samples:
        raise ValueError(f"no supported CAE archive files found in {root}")
    return samples


def run_cae_ingest_summary(archive_dir: Path) -> dict[str, Any]:
    """Discover, parse, and screen CAE archive files into a canonical report."""
    samples = discover_cae_samples(archive_dir)
    report = screen_cae_archive(samples)
    return report.document()


def load_experimental_coupons_json(
    dataset_path: Path,
) -> tuple[CalibrationModelFamily, list[ExperimentalCouponDataset]]:
    """Parse a JSON experimental coupon bundle into canonical datasets."""
    raw = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    model_family = cast(
        CalibrationModelFamily,
        str(raw.get("model_family", "incompressible_neo_hookean")),
    )
    coupons: list[ExperimentalCouponDataset] = []
    for item in raw.get("coupons", ()):
        coupons.append(
            ExperimentalCouponDataset(
                specimen_id=str(item["specimen_id"]),
                role=CouponRole(str(item["role"])),
                test_mode=ExperimentalTestMode(str(item["test_mode"])),
                stretches_or_strains=tuple(
                    float(v) for v in item["stretches_or_strains"]
                ),
                measured_stress_mpa=tuple(
                    float(v) for v in item["measured_stress_mpa"]
                ),
                measurement_std_mpa=float(item["measurement_std_mpa"]),
                metadata={
                    str(k): str(v) for k, v in dict(item.get("metadata", {})).items()
                },
            )
        )
    return model_family, coupons


def run_vv10_calibration_summary(
    dataset_path: Path,
    *,
    u_num_mpa: float = 0.05,
) -> dict[str, Any]:
    """Load an experimental coupon JSON and run ASME V&V 10 inverse calibration."""
    model_family, coupons = load_experimental_coupons_json(dataset_path)
    cfg = InverseCalibrationConfig(
        model_family=model_family,
        u_num_mpa=float(u_num_mpa),
    )
    dossier = calibrate_material_from_experiments(coupons, config=cfg)
    return dossier.document()


def render_cae_vv10_mode(st: Any, workspace_root: Path) -> None:
    """Render the Streamlit CAE Archive Ingest & Experimental V&V 10 mode."""
    st.title("CAE Archive Ingestion & Experimental ASME V&V 10")
    tab_ingest, tab_vv10 = st.tabs(["CAE Archive Screening", "Experimental V&V 10"])

    with tab_ingest:
        arch_str = st.text_input(
            "CAE Archive Directory (.vtu, .xdmf, .inp/.frd)",
            value=str(workspace_root / "cae_archive"),
            key="wb3d_ingest_dir",
        )
        if st.button("Screen CAE Archive", type="primary"):
            arch = Path(arch_str.strip())
            if not arch.is_dir():
                st.error(f"Directory does not exist: {arch}")
            else:
                doc = run_cae_ingest_summary(arch)
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Samples", doc["total_samples"])
                c2.metric("Accepted", doc["accepted_count"])
                c3.metric("Quarantined", doc["quarantined_count"])
                st.json(doc)

    with tab_vv10:
        ds_str = st.text_input(
            "Experimental Coupon Dataset JSON",
            value=str(workspace_root / "coupons.json"),
            key="wb3d_vv10_json",
        )
        u_num = float(
            st.number_input(
                "Numerical Uncertainty u_num (MPa)", value=0.05, min_value=0.0
            )
        )
        if st.button("Run Inverse Calibration & V&V 10 Budget", type="primary"):
            ds_path = Path(ds_str.strip())
            if not ds_path.is_file():
                st.error(f"Dataset file not found: {ds_path}")
            else:
                doc = run_vv10_calibration_summary(ds_path, u_num_mpa=u_num)
                c1, c2, c3 = st.columns(3)
                c1.metric("Experimental Tier", doc["experimental_validation_tier"])
                err_e = doc.get("holdout_comparison_error_e_mpa")
                c2.metric(
                    "Holdout Error E (MPa)",
                    f"{err_e:.4f}" if err_e is not None else "n/a",
                )
                c3.metric(
                    "Validation Uncertainty u_val (MPa)", f"{doc['u_val_mpa']:.4f}"
                )
                st.json(doc)


__all__ = (
    "discover_cae_samples",
    "load_experimental_coupons_json",
    "render_cae_vv10_mode",
    "run_cae_ingest_summary",
    "run_vv10_calibration_summary",
)
