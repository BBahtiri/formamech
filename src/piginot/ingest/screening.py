"""Automated statistical + physical admissibility screening for CAE archives (V3-3)."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from piginot.contracts.canonical import canonical_json_bytes
from piginot.ingest.contracts import (
    CAE_ARCHIVE_SCHEMA,
    INGEST_PIPELINE_MACHINE,
    CaeSampleSummary,
    IngestPipelineState,
    OutlierFlag,
    OutlierReasonCode,
    OutlierScreeningReport,
)
from piginot.ingest.readers import ParsedCaeSample

FloatArray = NDArray[np.float64]


def _extract_displacement(sample: ParsedCaeSample) -> FloatArray | None:
    for key in ("displacement", "displacement_mm", "u", "U"):
        if key in sample.point_fields:
            arr = np.asarray(sample.point_fields[key], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[0] == sample.nodes.shape[0]:
                return arr[:, : sample.spatial_dim]
    return None


def _compute_min_det_f(sample: ParsedCaeSample, disp: FloatArray) -> float | None:
    """Compute minimum element-level deformation gradient determinant ``det(I + grad u)``."""
    nodes = sample.nodes
    cells = sample.cells
    dim = sample.spatial_dim
    if sample.cell_type == "triangle" and dim == 2:
        x0 = nodes[cells[:, 0]]
        x1 = nodes[cells[:, 1]]
        x2 = nodes[cells[:, 2]]
        ref = np.stack((x1 - x0, x2 - x0), axis=-1)  # (E, 2, 2)
        d_nodes = nodes + disp
        y0 = d_nodes[cells[:, 0]]
        y1 = d_nodes[cells[:, 1]]
        y2 = d_nodes[cells[:, 2]]
        cur = np.stack((y1 - y0, y2 - y0), axis=-1)
        det_ref = np.linalg.det(ref)
        det_cur = np.linalg.det(cur)
        valid = np.abs(det_ref) > 1.0e-15
        if not np.any(valid):
            return None
        return float(np.min(det_cur[valid] / det_ref[valid]))
    if sample.cell_type in ("tet4", "tet10") and dim == 3:
        # Use corner tetrahedron vertices (indices 0..3) for robust orientation/folding check
        x0, x1, x2, x3 = (
            nodes[cells[:, 0]],
            nodes[cells[:, 1]],
            nodes[cells[:, 2]],
            nodes[cells[:, 3]],
        )
        ref = np.stack((x1 - x0, x2 - x0, x3 - x0), axis=-1)  # (E, 3, 3)
        d_nodes = nodes + disp
        y0, y1, y2, y3 = (
            d_nodes[cells[:, 0]],
            d_nodes[cells[:, 1]],
            d_nodes[cells[:, 2]],
            d_nodes[cells[:, 3]],
        )
        cur = np.stack((y1 - y0, y2 - y0, y3 - y0), axis=-1)
        det_ref = np.linalg.det(ref)
        det_cur = np.linalg.det(cur)
        valid = np.abs(det_ref) > 1.0e-15
        if not np.any(valid):
            return None
        return float(np.min(det_cur[valid] / det_ref[valid]))
    return None


def _global_qois(sample: ParsedCaeSample, disp: FloatArray | None) -> dict[str, float]:
    qois: dict[str, float] = {}
    if disp is not None and np.all(np.isfinite(disp)):
        qois["max_displacement"] = float(np.max(np.linalg.norm(disp, axis=1)))
        qois["rms_displacement"] = float(np.sqrt(np.mean(np.sum(disp * disp, axis=1))))
    for key in ("von_mises_mpa", "von_mises", "sigma_vm"):
        if key in sample.point_fields:
            arr = np.asarray(sample.point_fields[key], dtype=np.float64)
            if np.all(np.isfinite(arr)):
                qois["peak_von_mises"] = float(np.max(arr))
            break
    return qois


def screen_cae_archive(
    samples: Sequence[ParsedCaeSample],
    *,
    zscore_threshold: float = 3.0,
    det_f_floor: float = 1.0e-5,
    train_fraction: float = 0.70,
    dev_fraction: float = 0.15,
    output_dir: Path | str | None = None,
) -> OutlierScreeningReport:
    """Screen ingested CAE runs for physical admissibility, duplicates, & 3-sigma outliers."""
    if not samples:
        raise ValueError("samples must be non-empty")
    state = IngestPipelineState.DISCOVERED
    state = INGEST_PIPELINE_MACHINE.require_transition(
        state, IngestPipelineState.PARSED, trigger="parse_samples"
    ).to_state

    flags: list[OutlierFlag] = []
    quarantined_ids: set[str] = set()
    seen_hashes: dict[str, str] = {}
    prelim_qois: dict[str, dict[str, float]] = {}
    prelim_det: dict[str, float | None] = {}

    # Pass 1: exact deduplication, finiteness, and element-level physical det(F) > floor
    for sample in samples:
        if sample.source_sha256 in seen_hashes:
            quarantined_ids.add(sample.sample_id)
            flags.append(
                OutlierFlag(
                    sample_id=sample.sample_id,
                    reason_code=OutlierReasonCode.DUPLICATE_HASH,
                    metric_name="source_sha256",
                    value=None,
                    threshold=None,
                    detail=f"duplicate content hash of {seen_hashes[sample.source_sha256]!r}",
                )
            )
        else:
            seen_hashes[sample.source_sha256] = sample.sample_id

        all_finite = bool(np.all(np.isfinite(sample.nodes))) and all(
            bool(np.all(np.isfinite(v))) for v in sample.point_fields.values()
        )
        if not all_finite:
            quarantined_ids.add(sample.sample_id)
            flags.append(
                OutlierFlag(
                    sample_id=sample.sample_id,
                    reason_code=OutlierReasonCode.NONFINITE_FIELD,
                    metric_name="point_fields",
                    value=float("nan"),
                    threshold=None,
                    detail="non-finite node coordinates or point field values",
                )
            )

        disp = _extract_displacement(sample)
        min_det: float | None = None
        if disp is not None and all_finite:
            min_det = _compute_min_det_f(sample, disp)
            if min_det is not None and min_det <= det_f_floor:
                quarantined_ids.add(sample.sample_id)
                flags.append(
                    OutlierFlag(
                        sample_id=sample.sample_id,
                        reason_code=OutlierReasonCode.NEGATIVE_JACOBIAN,
                        metric_name="min_det_f",
                        value=min_det,
                        threshold=det_f_floor,
                        detail=f"folded/inverted element: min det(F)={min_det:.4g} <= {det_f_floor:.4g}",
                    )
                )
        prelim_det[sample.sample_id] = min_det
        prelim_qois[sample.sample_id] = _global_qois(sample, disp)

    # Pass 2: Robust modified Z-score (MAD) + standard Z-score on physically admissible runs
    clean_ids = [s.sample_id for s in samples if s.sample_id not in quarantined_ids]
    qoi_keys = sorted({k for sid in clean_ids for k in prelim_qois[sid]})
    if len(clean_ids) >= 4:
        for key in qoi_keys:
            pairs = [
                (sid, prelim_qois[sid][key])
                for sid in clean_ids
                if key in prelim_qois[sid]
            ]
            if len(pairs) < 4:
                continue
            vals = np.asarray([v for _, v in pairs], dtype=np.float64)
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            # Consistent Gaussian-equivalent scale: 1.4826 * MAD (fallback to std if MAD==0)
            scale = 1.4826 * mad if mad > 1.0e-12 else float(np.std(vals))
            if scale <= 1.0e-12:
                continue
            for sid, val in pairs:
                z = abs(val - med) / scale
                if z > zscore_threshold:
                    quarantined_ids.add(sid)
                    flags.append(
                        OutlierFlag(
                            sample_id=sid,
                            reason_code=OutlierReasonCode.STATISTICAL_ZSCORE,
                            metric_name=key,
                            value=z,
                            threshold=zscore_threshold,
                            detail=f"robust Z-score {z:.2f} > {zscore_threshold:.2f} (val={val:.4g}, med={med:.4g})",
                        )
                    )

    state = INGEST_PIPELINE_MACHINE.require_transition(
        state, IngestPipelineState.SCREENED, trigger="screen_outliers"
    ).to_state

    # Pass 3: Leak-free deterministic TRAIN / DEV / HOLDOUT role assignment
    admitted = [s for s in samples if s.sample_id not in quarantined_ids]
    # Deterministic hash sort for reproducible partition allocation
    ordered_admitted = sorted(
        admitted,
        key=lambda s: hashlib.sha256(
            f"{s.sample_id}:{s.source_sha256}".encode()
        ).hexdigest(),
    )
    n_ok = len(ordered_admitted)
    n_train = max(1, int(math.floor(n_ok * train_fraction))) if n_ok > 0 else 0
    n_dev = (
        max(0, min(n_ok - n_train, int(math.floor(n_ok * dev_fraction))))
        if n_ok > 1
        else 0
    )
    role_map: dict[str, str] = {}
    for idx, s in enumerate(ordered_admitted):
        if idx < n_train:
            role_map[s.sample_id] = "TRAIN"
        elif idx < n_train + n_dev:
            role_map[s.sample_id] = "DEV"
        else:
            role_map[s.sample_id] = "HOLDOUT"

    summaries: list[CaeSampleSummary] = []
    for sample in samples:
        is_ok = sample.sample_id not in quarantined_ids
        summaries.append(
            CaeSampleSummary(
                sample_id=sample.sample_id,
                source_format=sample.source_format,
                source_sha256=sample.source_sha256,
                spatial_dim=sample.spatial_dim,
                node_count=int(sample.nodes.shape[0]),
                cell_count=int(sample.cells.shape[0]),
                cell_type=sample.cell_type,
                field_names=tuple(sorted(sample.point_fields.keys())),
                parameters=sample.parameters,
                global_qois=prelim_qois[sample.sample_id],
                min_det_f=prelim_det[sample.sample_id],
                admissible=is_ok,
                partition_role=role_map.get(sample.sample_id, "QUARANTINED"),
            )
        )

    if n_ok == 0:
        state = INGEST_PIPELINE_MACHINE.require_transition(
            state, IngestPipelineState.QUARANTINED, trigger="quarantine_all"
        ).to_state
    else:
        state = INGEST_PIPELINE_MACHINE.require_transition(
            state, IngestPipelineState.PARTITIONED, trigger="assign_partitions"
        ).to_state

    report = OutlierScreeningReport(
        state=state,
        total_samples=len(samples),
        accepted_count=n_ok,
        quarantined_count=len(samples) - n_ok,
        zscore_threshold=zscore_threshold,
        det_f_floor=det_f_floor,
        flags=tuple(flags),
        samples=tuple(summaries),
    )

    if output_dir is not None:
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        screen_doc: dict[str, Any] = {
            **report.document(),
            "report_hash": report.report_hash,
        }
        (out / "outlier-screen.json").write_bytes(
            canonical_json_bytes(screen_doc) + b"\n"
        )
        archive_doc = {
            "schema": CAE_ARCHIVE_SCHEMA,
            "report_hash": report.report_hash,
            "accepted_count": report.accepted_count,
            "quarantined_count": report.quarantined_count,
            "partitions": {
                role: [s.sample_id for s in summaries if s.partition_role == role]
                for role in ("TRAIN", "DEV", "HOLDOUT", "QUARANTINED")
            },
        }
        (out / "archive-manifest.json").write_bytes(
            canonical_json_bytes(archive_doc) + b"\n"
        )
    return report


__all__ = ("screen_cae_archive",)
