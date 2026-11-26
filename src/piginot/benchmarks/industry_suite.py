"""PhysicsBench / AeroZoo-style Neutral Multi-Scale Industry Benchmark Suite (V5-3).

Evaluates 3D neural surrogates across multiple training-set scales ``N`` and an
explicit **Held-Out Boundary-Condition / Load-Case Shift** stress track, emitting
AIPhysics.org / PhysicsBench-compatible canonical JSON + Markdown reports with:

* Mean, p95, and **worst-case (max)** physical-unit errors (mm),
* Spatial BC-layout OOD AUROC (AeroZoo-style shift detection),
* Wall-clock break-even ``N*`` vs. reference FEM cost,
* Deterministic ``report_hash`` (RFC 8785 canonical JSON SHA-256).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from piginot.benchmarks.external_data import prepare_dataset_cache
from piginot.certify.regime_guards import (
    BcLayoutOODDetector3D,
    compute_bc_layout_descriptor_3d,
)
from piginot.contracts import Unit, canonical_json_bytes
from piginot.geometry.mesh3d import make_brick_tet4_mesh
from piginot.ingest.simops_fabric import (
    curate_archive_to_sharded_dataset,
    read_simjeb_sample,
)
from piginot.store.dataset_3d import ShardedMeshDataset3D
from piginot.training.archive_trainer import (
    ArchivePretrainConfig,
    train_operator_from_sharded_archive,
)
from piginot.verification.amortization import amortized_break_even_queries

INDUSTRY_BENCHMARK_SCHEMA: Final = "formamech.industry-benchmark-v1"


def _sha256_doc(doc: object) -> str:
    return hashlib.sha256(canonical_json_bytes(doc)).hexdigest()


def _roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Exact deterministic ROC-AUC (Mann-Whitney U) with zero sklearn dependency."""
    pos = [float(s) for y, s in zip(y_true, scores, strict=True) if y == 1]
    neg = [float(s) for y, s in zip(y_true, scores, strict=True) if y == 0]
    if not pos or not neg:
        return 1.0
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / float(len(pos) * len(neg))


@dataclass(frozen=True, slots=True)
class IndustryBenchmarkConfig:
    """Configuration for neutral multi-scale + BC-shift industry benchmark run."""

    dataset_name: str = "deepjeb"
    scales: tuple[int, ...] = (4, 6, 8)
    epochs_per_scale: int = 12
    hidden_dim: int = 24
    reference_fem_seconds: float = 2.0
    reference_inference_seconds: float = 0.005
    allow_noncommercial: bool = False
    allow_synthetic_fallback: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if len(self.scales) < 1 or any(
            type(s) is not int or s < 2 for s in self.scales
        ):
            raise ValueError("scales must be non-empty integers >= 2")


@dataclass(frozen=True, slots=True)
class ScaleBenchmarkPoint:
    """Held-out physical-unit metrics and break-even N* at one training scale N."""

    train_scale_n: int
    holdout_count: int
    mean_disp_mae_mm: float
    p95_disp_mae_mm: float
    worst_case_disp_mae_mm: float
    mean_rel_l2: float
    train_wall_time_s: float
    break_even_n_star: float | None

    def document(self) -> dict[str, Any]:
        return {
            "train_scale_n": self.train_scale_n,
            "holdout_count": self.holdout_count,
            "mean_disp_mae_mm": float(self.mean_disp_mae_mm),
            "p95_disp_mae_mm": float(self.p95_disp_mae_mm),
            "worst_case_disp_mae_mm": float(self.worst_case_disp_mae_mm),
            "mean_rel_l2": float(self.mean_rel_l2),
            "train_wall_time_s": float(self.train_wall_time_s),
            "break_even_n_star": (
                None
                if self.break_even_n_star is None
                else float(self.break_even_n_star)
            ),
        }


@dataclass(frozen=True, slots=True)
class BcShiftStressResult:
    """AeroZoo / SimJEB-style held-out boundary-condition layout shift stress track."""

    in_distribution_cases: int
    shifted_bc_cases: int
    ood_auroc: float
    shifted_recall_at_threshold: float

    def document(self) -> dict[str, Any]:
        return {
            "in_distribution_cases": self.in_distribution_cases,
            "shifted_bc_cases": self.shifted_bc_cases,
            "ood_auroc": float(self.ood_auroc),
            "shifted_recall_at_threshold": float(self.shifted_recall_at_threshold),
        }


@dataclass(frozen=True, slots=True)
class IndustryBenchmarkReport:
    """AIPhysics.org / PhysicsBench-compatible canonical benchmark leaderboard report."""

    dataset_name: str
    license_name: str
    source_mode: str
    scale_results: tuple[ScaleBenchmarkPoint, ...]
    bc_shift_result: BcShiftStressResult

    def deterministic_document(self) -> dict[str, Any]:
        """Wall-clock-stripped document for deterministic cross-run report_hash."""
        return {
            "schema": INDUSTRY_BENCHMARK_SCHEMA,
            "dataset_name": self.dataset_name,
            "license_name": self.license_name,
            "source_mode": self.source_mode,
            "scales": [
                {
                    "train_scale_n": s.train_scale_n,
                    "holdout_count": s.holdout_count,
                    "mean_disp_mae_mm": round(s.mean_disp_mae_mm, 8),
                    "p95_disp_mae_mm": round(s.p95_disp_mae_mm, 8),
                    "worst_case_disp_mae_mm": round(s.worst_case_disp_mae_mm, 8),
                    "mean_rel_l2": round(s.mean_rel_l2, 8),
                }
                for s in self.scale_results
            ],
            "bc_shift_result": self.bc_shift_result.document(),
        }

    @property
    def report_hash(self) -> str:
        return _sha256_doc(self.deterministic_document())

    def document(self) -> dict[str, Any]:
        return {
            "schema": INDUSTRY_BENCHMARK_SCHEMA,
            "dataset_name": self.dataset_name,
            "license_name": self.license_name,
            "source_mode": self.source_mode,
            "scale_results": [s.document() for s in self.scale_results],
            "bc_shift_result": self.bc_shift_result.document(),
            "report_hash": self.report_hash,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# FormaMech Neutral Industry Benchmark Report — `{self.dataset_name}`",
            "",
            f"- **License:** `{self.license_name}` | **Source Mode:** `{self.source_mode}`",
            f"- **Deterministic Report SHA-256:** `{self.report_hash}`",
            "",
            "## Multi-Scale Learning Curve (Physical Units: mm)",
            "",
            "| Train Scale `N` | Holdout `M` | Mean MAE (mm) | p95 MAE (mm) | Worst-Case MAE (mm) | Mean Rel-L2 | Break-Even `N*` |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for s in self.scale_results:
            n_star = (
                "n/a" if s.break_even_n_star is None else f"{s.break_even_n_star:.1f}"
            )
            lines.append(
                f"| {s.train_scale_n} | {s.holdout_count} | {s.mean_disp_mae_mm:.5f} | "
                f"{s.p95_disp_mae_mm:.5f} | {s.worst_case_disp_mae_mm:.5f} | "
                f"{s.mean_rel_l2:.4f} | {n_star} |"
            )
        b = self.bc_shift_result
        lines.extend(
            [
                "",
                "## Held-Out Boundary-Condition / Load-Case Shift Track",
                "",
                f"- **In-Distribution Probes:** {b.in_distribution_cases} | **Shifted-BC Probes:** {b.shifted_bc_cases}",
                f"- **BC-Layout OOD AUROC:** `{b.ood_auroc:.4f}` | **Shifted-BC Recall:** `{b.shifted_recall_at_threshold:.1%}`",
                "",
            ]
        )
        return "\n".join(lines)


def _run_bc_shift_stress_track() -> BcShiftStressResult:
    """Evaluate spatial BC-layout shift detector (train on x_min/x_max, test on shifted y_max/z_max)."""
    train_descs = []
    id_descs = []
    ood_descs = []
    for idx in range(8):
        lx = 16.0 + 1.0 * idx
        mesh = make_brick_tet4_mesh(lx, 8.0, 6.0, nx=2, ny=2, nz=2)
        d_train = compute_bc_layout_descriptor_3d(
            mesh, clamped_face_tags=("x_min",), loaded_face_tags=("x_max",)
        )
        if idx < 5:
            train_descs.append(d_train)
        else:
            id_descs.append(d_train)
            # Shifted load case: loaded patch moved from x_max to y_max or z_max
            shifted_tag = "y_max" if idx % 2 == 0 else "z_max"
            ood_descs.append(
                compute_bc_layout_descriptor_3d(
                    mesh, clamped_face_tags=("x_min",), loaded_face_tags=(shifted_tag,)
                )
            )
    det = BcLayoutOODDetector3D.fit(train_descs)
    y_true: list[int] = []
    scores: list[float] = []
    shifted_detected = 0
    for d in id_descs:
        ok, dist = det.is_in_distribution(d)
        del ok
        y_true.append(0)
        scores.append(dist if math.isfinite(dist) else 1.0e9)
    for d in ood_descs:
        ok, dist = det.is_in_distribution(d)
        y_true.append(1)
        scores.append(dist if math.isfinite(dist) else 1.0e9)
        if not ok:
            shifted_detected += 1
    return BcShiftStressResult(
        in_distribution_cases=len(id_descs),
        shifted_bc_cases=len(ood_descs),
        ood_auroc=_roc_auc(y_true, scores),
        shifted_recall_at_threshold=shifted_detected / max(len(ood_descs), 1),
    )


def run_industry_benchmark_suite(
    config: IndustryBenchmarkConfig | None = None,
    *,
    work_dir: Path | str,
) -> IndustryBenchmarkReport:
    """Run multi-scale learning curve + BC-shift stress suite & emit JSON + Markdown."""
    cfg = IndustryBenchmarkConfig() if config is None else config
    root = Path(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    max_n = max(cfg.scales)
    needed_total = max(max_n + 4, int(math.ceil(max_n / 0.65)))
    prepared = prepare_dataset_cache(
        cfg.dataset_name,
        root / "cache",
        allow_noncommercial=cfg.allow_noncommercial,
        allow_synthetic_fallback=cfg.allow_synthetic_fallback,
        synthetic_sample_count=needed_total,
    )

    entries = [
        read_simjeb_sample(
            p,
            length_unit=Unit.MM,
            load_case_id=f"lc_{(i % 4) + 1}",
            geometry_group_id=f"geom_{i:04d}",
        )
        for i, p in enumerate(prepared.sample_files)
    ]
    ds_dir = root / "curated_sharded"
    curate_archive_to_sharded_dataset(
        entries,
        ds_dir,
        train_fraction=0.75,
        dev_fraction=0.10,
    )
    ds = ShardedMeshDataset3D(ds_dir)

    scale_points: list[ScaleBenchmarkPoint] = []
    for n_scale in sorted(cfg.scales):
        _, train_rep = train_operator_from_sharded_archive(
            ds,
            config=ArchivePretrainConfig(
                parameter_dim=2,
                hidden_dim=cfg.hidden_dim,
                epochs=cfg.epochs_per_scale,
                max_train_records=n_scale,
                seed=cfg.seed,
            ),
        )
        hm = train_rep.holdout_metrics
        assert hm is not None
        amort = amortized_break_even_queries(
            training_wall_seconds=train_rep.wall_time_s,
            calibration_cases=train_rep.train_record_count,
            fem_seconds=cfg.reference_fem_seconds,
            inference_seconds=cfg.reference_inference_seconds,
        )
        scale_points.append(
            ScaleBenchmarkPoint(
                train_scale_n=train_rep.train_record_count,
                holdout_count=hm.sample_count,
                mean_disp_mae_mm=hm.mean_disp_mae_mm,
                p95_disp_mae_mm=hm.p95_disp_mae_mm,
                worst_case_disp_mae_mm=hm.worst_case_disp_mae_mm,
                mean_rel_l2=hm.mean_rel_l2,
                train_wall_time_s=train_rep.wall_time_s,
                break_even_n_star=amort.break_even_queries,
            )
        )

    bc_stress = _run_bc_shift_stress_track()
    report = IndustryBenchmarkReport(
        dataset_name=prepared.spec.name,
        license_name=prepared.spec.license_name,
        source_mode=prepared.source_mode,
        scale_results=tuple(scale_points),
        bc_shift_result=bc_stress,
    )
    (root / "industry_benchmark.json").write_bytes(
        canonical_json_bytes(report.document())
    )
    (root / "industry_benchmark.md").write_text(report.to_markdown(), encoding="utf-8")
    return report


__all__ = (
    "INDUSTRY_BENCHMARK_SCHEMA",
    "BcShiftStressResult",
    "IndustryBenchmarkConfig",
    "IndustryBenchmarkReport",
    "ScaleBenchmarkPoint",
    "run_industry_benchmark_suite",
)
