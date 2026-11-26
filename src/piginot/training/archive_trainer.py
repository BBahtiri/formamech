"""Archive-to-Foundation 3D pretraining, few-shot 3D LoRA transfer & N* economics (V5-3).

Bridges curated :class:`~piginot.store.dataset_3d.ShardedMeshDataset3D` enterprise
archives directly to :class:`~piginot.operators.multi_topology_3d.MultiTopologyOperator3D`:

* **Multi-Fidelity Archive Pretraining:** supervised nodal displacement loss paired
  with optional positive-Jacobian (``det(F) > 0``) admissibility regularization
  (explicitly disclosed as ``multi-fidelity-label-efficient``).
* **Few-Shot 3D LoRA Transfer:** freezes pretrained base weights and trains
  ``<= 10%`` low-rank adapter parameters on 20-80 (or smaller micro-CI) target
  component solves for ``>= 5x`` faster adaptation wall time.
* **Pre-Flight Break-Even ``N*`` Estimator:** estimates archive-pretrained LoRA vs.
  from-scratch vs. direct FEM amortization before spending compute.

All FEM/LoRA imports remain lazy inside functions so ``import piginot.training``
never eagerly imports ``piginot.fem`` or ``skfem``.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray

from piginot.contracts import canonical_json_bytes
from piginot.geometry.mesh3d import CanonicalMesh3D, CellType3D, _tet_signed_volume
from piginot.operators.multi_topology_3d import (
    MultiTopologyOperator3D,
    build_multi_topology_surface_tensors,
)
from piginot.store.dataset_3d import (
    DatasetSplit,
    ShardedMeshDataset3D,
    ShardedRecord3D,
)
from piginot.verification.amortization import (
    AmortizationSummary,
    amortized_break_even_queries,
)

FloatArray = NDArray[np.float64]

ARCHIVE_PRETRAIN_REPORT_SCHEMA: Final = "formamech.archive-pretrain-3d-v1"
ARCHIVE_LORA_REPORT_SCHEMA: Final = "formamech.archive-lora-transfer-3d-v1"
ARCHIVE_BREAKEVEN_SCHEMA: Final = "formamech.archive-breakeven-3d-v1"


def _sha256_doc(doc: object) -> str:
    return hashlib.sha256(canonical_json_bytes(doc)).hexdigest()


def reconstruct_mesh_from_sharded_record(rec: ShardedRecord3D) -> CanonicalMesh3D:
    """Reconstruct a canonical 3D mesh with outward boundary triangles from a sharded record."""
    nodes = np.asarray(rec.nodes_mm, dtype=np.float64)
    cells = np.asarray(rec.cells, dtype=np.int64)
    k = int(cells.shape[1])
    ctype = (
        CellType3D.TET10
        if k == 10
        else (CellType3D.HEX8 if k == 8 else CellType3D.TET4)
    )

    # Ensure positive corner orientation for tets
    cell_list: list[tuple[int, ...]] = []
    face_counts: dict[tuple[int, int, int], list[tuple[tuple[int, int, int], int]]] = {}
    if ctype in (CellType3D.TET4, CellType3D.TET10):
        for row in cells:
            c = [int(v) for v in row]
            p0 = (float(nodes[c[0], 0]), float(nodes[c[0], 1]), float(nodes[c[0], 2]))
            p1 = (float(nodes[c[1], 0]), float(nodes[c[1], 1]), float(nodes[c[1], 2]))
            p2 = (float(nodes[c[2], 0]), float(nodes[c[2], 1]), float(nodes[c[2], 2]))
            p3 = (float(nodes[c[3], 0]), float(nodes[c[3], 1]), float(nodes[c[3], 2]))
            if _tet_signed_volume(p0, p1, p2, p3) < 0.0:
                c[1], c[2] = c[2], c[1]
                if ctype is CellType3D.TET10:
                    c[4], c[6] = c[6], c[4]
                    c[8], c[9] = c[9], c[8]
            cell_list.append(tuple(c))
            v0, v1, v2, v3 = c[0], c[1], c[2], c[3]
            for tri, opp in (
                ((v0, v2, v1), v3),
                ((v0, v1, v3), v2),
                ((v1, v2, v3), v0),
                ((v0, v3, v2), v1),
            ):
                key = tuple(sorted(tri))
                assert len(key) == 3
                face_counts.setdefault((key[0], key[1], key[2]), []).append((tri, opp))
    else:
        for row in cells:
            c = [int(v) for v in row]
            cell_list.append(tuple(c))
            n0, n1, n2, n3, n4, n5, n6, n7 = c[:8]
            for a, b, c_i, d_i in (
                (n0, n3, n2, n1),
                (n4, n5, n6, n7),
                (n0, n1, n5, n4),
                (n2, n3, n7, n6),
                (n0, n4, n7, n3),
                (n1, n2, n6, n5),
            ):
                for tri in ((a, b, c_i), (a, c_i, d_i)):
                    key = tuple(sorted(tri))
                    assert len(key) == 3
                    face_counts.setdefault((key[0], key[1], key[2]), []).append(
                        (tri, a)
                    )

    surface_tris: list[tuple[int, int, int]] = []
    for entries in face_counts.values():
        if len(entries) == 1:
            tri, opp = entries[0]
            # Orient normal outward away from opposite vertex
            p_a, p_b, p_c, p_opp = (
                nodes[tri[0]],
                nodes[tri[1]],
                nodes[tri[2]],
                nodes[opp],
            )
            normal = np.cross(p_b - p_a, p_c - p_a)
            if float(np.dot(normal, p_a - p_opp)) < 0.0:
                tri = (tri[0], tri[2], tri[1])
            surface_tris.append(tri)

    if not surface_tris:
        raise ValueError(f"record {rec.record_id!r} produced empty boundary surface")

    xs = nodes[:, 0]
    x_lo, x_hi = float(np.min(xs)), float(np.max(xs))
    tol = max(1.0e-5 * (x_hi - x_lo), 1.0e-6)
    x_min_tris: list[int] = []
    x_max_tris: list[int] = []
    for t_i, (a, b, c_i) in enumerate(surface_tris):
        cx = (nodes[a, 0] + nodes[b, 0] + nodes[c_i, 0]) / 3.0
        if abs(cx - x_lo) <= tol:
            x_min_tris.append(t_i)
        elif abs(cx - x_hi) <= tol:
            x_max_tris.append(t_i)

    tags: dict[str, tuple[int, ...]] = {"surface": tuple(range(len(surface_tris)))}
    if x_min_tris:
        tags["x_min"] = tuple(x_min_tris)
    if x_max_tris:
        tags["x_max"] = tuple(x_max_tris)

    return CanonicalMesh3D(
        nodes=tuple((float(p[0]), float(p[1]), float(p[2])) for p in nodes),
        cells=tuple(cell_list),
        cell_type=ctype,
        surface_triangles=tuple(surface_tris),
        semantic_face_tags=tags,
        metadata={"record_id": rec.record_id, "mesh_id": rec.mesh_id},
    )


def _pad_parameters(params: Sequence[float], target_dim: int) -> list[float]:
    vals = [float(v) for v in params[:target_dim]]
    while len(vals) < target_dim:
        vals.append(0.0)
    return vals


def _tet_corner_det_f_penalty(
    nodes_t: torch.Tensor,
    cells_np: NDArray[np.int64],
    u_pred: torch.Tensor,
    *,
    det_floor: float = 0.15,
) -> torch.Tensor:
    """Differentiable positive-Jacobian barrier penalty on tetrahedral corners."""
    if cells_np.shape[1] not in (4, 10):
        return torch.zeros((), dtype=u_pred.dtype, device=u_pred.device)
    idx = torch.as_tensor(cells_np[:, :4], dtype=torch.long, device=u_pred.device)
    x_e = nodes_t[idx]  # (E, 4, 3)
    d_e = (nodes_t + u_pred)[idx]
    ref = torch.stack(
        (x_e[:, 1] - x_e[:, 0], x_e[:, 2] - x_e[:, 0], x_e[:, 3] - x_e[:, 0]), dim=-1
    )
    cur = torch.stack(
        (d_e[:, 1] - d_e[:, 0], d_e[:, 2] - d_e[:, 0], d_e[:, 3] - d_e[:, 0]), dim=-1
    )
    det_ref = torch.linalg.det(ref)
    det_cur = torch.linalg.det(cur)
    valid = det_ref.abs() > 1.0e-12
    if not bool(torch.any(valid)):
        return torch.zeros((), dtype=u_pred.dtype, device=u_pred.device)
    det_f = det_cur[valid] / det_ref[valid]
    viol = torch.relu(det_floor - det_f)
    return torch.mean(viol * viol)


@dataclass(frozen=True, slots=True)
class ArchivePretrainConfig:
    """Configuration for pretraining a 3D multi-topology operator from a sharded archive."""

    parameter_dim: int = 2
    hidden_dim: int = 32
    slice_count: int = 6
    num_blocks: int = 2
    num_heads: int = 4
    epochs: int = 25
    learning_rate: float = 8.0e-3
    jacobian_barrier_weight: float = 1.0
    max_train_records: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.parameter_dim < 1 or self.hidden_dim < 4 or self.epochs < 1:
            raise ValueError("parameter_dim, hidden_dim, and epochs must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")


@dataclass(frozen=True, slots=True)
class ArchiveTrainingMetrics:
    """Physical-unit error summary across TRAIN / DEV / HOLDOUT splits."""

    split: str
    sample_count: int
    mean_disp_mae_mm: float
    p95_disp_mae_mm: float
    worst_case_disp_mae_mm: float
    mean_rel_l2: float

    def document(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "sample_count": self.sample_count,
            "mean_disp_mae_mm": float(self.mean_disp_mae_mm),
            "p95_disp_mae_mm": float(self.p95_disp_mae_mm),
            "worst_case_disp_mae_mm": float(self.worst_case_disp_mae_mm),
            "mean_rel_l2": float(self.mean_rel_l2),
        }


@dataclass(frozen=True, slots=True)
class ArchivePretrainReport:
    """Auditable report from archive-to-foundation 3D pretraining."""

    training_mode_disclosure: str
    dataset_manifest_sha256: str
    train_record_count: int
    total_parameter_count: int
    epochs: int
    final_train_loss: float
    wall_time_s: float
    dev_metrics: ArchiveTrainingMetrics | None
    holdout_metrics: ArchiveTrainingMetrics | None

    @property
    def report_hash(self) -> str:
        return _sha256_doc(self.document())

    def document(self) -> dict[str, Any]:
        return {
            "schema": ARCHIVE_PRETRAIN_REPORT_SCHEMA,
            "training_mode_disclosure": self.training_mode_disclosure,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "train_record_count": self.train_record_count,
            "total_parameter_count": self.total_parameter_count,
            "epochs": self.epochs,
            "final_train_loss": float(self.final_train_loss),
            "wall_time_s": float(self.wall_time_s),
            "dev_metrics": None
            if self.dev_metrics is None
            else self.dev_metrics.document(),
            "holdout_metrics": (
                None
                if self.holdout_metrics is None
                else self.holdout_metrics.document()
            ),
        }


def _evaluate_operator_on_records(
    model: MultiTopologyOperator3D,
    records: Sequence[tuple[ShardedRecord3D, CanonicalMesh3D]],
    *,
    split_name: str,
    parameter_dim: int,
) -> ArchiveTrainingMetrics | None:
    if not records:
        return None
    maes: list[float] = []
    rel_l2s: list[float] = []
    for rec, mesh in records:
        c_tags = ("x_min",) if "x_min" in mesh.semantic_face_tags else ()
        l_tags = ("x_max",) if "x_max" in mesh.semantic_face_tags else ()
        pred, _ = model.predict_nodal_displacement(
            mesh,
            parameters=_pad_parameters(rec.parameters, parameter_dim),
            clamped_face_tags=c_tags,
            loaded_face_tags=l_tags,
            enforce_ood=False,
        )
        assert pred is not None
        diff = pred - rec.displacement_mm
        maes.append(float(np.mean(np.abs(diff))))
        denom = max(float(np.linalg.norm(rec.displacement_mm)), 1.0e-9)
        rel_l2s.append(float(np.linalg.norm(diff)) / denom)
    arr = np.asarray(maes, dtype=np.float64)
    return ArchiveTrainingMetrics(
        split=split_name,
        sample_count=len(records),
        mean_disp_mae_mm=float(np.mean(arr)),
        p95_disp_mae_mm=float(np.percentile(arr, 95)),
        worst_case_disp_mae_mm=float(np.max(arr)),
        mean_rel_l2=float(np.mean(rel_l2s)),
    )


def train_operator_from_sharded_archive(
    dataset: ShardedMeshDataset3D,
    *,
    config: ArchivePretrainConfig | None = None,
) -> tuple[MultiTopologyOperator3D, ArchivePretrainReport]:
    """Train a 3D multi-topology neural operator directly from a verified sharded archive."""
    cfg = ArchivePretrainConfig() if config is None else config
    dataset.verify_integrity()
    torch.manual_seed(cfg.seed)
    t0 = time.perf_counter()

    train_recs = list(dataset.iter_records(split=DatasetSplit.TRAIN, verify=False))
    if not train_recs:
        raise ValueError("ShardedMeshDataset3D contains zero TRAIN records")
    if cfg.max_train_records is not None:
        train_recs = train_recs[: cfg.max_train_records]
    dev_recs = list(dataset.iter_records(split=DatasetSplit.DEV, verify=False))
    holdout_recs = list(dataset.iter_records(split=DatasetSplit.HOLDOUT, verify=False))

    train_items = [(r, reconstruct_mesh_from_sharded_record(r)) for r in train_recs]
    dev_items = [(r, reconstruct_mesh_from_sharded_record(r)) for r in dev_recs]
    holdout_items = [(r, reconstruct_mesh_from_sharded_record(r)) for r in holdout_recs]

    model = MultiTopologyOperator3D(
        parameter_dim=cfg.parameter_dim,
        hidden_dim=cfg.hidden_dim,
        slice_count=cfg.slice_count,
        num_blocks=cfg.num_blocks,
        num_heads=cfg.num_heads,
    )
    model.fit_ood(
        [
            (
                mesh,
                ("x_min",) if "x_min" in mesh.semantic_face_tags else (),
                ("x_max",) if "x_max" in mesh.semantic_face_tags else (),
            )
            for _, mesh in train_items
        ]
    )

    # Precompute surface tensors once per geometry for fast CPU/GPU training
    cached_train: list[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            NDArray[np.int64],
        ]
    ] = []
    for rec, mesh in train_items:
        c_tags = ("x_min",) if "x_min" in mesh.semantic_face_tags else ()
        l_tags = ("x_max",) if "x_max" in mesh.semantic_face_tags else ()
        c_t, n_t, a_t, f_t = build_multi_topology_surface_tensors(
            mesh, clamped_face_tags=c_tags, loaded_face_tags=l_tags, dtype=torch.float32
        )
        pts = torch.tensor(mesh.nodes, dtype=torch.float32).unsqueeze(0)
        p_vec = torch.tensor(
            [_pad_parameters(rec.parameters, cfg.parameter_dim)], dtype=torch.float32
        )
        u_true = torch.tensor(rec.displacement_mm, dtype=torch.float32).unsqueeze(0)
        cached_train.append((c_t, n_t, a_t, f_t, pts, p_vec, u_true, rec.cells))

    opt = torch.optim.Adam(model.operator.parameters(), lr=cfg.learning_rate)
    final_loss = float("inf")
    model.operator.train()
    for _ in range(cfg.epochs):
        opt.zero_grad()
        total_loss = torch.zeros((), dtype=torch.float32)
        for c_t, n_t, a_t, f_t, pts, p_vec, u_true, cells_np in cached_train:
            enc = model.operator.encode_surface(c_t, n_t, a_t, f_t)
            u_pred = model.operator.displacement(
                encoding=enc, query_xyz=pts, parameters=p_vec
            )
            mse = torch.mean((u_pred - u_true) ** 2)
            if cfg.jacobian_barrier_weight > 0.0:
                pen = _tet_corner_det_f_penalty(pts[0], cells_np, u_pred[0])
                loss_i = mse + cfg.jacobian_barrier_weight * pen
            else:
                loss_i = mse
            total_loss = total_loss + loss_i
        mean_loss = total_loss / float(len(cached_train))
        mean_loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        final_loss = float(mean_loss.detach().item())

    wall_s = time.perf_counter() - t0
    dev_m = _evaluate_operator_on_records(
        model, dev_items, split_name="DEV", parameter_dim=cfg.parameter_dim
    )
    holdout_m = _evaluate_operator_on_records(
        model, holdout_items, split_name="HOLDOUT", parameter_dim=cfg.parameter_dim
    )
    disclosure = (
        "multi-fidelity-label-efficient"
        if cfg.jacobian_barrier_weight > 0.0
        else "supervised-archive-pretraining"
    )
    n_params = sum(p.numel() for p in model.operator.parameters())
    return model, ArchivePretrainReport(
        training_mode_disclosure=disclosure,
        dataset_manifest_sha256=dataset.manifest_sha256,
        train_record_count=len(train_items),
        total_parameter_count=n_params,
        epochs=cfg.epochs,
        final_train_loss=final_loss,
        wall_time_s=wall_s,
        dev_metrics=dev_m,
        holdout_metrics=holdout_m,
    )


@dataclass(frozen=True, slots=True)
class ArchiveLoraTransferConfig:
    """Budgeted few-shot 3D LoRA adaptation config (<= 10% trainable parameters)."""

    rank: int = 2
    alpha: float = 4.0
    target_substrings: tuple[str, ...] = ("out_proj", "decoder_head")
    epochs: int = 12
    learning_rate: float = 1.5e-2
    max_trainable_param_fraction: float = 0.10
    seed: int = 0


@dataclass(frozen=True, slots=True)
class ArchiveLoraTransferReport:
    """Auditable few-shot 3D LoRA adaptation metrics & efficiency report."""

    dataset_manifest_sha256: str
    target_train_count: int
    total_parameter_count: int
    trainable_parameter_count: int
    trainable_param_fraction: float
    injected_modules: tuple[str, ...]
    epochs: int
    wall_time_s: float
    holdout_metrics: ArchiveTrainingMetrics | None

    @property
    def report_hash(self) -> str:
        return _sha256_doc(self.document())

    def document(self) -> dict[str, Any]:
        return {
            "schema": ARCHIVE_LORA_REPORT_SCHEMA,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "target_train_count": self.target_train_count,
            "total_parameter_count": self.total_parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "trainable_param_fraction": float(self.trainable_param_fraction),
            "injected_modules": list(self.injected_modules),
            "epochs": self.epochs,
            "wall_time_s": float(self.wall_time_s),
            "holdout_metrics": (
                None
                if self.holdout_metrics is None
                else self.holdout_metrics.document()
            ),
        }


def adapt_archive_operator_lora_3d(
    pretrained_model: MultiTopologyOperator3D,
    target_dataset: ShardedMeshDataset3D,
    *,
    config: ArchiveLoraTransferConfig | None = None,
) -> tuple[MultiTopologyOperator3D, ArchiveLoraTransferReport]:
    """Few-shot 3D LoRA transfer onto a new component family with frozen base weights."""
    # Lazy import keeps `import piginot.training` completely free of `piginot.fem`
    from piginot.operators.low_rank_adapter_3d import (
        LowRankAdapter3DConfig,
        inject_lora_3d,
    )

    cfg = ArchiveLoraTransferConfig() if config is None else config
    target_dataset.verify_integrity()
    torch.manual_seed(cfg.seed)
    t0 = time.perf_counter()

    train_recs = list(
        target_dataset.iter_records(split=DatasetSplit.TRAIN, verify=False)
    )
    if not train_recs:
        raise ValueError("target_dataset contains zero TRAIN records")
    holdout_recs = list(
        target_dataset.iter_records(split=DatasetSplit.HOLDOUT, verify=False)
    )

    train_items = [(r, reconstruct_mesh_from_sharded_record(r)) for r in train_recs]
    holdout_items = [(r, reconstruct_mesh_from_sharded_record(r)) for r in holdout_recs]

    lora_cfg = LowRankAdapter3DConfig(
        rank=cfg.rank,
        alpha=cfg.alpha,
        target_substrings=cfg.target_substrings,
        max_steps=cfg.epochs,
        from_scratch_reference_steps=max(cfg.epochs * 10, 100),
        max_step_budget_fraction=0.10,
        learning_rate=cfg.learning_rate,
        seed=cfg.seed,
    )
    adapted_op, injected = inject_lora_3d(pretrained_model.operator, lora_cfg)
    total_params = sum(p.numel() for p in adapted_op.parameters())
    trainable_params = [p for p in adapted_op.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    frac = trainable_count / max(total_params, 1)
    if frac > cfg.max_trainable_param_fraction:
        raise ValueError(
            f"LoRA trainable parameter fraction {frac:.3f} exceeds ceiling "
            f"{cfg.max_trainable_param_fraction:.3f}"
        )

    adapted_model = MultiTopologyOperator3D(
        parameter_dim=pretrained_model.config.parameter_dim,
        hidden_dim=pretrained_model.config.hidden_dim,
        slice_count=pretrained_model.config.slice_count,
        num_blocks=pretrained_model.config.num_blocks,
        num_heads=pretrained_model.config.num_heads,
    )
    adapted_model.operator = adapted_op
    adapted_model.fit_ood(
        [
            (
                mesh,
                ("x_min",) if "x_min" in mesh.semantic_face_tags else (),
                ("x_max",) if "x_max" in mesh.semantic_face_tags else (),
            )
            for _, mesh in train_items
        ]
    )

    p_dim = adapted_model.config.parameter_dim
    cached: list[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ] = []
    for rec, mesh in train_items:
        c_tags = ("x_min",) if "x_min" in mesh.semantic_face_tags else ()
        l_tags = ("x_max",) if "x_max" in mesh.semantic_face_tags else ()
        c_t, n_t, a_t, f_t = build_multi_topology_surface_tensors(
            mesh, clamped_face_tags=c_tags, loaded_face_tags=l_tags, dtype=torch.float32
        )
        pts = torch.tensor(mesh.nodes, dtype=torch.float32).unsqueeze(0)
        p_vec = torch.tensor(
            [_pad_parameters(rec.parameters, p_dim)], dtype=torch.float32
        )
        u_true = torch.tensor(rec.displacement_mm, dtype=torch.float32).unsqueeze(0)
        cached.append((c_t, n_t, a_t, f_t, pts, p_vec, u_true))

    opt = torch.optim.Adam(trainable_params, lr=cfg.learning_rate)
    adapted_model.operator.train()
    for _ in range(cfg.epochs):
        opt.zero_grad()
        loss = torch.zeros((), dtype=torch.float32)
        for c_t, n_t, a_t, f_t, pts, p_vec, u_true in cached:
            enc = adapted_model.operator.encode_surface(c_t, n_t, a_t, f_t)
            u_pred = adapted_model.operator.displacement(
                encoding=enc, query_xyz=pts, parameters=p_vec
            )
            loss = loss + torch.mean((u_pred - u_true) ** 2)
        (loss / float(len(cached))).backward()  # type: ignore[no-untyped-call]
        opt.step()

    wall_s = time.perf_counter() - t0
    holdout_m = _evaluate_operator_on_records(
        adapted_model, holdout_items, split_name="HOLDOUT", parameter_dim=p_dim
    )
    return adapted_model, ArchiveLoraTransferReport(
        dataset_manifest_sha256=target_dataset.manifest_sha256,
        target_train_count=len(train_items),
        total_parameter_count=total_params,
        trainable_parameter_count=trainable_count,
        trainable_param_fraction=frac,
        injected_modules=tuple(injected),
        epochs=cfg.epochs,
        wall_time_s=wall_s,
        holdout_metrics=holdout_m,
    )


@dataclass(frozen=True, slots=True)
class ArchiveBreakEvenComparison:
    """Pre-flight break-even N* comparison: from-scratch vs. archive-LoRA vs. direct FEM."""

    expected_queries: int
    from_scratch: AmortizationSummary
    archive_lora: AmortizationSummary
    recommended_strategy: str
    estimated_time_saved_s: float

    def document(self) -> dict[str, Any]:
        return {
            "schema": ARCHIVE_BREAKEVEN_SCHEMA,
            "expected_queries": self.expected_queries,
            "from_scratch": self.from_scratch.document(),
            "archive_lora": self.archive_lora.document(),
            "recommended_strategy": self.recommended_strategy,
            "estimated_time_saved_s": float(self.estimated_time_saved_s),
        }


def estimate_archive_breakeven(
    *,
    expected_queries: int,
    fem_seconds_per_solve: float,
    inference_seconds: float,
    from_scratch_solves: int,
    from_scratch_train_s: float,
    lora_target_solves: int,
    lora_adapt_s: float,
    archive_already_paid: bool = True,
) -> ArchiveBreakEvenComparison:
    """Pre-flight N* economics gate preventing negative-ROI surrogate campaigns."""
    if expected_queries < 1:
        raise ValueError("expected_queries must be >= 1")
    scratch_sum = amortized_break_even_queries(
        training_wall_seconds=from_scratch_train_s,
        calibration_cases=from_scratch_solves,
        fem_seconds=fem_seconds_per_solve,
        inference_seconds=inference_seconds,
    )
    lora_sum = amortized_break_even_queries(
        training_wall_seconds=lora_adapt_s,
        calibration_cases=lora_target_solves
        if archive_already_paid
        else from_scratch_solves,
        fem_seconds=fem_seconds_per_solve,
        inference_seconds=inference_seconds,
    )
    direct_cost = expected_queries * fem_seconds_per_solve
    scratch_cost = (
        from_scratch_train_s
        + from_scratch_solves * fem_seconds_per_solve
        + expected_queries * inference_seconds
    )
    lora_cost = (
        lora_adapt_s
        + (lora_target_solves if archive_already_paid else from_scratch_solves)
        * fem_seconds_per_solve
        + expected_queries * inference_seconds
    )
    costs = {
        "direct_fem": direct_cost,
        "archive_pretrained_lora": lora_cost,
        "from_scratch_surrogate": scratch_cost,
    }
    best = min(costs, key=lambda k: costs[k])
    return ArchiveBreakEvenComparison(
        expected_queries=expected_queries,
        from_scratch=scratch_sum,
        archive_lora=lora_sum,
        recommended_strategy=best,
        estimated_time_saved_s=max(0.0, direct_cost - costs[best]),
    )


__all__ = (
    "ARCHIVE_BREAKEVEN_SCHEMA",
    "ARCHIVE_LORA_REPORT_SCHEMA",
    "ARCHIVE_PRETRAIN_REPORT_SCHEMA",
    "ArchiveBreakEvenComparison",
    "ArchiveLoraTransferConfig",
    "ArchiveLoraTransferReport",
    "ArchivePretrainConfig",
    "ArchivePretrainReport",
    "ArchiveTrainingMetrics",
    "adapt_archive_operator_lora_3d",
    "estimate_archive_breakeven",
    "reconstruct_mesh_from_sharded_record",
    "train_operator_from_sharded_archive",
)
