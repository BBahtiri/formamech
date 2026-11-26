"""Gate G-V5-3b tests: Archive-to-Foundation pretraining, 3D LoRA transfer & N* economics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from piginot.geometry.mesh3d import make_brick_tet4_mesh
from piginot.store.dataset_3d import (
    DatasetSplit,
    ShardedMeshDataset3D,
    ShardedRecord3D,
)
from piginot.training.archive_trainer import (
    ArchiveLoraTransferConfig,
    ArchivePretrainConfig,
    adapt_archive_operator_lora_3d,
    estimate_archive_breakeven,
    train_operator_from_sharded_archive,
)


def _populate_synthetic_dataset(
    ds_dir: Path,
    *,
    n_train: int,
    n_holdout: int,
    stiffness_scale: float = 1.0,
) -> ShardedMeshDataset3D:
    ds = ShardedMeshDataset3D(ds_dir, max_records_per_shard=8)
    recs: list[ShardedRecord3D] = []
    for idx in range(n_train + n_holdout):
        split = DatasetSplit.TRAIN if idx < n_train else DatasetSplit.HOLDOUT
        lx = 15.0 + 1.2 * idx
        ly = 8.0 + 0.4 * (idx % 3)
        lz = 6.0
        mesh = make_brick_tet4_mesh(lx, ly, lz, nx=2, ny=1, nz=1)
        nodes = np.asarray(mesh.nodes, dtype=np.float64)
        cells = np.asarray(mesh.cells, dtype=np.int64)
        load = 1.0 + 0.3 * (idx % 4)
        xn = nodes[:, 0] / lx
        disp = np.zeros_like(nodes)
        disp[:, 0] = 0.02 * load * stiffness_scale * xn
        disp[:, 1] = -0.05 * load * stiffness_scale * (xn**2)
        recs.append(
            ShardedRecord3D(
                record_id=f"rec-{idx:03d}",
                mesh_id=f"mesh-{idx:03d}",
                split=split,
                parameters=(lx / 20.0, load),
                nodes_mm=nodes,
                cells=cells,
                displacement_mm=disp,
                metadata={"family": f"scale-{stiffness_scale}"},
            )
        )
    ds.append_records(recs)
    return ds


def test_archive_pretrain_and_few_shot_lora_transfer(tmp_path: Path) -> None:
    """Gate G-V5-3b: Pretrain from ShardedMeshDataset3D, adapt <=10% LoRA params faster than scratch."""
    base_ds = _populate_synthetic_dataset(
        tmp_path / "base_archive", n_train=6, n_holdout=2, stiffness_scale=1.0
    )
    pretrained, pre_report = train_operator_from_sharded_archive(
        base_ds,
        config=ArchivePretrainConfig(
            parameter_dim=2,
            hidden_dim=24,
            epochs=20,
            learning_rate=1.0e-2,
            jacobian_barrier_weight=1.0,
            seed=0,
        ),
    )
    assert pre_report.training_mode_disclosure == "multi-fidelity-label-efficient"
    assert pre_report.holdout_metrics is not None
    assert (
        pre_report.holdout_metrics.worst_case_disp_mae_mm
        >= pre_report.holdout_metrics.mean_disp_mae_mm
    )
    assert len(pre_report.report_hash) == 64

    # Target enterprise component family with slightly shifted stiffness (few-shot adaptation)
    target_ds = _populate_synthetic_dataset(
        tmp_path / "target_family", n_train=3, n_holdout=2, stiffness_scale=1.25
    )
    _, lora_report = adapt_archive_operator_lora_3d(
        pretrained,
        target_ds,
        config=ArchiveLoraTransferConfig(
            rank=2,
            target_substrings=("out_proj", "decoder_head"),
            epochs=10,
            learning_rate=2.0e-2,
            max_trainable_param_fraction=0.10,
            seed=0,
        ),
    )
    assert lora_report.trainable_param_fraction <= 0.10
    assert lora_report.trainable_parameter_count > 0
    assert len(lora_report.injected_modules) >= 1
    assert lora_report.holdout_metrics is not None
    assert len(lora_report.report_hash) == 64

    # Compare against from-scratch training on the same target family
    _, scratch_report = train_operator_from_sharded_archive(
        target_ds,
        config=ArchivePretrainConfig(
            parameter_dim=2,
            hidden_dim=24,
            epochs=50,
            learning_rate=8.0e-3,
            seed=1,
        ),
    )
    assert scratch_report.holdout_metrics is not None
    # LoRA updates <=10% params in 5x fewer epochs than from-scratch
    assert lora_report.wall_time_s < scratch_report.wall_time_s


def test_preflight_breakeven_n_star_matches_amortization_formula() -> None:
    """Gate G-V5-3b: Pre-flight N* economics selects optimal strategy accurately."""
    comp = estimate_archive_breakeven(
        expected_queries=250,
        fem_seconds_per_solve=10.0,
        inference_seconds=0.01,
        from_scratch_solves=400,
        from_scratch_train_s=1200.0,
        lora_target_solves=30,
        lora_adapt_s=120.0,
        archive_already_paid=True,
    )
    # Exact analytical N* for LoRA = (120 + 30*10) / (10 - 0.01) = 420 / 9.99
    expected_lora_n_star = (120.0 + 30 * 10.0) / (10.0 - 0.01)
    assert comp.archive_lora.break_even_queries is not None
    assert math.isclose(
        comp.archive_lora.break_even_queries, expected_lora_n_star, rel_tol=1.0e-9
    )
    assert comp.recommended_strategy == "archive_pretrained_lora"
    assert comp.estimated_time_saved_s > 2000.0

    # When only 5 queries are expected, direct FEM wins — preventing negative-ROI training
    small = estimate_archive_breakeven(
        expected_queries=5,
        fem_seconds_per_solve=10.0,
        inference_seconds=0.01,
        from_scratch_solves=400,
        from_scratch_train_s=1200.0,
        lora_target_solves=30,
        lora_adapt_s=120.0,
    )
    assert small.recommended_strategy == "direct_fem"
