"""Verification tests for Phase 5 (V4-5) ShardedMeshDataset3D & Local/Slurm/K8s fleet."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from piginot.cli import main as cli_main
from piginot.geometry.mesh3d import make_brick_tet4_mesh
from piginot.store.dataset_3d import (
    DatasetSplit,
    ShardedMeshDataset3D,
    ShardedRecord3D,
    ShardIntegrityError,
    SplitLeakageError,
)
from piginot.training.fleet import (
    FleetJobState,
    FleetTaskSpec,
    KubernetesJobFleet,
    LocalMultiprocessFleet,
    SlurmScriptFleet,
)


def _make_record(rec_id: str, lx: float, split: DatasetSplit) -> ShardedRecord3D:
    mesh = make_brick_tet4_mesh(lx, 8.0, 8.0, nx=2, ny=1, nz=1)
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64)
    disp = 0.01 * nodes
    return ShardedRecord3D(
        record_id=rec_id,
        mesh_id=mesh.mesh_id,
        split=split,
        parameters=(lx,),
        nodes_mm=nodes,
        cells=cells,
        displacement_mm=disp,
        metadata={"lx": lx},
    )


def test_sharded_dataset_3d_streaming_split_lock_and_tamper_detection(
    tmp_path: Path,
) -> None:
    """ShardedMeshDataset3D streams lazily, blocks HOLDOUT leakage, and catches shard tamper."""
    ds = ShardedMeshDataset3D(tmp_path / "ds3d", max_records_per_shard=2)
    train_recs = [
        _make_record("tr-0", 16.0, DatasetSplit.TRAIN),
        _make_record("tr-1", 18.0, DatasetSplit.TRAIN),
        _make_record("tr-2", 20.0, DatasetSplit.TRAIN),
    ]
    hold_recs = [_make_record("ho-0", 24.0, DatasetSplit.HOLDOUT)]

    shards_tr = ds.append_records(train_recs)
    shards_ho = ds.append_records(hold_recs)
    assert len(shards_tr) == 2
    assert len(shards_ho) == 1

    ver = ds.verify_integrity()
    assert ver["ok"] is True and ver["total_records"] == 4

    # Lazy streaming batches
    batches = list(ds.iter_batches(batch_size=2, split=DatasetSplit.TRAIN))
    assert [len(b) for b in batches] == [2, 1]
    assert [r.record_id for r in ds.iter_records(split=DatasetSplit.HOLDOUT)] == [
        "ho-0"
    ]

    # Zero-leakage split lock: reusing HOLDOUT geometry (lx=24.0) inside TRAIN is REFUSED!
    leaked = _make_record("tr-leak", 24.0, DatasetSplit.TRAIN)
    with pytest.raises(SplitLeakageError):
        ds.append_records([leaked])

    # Tampering a single byte of shard-0000.npz raises ShardIntegrityError
    shard0 = ds.shards_dir / "shard-0000.npz"
    raw = bytearray(shard0.read_bytes())
    raw[-1] ^= 0xFF
    shard0.write_bytes(bytes(raw))
    with pytest.raises(ShardIntegrityError):
        ds.verify_integrity()


def test_fleet_local_multiprocess_slurm_kubernetes_and_cli(tmp_path: Path) -> None:
    """LocalMultiprocessFleet solves parallel 3D FEM; Slurm & K8s generate valid manifests."""
    tasks = [
        FleetTaskSpec(task_id="b0", payload={"lx_mm": 16.0, "traction_x_mpa": 0.3}),
        FleetTaskSpec(task_id="b1", payload={"lx_mm": 22.0, "traction_x_mpa": 0.4}),
    ]

    # 1) Genuine multi-process local execution
    local = LocalMultiprocessFleet(max_workers=2, use_threads=False)
    rep_loc = local.submit_batch("job-mp", tasks, workdir=tmp_path / "local")
    assert rep_loc.state is FleetJobState.VERIFIED
    assert rep_loc.succeeded_count == 2 and rep_loc.failed_count == 0
    assert all(
        r.ok and r.result is not None and r.result["converged"] for r in rep_loc.results
    )

    # 2) Slurm array script + completed result collection
    slurm = SlurmScriptFleet(partition="gpu-mech")
    rep_sl = slurm.submit_batch("job-slurm", tasks, workdir=tmp_path / "slurm")
    assert rep_sl.state is FleetJobState.DISPATCHED
    sbatch = Path(rep_sl.artifacts[0])
    assert "#SBATCH --array=0-1" in sbatch.read_text(encoding="utf-8")
    # Simulate worker outputs and verify collect_completed
    res_dir = sbatch.parent / "results"
    for idx, t in enumerate(tasks):
        (res_dir / f"task-{idx:04d}.json").write_text(
            json.dumps({"ok": True, "task_id": t.task_id, "max_displacement_mm": 0.05}),
            encoding="utf-8",
        )
    collected = slurm.collect_completed(sbatch.parent)
    assert collected.state is FleetJobState.VERIFIED and collected.succeeded_count == 2

    # 3) Kubernetes Indexed Job YAML
    k8s = KubernetesJobFleet(namespace="formamech-prod")
    rep_k8s = k8s.submit_batch("job-k8s", tasks, workdir=tmp_path / "k8s")
    assert rep_k8s.state is FleetJobState.DISPATCHED
    yaml_txt = Path(rep_k8s.artifacts[0]).read_text(encoding="utf-8")
    assert "completionMode: Indexed" in yaml_txt and "completions: 2" in yaml_txt

    # 4) CLI smoke test
    out_json = tmp_path / "cli_fleet.json"
    rc = cli_main(
        [
            "fleet",
            "submit-local",
            "--job-id",
            "cli-job",
            "--workdir",
            str(tmp_path / "cli_ws"),
            "--use-threads",
            "--output",
            str(out_json),
        ]
    )
    assert rc == 0
    assert json.loads(out_json.read_text(encoding="utf-8"))["state"] == "VERIFIED"
