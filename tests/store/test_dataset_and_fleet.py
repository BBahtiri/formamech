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


def test_v4_5_audit_regressions_dataset_traversal_and_fleet_validation(
    tmp_path: Path,
) -> None:
    """Dataset blocks shard path traversal & mesh_id split leakage; fleet blocks injection."""
    ds = ShardedMeshDataset3D(tmp_path / "ds_sec", max_records_per_shard=4)
    r_base = _make_record("r1", 10.0, DatasetSplit.TRAIN)
    r_train = ShardedRecord3D(
        record_id="r1",
        mesh_id="mesh-shared",
        split=DatasetSplit.TRAIN,
        parameters=r_base.parameters,
        nodes_mm=r_base.nodes_mm,
        cells=r_base.cells,
        displacement_mm=r_base.displacement_mm,
        metadata={},
    )
    ds.append_records([r_train])

    # 1) Same mesh_id with slightly different geometry still triggers SplitLeakageError
    r_other = _make_record("r2", 12.0, DatasetSplit.HOLDOUT)
    r_leak_id = ShardedRecord3D(
        record_id="r2",
        mesh_id="mesh-shared",
        split=DatasetSplit.HOLDOUT,
        parameters=r_other.parameters,
        nodes_mm=r_other.nodes_mm,
        cells=r_other.cells,
        displacement_mm=r_other.displacement_mm,
        metadata={},
    )
    with pytest.raises(SplitLeakageError, match="mesh_id"):
        ds.append_records([r_leak_id])

    # 2) -0.0 and +0.0 produce identical geometry_fingerprint
    r_neg0 = ShardedRecord3D(
        record_id="r-neg0",
        mesh_id="m-z",
        split=DatasetSplit.TRAIN,
        parameters=(1.0,),
        nodes_mm=np.where(r_train.nodes_mm == 0.0, -0.0, r_train.nodes_mm),
        cells=r_train.cells,
        displacement_mm=r_train.displacement_mm,
        metadata={},
    )
    assert r_neg0.geometry_fingerprint() == r_train.geometry_fingerprint()

    # 3) Manifest shard path traversal raises ShardIntegrityError
    manifest = json.loads(ds.manifest_path.read_text(encoding="utf-8"))
    manifest["shards"][0]["relpath"] = "../../outside.npz"
    ds.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ShardIntegrityError, match="escapes"):
        ds.verify_integrity()

    # 4) Fleet blocks shell/YAML/path injection and fails closed on missing 'ok' in results
    slurm = SlurmScriptFleet()
    tasks = (FleetTaskSpec("t0", {"lx_mm": 10.0}),)
    with pytest.raises(ValueError, match="invalid job_id"):
        slurm.submit_batch("../evil;rm", tasks, workdir=tmp_path / "s_bad")
    with pytest.raises(ValueError, match="invalid partition"):
        SlurmScriptFleet(partition="gpu\nrm -rf /")

    k8s = KubernetesJobFleet(namespace="prod")
    rep_k8s = k8s.submit_batch("k8s-collect", tasks, workdir=tmp_path / "k8s_c")
    k8s_root = Path(rep_k8s.artifacts[0]).parent
    # Missing 'ok' key must NOT default to True
    (k8s_root / "results" / "task-0000.json").write_text(
        json.dumps({"task_id": "t0", "value": 1}), encoding="utf-8"
    )
    col_bad = k8s.collect_completed(k8s_root)
    assert col_bad.state is FleetJobState.FAILED and col_bad.failed_count == 1
    # Explicit ok=True verifies cleanly
    (k8s_root / "results" / "task-0000.json").write_text(
        json.dumps({"task_id": "t0", "ok": True, "value": 1}), encoding="utf-8"
    )
    col_ok = k8s.collect_completed(k8s_root)
    assert col_ok.state is FleetJobState.VERIFIED and col_ok.succeeded_count == 1
