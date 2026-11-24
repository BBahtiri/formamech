"""Verification tests for Phase 5 (V4-5) fail-closed ContinualQualificationLoop (CLPM)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from piginot.cli import main as cli_main
from piginot.geometry.mesh3d import make_brick_tet4_mesh
from piginot.harness import FormaMechAgentHarness
from piginot.serving.audit_ledger import AuditEventKind, SignedAuditLedger
from piginot.store.dataset_3d import (
    DatasetSplit,
    ShardedMeshDataset3D,
    ShardedRecord3D,
)
from piginot.training.continual_clpm import (
    ContinualLoopState,
    ContinualQualificationConfig,
    ContinualQualificationLoop,
)


def _rec(rec_id: str, lx: float, split: DatasetSplit) -> ShardedRecord3D:
    mesh = make_brick_tet4_mesh(lx, 8.0, 8.0, nx=2, ny=1, nz=1)
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64)
    return ShardedRecord3D(
        record_id=rec_id,
        mesh_id=mesh.mesh_id,
        split=split,
        parameters=(lx,),
        nodes_mm=nodes,
        cells=cells,
        displacement_mm=0.01 * nodes,
        metadata={"lx": lx},
    )


def test_continual_qualification_promotes_clean_and_rolls_back_holdout_regression(
    tmp_path: Path,
) -> None:
    """CLPM promotes shadow when holdout holds & fallbacks improve; rolls back any regression."""
    active_v1 = tmp_path / "ckpt-v1"
    active_v1.mkdir()
    (active_v1 / "weights.json").write_text('{"version": 1}', encoding="utf-8")

    ledger = SignedAuditLedger(tmp_path / "audit-ledger.jsonl")
    ds = ShardedMeshDataset3D(tmp_path / "ds3d", max_records_per_shard=4)
    # Seed initial TRAIN + frozen blind HOLDOUT
    ds.append_records(
        [
            _rec("tr-init-0", 16.0, DatasetSplit.TRAIN),
            _rec("ho-blind-0", 30.0, DatasetSplit.HOLDOUT),
            _rec("ho-blind-1", 32.0, DatasetSplit.HOLDOUT),
        ]
    )

    # Log a live OOD fallback event into the signed ledger
    ledger.append(
        event_kind=AuditEventKind.FEM_FALLBACK,
        bundle_name="ckpt-v1",
        mode="certified",
        verdict="FALLBACK_SOLVED",
        inputs={"request_id": "prod-req-42", "lx": 25.0},
    )

    loop = ContinualQualificationLoop(
        active_checkpoint_dir=active_v1,
        audit_ledger=ledger,
        dataset=ds,
        workdir=tmp_path / "clpm",
        config=ContinualQualificationConfig(min_fallbacks_to_trigger=1),
    )

    # 1) Cycle 1: Regressing shadow (catastrophic forgetting on frozen HOLDOUT) => ROLLED_BACK!
    def _regressing_eval(ckpt: Path, records: Sequence[ShardedRecord3D]) -> float:
        is_shadow = "shadow-" in ckpt.name
        is_holdout = records[0].split is DatasetSplit.HOLDOUT
        if is_holdout:
            return 0.14 if is_shadow else 0.05  # regressed from 5% -> 14%!
        return 0.04 if is_shadow else 0.22  # improved on fallback, but broke holdout!

    fb1 = [_rec("fb-001", 25.0, DatasetSplit.TRAIN)]
    res_rb = loop.run_cycle(
        "cycle-01-regress", solved_fallbacks=fb1, evaluator=_regressing_eval
    )
    assert res_rb.state is ContinualLoopState.ROLLED_BACK
    assert res_rb.promoted is False
    assert (
        res_rb.shadow_metrics is not None
        and res_rb.shadow_metrics.holdout_regressed is True
    )
    # Active pointer MUST remain untouched at ckpt-v1
    assert loop.current_active_checkpoint() == active_v1.resolve()

    # 2) Cycle 2: Non-regressing shadow improving fallback accuracy => PROMOTED atomically!
    def _clean_eval(ckpt: Path, records: Sequence[ShardedRecord3D]) -> float:
        is_shadow = "shadow-" in ckpt.name
        is_holdout = records[0].split is DatasetSplit.HOLDOUT
        if is_holdout:
            return 0.048 if is_shadow else 0.050  # zero regression on frozen HOLDOUT
        return 0.035 if is_shadow else 0.220  # major improvement on new fallbacks

    fb2 = [_rec("fb-002", 26.5, DatasetSplit.TRAIN)]
    res_pr = loop.run_cycle(
        "cycle-02-clean", solved_fallbacks=fb2, evaluator=_clean_eval
    )
    assert res_pr.state is ContinualLoopState.PROMOTED
    assert res_pr.promoted is True
    assert res_pr.shadow_checkpoint_dir is not None
    assert (
        loop.current_active_checkpoint() == Path(res_pr.shadow_checkpoint_dir).resolve()
    )
    chain_ok, _ = ledger.verify_chain()
    assert chain_ok is True

    # 3) CLI continual status & Harness governance.continual_status work cleanly
    out_json = tmp_path / "continual_status.json"
    rc = cli_main(
        [
            "continual",
            "status",
            "--ledger",
            str(tmp_path / "audit-ledger.jsonl"),
            "--dataset",
            str(tmp_path / "ds3d"),
            "--output",
            str(out_json),
        ]
    )
    assert rc == 0
    doc = json.loads(out_json.read_text(encoding="utf-8"))
    assert doc["ledger_valid"] is True and doc["fallback_count"] >= 1

    harness = FormaMechAgentHarness(tmp_path / "agent_clpm")
    env_fleet = harness.invoke(
        "governance.fleet_submit_3d", {"backend": "local-threads"}
    )
    assert (
        env_fleet.ok
        and env_fleet.result is not None
        and env_fleet.result["state"] == "VERIFIED"
    )
    env_cs = harness.invoke("governance.continual_status", {})
    assert (
        env_cs.ok
        and env_cs.result is not None
        and env_cs.result["ledger_valid"] is True
    )
