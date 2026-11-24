"""Verification tests for Phase 2 (V4-2) shipped 3D foundation checkpoint & benchmark suite."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from piginot.benchmarks import BENCHMARK_REPORT_SCHEMA, run_benchmark_suite_3d
from piginot.cad.importing_3d import classify_faces_by_bbox, import_cad_3d
from piginot.cad.parametric_3d import generate_3d_holed_block_step
from piginot.cli import main as cli_main
from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.geometry.mesh3d import CellType3D, make_brick_tet4_mesh
from piginot.operators.foundation_hub import load_foundation_3d_checkpoint
from piginot.operators.low_rank_adapter_3d import Adapter3DState, LowRankAdapter3DConfig
from piginot.training.foundation_pretrain import (
    adapt_foundation_to_3d_part,
    default_foundation_checkpoint_dir,
)


def test_shipped_foundation_3d_checkpoint_integrity_tamper_and_lora_transfer(
    tmp_path: Path,
) -> None:
    ckpt_dir = default_foundation_checkpoint_dir()
    assert (ckpt_dir / "foundation.json").is_file()
    assert (ckpt_dir / "weights.safetensors").is_file()
    assert (ckpt_dir / "ood-detector-3d.json").is_file()
    # Keep shipped checkpoint compact (< 2 MB)
    assert (ckpt_dir / "weights.safetensors").stat().st_size < 2_000_000

    loaded = load_foundation_3d_checkpoint(ckpt_dir)
    assert loaded.spec.checkpoint_id == "struct3d-base-v1"
    assert loaded.spec.claim_level == "NOT_VERIFIED"

    # Tampering a single byte of weights.safetensors is refused
    tampered = tmp_path / "tampered_ckpt"
    shutil.copytree(ckpt_dir, tampered)
    w_bytes = bytearray((tampered / "weights.safetensors").read_bytes())
    w_bytes[-1] ^= 0xFF
    (tampered / "weights.safetensors").write_bytes(bytes(w_bytes))
    with pytest.raises(ValueError, match="SHA-256"):
        load_foundation_3d_checkpoint(tampered)

    mat = MaterialSpec(
        name="transfer-nh",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(50.0, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(0.32, Unit.ONE).to_canonical(),
        },
    )

    # Topology-mismatched genus-0 brick (chi=2 != 0) is fail-closed REFUSED
    brick = make_brick_tet4_mesh(16.0, 7.0, 7.0, nx=2, ny=1, nz=1)
    refused_op, refused_res = adapt_foundation_to_3d_part(
        ckpt_dir,
        mesh=brick,
        material=mat,
        clamped_face_tags=("x_min",),
        loaded_face_tags=("x_max",),
        adapter_config=LowRankAdapter3DConfig(
            rank=4,
            max_steps=6,
            from_scratch_reference_steps=100,
            allow_ood_adaptation=True,
        ),
    )
    assert refused_op is None
    assert refused_res.state is Adapter3DState.REFUSED

    # Budgeted 3D LoRA transfer adapts cleanly on an unseen genus-1 holed block
    step_p = generate_3d_holed_block_step(
        tmp_path / "unseen_holed.step", lx=26.0, ly=14.0, lz=10.0, hole_radius=2.8
    )
    imp = import_cad_3d(step_p, mesh_size_mm=9.0, element_type=CellType3D.TET4)
    faces = classify_faces_by_bbox(imp)
    adapted_op, res = adapt_foundation_to_3d_part(
        ckpt_dir,
        mesh=imp.mesh,
        material=mat,
        clamped_face_tags=tuple(faces["x_min"]),
        loaded_face_tags=tuple(faces["x_max"]),
        adapter_config=LowRankAdapter3DConfig(
            rank=4,
            max_steps=6,
            from_scratch_reference_steps=100,
            allow_ood_adaptation=True,
        ),
    )
    assert adapted_op is not None
    assert res.state is Adapter3DState.COMPLETED
    assert res.adapter_parameter_count / res.base_parameter_count <= 0.25


def test_benchmark_suite_3d_and_cli_runner(tmp_path: Path) -> None:
    rep = run_benchmark_suite_3d(tmp_path / "b_run", profile="quick")
    doc = rep.document()
    assert doc["schema"] == BENCHMARK_REPORT_SCHEMA
    assert all(row["converged"] for row in rep.solver_scaling)
    assert rep.ood_stress["ood_refusal_rate"] == 1.0
    assert rep.cross_backend_parity["torch_vs_analytical_rel"] < 1.0e-7
    assert rep.cross_backend_parity["torch_vs_skfem_rel"] < 1.0e-7
    md = rep.to_markdown()
    assert "## B1 — 3D Linear Solver Scaling" in md
    assert "## B4 — Cross-Backend 3D Parity" in md

    out_json = tmp_path / "bench.json"
    out_md = tmp_path / "bench.md"
    rc = cli_main(
        [
            "bench",
            "run",
            "--profile",
            "quick",
            "--workdir",
            str(tmp_path / "cli_b"),
            "--output",
            str(out_json),
            "--markdown",
            str(out_md),
        ]
    )
    assert rc == 0
    assert json.loads(out_json.read_text(encoding="utf-8"))["report_hash"]
    assert out_md.is_file()
