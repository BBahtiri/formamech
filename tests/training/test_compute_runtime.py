"""Verification tests for Phase 1 (V3-1) GPU compute runtime, chunked quadrature & DDP."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from piginot.cad.importing_3d import (
    classify_faces_by_bbox,
    generate_3d_box_step,
    import_cad_3d,
)
from piginot.compiler.config_types import CompilerConfigError, _profile_metadata
from piginot.contracts import ExecutionProfile, MaterialSpec, Quantity, Unit
from piginot.fem.backend_3d import TractionBC3D
from piginot.geometry.mesh3d import CellType3D
from piginot.operators.surface_operator_3d import Surface3DConfig
from piginot.training.compute import (
    COMPUTE_RUNTIME_MACHINE,
    ComputeRuntimeConfig,
    ComputeRuntimeSession,
    ComputeRuntimeState,
    SimulatedDistributedStrategy,
    SingleProcessStrategy,
    autocast_context,
    chunked_quadrature_slices,
)
from piginot.training.energy_engine_3d import (
    LoadCase3D,
    PlanarClampLifting3D,
    Surface3DTrainingConfig,
    train_surface_3d_operator,
)


def test_compute_runtime_config_and_fsm_guards() -> None:
    cfg_cpu = ComputeRuntimeConfig()
    assert cfg_cpu.device == "cpu"
    assert cfg_cpu.amp_enabled is False
    assert cfg_cpu.document()["schema"] == "formamech.compute-runtime-v1"
    assert COMPUTE_RUNTIME_MACHINE.machine_id == "formamech.training.compute_runtime"

    # CUDA requires gpu-accelerated-tolerance
    with pytest.raises(ValueError, match="gpu-accelerated-tolerance"):
        ComputeRuntimeConfig(device="cuda:0", replay_tier="cpu-deterministic-bitwise")

    # AMP requires gpu-accelerated-tolerance
    with pytest.raises(ValueError, match="mixed precision"):
        ComputeRuntimeConfig(
            device="cpu",
            replay_tier="cpu-deterministic-bitwise",
            amp_dtype="bfloat16",
        )

    # Valid CUDA config without hardware fails closed at session creation when require_available=True
    cfg_gpu = ComputeRuntimeConfig(
        device="cuda:0",
        replay_tier="gpu-accelerated-tolerance",
        amp_dtype="bfloat16",
        quadrature_chunk_size=128,
    )
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="cuda.is_available"):
            ComputeRuntimeSession(cfg_gpu, require_available=True)

    # Valid dry-run session without hardware succeeds when require_available=False
    session = ComputeRuntimeSession(cfg_gpu, require_available=False)
    assert session.state is ComputeRuntimeState.VALIDATED
    session.activate()
    assert session.state is ComputeRuntimeState.ACTIVE
    assert session.snapshot().state == "ACTIVE"
    session.complete()
    assert session.state is ComputeRuntimeState.COMPLETED


def test_chunked_quadrature_slices_and_ddp_strategy() -> None:
    slices = chunked_quadrature_slices(10, 4)
    assert slices == (slice(0, 4), slice(4, 8), slice(8, 10))
    assert chunked_quadrature_slices(10, None) == (slice(0, 10),)
    assert chunked_quadrature_slices(10, 20) == (slice(0, 10),)

    single = SingleProcessStrategy()
    assert single.world_size == 1 and single.is_main_process
    sim = SimulatedDistributedStrategy(rank=0, world_size=2)
    reduced = sim.reduce_rank_tensors(
        [torch.tensor([1.0, 3.0]), torch.tensor([3.0, 5.0])]
    )
    assert torch.allclose(reduced, torch.tensor([2.0, 4.0]))
    assert sim.all_reduce_calls == 1


def test_compiler_profile_metadata_accepts_cuda_with_gpu_replay_tier() -> None:
    libs = {
        "numpy": "2.0.0",
        "scikit-fem": "12.0.0",
        "scipy": "1.18.0",
        "sympy": "1.13.0",
        "torch": "2.6.0",
        "safetensors": "0.8.0",
        "threadpoolctl": "3.5.0",
    }
    fp = {"os": "Linux", "machine": "x86_64", "cpu_capability": "AVX2"}

    # Bare cuda without replay_tier is refused (preserving existing contract)
    bad = ExecutionProfile(
        python_version="3.13.0",
        library_versions=libs,
        blas_threads=1,
        device="cuda:0",
        dtype="float32",
        rng_algorithm="torch-cuda-philox",
        seed=0,
        platform_fingerprint=fp,
    )
    with pytest.raises(CompilerConfigError, match="gpu-accelerated-tolerance"):
        _profile_metadata(bad)

    # CUDA with gpu-accelerated-tolerance passes metadata validation
    good = ExecutionProfile(
        python_version="3.13.0",
        library_versions=libs,
        blas_threads=4,
        device="cuda:0",
        dtype="float32",
        rng_algorithm="torch-cuda-philox",
        seed=0,
        platform_fingerprint=fp,
        replay_tier="gpu-accelerated-tolerance",
    )
    _profile_metadata(good)


def test_chunked_volumetric_quadrature_matches_full_batch_gradients(
    tmp_path: Path,
) -> None:
    step = generate_3d_box_step(tmp_path / "chunk_box.step", lx=10.0, ly=6.0, lz=6.0)
    imported = import_cad_3d(step, mesh_size_mm=6.0, element_type=CellType3D.TET10)
    faces = classify_faces_by_bbox(imported)
    mat = MaterialSpec(
        name="nh-chunk",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(10.0, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(0.35, Unit.ONE).to_canonical(),
        },
    )
    tbc = (TractionBC3D("load", faces["x_max"], (0.3, 0.0, 0.0)),)
    cases = [LoadCase3D("c1", (0.3,), tbc)]
    lifting = PlanarClampLifting3D(axis="x", plane_coord=0.0, extent_mm=10.0)
    op_cfg = Surface3DConfig(
        parameter_dim=1, hidden_dim=16, slice_count=4, num_blocks=1, num_heads=2
    )

    b_full = train_surface_3d_operator(
        imported.mesh,
        mat,
        clamped_face_tags=faces["x_min"],
        loaded_face_tags=faces["x_max"],
        lifting=lifting,
        train_cases=cases,
        operator_config=op_cfg,
        training_config=Surface3DTrainingConfig(
            epochs=3, learning_rate=1e-3, quadrature_chunk_size=None, seed=42
        ),
        dtype=torch.float64,
    )
    b_chunked = train_surface_3d_operator(
        imported.mesh,
        mat,
        clamped_face_tags=faces["x_min"],
        loaded_face_tags=faces["x_max"],
        lifting=lifting,
        train_cases=cases,
        operator_config=op_cfg,
        training_config=Surface3DTrainingConfig(
            epochs=3, learning_rate=1e-3, quadrature_chunk_size=16, seed=42
        ),
        dtype=torch.float64,
    )
    # Exact additive gradient equivalence in float64
    for l1, l2 in zip(
        b_full.training_summary.loss_history,
        b_chunked.training_summary.loss_history,
        strict=True,
    ):
        assert abs(l1 - l2) < 1.0e-11
    for p1, p2 in zip(
        b_full.operator.parameters(), b_chunked.operator.parameters(), strict=True
    ):
        assert torch.allclose(p1, p2, atol=1.0e-11, rtol=1.0e-11)


def test_cpu_autocast_bfloat16_and_fp16_guard() -> None:
    cfg_bf16 = ComputeRuntimeConfig(
        device="cpu",
        replay_tier="gpu-accelerated-tolerance",
        amp_dtype="bfloat16",
    )
    with autocast_context(cfg_bf16):
        y = torch.nn.Linear(4, 4)(torch.ones(2, 4))
        assert y.dtype == torch.bfloat16

    cfg_fp16 = ComputeRuntimeConfig(
        device="cpu",
        replay_tier="gpu-accelerated-tolerance",
        amp_dtype="float16",
    )
    with pytest.raises(ValueError, match="unsupported on CPU"):
        with autocast_context(cfg_fp16):
            pass
