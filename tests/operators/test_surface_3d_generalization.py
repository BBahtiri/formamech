"""Verification tests for Phase 2 (V3-2) 3D masked batching, ADF lifting, GALE, LoRA & Foundation Hub."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from piginot.cad.importing_3d import (
    classify_faces_by_bbox,
    generate_3d_box_step,
    import_cad_3d,
)
from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.fem.backend_3d import TractionBC3D
from piginot.geometry.mesh3d import CellType3D
from piginot.lifting.adf_3d import (
    FacePatch3D,
    GeneralFaceADFLifting3D,
    build_adf_lifting_from_mesh_faces,
)
from piginot.operators.foundation_hub import (
    FoundationRegistry,
    load_foundation_3d_checkpoint,
    save_foundation_3d_checkpoint,
)
from piginot.operators.geometry_3d import (
    SURFACE_3D_GALE_FEATURE_DIM,
    GeometryOOD3DStatus,
    GeometryOODDetector3D,
    build_gale_surface_3d_tensors,
    compute_geometry_descriptor_3d,
    pad_and_stack_surface_3d_batch,
)
from piginot.operators.low_rank_adapter_3d import (
    ADAPTER_3D_MACHINE,
    Adapter3DState,
    LowRankAdapter3DConfig,
    adapt_surface_3d_operator,
    inject_lora_3d,
    load_lora_3d_artifact,
    save_lora_3d_artifact,
)
from piginot.operators.surface_operator_3d import (
    Surface3DConfig,
    Surface3DEnergyOperator,
)


def _material() -> MaterialSpec:
    return MaterialSpec(
        name="nh-v32",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(10.0, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(0.35, Unit.ONE).to_canonical(),
        },
    )


def test_masked_surface_3d_operator_batch_matches_unbatched_exactly() -> None:
    torch.manual_seed(0)
    cfg = Surface3DConfig(
        parameter_dim=2,
        surface_feature_dim=SURFACE_3D_GALE_FEATURE_DIM,
        hidden_dim=16,
        slice_count=4,
        num_blocks=2,
        num_heads=2,
    )
    op = Surface3DEnergyOperator(cfg).to(dtype=torch.float64)

    # Create 3 heterogeneous synthetic surfaces of sizes M=12, 20, 7
    items = []
    for m in (12, 20, 7):
        c = torch.randn(1, m, 3, dtype=torch.float64)
        n = torch.nn.functional.normalize(
            torch.randn(1, m, 3, dtype=torch.float64), dim=-1
        )
        a = torch.rand(1, m, dtype=torch.float64) + 0.1
        f = torch.randn(1, m, SURFACE_3D_GALE_FEATURE_DIM, dtype=torch.float64)
        items.append((c, n, a, f))

    c_b, n_b, a_b, f_b, mask_b = pad_and_stack_surface_3d_batch(items)
    enc_batch = op.encode_surface(c_b, n_b, a_b, f_b, surface_mask=mask_b)

    for i, (c, n, a, f) in enumerate(items):
        enc_single = op.encode_surface(c, n, a, f)
        assert torch.allclose(
            enc_batch.origin[i : i + 1], enc_single.origin, atol=1e-12
        )
        assert torch.allclose(
            enc_batch.extent[i : i + 1], enc_single.extent, atol=1e-12
        )
        assert torch.allclose(
            enc_batch.slice_tokens[i : i + 1], enc_single.slice_tokens, atol=1e-12
        )


def test_general_face_adf_lifting_3d_exact_bc_and_fd_gradient() -> None:
    plane = FacePatch3D(
        name="clamp_z0",
        kind="plane",
        prescribed_mm=(0.0, 0.0, 0.0),
        origin_mm=(0.0, 0.0, 0.0),
        axis_or_normal=(0.0, 0.0, 1.0),
    )
    cyl = FacePatch3D(
        name="bore_r3",
        kind="cylinder",
        prescribed_mm=(0.5, 0.0, 0.0),
        origin_mm=(5.0, 5.0, 0.0),
        axis_or_normal=(0.0, 0.0, 1.0),
        radius_mm=3.0,
    )
    lift = GeneralFaceADFLifting3D(patches=(plane, cyl), extent_mm=10.0)
    # Round-trip serialization
    lift_rt = GeneralFaceADFLifting3D.from_document(lift.document())
    assert lift_rt == lift

    # Points exactly on z=0 plane -> scale==0, offset==(0,0,0)
    pts_plane = torch.tensor([[1.0, 2.0, 0.0], [9.0, 1.0, 0.0]], dtype=torch.float64)
    sc_p, off_p = lift.scale_and_offset(pts_plane)
    assert torch.allclose(sc_p, torch.zeros_like(sc_p), atol=1e-13)
    assert torch.allclose(off_p, torch.zeros_like(off_p), atol=1e-13)

    # Points on cylinder r=3 centered at (5,5) -> scale==0, offset==(0.5,0,0)
    pts_cyl = torch.tensor([[8.0, 5.0, 4.0], [5.0, 8.0, 4.0]], dtype=torch.float64)
    sc_c, off_c = lift.scale_and_offset(pts_cyl)
    assert torch.allclose(sc_c, torch.zeros_like(sc_c), atol=1e-13)
    assert torch.allclose(
        off_c,
        torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float64),
        atol=1e-12,
    )

    # Central FD gradient check at an interior point
    pt = torch.tensor([[3.5, 4.2, 2.7]], dtype=torch.float64, requires_grad=True)
    sc, off = lift.scale_and_offset(pt)
    val = (sc + off).sum()
    grad_auto = torch.autograd.grad(val, pt)[0][0].detach().numpy()
    eps = 1.0e-6
    grad_fd = np.zeros(3, dtype=np.float64)
    for k in range(3):
        p_plus = pt.detach().clone()
        p_minus = pt.detach().clone()
        p_plus[0, k] += eps
        p_minus[0, k] -= eps
        sp, op = lift.scale_and_offset(p_plus)
        sm, om = lift.scale_and_offset(p_minus)
        grad_fd[k] = float(((sp + op).sum() - (sm + om).sum()).item()) / (2.0 * eps)
    assert float(np.max(np.abs(grad_auto - grad_fd))) < 1.0e-8


def test_3d_lora_step0_bit_identity_budget_guard_and_foundation_hub(
    tmp_path: Path,
) -> None:
    step_a = generate_3d_box_step(tmp_path / "box_a.step", lx=12.0, ly=7.0, lz=6.0)
    step_b = generate_3d_box_step(tmp_path / "box_b.step", lx=14.0, ly=8.0, lz=6.0)
    imp_a = import_cad_3d(step_a, mesh_size_mm=6.5, element_type=CellType3D.TET10)
    imp_b = import_cad_3d(step_b, mesh_size_mm=6.5, element_type=CellType3D.TET10)
    faces_a = classify_faces_by_bbox(imp_a)
    faces_b = classify_faces_by_bbox(imp_b)

    op = Surface3DEnergyOperator(
        Surface3DConfig(
            parameter_dim=1,
            surface_feature_dim=SURFACE_3D_GALE_FEATURE_DIM,
            hidden_dim=16,
            slice_count=4,
            num_blocks=1,
            num_heads=2,
        )
    ).to(dtype=torch.float64)

    # Budget > 10% refused
    with pytest.raises(ValueError, match="exceeds budget cap"):
        LowRankAdapter3DConfig(max_steps=25, from_scratch_reference_steps=200)

    # Step-0 zero-init LoRA bit-identity
    cfg = LowRankAdapter3DConfig(rank=2, max_steps=4, from_scratch_reference_steps=100)
    lora_op, replaced = inject_lora_3d(op, cfg)
    assert len(replaced) > 0
    sc, sn, sa, sf = build_gale_surface_3d_tensors(
        imp_a.mesh,
        clamped_face_tags=faces_a["x_min"],
        loaded_face_tags=faces_a["x_max"],
    )
    q = torch.tensor(imp_a.mesh.nodes[:10], dtype=torch.float64).unsqueeze(0)
    p = torch.tensor([[0.3]], dtype=torch.float64)
    with torch.no_grad():
        u0 = op.displacement(
            encoding=op.encode_surface(sc, sn, sa, sf), query_xyz=q, parameters=p
        )
        u1 = lora_op.displacement(
            encoding=lora_op.encode_surface(sc, sn, sa, sf), query_xyz=q, parameters=p
        )
    assert torch.equal(u0, u1)

    # Run budgeted 3D LoRA adaptation & persist tamper-checked artifact
    lift_a = build_adf_lifting_from_mesh_faces(imp_a.mesh, faces_a["x_min"])
    adapted, res = adapt_surface_3d_operator(
        op,
        mesh=imp_a.mesh,
        material=_material(),
        clamped_face_tags=faces_a["x_min"],
        loaded_face_tags=faces_a["x_max"],
        lifting=lift_a,
        parameter_vector=(0.3,),
        traction_bcs=(TractionBC3D("load", faces_a["x_max"], (0.3, 0.0, 0.0)),),
        config=cfg,
    )
    assert adapted is not None
    assert res.state is Adapter3DState.COMPLETED
    assert res.snapshot().state == "COMPLETED"
    assert ADAPTER_3D_MACHINE.machine_id == "formamech.operators.surface_3d_lora"
    assert res.final_loss <= res.initial_loss + 1e-6

    save_lora_3d_artifact(
        tmp_path / "lora3d", adapted_operator=adapted, config=cfg, result=res
    )
    reloaded, manifest = load_lora_3d_artifact(tmp_path / "lora3d", op)
    assert manifest["claim_level"] == "NOT_VERIFIED"
    with torch.no_grad():
        ua = adapted.displacement(
            encoding=adapted.encode_surface(sc, sn, sa, sf), query_xyz=q, parameters=p
        )
        ub = reloaded.displacement(
            encoding=reloaded.encode_surface(sc, sn, sa, sf), query_xyz=q, parameters=p
        )
    assert torch.allclose(ua, ub, atol=1e-13)

    # Foundation checkpoint save / registry lookup / tamper refusal
    desc_a = compute_geometry_descriptor_3d(
        imp_a.mesh,
        clamped_face_tags=faces_a["x_min"],
        loaded_face_tags=faces_a["x_max"],
    )
    desc_b = compute_geometry_descriptor_3d(
        imp_b.mesh,
        clamped_face_tags=faces_b["x_min"],
        loaded_face_tags=faces_b["x_max"],
    )
    ood = GeometryOODDetector3D.fit([desc_a, desc_b])
    assert ood.score(desc_a).status is GeometryOOD3DStatus.IN_DISTRIBUTION

    hub_dir = tmp_path / "hub"
    save_foundation_3d_checkpoint(
        hub_dir / "mechfm-box-v1",
        checkpoint_id="mechfm-box-v1",
        family_domain="3d-box-hyperelastic",
        operator=op,
        ood_detector=ood,
    )
    reg = FoundationRegistry(hub_dir)
    specs = reg.list_checkpoints()
    assert len(specs) == 1 and specs[0].checkpoint_id == "mechfm-box-v1"
    loaded_f = reg.load("mechfm-box-v1")
    assert loaded_f.ood_detector is not None
    assert loaded_f.ood_detector.detector_hash == ood.detector_hash

    # Tampering weights fails closed
    w_file = hub_dir / "mechfm-box-v1" / "weights.safetensors"
    w_file.write_bytes(w_file.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_foundation_3d_checkpoint(hub_dir / "mechfm-box-v1")
