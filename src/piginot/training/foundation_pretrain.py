"""Multi-family 3D structural foundation pretraining & budgeted LoRA transfer (V4-2).

Trains one shared 16-channel GALE :class:`Surface3DEnergyOperator` across
multiple distinct OpenCASCADE 3D structural component families (holed block,
L-bracket, mounting lug), fits a multi-family :class:`GeometryOODDetector3D`,
saves tamper-checked ``safetensors`` checkpoints via ``foundation_hub``, and
provides one-call budgeted 3D LoRA transfer onto unseen user 3D parts.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from piginot.cad.parametric_3d import (
    ParametricCadSpec3D,
    ParametricInstance3D,
    default_holed_block_3d_spec,
    default_l_bracket_3d_spec,
    default_mounting_lug_3d_spec,
    sample_parametric_cad_3d_doe,
)
from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.fem.backend_3d import TractionBC3D, assemble_surface_tractions_3d
from piginot.geometry.mesh3d import CanonicalMesh3D
from piginot.lifting.adf_3d import (
    GeneralFaceADFLifting3D,
    build_adf_lifting_from_mesh_faces,
)
from piginot.operators.foundation_hub import (
    FoundationCheckpointSpec,
    load_foundation_3d_checkpoint,
    save_foundation_3d_checkpoint,
)
from piginot.operators.geometry_3d import (
    SURFACE_3D_GALE_FEATURE_DIM,
    GeometryOODDetector3D,
    build_gale_surface_3d_tensors,
    compute_geometry_descriptor_3d,
    pad_and_stack_surface_3d_batch,
)
from piginot.operators.low_rank_adapter_3d import (
    Adapter3DResult,
    LowRankAdapter3DConfig,
    adapt_surface_3d_operator,
)
from piginot.operators.surface_operator_3d import (
    Surface3DConfig,
    Surface3DEnergyOperator,
)
from piginot.physics import energy_density, reference_modulus
from piginot.training.energy_engine_3d import _volumetric_quadrature_3d

FOUNDATION_PARAM_DIM: Final = 4
DEFAULT_FOUNDATION_ID: Final = "struct3d-base-v1"


@dataclass(frozen=True, slots=True)
class FoundationPretrainConfig:
    """Hyperparameters for multi-family 3D structural foundation pretraining."""

    checkpoint_id: str = DEFAULT_FOUNDATION_ID
    family_domain: str = "structural-3d-multi-family"
    samples_per_family: int = 2
    mesh_size_mm: float = 9.0
    epochs: int = 18
    learning_rate: float = 3.0e-3
    hidden_dim: int = 32
    slice_count: int = 6
    num_blocks: int = 2
    num_heads: int = 4
    quadrature_subsample: int = 48
    youngs_modulus_mpa: float = 20.0
    poisson_ratio: float = 0.35
    traction_x_mpa: float = 0.30
    seed: int = 7


def _canonical_param_vector(
    spec: ParametricCadSpec3D, params: dict[str, float]
) -> list[float]:
    """Map variable-length CAD parameters into fixed ``FOUNDATION_PARAM_DIM`` space."""
    vec: list[float] = []
    for p in spec.parameters[:FOUNDATION_PARAM_DIM]:
        span = max(float(p.upper - p.lower), 1.0e-12)
        vec.append(2.0 * (float(params[p.name]) - float(p.lower)) / span - 1.0)
    while len(vec) < FOUNDATION_PARAM_DIM:
        vec.append(0.0)
    return vec


def pretrain_struct3d_foundation(
    destination_dir: Path | str,
    *,
    config: FoundationPretrainConfig | None = None,
) -> tuple[FoundationCheckpointSpec, dict[str, Any]]:
    """Pretrain a multi-family 3D structural foundation model and persist its checkpoint."""
    cfg = FoundationPretrainConfig() if config is None else config
    torch.manual_seed(cfg.seed)
    dtype = torch.float32

    specs = (
        default_holed_block_3d_spec(mesh_size_mm=cfg.mesh_size_mm),
        default_l_bracket_3d_spec(mesh_size_mm=cfg.mesh_size_mm),
        default_mounting_lug_3d_spec(mesh_size_mm=cfg.mesh_size_mm),
    )

    instances: list[tuple[ParametricCadSpec3D, ParametricInstance3D]] = []
    for idx, spec in enumerate(specs):
        doe = sample_parametric_cad_3d_doe(
            spec,
            count=cfg.samples_per_family,
            seed_offset=cfg.seed + 10 * idx + 1,
        )
        for inst in doe.instances:
            instances.append((spec, inst))

    op_cfg = Surface3DConfig(
        parameter_dim=FOUNDATION_PARAM_DIM,
        surface_feature_dim=SURFACE_3D_GALE_FEATURE_DIM,
        hidden_dim=cfg.hidden_dim,
        slice_count=cfg.slice_count,
        num_blocks=cfg.num_blocks,
        num_heads=cfg.num_heads,
    )
    operator = Surface3DEnergyOperator(op_cfg).to(dtype=dtype)
    mat = MaterialSpec(
        name="foundation-3d-nh",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(
                cfg.youngs_modulus_mpa, Unit.MPA
            ).to_canonical(),
            "poisson_ratio": Quantity(cfg.poisson_ratio, Unit.ONE).to_canonical(),
        },
    )
    e_ref = reference_modulus(mat)

    surf_items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    liftings: list[GeneralFaceADFLifting3D] = []
    quad_data: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    nodes_list: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    p_rows: list[list[float]] = []

    for spec, inst in instances:
        mesh = inst.imported.mesh
        surf_items.append(
            build_gale_surface_3d_tensors(
                mesh,
                clamped_face_tags=inst.clamped_face_tags,
                loaded_face_tags=inst.loaded_face_tags,
                dtype=dtype,
            )
        )
        lift = build_adf_lifting_from_mesh_faces(
            mesh, inst.clamped_face_tags, displacement_scale_mm=1.0
        )
        liftings.append(lift)
        q_pts, q_wts, vol = _volumetric_quadrature_3d(mesh, dtype=dtype)
        quad_data.append((q_pts, q_wts, vol))
        n_t = torch.tensor(mesh.nodes, dtype=dtype).unsqueeze(0)
        n_sc, n_off = lift.scale_and_offset(n_t)
        f_ext = torch.tensor(
            assemble_surface_tractions_3d(
                mesh,
                (
                    TractionBC3D(
                        "loaded",
                        inst.loaded_face_tags,
                        (cfg.traction_x_mpa, 0.0, 0.0),
                    ),
                ),
            ),
            dtype=dtype,
        ).unsqueeze(0)
        nodes_list.append((n_t, n_sc, n_off, f_ext))
        p_rows.append(_canonical_param_vector(spec, dict(inst.parameters)))

    s_cent, s_norm, s_area, s_feat, s_mask = pad_and_stack_surface_3d_batch(surf_items)
    p_batch = torch.tensor(p_rows, dtype=dtype)

    opt = torch.optim.Adam(operator.parameters(), lr=cfg.learning_rate)
    loss_history: list[float] = []
    gen = torch.Generator().manual_seed(cfg.seed)

    for _epoch in range(cfg.epochs):
        opt.zero_grad(set_to_none=True)
        enc = operator.encode_surface(
            s_cent, s_norm, s_area, s_feat, surface_mask=s_mask
        )
        losses: list[torch.Tensor] = []
        for idx in range(len(instances)):
            q_pts, q_wts, vol = quad_data[idx]
            n_q = int(q_pts.shape[0])
            if n_q > cfg.quadrature_subsample:
                sel = torch.randperm(n_q, generator=gen)[: cfg.quadrature_subsample]
                q_use = q_pts[sel].unsqueeze(0)
                w_use = q_wts[sel] * (vol / float(q_wts[sel].sum().item()))
            else:
                q_use = q_pts.unsqueeze(0)
                w_use = q_wts
            enc_i = type(enc)(
                slice_tokens=enc.slice_tokens[idx : idx + 1],
                origin=enc.origin[idx : idx + 1],
                extent=enc.extent[idx : idx + 1],
            )
            p_i = p_batch[idx : idx + 1]
            pts = q_use.detach().requires_grad_(True)
            q_sc, q_off = liftings[idx].scale_and_offset(pts)
            u_q = operator.displacement(
                encoding=enc_i,
                query_xyz=pts,
                parameters=p_i,
                lifting_offset=q_off,
                lifting_scale=q_sc,
            )
            grads: list[torch.Tensor] = []
            for comp in range(3):
                g = torch.autograd.grad(
                    u_q[..., comp].sum(),
                    pts,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                grads.append(g)
            grad_u = torch.stack(grads, dim=-2)
            f_mat = grad_u.squeeze(0) + torch.eye(3, dtype=dtype)
            det_f = torch.linalg.det(f_mat)
            w_int = energy_density(f_mat, mat)
            e_int = torch.sum(w_int * w_use)
            barrier = 200.0 * torch.relu(0.25 - det_f).square().mean()

            n_t, n_sc, n_off, f_ext = nodes_list[idx]
            u_nod = operator.displacement(
                encoding=enc_i,
                query_xyz=n_t,
                parameters=p_i,
                lifting_offset=n_off,
                lifting_scale=n_sc,
            )
            w_ext = torch.sum(u_nod * f_ext)
            losses.append((e_int - w_ext) / max(e_ref * vol, 1.0e-8) + barrier)

        total_loss = torch.stack(losses).mean()
        if not math.isfinite(float(total_loss.item())):
            raise RuntimeError("non-finite loss during 3D foundation pretraining")
        torch.autograd.backward(total_loss)
        torch.nn.utils.clip_grad_norm_(operator.parameters(), max_norm=5.0)
        opt.step()
        loss_history.append(float(total_loss.item()))

    # Fit GeometryOODDetector3D on the primary genus-1 (chi=0) TRAIN descriptors
    # (holed-block + L-bracket) in pure 11-dim intrinsic descriptor space so
    # fail-closed scoring works directly on any GeometryDescriptor3D.
    descriptors = [inst.descriptor for _, inst in instances]
    primary_descriptors = [
        d for d in descriptors if d.surface_euler_characteristic == 0
    ]
    if len(primary_descriptors) < 2:
        primary_descriptors = descriptors
    ood_detector = GeometryOODDetector3D.fit(primary_descriptors)

    spec_saved = save_foundation_3d_checkpoint(
        destination_dir,
        checkpoint_id=cfg.checkpoint_id,
        family_domain=cfg.family_domain,
        operator=operator,
        ood_detector=ood_detector,
        training_mode="variational-energy-label-free",
        claim_level="NOT_VERIFIED",
    )
    summary = {
        "checkpoint_id": spec_saved.checkpoint_id,
        "families": [s.template_name for s in specs],
        "instance_count": len(instances),
        "epochs": cfg.epochs,
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "weights_sha256": spec_saved.weights_sha256,
        "parameter_count": sum(p.numel() for p in operator.parameters()),
    }
    return spec_saved, summary


def default_foundation_checkpoint_dir(repo_root: Path | None = None) -> Path:
    """Resolve the shipped ``bundles/foundation/struct3d-base-v1`` checkpoint directory."""
    root = (
        Path(__file__).resolve().parents[3]
        if repo_root is None
        else Path(repo_root).resolve()
    )
    return root / "bundles" / "foundation" / DEFAULT_FOUNDATION_ID


def adapt_foundation_to_3d_part(
    checkpoint_dir: Path | str,
    *,
    mesh: CanonicalMesh3D,
    material: MaterialSpec,
    clamped_face_tags: Sequence[str],
    loaded_face_tags: Sequence[str],
    traction_mpa: tuple[float, float, float] = (0.30, 0.0, 0.0),
    parameter_vector: Sequence[float] | None = None,
    adapter_config: LowRankAdapter3DConfig | None = None,
) -> tuple[Surface3DEnergyOperator | None, Adapter3DResult]:
    """Load a verified 3D foundation checkpoint and adapt via budgeted 3D LoRA."""
    loaded = load_foundation_3d_checkpoint(checkpoint_dir)
    lift = build_adf_lifting_from_mesh_faces(
        mesh, clamped_face_tags, displacement_scale_mm=1.0
    )
    p_dim = loaded.spec.operator_config.parameter_dim
    p_vec = (
        [0.0] * p_dim
        if parameter_vector is None
        else list(parameter_vector[:p_dim])
        + [0.0] * max(0, p_dim - len(parameter_vector))
    )
    ood_report = None
    if loaded.ood_detector is not None:
        desc = compute_geometry_descriptor_3d(
            mesh,
            clamped_face_tags=clamped_face_tags,
            loaded_face_tags=loaded_face_tags,
        )
        ood_report = loaded.ood_detector.score(desc)
    cfg = (
        LowRankAdapter3DConfig(
            rank=4,
            max_steps=8,
            from_scratch_reference_steps=100,
            allow_ood_adaptation=True,
        )
        if adapter_config is None
        else adapter_config
    )
    return adapt_surface_3d_operator(
        loaded.operator,
        mesh=mesh,
        material=material,
        clamped_face_tags=clamped_face_tags,
        loaded_face_tags=loaded_face_tags,
        lifting=lift,
        parameter_vector=p_vec,
        traction_bcs=(TractionBC3D("loaded", tuple(loaded_face_tags), traction_mpa),),
        config=cfg,
        ood_report=ood_report,
    )


__all__ = (
    "DEFAULT_FOUNDATION_ID",
    "FOUNDATION_PARAM_DIM",
    "FoundationPretrainConfig",
    "adapt_foundation_to_3d_part",
    "default_foundation_checkpoint_dir",
    "pretrain_struct3d_foundation",
)
