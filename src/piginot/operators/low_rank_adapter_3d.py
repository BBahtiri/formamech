"""Budgeted low-rank physics-informed adapters (3D LoRA) for ``Surface3DEnergyOperator`` (V3-2).

Injects zero-initialized :class:`~piginot.operators.low_rank_adapter.LowRankLinear`
residuals into frozen 3D slice-attention and decoder layers:

* Exact zero-shot bit-identity at step 0 (zero-initialized output factor ``B``),
* Strictly frozen base operator weights,
* Budget ceiling (default ``<= 10%`` of from-scratch step count),
* Fail-closed refusal on topology-refused 3D CAD variants,
* Tamper-checked ``adapter-3d.safetensors`` + canonical ``adapter-3d.json``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
from torch import nn

from piginot.contracts import MaterialSpec, canonical_json_bytes
from piginot.contracts.state_machine import (
    StateMachineSpec,
    StateSnapshot,
    TransitionSpec,
    register_state_machine,
)
from piginot.fem.backend_3d import TractionBC3D, assemble_surface_tractions_3d
from piginot.geometry.mesh3d import CanonicalMesh3D
from piginot.lifting.adf_3d import GeneralFaceADFLifting3D
from piginot.operators.geometry_3d import (
    SURFACE_3D_GALE_FEATURE_DIM,
    GeometryOOD3DStatus,
    GeometryOODReport3D,
    build_gale_surface_3d_tensors,
)
from piginot.operators.low_rank_adapter import LowRankLinear
from piginot.operators.surface_operator_3d import (
    Surface3DEncoding,
    Surface3DEnergyOperator,
)
from piginot.physics import energy_density, reference_modulus
from piginot.training.energy_engine_3d import (
    Lifting3D,
    _volumetric_quadrature_3d,
    build_surface_3d_tensors,
    lifted_displacement_and_gradient,
)

ADAPTER_3D_SCHEMA: Final = "formamech.surface-3d-lora-v1"
DEFAULT_3D_LORA_TARGETS: Final[tuple[str, ...]] = (
    "out_proj",
    "decoder_head",
    "query_in",
)


class Adapter3DState(StrEnum):
    """Explicit lifecycle state machine for budgeted 3D geometry adaptation."""

    INITIALIZED = "INITIALIZED"
    OOD_CHECKED = "OOD_CHECKED"
    ADAPTING = "ADAPTING"
    COMPLETED = "COMPLETED"
    REFUSED = "REFUSED"


ADAPTER_3D_MACHINE: Final[StateMachineSpec[Adapter3DState]] = register_state_machine(
    StateMachineSpec(
        machine_id="formamech.operators.surface_3d_lora",
        description=(
            "Budgeted low-rank physics-informed 3D CAD adaptation lifecycle: "
            "INITIALIZED -> OOD_CHECKED -> ADAPTING -> COMPLETED | REFUSED."
        ),
        states=tuple(Adapter3DState),
        initial_states=(Adapter3DState.INITIALIZED,),
        terminal_states=(Adapter3DState.COMPLETED, Adapter3DState.REFUSED),
        transitions=(
            TransitionSpec(
                Adapter3DState.INITIALIZED,
                Adapter3DState.OOD_CHECKED,
                "check_ood",
                "Evaluate 3D geometry OOD and topology admissibility.",
            ),
            TransitionSpec(
                Adapter3DState.INITIALIZED,
                Adapter3DState.REFUSED,
                "refuse_invalid",
                "Refuse invalid adapter configuration or unsupported topology.",
            ),
            TransitionSpec(
                Adapter3DState.OOD_CHECKED,
                Adapter3DState.ADAPTING,
                "begin_adapt",
                "Topology admitted; inject zero-initialized 3D LoRA modules.",
            ),
            TransitionSpec(
                Adapter3DState.OOD_CHECKED,
                Adapter3DState.REFUSED,
                "refuse_topology",
                "Refuse topology-broken or disallowed OOD 3D geometry.",
            ),
            TransitionSpec(
                Adapter3DState.ADAPTING,
                Adapter3DState.COMPLETED,
                "complete_adapt",
                "Budgeted 3D LoRA optimization completed with finite admissible fields.",
            ),
            TransitionSpec(
                Adapter3DState.ADAPTING,
                Adapter3DState.REFUSED,
                "abort_nonfinite",
                "Adaptation aborted due to non-finite loss or inverted deformation.",
            ),
        ),
    )
)


@dataclass(frozen=True, slots=True)
class LowRankAdapter3DConfig:
    """Budgeted low-rank adapter specification for one unseen 3D CAD part."""

    rank: int = 4
    alpha: float = 4.0
    target_substrings: tuple[str, ...] = DEFAULT_3D_LORA_TARGETS
    max_steps: int = 16
    from_scratch_reference_steps: int = 200
    max_step_budget_fraction: float = 0.10
    learning_rate: float = 8.0e-3
    barrier_weight: float = 400.0
    barrier_threshold: float = 0.2
    quadrature_subsample: int | None = 128
    allow_ood_adaptation: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be a positive integer")
        if not math.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("alpha must be finite and positive")
        if not self.target_substrings:
            raise ValueError("target_substrings must be non-empty")
        if type(self.max_steps) is not int or self.max_steps <= 0:
            raise ValueError("max_steps must be >= 1")
        if (
            type(self.from_scratch_reference_steps) is not int
            or self.from_scratch_reference_steps <= 0
        ):
            raise ValueError("from_scratch_reference_steps must be >= 1")
        if not (0.0 < self.max_step_budget_fraction <= 1.0):
            raise ValueError("max_step_budget_fraction must lie in (0, 1]")
        budget_cap = math.floor(
            self.from_scratch_reference_steps * self.max_step_budget_fraction + 1.0e-12
        )
        if self.max_steps > budget_cap:
            raise ValueError(
                f"max_steps={self.max_steps} exceeds budget cap {budget_cap} "
                f"({self.max_step_budget_fraction:.0%} of {self.from_scratch_reference_steps})"
            )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")

    def document(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "alpha": float(self.alpha),
            "target_substrings": list(self.target_substrings),
            "max_steps": self.max_steps,
            "from_scratch_reference_steps": self.from_scratch_reference_steps,
            "max_step_budget_fraction": float(self.max_step_budget_fraction),
            "learning_rate": float(self.learning_rate),
            "barrier_weight": float(self.barrier_weight),
            "barrier_threshold": float(self.barrier_threshold),
            "quadrature_subsample": self.quadrature_subsample,
            "allow_ood_adaptation": self.allow_ood_adaptation,
            "seed": self.seed,
        }


def inject_lora_3d(
    base_operator: Surface3DEnergyOperator,
    config: LowRankAdapter3DConfig,
) -> tuple[Surface3DEnergyOperator, list[str]]:
    """Clone ``base_operator``, freeze all base weights, and inject zero-init LoRA."""
    adapted = copy.deepcopy(base_operator)
    for param in adapted.parameters():
        param.requires_grad_(False)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(config.seed)
    replaced: list[str] = []

    def _walk(module: nn.Module, prefix: str) -> None:
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and any(
                sub in full_name for sub in config.target_substrings
            ):
                setattr(
                    module,
                    child_name,
                    LowRankLinear(
                        child, rank=config.rank, alpha=config.alpha, generator=gen
                    ),
                )
                replaced.append(full_name)
            else:
                _walk(child, full_name)

    _walk(adapted, "")
    if not replaced:
        raise ValueError(
            f"no nn.Linear modules matched target_substrings={config.target_substrings!r}"
        )
    return adapted, replaced


def extract_lora_3d_state_dict(
    adapted_operator: Surface3DEnergyOperator,
) -> dict[str, torch.Tensor]:
    """Extract only trainable low-rank ``lora_a`` / ``lora_b`` tensors on CPU."""
    out: dict[str, torch.Tensor] = {}
    for name, module in adapted_operator.named_modules():
        if isinstance(module, LowRankLinear):
            out[f"{name}.lora_a"] = module.lora_a.detach().contiguous().cpu()
            out[f"{name}.lora_b"] = module.lora_b.detach().contiguous().cpu()
    return out


def apply_lora_3d_state_dict(
    base_operator: Surface3DEnergyOperator,
    config: LowRankAdapter3DConfig,
    lora_tensors: Mapping[str, torch.Tensor],
) -> Surface3DEnergyOperator:
    """Re-inject and load persisted 3D LoRA weights onto a frozen base operator."""
    adapted, _ = inject_lora_3d(base_operator, config)
    with torch.no_grad():
        for name, module in adapted.named_modules():
            if isinstance(module, LowRankLinear):
                ka = f"{name}.lora_a"
                kb = f"{name}.lora_b"
                if ka not in lora_tensors or kb not in lora_tensors:
                    raise ValueError(f"missing 3D LoRA tensors for layer {name!r}")
                module.lora_a.copy_(
                    lora_tensors[ka].to(
                        dtype=module.lora_a.dtype, device=module.lora_a.device
                    )
                )
                module.lora_b.copy_(
                    lora_tensors[kb].to(
                        dtype=module.lora_b.dtype, device=module.lora_b.device
                    )
                )
    return adapted


@dataclass(frozen=True, slots=True)
class Adapter3DResult:
    """Auditable outcome of a budgeted 3D physics-informed LoRA run."""

    mesh_id: str
    state: Adapter3DState
    adapter_parameter_count: int
    base_parameter_count: int
    replaced_modules: tuple[str, ...]
    steps_executed: int
    initial_loss: float
    final_loss: float
    min_det_f: float
    wall_time_s: float
    training_mode: str
    claim_level: str
    reason: str

    def snapshot(self) -> StateSnapshot[Adapter3DState]:
        return ADAPTER_3D_MACHINE.snapshot(
            self.state,
            blockers=() if self.state is Adapter3DState.COMPLETED else (self.reason,),
            metadata={
                "steps_executed": self.steps_executed,
                "adapter_parameter_count": self.adapter_parameter_count,
            },
        )

    def document(self) -> dict[str, Any]:
        return {
            "schema": ADAPTER_3D_SCHEMA,
            "mesh_id": self.mesh_id,
            "state": self.state.value,
            "adapter_parameter_count": self.adapter_parameter_count,
            "base_parameter_count": self.base_parameter_count,
            "replaced_modules": list(self.replaced_modules),
            "steps_executed": self.steps_executed,
            "initial_loss": (
                float(self.initial_loss) if math.isfinite(self.initial_loss) else None
            ),
            "final_loss": (
                float(self.final_loss) if math.isfinite(self.final_loss) else None
            ),
            "min_det_f": (
                float(self.min_det_f) if math.isfinite(self.min_det_f) else None
            ),
            "wall_time_s": float(self.wall_time_s),
            "training_mode": self.training_mode,
            "claim_level": self.claim_level,
            "reason": self.reason,
        }


def adapt_surface_3d_operator(
    base_operator: Surface3DEnergyOperator,
    *,
    mesh: CanonicalMesh3D,
    material: MaterialSpec,
    clamped_face_tags: Sequence[str],
    loaded_face_tags: Sequence[str],
    lifting: Lifting3D | GeneralFaceADFLifting3D,
    parameter_vector: Sequence[float],
    traction_bcs: Sequence[TractionBC3D],
    config: LowRankAdapter3DConfig | None = None,
    ood_report: GeometryOODReport3D | None = None,
) -> tuple[Surface3DEnergyOperator | None, Adapter3DResult]:
    """Run budgeted label-free 3D Deep Ritz LoRA adaptation on an unseen 3D CAD part."""
    t0 = time.perf_counter()
    cfg = LowRankAdapter3DConfig() if config is None else config
    base_params = sum(p.numel() for p in base_operator.parameters())
    state = Adapter3DState.INITIALIZED

    # Locking & OOD / topology gate
    nu_prop = material.properties.get("poisson_ratio")
    if nu_prop is not None:
        mesh.validate_incompressibility_locking(float(nu_prop.value))

    state = ADAPTER_3D_MACHINE.require_transition(
        state, Adapter3DState.OOD_CHECKED, trigger="check_ood"
    ).to_state
    if ood_report is not None:
        if ood_report.status is GeometryOOD3DStatus.TOPOLOGY_REFUSED:
            state = ADAPTER_3D_MACHINE.require_transition(
                state, Adapter3DState.REFUSED, trigger="refuse_topology"
            ).to_state
            return (
                None,
                Adapter3DResult(
                    mesh_id=mesh.mesh_id,
                    state=state,
                    adapter_parameter_count=0,
                    base_parameter_count=base_params,
                    replaced_modules=(),
                    steps_executed=0,
                    initial_loss=float("nan"),
                    final_loss=float("nan"),
                    min_det_f=float("nan"),
                    wall_time_s=time.perf_counter() - t0,
                    training_mode="variational-energy-label-free",
                    claim_level="NOT_VERIFIED",
                    reason=f"topology refused: {ood_report.reason}",
                ),
            )
        if (
            ood_report.status is GeometryOOD3DStatus.OUT_OF_DISTRIBUTION
            and not cfg.allow_ood_adaptation
        ):
            state = ADAPTER_3D_MACHINE.require_transition(
                state, Adapter3DState.REFUSED, trigger="refuse_topology"
            ).to_state
            return (
                None,
                Adapter3DResult(
                    mesh_id=mesh.mesh_id,
                    state=state,
                    adapter_parameter_count=0,
                    base_parameter_count=base_params,
                    replaced_modules=(),
                    steps_executed=0,
                    initial_loss=float("nan"),
                    final_loss=float("nan"),
                    min_det_f=float("nan"),
                    wall_time_s=time.perf_counter() - t0,
                    training_mode="variational-energy-label-free",
                    claim_level="NOT_VERIFIED",
                    reason=f"OOD adaptation disallowed: {ood_report.reason}",
                ),
            )

    adapted, replaced = inject_lora_3d(base_operator, cfg)
    trainable = [p for p in adapted.parameters() if p.requires_grad]
    n_adapter = sum(p.numel() for p in trainable)
    state = ADAPTER_3D_MACHINE.require_transition(
        state, Adapter3DState.ADAPTING, trigger="begin_adapt"
    ).to_state

    dtype = next(base_operator.parameters()).dtype
    device = next(base_operator.parameters()).device
    if base_operator.config.surface_feature_dim == SURFACE_3D_GALE_FEATURE_DIM:
        s_cent, s_norm, s_area, s_feat = build_gale_surface_3d_tensors(
            mesh,
            clamped_face_tags=clamped_face_tags,
            loaded_face_tags=loaded_face_tags,
            dtype=dtype,
            device=device,
        )
    else:
        s_cent, s_norm, s_area, s_feat = build_surface_3d_tensors(
            mesh,
            clamped_face_tags=clamped_face_tags,
            loaded_face_tags=loaded_face_tags,
            dtype=dtype,
            device=device,
        )
    q_pts_all, q_wts_all, total_vol = _volumetric_quadrature_3d(
        mesh, dtype=dtype, device=device
    )
    nodes_t = torch.tensor(mesh.nodes, dtype=dtype, device=device).unsqueeze(0)
    node_scale, node_offset = lifting.scale_and_offset(nodes_t)
    f_ext = torch.tensor(
        assemble_surface_tractions_3d(mesh, traction_bcs), dtype=dtype, device=device
    ).unsqueeze(0)
    p_t = torch.tensor([list(parameter_vector)], dtype=dtype, device=device)
    energy_scale = max(reference_modulus(material) * total_vol, 1.0e-12)
    eye3 = torch.eye(3, dtype=dtype, device=device)

    optimizer = torch.optim.Adam(trainable, lr=cfg.learning_rate)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(cfg.seed)
    n_q = int(q_pts_all.shape[0])

    losses: list[float] = []
    min_det_seen = float("inf")
    for _ in range(cfg.max_steps):
        optimizer.zero_grad(set_to_none=True)
        if cfg.quadrature_subsample is not None and cfg.quadrature_subsample < n_q:
            idx = torch.randperm(n_q, generator=gen)[: cfg.quadrature_subsample].to(
                device=device
            )
            q_pts = q_pts_all[idx].unsqueeze(0)
            q_wts = q_wts_all[idx] * (float(n_q) / float(cfg.quadrature_subsample))
        else:
            q_pts = q_pts_all.unsqueeze(0)
            q_wts = q_wts_all

        enc = adapted.encode_surface(s_cent, s_norm, s_area, s_feat)
        enc_1 = Surface3DEncoding(
            slice_tokens=enc.slice_tokens, origin=enc.origin, extent=enc.extent
        )
        _, grad_u = lifted_displacement_and_gradient(
            adapted,
            encoding=enc_1,
            query_xyz=q_pts,
            parameters=p_t,
            lifting=lifting,  # type: ignore[arg-type]
            create_graph=True,
        )
        f_def = eye3 + grad_u
        det_f = torch.linalg.det(f_def)
        min_det_seen = min(min_det_seen, float(det_f.detach().min().item()))
        adm = det_f > 1.0e-5
        safe_f = torch.where(adm[..., None, None], f_def, eye3.expand_as(f_def))
        w_d = torch.where(
            adm, energy_density(safe_f, material), torch.zeros_like(det_f)
        )
        strain_e = torch.sum(w_d * q_wts.unsqueeze(0), dim=-1)
        u_nodes = adapted.displacement(
            encoding=enc_1,
            query_xyz=nodes_t,
            parameters=p_t,
            lifting_offset=node_offset,
            lifting_scale=node_scale,
        )
        ext_w = torch.sum(f_ext * u_nodes, dim=(-2, -1))
        barrier = torch.mean(torch.relu(cfg.barrier_threshold - det_f).square())
        loss = (
            torch.mean((strain_e - ext_w) / energy_scale) + cfg.barrier_weight * barrier
        )
        loss_val = float(loss.detach().item())
        if not math.isfinite(loss_val):
            state = ADAPTER_3D_MACHINE.require_transition(
                state, Adapter3DState.REFUSED, trigger="abort_nonfinite"
            ).to_state
            return (
                None,
                Adapter3DResult(
                    mesh_id=mesh.mesh_id,
                    state=state,
                    adapter_parameter_count=n_adapter,
                    base_parameter_count=base_params,
                    replaced_modules=tuple(replaced),
                    steps_executed=len(losses),
                    initial_loss=losses[0] if losses else float("nan"),
                    final_loss=loss_val,
                    min_det_f=min_det_seen,
                    wall_time_s=time.perf_counter() - t0,
                    training_mode="variational-energy-label-free",
                    claim_level="NOT_VERIFIED",
                    reason="non-finite 3D LoRA loss",
                ),
            )
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
        optimizer.step()
        losses.append(loss_val)

    state = ADAPTER_3D_MACHINE.require_transition(
        state, Adapter3DState.COMPLETED, trigger="complete_adapt"
    ).to_state
    return (
        adapted,
        Adapter3DResult(
            mesh_id=mesh.mesh_id,
            state=state,
            adapter_parameter_count=n_adapter,
            base_parameter_count=base_params,
            replaced_modules=tuple(replaced),
            steps_executed=len(losses),
            initial_loss=losses[0],
            final_loss=losses[-1],
            min_det_f=min_det_seen,
            wall_time_s=time.perf_counter() - t0,
            training_mode="variational-energy-label-free",
            claim_level="NOT_VERIFIED",
            reason="completed",
        ),
    )


def save_lora_3d_artifact(
    output_dir: Path | str,
    *,
    adapted_operator: Surface3DEnergyOperator,
    config: LowRankAdapter3DConfig,
    result: Adapter3DResult,
) -> dict[str, Any]:
    """Persist tamper-checked ``adapter-3d.safetensors`` + ``adapter-3d.json``."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tensors = extract_lora_3d_state_dict(adapted_operator)
    raw = save_safetensors(tensors, metadata={"schema": ADAPTER_3D_SCHEMA})
    w_path = out / "adapter-3d.safetensors"
    w_path.write_bytes(raw)
    w_sha = hashlib.sha256(raw).hexdigest()
    manifest = {
        "schema": ADAPTER_3D_SCHEMA,
        "weights_file": "adapter-3d.safetensors",
        "weights_sha256": w_sha,
        "config": config.document(),
        "result": result.document(),
        "claim_level": "NOT_VERIFIED",
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    (out / "adapter-3d.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def load_lora_3d_artifact(
    artifact_dir: Path | str,
    base_operator: Surface3DEnergyOperator,
) -> tuple[Surface3DEnergyOperator, dict[str, Any]]:
    """Verify SHA-256 digests and apply persisted 3D LoRA weights to ``base_operator``."""
    root = Path(artifact_dir).resolve()
    manifest = json.loads((root / "adapter-3d.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != ADAPTER_3D_SCHEMA:
        raise ValueError(
            f"unexpected 3D LoRA manifest schema {manifest.get('schema')!r}"
        )
    recorded_m = manifest.get("manifest_sha256")
    without_m = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if hashlib.sha256(canonical_json_bytes(without_m)).hexdigest() != recorded_m:
        raise ValueError("3D LoRA manifest SHA-256 mismatch (tampered artifact)")
    w_bytes = (root / str(manifest["weights_file"])).read_bytes()
    if hashlib.sha256(w_bytes).hexdigest() != manifest["weights_sha256"]:
        raise ValueError("3D LoRA weights SHA-256 mismatch (tampered safetensors)")
    cfg_doc = dict(manifest["config"])
    cfg_doc["target_substrings"] = tuple(cfg_doc["target_substrings"])
    cfg = LowRankAdapter3DConfig(**cfg_doc)
    tensors = load_safetensors(w_bytes)
    adapted = apply_lora_3d_state_dict(base_operator, cfg, tensors)
    return adapted, manifest


__all__ = (
    "ADAPTER_3D_MACHINE",
    "ADAPTER_3D_SCHEMA",
    "DEFAULT_3D_LORA_TARGETS",
    "Adapter3DResult",
    "Adapter3DState",
    "LowRankAdapter3DConfig",
    "adapt_surface_3d_operator",
    "apply_lora_3d_state_dict",
    "extract_lora_3d_state_dict",
    "inject_lora_3d",
    "load_lora_3d_artifact",
    "save_lora_3d_artifact",
)
