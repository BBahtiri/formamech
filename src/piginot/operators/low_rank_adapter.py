"""Low-rank per-geometry adapters (Delta-theta_g) for ``BoundaryEnergyOperator``.

Adapts only low-rank residual matrices injected into frozen decoder linear layers
using the pure label-free plane-stress potential energy objective, preserving:

* exact zero-shot bit-identity at step 0 (zero-initialized output factor ``B``),
* strictly frozen base operator parameters (no full-network retraining),
* exact component-wise affine Dirichlet lifting,
* explicit budget ceilings (default <= 10% of from-scratch optimizer steps), and
* explicit refusal on unsupported geometry topology.
"""

from __future__ import annotations

import copy
import enum
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast

import torch
from torch import nn

from piginot.contracts import MaterialSpec, canonical_json_bytes
from piginot.lifting.base import DirichletLifting
from piginot.operators.architecture import OperatorOutput
from piginot.operators.boundary_energy_operator import (
    BoundaryEnergyEncoding,
    BoundaryEnergyOperator,
    boundary_energy_module_digest,
)
from piginot.operators.energy_operator import QueryFrame
from piginot.operators.geometry_tokenizer import (
    GeometryOODReport,
    GeometryOODStatus,
)

if TYPE_CHECKING:
    from piginot.objectives.energy import PotentialEnergyConfig, TriangleDomain

ADAPTER_SCHEMA: Final = "piginot.low-rank-geometry-adapter-v1"
ADAPTATION_PLAN_SCHEMA: Final = "piginot.geometry-adaptation-plan-v1"
DEFAULT_TARGET_MODULES: Final[tuple[str, ...]] = (
    "out_proj",
    "films",
    "output",
)


@dataclass(frozen=True, slots=True)
class LowRankAdapterConfig:
    """Budgeted low-rank adapter contract for one unseen geometry."""

    rank: int = 4
    alpha: float = 4.0
    target_substrings: tuple[str, ...] = DEFAULT_TARGET_MODULES
    max_steps: int = 20
    from_scratch_reference_steps: int = 200
    max_step_budget_fraction: float = 0.10
    learning_rate: float = 1e-2
    seed: int = 0
    use_cosine_schedule: bool = True
    grad_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be a positive integer")
        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, (int, float))
            or not math.isfinite(float(self.alpha))
            or float(self.alpha) <= 0.0
        ):
            raise ValueError("alpha must be finite and positive")
        if (
            type(self.target_substrings) is not tuple
            or not self.target_substrings
            or any(type(s) is not str or not s.strip() for s in self.target_substrings)
        ):
            raise ValueError("target_substrings must be a non-empty tuple of strings")
        if type(self.max_steps) is not int or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if (
            type(self.from_scratch_reference_steps) is not int
            or self.from_scratch_reference_steps <= 0
        ):
            raise ValueError("from_scratch_reference_steps must be a positive integer")
        if (
            isinstance(self.max_step_budget_fraction, bool)
            or not isinstance(self.max_step_budget_fraction, (int, float))
            or not (0.0 < float(self.max_step_budget_fraction) <= 1.0)
        ):
            raise ValueError("max_step_budget_fraction must lie in (0, 1]")
        if self.max_steps > math.floor(
            self.from_scratch_reference_steps * self.max_step_budget_fraction + 1e-12
        ):
            raise ValueError(
                f"max_steps={self.max_steps} exceeds budget fraction "
                f"{self.max_step_budget_fraction} of "
                f"from_scratch_reference_steps={self.from_scratch_reference_steps}"
            )
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(float(self.learning_rate))
            or float(self.learning_rate) <= 0.0
        ):
            raise ValueError("learning_rate must be finite and positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.use_cosine_schedule) is not bool:
            raise TypeError("use_cosine_schedule must be a boolean")
        if (
            isinstance(self.grad_clip_norm, bool)
            or not isinstance(self.grad_clip_norm, (int, float))
            or not math.isfinite(float(self.grad_clip_norm))
            or float(self.grad_clip_norm) <= 0.0
        ):
            raise ValueError("grad_clip_norm must be finite and positive")

    @property
    def step_budget_ratio(self) -> float:
        return self.max_steps / self.from_scratch_reference_steps

    def document(self) -> dict[str, Any]:
        return {
            "schema": ADAPTER_SCHEMA,
            "rank": self.rank,
            "alpha": float(self.alpha),
            "target_substrings": list(self.target_substrings),
            "max_steps": self.max_steps,
            "from_scratch_reference_steps": self.from_scratch_reference_steps,
            "max_step_budget_fraction": float(self.max_step_budget_fraction),
            "learning_rate": float(self.learning_rate),
            "seed": self.seed,
            "use_cosine_schedule": self.use_cosine_schedule,
            "grad_clip_norm": float(self.grad_clip_norm),
        }


class LowRankLinear(nn.Module):
    """Frozen ``nn.Linear`` augmented with a zero-initialized low-rank residual."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be an nn.Linear")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.rank = rank
        self.scaling = float(alpha) / float(rank)
        dtype = base.weight.dtype
        device = base.weight.device
        a_init = torch.empty((rank, base.in_features), dtype=dtype, device=device)
        a_init.normal_(0.0, 1.0 / math.sqrt(base.in_features), generator=generator)
        self.lora_a = nn.Parameter(a_init)
        self.lora_b = nn.Parameter(
            torch.zeros((base.out_features, rank), dtype=dtype, device=device)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_out: torch.Tensor = self.base(inputs)
        low_rank = torch.nn.functional.linear(
            torch.nn.functional.linear(inputs, self.lora_a), self.lora_b
        )
        return base_out + self.scaling * low_rank


def _replace_target_linears(
    root: nn.Module,
    target_substrings: tuple[str, ...],
    *,
    rank: int,
    alpha: float,
    generator: torch.Generator,
) -> tuple[str, ...]:
    replaced: list[str] = []

    def _walk(module: nn.Module, prefix: str) -> None:
        for child_name, child in list(module.named_children()):
            full = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and any(
                sub in full for sub in target_substrings
            ):
                setattr(
                    module,
                    child_name,
                    LowRankLinear(child, rank=rank, alpha=alpha, generator=generator),
                )
                replaced.append(full)
            else:
                _walk(child, full)

    _walk(root, "")
    return tuple(sorted(replaced))


class AdaptedBoundaryEnergyOperator(nn.Module):
    """Frozen ``BoundaryEnergyOperator`` equipped with trainable ``Delta-theta_g``."""

    def __init__(
        self,
        base_operator: BoundaryEnergyOperator,
        config: LowRankAdapterConfig,
    ) -> None:
        super().__init__()
        if type(base_operator) is not BoundaryEnergyOperator:
            raise TypeError("base_operator must be an exact BoundaryEnergyOperator")
        if type(config) is not LowRankAdapterConfig:
            raise TypeError("config must be a LowRankAdapterConfig")

        self.adapter_config = config
        self.base_digest = boundary_energy_module_digest(base_operator)
        self.operator = copy.deepcopy(base_operator)
        for param in self.operator.parameters():
            param.requires_grad_(False)

        gen = torch.Generator().manual_seed(config.seed)
        adapted_layers = _replace_target_linears(
            self.operator.decoder,
            config.target_substrings,
            rank=config.rank,
            alpha=config.alpha,
            generator=gen,
        )
        if not adapted_layers:
            raise ValueError(
                f"no nn.Linear layers matched target_substrings={config.target_substrings}"
            )
        self.adapted_layers: tuple[str, ...] = adapted_layers

    def adapter_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            param for param in self.operator.parameters() if param.requires_grad
        )

    def adapter_parameter_count(self) -> int:
        return sum(int(p.numel()) for p in self.adapter_parameters())

    def base_parameter_count(self) -> int:
        return sum(
            int(p.numel()) for p in self.operator.parameters() if not p.requires_grad
        )

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only the low-rank adapter tensors (safetensors-serializable)."""
        return {
            name: tensor.detach().clone()
            for name, tensor in sorted(self.operator.state_dict().items())
            if name.endswith(".lora_a") or name.endswith(".lora_b")
        }

    def load_adapter_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        expected = self.adapter_state_dict()
        if set(state) != set(expected):
            raise ValueError("adapter state keys do not match adapted layers")
        with torch.no_grad():
            params = dict(self.operator.named_parameters())
            for name, tensor in state.items():
                target = params[name]
                if tuple(tensor.shape) != tuple(target.shape):
                    raise ValueError(f"adapter tensor {name!r} shape mismatch")
                target.copy_(tensor.to(dtype=target.dtype, device=target.device))

    def adapter_digest(self) -> str:
        """Content digest over adapter schema and exact byte values."""
        hasher = hashlib.blake2b(digest_size=32)
        hasher.update(canonical_json_bytes(self.adapter_config.document()))
        hasher.update(self.base_digest.encode("ascii"))
        for name, tensor in sorted(self.adapter_state_dict().items()):
            hasher.update(name.encode("utf-8"))
            hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return f"blake2b256:{hasher.hexdigest()}"

    def encode_boundary(
        self,
        *,
        boundary_xy: torch.Tensor,
        boundary_normals: torch.Tensor,
        boundary_features: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
        frame: QueryFrame,
    ) -> BoundaryEnergyEncoding:
        return self.operator.encode_boundary(
            boundary_xy=boundary_xy,
            boundary_normals=boundary_normals,
            boundary_features=boundary_features,
            weights=weights,
            mask=mask,
            frame=frame,
        )

    def displacement(
        self,
        *,
        encoding: BoundaryEnergyEncoding,
        query_xy: torch.Tensor,
        parameters: torch.Tensor,
        lifting: DirichletLifting,
    ) -> torch.Tensor:
        return self.operator.displacement(
            encoding=encoding,
            query_xy=query_xy,
            parameters=parameters,
            lifting=lifting,
        )

    def displacement_and_gradient(
        self,
        *,
        encoding: BoundaryEnergyEncoding,
        query_xy: torch.Tensor,
        parameters: torch.Tensor,
        lifting: DirichletLifting,
        create_graph: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.operator.displacement_and_gradient(
            encoding=encoding,
            query_xy=query_xy,
            parameters=parameters,
            lifting=lifting,
            create_graph=create_graph,
        )

    def forward(
        self,
        *,
        encoding: BoundaryEnergyEncoding,
        query_xy: torch.Tensor,
        parameters: torch.Tensor,
        lifting: DirichletLifting,
        materials: Sequence[MaterialSpec],
    ) -> OperatorOutput:
        return cast(
            OperatorOutput,
            self.operator(
                encoding=encoding,
                query_xy=query_xy,
                parameters=parameters,
                lifting=lifting,
                materials=materials,
            ),
        )


class AdaptationStatus(enum.StrEnum):
    COMPLETED = "COMPLETED"
    REFUSED_UNSUPPORTED_TOPOLOGY = "REFUSED_UNSUPPORTED_TOPOLOGY"
    FAILED_INADMISSIBLE = "FAILED_INADMISSIBLE"


@dataclass(frozen=True, slots=True)
class AdaptationPlan:
    """Declarative, content-hashed adaptation plan for one unseen geometry."""

    base_model_digest: str
    target_mesh_id: str
    adapter_config: LowRankAdapterConfig
    energy_config: PotentialEnergyConfig
    ood_report: GeometryOODReport | None = None

    def document(self) -> dict[str, Any]:
        return {
            "schema": ADAPTATION_PLAN_SCHEMA,
            "base_model_digest": self.base_model_digest,
            "target_mesh_id": self.target_mesh_id,
            "adapter": self.adapter_config.document(),
            "objective_hash": self.energy_config.config_hash,
            "ood_status": (
                None if self.ood_report is None else self.ood_report.status.value
            ),
            "ood_report_hash": (
                None if self.ood_report is None else self.ood_report.report_hash
            ),
        }

    @property
    def plan_hash(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.document())).hexdigest()


@dataclass(frozen=True, slots=True)
class AdaptationResult:
    """Deterministic provenance record of one low-rank energy adaptation run."""

    plan_hash: str
    status: AdaptationStatus
    claim_level: str
    steps_executed: int
    step_budget_ratio: float
    adapter_parameter_count: int
    base_parameter_count: int
    initial_energy: float | None
    final_energy: float | None
    minimum_determinant: float | None
    energy_history: tuple[float, ...]
    adapter_digest: str | None
    reason: str

    def document(self) -> dict[str, Any]:
        return {
            "plan_hash": self.plan_hash,
            "status": self.status.value,
            "claim_level": self.claim_level,
            "steps_executed": self.steps_executed,
            "step_budget_ratio": self.step_budget_ratio,
            "adapter_parameter_count": self.adapter_parameter_count,
            "base_parameter_count": self.base_parameter_count,
            "initial_energy": self.initial_energy,
            "final_energy": self.final_energy,
            "minimum_determinant": self.minimum_determinant,
            "energy_history": list(self.energy_history),
            "adapter_digest": self.adapter_digest,
            "reason": self.reason,
        }


def adapt_boundary_operator_energy(
    base_operator: BoundaryEnergyOperator,
    *,
    plan: AdaptationPlan,
    domain: TriangleDomain,
    boundary_xy: torch.Tensor,
    boundary_normals: torch.Tensor,
    boundary_features: torch.Tensor,
    boundary_weights: torch.Tensor,
    boundary_mask: torch.Tensor,
    frame: QueryFrame,
    parameters: torch.Tensor,
    lifting: DirichletLifting,
    material: MaterialSpec,
) -> tuple[AdaptedBoundaryEnergyOperator | None, AdaptationResult]:
    """Optimize only low-rank adapters ``Delta-theta_g`` via pure potential energy."""
    from piginot.objectives.energy import (
        plane_stress_potential_energy,
        stratified_triangle_quadrature,
    )

    if plan.base_model_digest != boundary_energy_module_digest(base_operator):
        raise ValueError("plan.base_model_digest does not match base_operator")

    if (
        plan.ood_report is not None
        and plan.ood_report.status is GeometryOODStatus.UNSUPPORTED_TOPOLOGY
    ):
        return None, AdaptationResult(
            plan_hash=plan.plan_hash,
            status=AdaptationStatus.REFUSED_UNSUPPORTED_TOPOLOGY,
            claim_level="NOT_VERIFIED",
            steps_executed=0,
            step_budget_ratio=0.0,
            adapter_parameter_count=0,
            base_parameter_count=sum(
                int(p.numel()) for p in base_operator.parameters()
            ),
            initial_energy=None,
            final_energy=None,
            minimum_determinant=None,
            energy_history=(),
            adapter_digest=None,
            reason=f"refused unsupported topology: {plan.ood_report.reason}",
        )

    adapted = AdaptedBoundaryEnergyOperator(base_operator, plan.adapter_config)
    optimizer = torch.optim.Adam(
        adapted.adapter_parameters(),
        lr=plan.adapter_config.learning_rate,
        foreach=False,
        fused=False,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(plan.adapter_config.max_steps, 1),
            eta_min=0.1 * plan.adapter_config.learning_rate,
        )
        if plan.adapter_config.use_cosine_schedule
        else None
    )
    quad_gen = torch.Generator(device=domain.vertices.device).manual_seed(
        plan.adapter_config.seed + 17
    )

    # Encode frozen boundary geometry once (encoder weights are strictly frozen).
    with torch.no_grad():
        encoding = adapted.encode_boundary(
            boundary_xy=boundary_xy,
            boundary_normals=boundary_normals,
            boundary_features=boundary_features,
            weights=boundary_weights,
            mask=boundary_mask,
            frame=frame,
        )

    history: list[float] = []
    min_det = float("inf")
    for step in range(plan.adapter_config.max_steps):
        pts, wts = stratified_triangle_quadrature(
            domain, plan.energy_config.points_per_cell, generator=quad_gen
        )
        optimizer.zero_grad(set_to_none=True)
        _, grad = adapted.displacement_and_gradient(
            encoding=encoding,
            query_xy=pts.unsqueeze(0),
            parameters=parameters,
            lifting=lifting,
            create_graph=True,
        )
        terms = plane_stress_potential_energy(
            grad[0],
            wts,
            material,
            config=plan.energy_config,
            reference_area=domain.area,
        )
        val = float(terms.total.detach().item())
        if not math.isfinite(val):
            return adapted, AdaptationResult(
                plan_hash=plan.plan_hash,
                status=AdaptationStatus.FAILED_INADMISSIBLE,
                claim_level="NOT_VERIFIED",
                steps_executed=step,
                step_budget_ratio=step
                / plan.adapter_config.from_scratch_reference_steps,
                adapter_parameter_count=adapted.adapter_parameter_count(),
                base_parameter_count=adapted.base_parameter_count(),
                initial_energy=history[0] if history else None,
                final_energy=history[-1] if history else None,
                minimum_determinant=min_det if math.isfinite(min_det) else None,
                energy_history=tuple(history),
                adapter_digest=adapted.adapter_digest(),
                reason="non-finite potential energy encountered during adaptation",
            )
        history.append(val)
        min_det = min(min_det, terms.minimum_determinant)
        terms.total.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(
            adapted.adapter_parameters(), max_norm=plan.adapter_config.grad_clip_norm
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    return adapted, AdaptationResult(
        plan_hash=plan.plan_hash,
        status=AdaptationStatus.COMPLETED,
        claim_level="NOT_VERIFIED",
        steps_executed=plan.adapter_config.max_steps,
        step_budget_ratio=plan.adapter_config.step_budget_ratio,
        adapter_parameter_count=adapted.adapter_parameter_count(),
        base_parameter_count=adapted.base_parameter_count(),
        initial_energy=history[0],
        final_energy=history[-1],
        minimum_determinant=min_det,
        energy_history=tuple(history),
        adapter_digest=adapted.adapter_digest(),
        reason="low-rank physics-energy adaptation completed within step budget",
    )


__all__ = (
    "ADAPTATION_PLAN_SCHEMA",
    "ADAPTER_SCHEMA",
    "DEFAULT_TARGET_MODULES",
    "AdaptationPlan",
    "AdaptationResult",
    "AdaptationStatus",
    "AdaptedBoundaryEnergyOperator",
    "LowRankAdapterConfig",
    "LowRankLinear",
    "adapt_boundary_operator_energy",
)
