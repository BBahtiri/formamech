"""Zero-shot test-time physics LoRA adaptation (Tier 1.5) before Tier 2 FEM escalation.

Injects zero-initialized low-rank adapters into a frozen clone of an
``EnergyFormOperator`` (or ``BoundaryEnergyOperator``) and executes a bounded
pass (typically 20-200 steps) of purely label-free potential-energy fine-tuning
for the current query geometry/loading. If energy fails to decrease or the
minimum deformation Jacobian falls below ``min_det_f``, rolls back cleanly to
the unmodified base weights.
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, TypeVar

import torch
from torch import nn

from piginot.contracts import MaterialSpec, canonical_json_bytes
from piginot.contracts.state_machine import (
    StateMachineSpec,
    TransitionSpec,
    register_state_machine,
)
from piginot.lifting.base import DirichletLifting
from piginot.objectives.energy import (
    PotentialEnergyConfig,
    TriangleDomain,
    plane_stress_potential_energy,
)
from piginot.objectives.hierarchical_energy import deterministic_triangle_quadrature
from piginot.operators.boundary_energy_operator import (
    BoundaryEnergyEncoding,
    BoundaryEnergyOperator,
)
from piginot.operators.energy_operator import EnergyFormOperator, QueryFrame
from piginot.operators.low_rank_adapter import (
    DEFAULT_TARGET_MODULES,
    LowRankLinear,
    _replace_target_linears,
)


class TestTimeLoraStatus(StrEnum):
    """Explicit outcome states for Tier 1.5 test-time physics LoRA adaptation."""

    ADAPTING = "ADAPTING"
    APPLIED = "APPLIED"
    ROLLED_BACK = "ROLLED_BACK"


TestTimeLoraStatus.__test__ = False  # type: ignore[attr-defined]


TTA_LORA_MACHINE: Final[StateMachineSpec[TestTimeLoraStatus]] = register_state_machine(
    StateMachineSpec(
        machine_id="formamech.serving.test_time_lora",
        description=(
            "Tier 1.5 label-free test-time physics LoRA lifecycle: adapt under "
            "variational energy and commit only on admissible energy drop."
        ),
        states=tuple(TestTimeLoraStatus),
        initial_states=(TestTimeLoraStatus.ADAPTING,),
        terminal_states=(TestTimeLoraStatus.APPLIED, TestTimeLoraStatus.ROLLED_BACK),
        transitions=(
            TransitionSpec(
                TestTimeLoraStatus.ADAPTING,
                TestTimeLoraStatus.APPLIED,
                "commit_improved",
                "Admissible energy strictly decreased; keep adapted LoRA weights.",
            ),
            TransitionSpec(
                TestTimeLoraStatus.ADAPTING,
                TestTimeLoraStatus.ROLLED_BACK,
                "rollback_to_base",
                "No admissible energy drop (or non-finite loss); zero LoRA B weights.",
            ),
        ),
    )
)


@dataclass(frozen=True, slots=True)
class TestTimeLoraConfig:
    """Bounded budget and safety contract for Tier 1.5 test-time physics LoRA."""

    __test__ = False

    steps: int = 50
    rank: int = 4
    alpha: float = 4.0
    learning_rate: float = 5.0e-3
    target_substrings: tuple[str, ...] = DEFAULT_TARGET_MODULES
    quadrature_degree: int = 2
    min_det_f: float = 0.05
    grad_clip_norm: float = 1.0
    seed: int = 0
    max_steps_ceiling: int = 200

    def __post_init__(self) -> None:
        if type(self.steps) is not int or not (
            1 <= self.steps <= self.max_steps_ceiling
        ):
            raise ValueError(
                f"steps must be an integer in [1, {self.max_steps_ceiling}]"
            )
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("rank must be a positive integer")
        if not math.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("alpha must be finite and positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.min_det_f) or self.min_det_f <= 0.0:
            raise ValueError("min_det_f must be finite and positive")

    def document(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "rank": self.rank,
            "alpha": self.alpha,
            "learning_rate": self.learning_rate,
            "target_substrings": list(self.target_substrings),
            "quadrature_degree": self.quadrature_degree,
            "min_det_f": self.min_det_f,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class TestTimeLoraReport:
    """Auditable provenance record for one Tier 1.5 test-time LoRA adaptation."""

    __test__ = False

    steps_taken: int
    adapted_layers: tuple[str, ...]
    adapter_parameter_count: int
    initial_energy: float
    final_energy: float
    energy_drop: float
    final_min_det_f: float
    improved: bool
    rolled_back: bool
    adapter_digest: str
    wall_seconds: float

    def __post_init__(self) -> None:
        if self.improved == self.rolled_back:
            raise ValueError(
                "TestTimeLoraReport invariant violated: improved and rolled_back "
                f"must be strictly opposite (got improved={self.improved}, "
                f"rolled_back={self.rolled_back})"
            )

    @property
    def status(self) -> TestTimeLoraStatus:
        """Explicit finite-state outcome (APPLIED vs ROLLED_BACK)."""
        return (
            TestTimeLoraStatus.APPLIED
            if self.improved
            else TestTimeLoraStatus.ROLLED_BACK
        )

    def document(self) -> dict[str, Any]:
        return {
            "steps_taken": self.steps_taken,
            "adapted_layers": list(self.adapted_layers),
            "adapter_parameter_count": self.adapter_parameter_count,
            "initial_energy": self.initial_energy,
            "final_energy": self.final_energy,
            "energy_drop": self.energy_drop,
            "final_min_det_f": self.final_min_det_f,
            "improved": self.improved,
            "rolled_back": self.rolled_back,
            "adapter_digest": self.adapter_digest,
            "wall_seconds": self.wall_seconds,
        }


def _adapter_digest(module: nn.Module, config: TestTimeLoraConfig) -> str:
    hasher = hashlib.blake2b(digest_size=32)
    hasher.update(canonical_json_bytes(config.document()))
    for name, tensor in sorted(module.state_dict().items()):
        if name.endswith(".lora_a") or name.endswith(".lora_b"):
            hasher.update(name.encode("utf-8"))
            hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return f"blake2b256:{hasher.hexdigest()}"


OpT = TypeVar("OpT", EnergyFormOperator, BoundaryEnergyOperator)


def adapt_operator_at_test_time(
    base_operator: OpT,
    *,
    domain: TriangleDomain,
    parameters: torch.Tensor,
    frame: QueryFrame,
    lifting: DirichletLifting,
    material: MaterialSpec,
    boundary_encoding: BoundaryEnergyEncoding | None = None,
    config: TestTimeLoraConfig | None = None,
    energy_config: PotentialEnergyConfig | None = None,
) -> tuple[OpT, TestTimeLoraReport]:
    """Run zero-shot label-free physics LoRA adaptation on a frozen operator clone."""
    cfg = TestTimeLoraConfig() if config is None else config
    ecfg = PotentialEnergyConfig() if energy_config is None else energy_config
    t0 = time.perf_counter()

    cloned = copy.deepcopy(base_operator)
    for param in cloned.parameters():
        param.requires_grad_(False)

    target_root: nn.Module = (
        cloned.decoder if isinstance(cloned, BoundaryEnergyOperator) else cloned
    )
    gen = torch.Generator().manual_seed(cfg.seed)
    adapted_layers = _replace_target_linears(
        target_root,
        cfg.target_substrings,
        rank=cfg.rank,
        alpha=cfg.alpha,
        generator=gen,
    )
    if not adapted_layers:
        raise ValueError(
            f"no nn.Linear layers matched target_substrings={cfg.target_substrings}"
        )

    adapter_params = [p for p in cloned.parameters() if p.requires_grad]
    param_count = sum(int(p.numel()) for p in adapter_params)
    pts, weights = deterministic_triangle_quadrature(
        domain, degree=cfg.quadrature_degree
    )
    pts = pts.to(dtype=parameters.dtype, device=parameters.device)
    weights = weights.to(dtype=parameters.dtype, device=parameters.device)
    query_batched = pts.unsqueeze(0)

    detached_encoding = (
        None
        if boundary_encoding is None
        else BoundaryEnergyEncoding(
            tokens=boundary_encoding.tokens.detach(),
            frame=QueryFrame(
                origin=boundary_encoding.frame.origin.detach(),
                extent=boundary_encoding.frame.extent.detach(),
            ),
        )
    )

    def _eval_energy(*, create_graph: bool) -> Any:
        if isinstance(cloned, BoundaryEnergyOperator):
            if detached_encoding is None:
                raise ValueError(
                    "boundary_encoding is required for BoundaryEnergyOperator TTA"
                )
            _, grad_u = cloned.displacement_and_gradient(
                encoding=detached_encoding,
                query_xy=query_batched,
                parameters=parameters,
                lifting=lifting,
                create_graph=create_graph,
            )
        else:
            _, grad_u = cloned.displacement_and_gradient(
                query_xy=query_batched,
                parameters=parameters,
                frame=frame,
                lifting=lifting,
                create_graph=create_graph,
            )
        return plane_stress_potential_energy(
            grad_u[0],
            weights,
            material,
            config=ecfg,
            reference_area=domain.area,
        )

    init_terms = _eval_energy(create_graph=False)
    init_energy = float(init_terms.total.detach().item())

    optimizer = torch.optim.Adam(adapter_params, lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.steps, 1), eta_min=cfg.learning_rate * 0.1
    )

    best_energy = init_energy
    best_det = init_terms.minimum_determinant
    best_state = {
        name: t.detach().clone()
        for name, t in cloned.state_dict().items()
        if name.endswith(".lora_a") or name.endswith(".lora_b")
    }

    steps_taken = 0
    for _ in range(cfg.steps):
        optimizer.zero_grad(set_to_none=True)
        terms = _eval_energy(create_graph=True)
        loss = terms.total
        if not bool(torch.isfinite(loss).item()):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter_params, cfg.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        steps_taken += 1

        val = float(loss.detach().item())
        if val < best_energy and terms.minimum_determinant >= cfg.min_det_f:
            best_energy = val
            best_det = terms.minimum_determinant
            best_state = {
                name: t.detach().clone()
                for name, t in cloned.state_dict().items()
                if name.endswith(".lora_a") or name.endswith(".lora_b")
            }

    improved = best_energy < init_energy - 1.0e-12 and best_det >= cfg.min_det_f
    rolled_back = not improved
    if improved:
        with torch.no_grad():
            params_map = dict(cloned.named_parameters())
            for name, tensor in best_state.items():
                params_map[name].copy_(tensor)
        final_energy = best_energy
        final_det = best_det
    else:
        # Reset LoRA B factors strictly to zero -> exact base-model bit-identity.
        with torch.no_grad():
            for mod in cloned.modules():
                if isinstance(mod, LowRankLinear):
                    mod.lora_b.zero_()
        final_energy = init_energy
        final_det = init_terms.minimum_determinant

    report = TestTimeLoraReport(
        steps_taken=steps_taken,
        adapted_layers=adapted_layers,
        adapter_parameter_count=param_count,
        initial_energy=init_energy,
        final_energy=final_energy,
        energy_drop=init_energy - final_energy,
        final_min_det_f=final_det,
        improved=improved,
        rolled_back=rolled_back,
        adapter_digest=_adapter_digest(cloned, cfg),
        wall_seconds=time.perf_counter() - t0,
    )
    return cloned, report


__all__ = (
    "TestTimeLoraConfig",
    "TestTimeLoraReport",
    "adapt_operator_at_test_time",
)
