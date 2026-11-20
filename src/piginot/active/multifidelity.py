"""Multi-fidelity physics + acquired-FEM correction loss (Feature C / P3).

Keeps the variational plane-stress potential energy objective primary and adds
an optional low-weight FEM displacement/stress distillation term strictly on
acquired ``TRAIN`` samples. Explicitly discloses ``multi_fidelity_label_efficient``
status so supervised multi-fidelity arms are never misrepresented as label-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch

from piginot.contracts import MaterialSpec
from piginot.objectives.energy import (
    PotentialEnergyConfig,
    PotentialEnergyTerms,
    plane_stress_potential_energy,
)
from piginot.operators.energy_operator import constitutive_in_plane_stress

MULTIFIDELITY_DISCLOSURE: Final = "multi-fidelity-label-efficient-not-label-free"


@dataclass(frozen=True, slots=True)
class MultiFidelityConfig:
    """Configuration for primary energy + low-weight acquired-FEM distillation."""

    base_energy: PotentialEnergyConfig = PotentialEnergyConfig()
    fem_displacement_weight: float = 0.25
    fem_stress_weight: float = 0.05
    disclosure: str = MULTIFIDELITY_DISCLOSURE

    def __post_init__(self) -> None:
        if type(self.base_energy) is not PotentialEnergyConfig:
            raise TypeError("base_energy must be a PotentialEnergyConfig")
        for label in ("fem_displacement_weight", "fem_stress_weight"):
            val = getattr(self, label)
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(float(val))
                or float(val) < 0.0
            ):
                raise ValueError(f"{label} must be finite and nonnegative")
            object.__setattr__(self, label, float(val))
        if (
            self.fem_displacement_weight > 0.0 or self.fem_stress_weight > 0.0
        ) and self.disclosure != MULTIFIDELITY_DISCLOSURE:
            raise ValueError(
                f"multi-fidelity loss must preserve disclosure {MULTIFIDELITY_DISCLOSURE!r}"
            )


@dataclass(frozen=True, slots=True)
class MultiFidelityTerms:
    """Graph-connected energy + multi-fidelity FEM correction objective terms."""

    total: torch.Tensor
    primary: PotentialEnergyTerms
    displacement_distillation: torch.Tensor
    stress_distillation: torch.Tensor
    disclosure: str


def multifidelity_energy_loss(
    *,
    u_pred: torch.Tensor,
    grad_pred: torch.Tensor,
    weights: torch.Tensor,
    material: MaterialSpec,
    reference_area: float,
    config: MultiFidelityConfig,
    u_fem_target: torch.Tensor | None = None,
    p_fem_target: torch.Tensor | None = None,
    partition: str = "TRAIN",
) -> MultiFidelityTerms:
    """Combine variational potential energy with optional TRAIN-only FEM targets."""
    if partition != "TRAIN":
        raise ValueError(
            f"FEM supervision is strictly restricted to TRAIN cases, got {partition!r}"
        )
    primary = plane_stress_potential_energy(
        grad_pred,
        weights,
        material,
        config=config.base_energy,
        reference_area=reference_area,
    )
    zero = primary.total.new_zeros(())
    u_loss = zero
    p_loss = zero

    if u_fem_target is not None and config.fem_displacement_weight > 0.0:
        if u_fem_target.shape != u_pred.shape:
            raise ValueError("u_fem_target must match u_pred shape")
        w_norm = weights / torch.clamp(weights.sum(), min=1e-15)
        diff_sq = torch.sum((u_pred - u_fem_target).square(), dim=-1)
        ref_sq = torch.clamp(torch.sum(u_fem_target.square(), dim=-1), min=1e-12)
        u_loss = torch.sum(w_norm * diff_sq) / torch.clamp(
            torch.sum(w_norm * ref_sq), min=1e-12
        )

    if p_fem_target is not None and config.fem_stress_weight > 0.0:
        p_pred = constitutive_in_plane_stress(grad_pred, material)
        if p_fem_target.shape != p_pred.shape:
            raise ValueError("p_fem_target must match constitutive stress shape")
        w_norm = weights / torch.clamp(weights.sum(), min=1e-15)
        diff_sq = torch.sum((p_pred - p_fem_target).square(), dim=(-2, -1))
        ref_sq = torch.sum(p_fem_target.square(), dim=(-2, -1))
        p_loss = torch.sum(w_norm * diff_sq) / torch.clamp(
            torch.sum(w_norm * ref_sq), min=1e-12
        )

    total = (
        primary.total
        + config.fem_displacement_weight * u_loss
        + config.fem_stress_weight * p_loss
    )
    return MultiFidelityTerms(
        total=total,
        primary=primary,
        displacement_distillation=u_loss,
        stress_distillation=p_loss,
        disclosure=config.disclosure,
    )


__all__ = (
    "MULTIFIDELITY_DISCLOSURE",
    "MultiFidelityConfig",
    "MultiFidelityTerms",
    "multifidelity_energy_loss",
)
