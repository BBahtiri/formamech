"""Objective registry construction and role-authorized scalar reduction."""

from __future__ import annotations

import torch

from piginot.doe import EvidenceRole
from piginot.objectives.models import (
    EquilibriumForm,
    ObjectiveEvaluation,
    ObjectiveMode,
    ObjectiveRegistry,
    ObjectiveResiduals,
)

_OBJECTIVE_ROLES = frozenset(
    (EvidenceRole.PILOT, EvidenceRole.TRAIN, EvidenceRole.DEVELOPMENT)
)


def default_objective_registry(
    *,
    mode: ObjectiveMode = ObjectiveMode.HYBRID,
    equilibrium_form: EquilibriumForm = EquilibriumForm.WEAK,
) -> ObjectiveRegistry:
    """Build a registry; HYBRID with weak equilibrium is deliberately default."""

    return ObjectiveRegistry(mode=mode, equilibrium_form=equilibrium_form)


def evaluate_objective(
    registry: ObjectiveRegistry,
    residuals: ObjectiveResiduals,
    *,
    role: EvidenceRole,
) -> ObjectiveEvaluation:
    """Reduce required residuals only after fail-closed evidence-role authorization."""

    if type(registry) is not ObjectiveRegistry:
        raise TypeError("registry must be an exact ObjectiveRegistry")
    if type(role) is not EvidenceRole:
        raise TypeError("role must be an EvidenceRole")
    if role not in _OBJECTIVE_ROLES:
        raise PermissionError(f"{role.value} evidence cannot enter an objective")
    if type(residuals) is not ObjectiveResiduals:
        raise TypeError("residuals must be exact ObjectiveResiduals")
    missing = tuple(
        term for term in registry.required_terms if term not in residuals.values
    )
    if missing:
        raise ValueError(
            "missing required objective residual: "
            + ", ".join(term.value for term in missing)
        )
    losses = {
        term: residuals.values[term].square().mean() * registry.term_weights[term]
        for term in registry.required_terms
    }
    total = torch.stack(tuple(losses.values())).sum()
    return ObjectiveEvaluation(registry.mode, registry.equilibrium_form, losses, total)


__all__ = ("default_objective_registry", "evaluate_objective")
