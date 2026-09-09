from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from piginot.doe import EvidenceRole
from piginot.objectives import (
    EquilibriumForm,
    ObjectiveMode,
    ObjectiveResiduals,
    ObjectiveTerm,
    default_objective_registry,
)


def _residuals() -> ObjectiveResiduals:
    return ObjectiveResiduals(
        {
            term: torch.tensor([float(index + 1)], dtype=torch.float64)
            for index, term in enumerate(ObjectiveTerm)
        }
    )


def test_default_registry_is_hybrid_and_reports_weak_equilibrium() -> None:
    registry = default_objective_registry()

    assert registry.mode is ObjectiveMode.HYBRID
    assert registry.equilibrium_form is EquilibriumForm.WEAK
    assert registry.required_terms == (
        ObjectiveTerm.DATA_U,
        ObjectiveTerm.DATA_P,
        ObjectiveTerm.EQUILIBRIUM_WEAK,
        ObjectiveTerm.CONSTITUTIVE,
        ObjectiveTerm.ANGULAR,
        ObjectiveTerm.DIRICHLET,
        ObjectiveTerm.TRACTION_FREE,
    )
    assert registry.reported_equilibrium_term is ObjectiveTerm.EQUILIBRIUM_WEAK


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        (
            ObjectiveMode.DATA_ONLY,
            (ObjectiveTerm.DATA_U, ObjectiveTerm.DATA_P),
        ),
        (
            ObjectiveMode.PHYSICS_ONLY,
            (
                ObjectiveTerm.EQUILIBRIUM_WEAK,
                ObjectiveTerm.CONSTITUTIVE,
                ObjectiveTerm.ANGULAR,
                ObjectiveTerm.DIRICHLET,
                ObjectiveTerm.TRACTION_FREE,
            ),
        ),
    ),
)
def test_registry_modes_have_fixed_named_term_sets(
    mode: ObjectiveMode, expected: tuple[ObjectiveTerm, ...]
) -> None:
    assert default_objective_registry(mode=mode).required_terms == expected


def test_strong_form_is_explicit_and_never_the_default() -> None:
    registry = default_objective_registry(
        mode=ObjectiveMode.PHYSICS_ONLY,
        equilibrium_form=EquilibriumForm.STRONG,
    )

    assert registry.reported_equilibrium_term is ObjectiveTerm.EQUILIBRIUM_STRONG
    assert ObjectiveTerm.EQUILIBRIUM_WEAK not in registry.required_terms
    assert ObjectiveTerm.EQUILIBRIUM_STRONG in registry.required_terms


def test_evaluation_is_nondimensional_weighted_mse_and_immutable() -> None:
    registry = default_objective_registry(mode=ObjectiveMode.DATA_ONLY)
    evaluation = registry.evaluate(_residuals(), role=EvidenceRole.TRAIN)

    assert evaluation.losses[ObjectiveTerm.DATA_U].item() == pytest.approx(1.0)
    assert evaluation.losses[ObjectiveTerm.DATA_P].item() == pytest.approx(4.0)
    assert evaluation.total.item() == pytest.approx(5.0)
    with pytest.raises(TypeError):
        evaluation.losses[ObjectiveTerm.DATA_U] = torch.tensor(0.0)  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        registry.mode = ObjectiveMode.PHYSICS_ONLY  # type: ignore[misc]


def test_registry_rejects_missing_terms_and_non_tensor_residuals() -> None:
    registry = default_objective_registry(mode=ObjectiveMode.DATA_ONLY)
    with pytest.raises(ValueError, match="missing required objective residual"):
        registry.evaluate(
            ObjectiveResiduals({ObjectiveTerm.DATA_U: torch.ones(1)}),
            role=EvidenceRole.TRAIN,
        )
    with pytest.raises(TypeError, match="torch.Tensor"):
        ObjectiveResiduals({ObjectiveTerm.DATA_U: 1.0})  # type: ignore[dict-item]


@pytest.mark.parametrize("role", (EvidenceRole.HOLDOUT, EvidenceRole.LEGACY))
def test_ineligible_roles_are_refused_before_objective_reduction(
    role: EvidenceRole,
) -> None:
    registry = default_objective_registry(mode=ObjectiveMode.DATA_ONLY)

    with pytest.raises(PermissionError, match=role.value):
        registry.evaluate(_residuals(), role=role)


def test_holdout_authorization_precedes_residual_type_or_payload_access() -> None:
    registry = default_objective_registry(mode=ObjectiveMode.DATA_ONLY)

    with pytest.raises(PermissionError, match="HOLDOUT"):
        registry.evaluate(object(), role=EvidenceRole.HOLDOUT)  # type: ignore[arg-type]
