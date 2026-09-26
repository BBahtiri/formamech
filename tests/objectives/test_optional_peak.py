from __future__ import annotations

import pytest
import torch

from piginot.contracts import hash_spec
from piginot.doe import EvidenceRole
from piginot.objectives import (
    EquilibriumForm,
    ObjectiveMode,
    ObjectiveRegistry,
    ObjectiveResiduals,
    ObjectiveTerm,
)

_LEGACY_S13_REGISTRY_HASH = (
    "b6bb39d2b12b08c81ebc5fb81fa78782c6ea6f4abe3ced8325483b9ead08d887"
)


def test_default_registry_keeps_frozen_s13_identity_and_terms() -> None:
    registry = ObjectiveRegistry(
        mode=ObjectiveMode.HYBRID,
        equilibrium_form=EquilibriumForm.STRONG,
    )

    assert hash_spec(registry, "recipe") == _LEGACY_S13_REGISTRY_HASH
    assert ObjectiveTerm.DATA_PEAK not in registry.term_weights
    assert ObjectiveTerm.DATA_PEAK not in registry.required_terms


def test_explicit_peak_weight_changes_identity_and_is_required_data() -> None:
    baseline = ObjectiveRegistry(mode=ObjectiveMode.DATA_ONLY)
    registry = ObjectiveRegistry(
        mode=ObjectiveMode.DATA_ONLY,
        term_weights={ObjectiveTerm.DATA_PEAK: 2.0},
    )

    assert registry.required_terms == (
        ObjectiveTerm.DATA_U,
        ObjectiveTerm.DATA_P,
        ObjectiveTerm.DATA_PEAK,
    )
    assert hash_spec(registry, "recipe") != hash_spec(baseline, "recipe")
    residuals = ObjectiveResiduals(
        {
            ObjectiveTerm.DATA_U: torch.tensor([1.0], dtype=torch.float64),
            ObjectiveTerm.DATA_P: torch.tensor([2.0], dtype=torch.float64),
            ObjectiveTerm.DATA_PEAK: torch.tensor([3.0], dtype=torch.float64),
        }
    )
    evaluation = registry.evaluate(residuals, role=EvidenceRole.TRAIN)
    assert evaluation.losses[ObjectiveTerm.DATA_PEAK].item() == pytest.approx(18.0)
    assert evaluation.total.item() == pytest.approx(23.0)


def test_registered_peak_refuses_missing_residual() -> None:
    registry = ObjectiveRegistry(
        mode=ObjectiveMode.DATA_ONLY,
        term_weights={ObjectiveTerm.DATA_PEAK: 1.0},
    )

    with pytest.raises(ValueError, match="data_peak"):
        registry.evaluate(
            ObjectiveResiduals(
                {
                    ObjectiveTerm.DATA_U: torch.ones(1),
                    ObjectiveTerm.DATA_P: torch.ones(1),
                }
            ),
            role=EvidenceRole.TRAIN,
        )


def test_physics_only_cannot_silently_ignore_explicit_peak_weight() -> None:
    with pytest.raises(ValueError, match="data_peak"):
        ObjectiveRegistry(
            mode=ObjectiveMode.PHYSICS_ONLY,
            term_weights={ObjectiveTerm.DATA_PEAK: 1.0},
        )
