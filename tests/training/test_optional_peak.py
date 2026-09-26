from __future__ import annotations

import pytest

from piginot.objectives import (
    EquilibriumForm,
    ObjectiveMode,
    ObjectiveRegistry,
    ObjectiveTerm,
)
from piginot.training.engine import _scaled_registry


def test_physics_scale_does_not_scale_registered_peak_data_term() -> None:
    registry = ObjectiveRegistry(
        mode=ObjectiveMode.HYBRID,
        equilibrium_form=EquilibriumForm.STRONG,
        term_weights={ObjectiveTerm.DATA_PEAK: 2.0},
    )

    scaled = _scaled_registry(registry, 1.0e-4)

    assert ObjectiveTerm.DATA_PEAK in scaled.required_terms
    assert scaled.term_weights[ObjectiveTerm.DATA_PEAK] == pytest.approx(2.0)
    assert scaled.term_weights[ObjectiveTerm.EQUILIBRIUM_STRONG] == pytest.approx(
        1.0e-4
    )
