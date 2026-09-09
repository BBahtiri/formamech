from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest
import torch

from piginot.doe import EvidenceRole
from piginot.fem.mms_runner import verification_material
from piginot.objectives import (
    CoordinateConversion,
    EquilibriumForm,
    NondimensionalScales,
    ObjectiveMode,
    ObjectiveRegistry,
    ObjectiveResiduals,
    ObjectiveTerm,
    S12GateMeasurements,
    angular_momentum_residual,
    constitutive_residual,
    dirichlet_residual,
    displacement_gradient,
    evaluate_objective,
    evaluate_s12_gate,
    run_s12_verification,
    strong_equilibrium_residual,
    supervised_displacement_residual,
    traction_free_residual,
    weak_equilibrium_residual,
)

SCALES = NondimensionalScales(1.0, 1.0, 1.0)
CONVERSION = CoordinateConversion((1.0, 1.0))


@pytest.mark.parametrize(
    "value",
    ([1.0, 1.0], (1.0,), (0.0, 1.0), (True, 1.0), (float("nan"), 1.0)),
)
def test_coordinate_conversion_rejects_noncanonical_factors(value: object) -> None:
    with pytest.raises(ValueError):
        CoordinateConversion(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    (
        [],
        {"data_u": torch.ones(1)},
        {ObjectiveTerm.DATA_U: torch.ones(1, dtype=torch.int64)},
        {ObjectiveTerm.DATA_U: torch.empty(0)},
        {ObjectiveTerm.DATA_U: torch.tensor([float("nan")])},
    ),
)
def test_objective_residual_contract_rejects_malformed_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ObjectiveResiduals(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    (
        {"mode": "HYBRID"},
        {"equilibrium_form": "weak"},
        {"term_weights": []},
        {"term_weights": {"data_u": 1.0}},
        {"term_weights": {ObjectiveTerm.DATA_U: 0.0}},
        {"term_weights": {ObjectiveTerm.DATA_U: float("inf")}},
    ),
)
def test_registry_contract_rejects_untyped_or_nonpositive_policy(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        ObjectiveRegistry(**kwargs)  # type: ignore[arg-type]


def test_registry_weight_is_applied_and_functional_boundary_is_exact() -> None:
    registry = ObjectiveRegistry(
        mode=ObjectiveMode.DATA_ONLY,
        equilibrium_form=EquilibriumForm.WEAK,
        term_weights={ObjectiveTerm.DATA_U: 2.0},
    )
    residuals = ObjectiveResiduals(
        {
            ObjectiveTerm.DATA_U: torch.ones(1),
            ObjectiveTerm.DATA_P: torch.ones(1),
        }
    )

    assert registry.evaluate(residuals, role=EvidenceRole.TRAIN).total.item() == 3.0
    with pytest.raises(TypeError, match="registry"):
        evaluate_objective(object(), residuals, role=EvidenceRole.TRAIN)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="role"):
        evaluate_objective(registry, residuals, role="TRAIN")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="residuals"):
        evaluate_objective(registry, object(), role=EvidenceRole.TRAIN)  # type: ignore[arg-type]


def test_local_term_boundaries_reject_shape_mask_normal_and_context_errors() -> None:
    displacement = torch.ones((1, 1, 2), dtype=torch.float64)
    with pytest.raises(ValueError, match="identical shapes"):
        supervised_displacement_residual(
            displacement, torch.ones((1, 2, 2), dtype=torch.float64), SCALES
        )
    with pytest.raises(ValueError, match="dense floating"):
        supervised_displacement_residual(
            displacement.to(torch.int64), displacement.to(torch.int64), SCALES
        )
    with pytest.raises(ValueError, match="finite"):
        supervised_displacement_residual(
            torch.full_like(displacement, float("inf")), displacement, SCALES
        )
    with pytest.raises(TypeError, match="component_mask"):
        dirichlet_residual(displacement, displacement, object(), SCALES)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="boolean"):
        dirichlet_residual(displacement, displacement, torch.ones_like(displacement), SCALES)
    with pytest.raises(ValueError, match="align"):
        dirichlet_residual(
            displacement,
            displacement,
            torch.ones((1, 2, 2), dtype=torch.bool),
            SCALES,
        )
    with pytest.raises(ValueError, match="select"):
        dirichlet_residual(
            displacement, displacement, torch.zeros_like(displacement, dtype=torch.bool), SCALES
        )

    stress = torch.zeros((1, 1, 2, 2), dtype=torch.float64)
    with pytest.raises(ValueError, match="align"):
        traction_free_residual(
            stress, torch.ones((1, 2, 2), dtype=torch.float64), SCALES
        )
    with pytest.raises(ValueError, match="unit"):
        traction_free_residual(
            stress, torch.tensor([[[2.0, 0.0]]], dtype=torch.float64), SCALES
        )
    with pytest.raises(ValueError, match="dtype"):
        traction_free_residual(
            stress, torch.tensor([[[1.0, 0.0]]], dtype=torch.float32), SCALES
        )


def test_constitutive_and_angular_boundaries_reject_incompatible_fields() -> None:
    stress = torch.zeros((1, 1, 2, 2), dtype=torch.float64)
    gradient = torch.zeros_like(stress)
    with pytest.raises(ValueError, match="align"):
        constitutive_residual(
            stress,
            torch.zeros((1, 2, 2, 2), dtype=torch.float64),
            verification_material(),
            SCALES,
        )
    with pytest.raises(ValueError, match="dtype"):
        constitutive_residual(
            stress,
            gradient.to(torch.float32),
            verification_material(),
            SCALES,
        )
    with pytest.raises(ValueError, match="align"):
        angular_momentum_residual(
            stress, torch.zeros((1, 2, 2, 2), dtype=torch.float64), SCALES
        )


def test_strong_and_kinematic_derivatives_fail_closed_on_detach_and_mismatch() -> None:
    coordinates = torch.ones((1, 2, 2), dtype=torch.float64, requires_grad=True)
    connected_stress = coordinates[..., 0, None, None] * torch.ones(
        (1, 2, 2, 2), dtype=torch.float64
    )
    with pytest.raises(ValueError, match="body_force"):
        strong_equilibrium_residual(
            connected_stress,
            coordinates,
            torch.zeros((1, 1, 2), dtype=torch.float64),
            SCALES,
            CONVERSION,
        )
    with pytest.raises(ValueError, match="require gradients"):
        strong_equilibrium_residual(
            connected_stress.detach(),
            coordinates.detach(),
            torch.zeros_like(coordinates),
            SCALES,
            CONVERSION,
        )
    with pytest.raises(ValueError, match="connected"):
        strong_equilibrium_residual(
            torch.ones((1, 2, 2, 2), dtype=torch.float64),
            coordinates,
            torch.zeros_like(coordinates),
            SCALES,
            CONVERSION,
        )
    with pytest.raises(ValueError, match="identical shapes"):
        displacement_gradient(
            coordinates[:, :1], coordinates, CONVERSION
        )
    with pytest.raises(ValueError, match="connected"):
        displacement_gradient(torch.ones_like(coordinates), coordinates, CONVERSION)
    with pytest.raises(ValueError, match="vector field"):
        displacement_gradient(
            torch.ones((1, 2, 3), dtype=torch.float64), coordinates, CONVERSION
        )


def _weak_arguments() -> dict[str, torch.Tensor | NondimensionalScales]:
    return {
        "stress": torch.zeros((1, 2, 2, 2), dtype=torch.float64),
        "body_force": torch.zeros((1, 2, 2), dtype=torch.float64),
        "test_values": torch.ones((1, 2, 1), dtype=torch.float64),
        "test_gradients": torch.ones((1, 2, 1, 2), dtype=torch.float64),
        "volume_weights": torch.ones((1, 2), dtype=torch.float64),
        "scales": SCALES,
    }


def _call_weak(
    arguments: Mapping[str, object],
    boundary: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    supplied = {} if boundary is None else boundary
    return weak_equilibrium_residual(
        stress=cast(torch.Tensor, arguments["stress"]),
        body_force=cast(torch.Tensor, arguments["body_force"]),
        test_values=cast(torch.Tensor, arguments["test_values"]),
        test_gradients=cast(torch.Tensor, arguments["test_gradients"]),
        volume_weights=cast(torch.Tensor, arguments["volume_weights"]),
        scales=cast(NondimensionalScales, arguments["scales"]),
        traction=supplied.get("traction"),
        boundary_test_values=supplied.get("boundary_test_values"),
        boundary_weights=supplied.get("boundary_weights"),
    )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("test_values", torch.ones((1, 2), dtype=torch.float64), "test_values"),
        ("volume_weights", torch.ones((1, 2, 1), dtype=torch.float64), "volume_weights"),
        ("body_force", torch.zeros((1, 1, 2), dtype=torch.float64), "volume fields"),
        ("test_gradients", torch.ones((1, 1, 1, 2), dtype=torch.float64), "test_gradients"),
        ("volume_weights", torch.ones((1, 1), dtype=torch.float64), "volume_weights"),
    ),
)
def test_weak_volume_contract_rejects_implicit_broadcasting(
    field: str, replacement: torch.Tensor, message: str
) -> None:
    arguments = _weak_arguments()
    arguments[field] = replacement
    with pytest.raises(ValueError, match=message):
        _call_weak(arguments)


def test_weak_boundary_contract_requires_a_complete_aligned_triple() -> None:
    arguments = _weak_arguments()
    with pytest.raises(ValueError, match="all required"):
        _call_weak(
            arguments,
            {"traction": torch.ones((1, 1, 2), dtype=torch.float64)},
        )

    boundary = {
        "traction": torch.ones((1, 1, 2), dtype=torch.float64),
        "boundary_test_values": torch.ones((1, 1, 1), dtype=torch.float64),
        "boundary_weights": torch.ones((1, 1), dtype=torch.float64),
    }
    for field, replacement in (
        ("traction", torch.ones((1, 1, 1, 2), dtype=torch.float64)),
        ("boundary_test_values", torch.ones((1, 2, 1), dtype=torch.float64)),
        ("boundary_weights", torch.ones((1, 2), dtype=torch.float64)),
    ):
        changed = {**boundary, field: replacement}
        with pytest.raises(ValueError):
            _call_weak(arguments, changed)


def test_gate_measurement_mapping_order_is_not_semantic() -> None:
    report = run_s12_verification()
    measurements = S12GateMeasurements(
        exact_max_residuals=dict(reversed(tuple(report.exact_max_residuals.items()))),
        perturbation_fired_terms=dict(
            reversed(tuple(report.perturbation_fired_terms.items()))
        ),
        finite_difference_relative_errors=dict(
            reversed(tuple(report.finite_difference_relative_errors.items()))
        ),
        detached_terms=(),
        reachable_parameter_names=report.reachable_parameter_names,
        required_parameter_names=report.required_parameter_names,
        anisotropy_relative_change=report.anisotropy_relative_change,
        coordinate_anisotropy_mutant_fired=True,
        fem_solve_success=True,
        weak_strong_difference=report.weak_strong_difference,
        s8_discretization_error_bound=report.s8_discretization_error_bound,
        constitutive_residual_shape=(2, 2),
        constitutive_rejects_out_of_plane=True,
        holdout_refused=True,
        holdout_access_mutant_fired=True,
        code_coverage_percent=85.0,
    )

    assert evaluate_s12_gate(measurements).verdict.value == "PASS"
