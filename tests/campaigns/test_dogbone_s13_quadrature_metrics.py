from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from typing import Any

import pytest
import torch

from campaigns.dogbone_v1 import s13_quadrature_metrics as metrics
from piginot.contracts import canonical_json_bytes
from piginot.objectives import NondimensionalScales
from piginot.physics import first_piola, plane_stress_closure
from piginot.physics.errors import NewtonNonConvergenceError
from tests.physics.reference import make_material


def _fields() -> dict[str, Any]:
    gradient = (
        torch.tensor(((0.2, 0.1), (0.05, -0.1)), dtype=torch.float64)
        .expand(2, 2, 2, 2)
        .clone()
    )
    stress = 760.0 * gradient
    return {
        "grad_model": gradient * 2.0,
        "grad_fem": gradient,
        "served_p": stress.clone(),
        "raw_fem_p": stress,
        "area_weights": torch.tensor(((1.0, 3.0), (2.0, 4.0)), dtype=torch.float64),
        "root_cell_mask": torch.tensor((True, False)),
        "xy_mm": torch.arange(8, dtype=torch.float64).reshape(2, 2, 2),
        "material": make_material(poisson_ratio=0.23),
        "scales": NondimensionalScales(54.0, 5.4, 760.0),
    }


@pytest.fixture
def linear_material(monkeypatch: pytest.MonkeyPatch) -> None:
    def linear(
        gradient: torch.Tensor, material: object, scales: object
    ) -> torch.Tensor:
        assert torch.linalg.det(torch.eye(2) + gradient).min() > 0.0
        return 760.0 * gradient

    monkeypatch.setattr(metrics, "_material_stress", linear)


def test_weighted_gradient_stress_and_signed_decomposition(
    linear_material: None,
) -> None:
    fields = _fields()
    fields["served_p"][0, 0, 0, 0] += 76.0
    fields["u_fem"] = torch.ones((2, 2, 2), dtype=torch.float64)
    fields["u_model"] = fields["u_fem"] * 1.2

    result = metrics.summarize_quadrature_case(**fields)

    assert result.all_points.cell_count == 2
    assert result.root_incident.point_count == 2
    assert result.nonroot.area_mm2 == pytest.approx(6.0)
    for population in (result.all_points, result.root_incident, result.nonroot):
        assert population.gradient_relative_h1 == pytest.approx(1.0)
        assert population.gradient_component_rms == pytest.approx(math.sqrt(0.0625 / 4))
        torch.testing.assert_close(
            torch.tensor(population.gradient_bias),
            torch.tensor(((0.2, 0.1), (0.05, -0.1))),
        )
        assert population.u_relative_l2 == pytest.approx(0.2)
        assert population.u_component_rms_mm == pytest.approx(0.2)
        assert population.invalid_count == 0
        assert population.admissible_count == population.point_count
        assert population.admissible_area_fraction == 1.0
        assert population.decomposition is not None
        assert population.decomposition.c_normalized_component_rms == 0.0
        assert population.decomposition.material_fem_relative_l2 == 0.0
        assert population.decomposition.material_model_relative_l2 == pytest.approx(1.0)
        assert population.decomposition.identity_absolute_error < 1e-16
        assert population.decomposition.reconstruction_max_abs_mpa < 1e-13
        assert population.material_fem_relative_l2 == 0.0
        assert population.material_fem_normalized_component_rms == 0.0
    all_points = result.all_points
    assert all_points.p_relative_l2 == pytest.approx(math.sqrt(0.01 / (10 * 0.0625)))
    assert all_points.p_normalized_component_rms == pytest.approx(math.sqrt(0.01 / 40))
    decomposition = all_points.decomposition
    assert decomposition is not None
    assert decomposition.cross_ab_normalized_mean < 0.0
    assert decomposition.cross_ac_normalized_mean == 0.0
    assert decomposition.cross_bc_normalized_mean == 0.0
    assert decomposition.total_normalized_mean_squared == pytest.approx(0.01 / 40)
    assert decomposition.reconstructed_normalized_mean_squared == pytest.approx(
        0.01 / 40
    )
    with pytest.raises(FrozenInstanceError):
        all_points.point_count = 4  # type: ignore[misc]
    assert b'"root_incident"' in canonical_json_bytes(result)


def test_nonzero_material_crosscheck_and_cross_terms(linear_material: None) -> None:
    fields = _fields()
    fields["raw_fem_p"] = fields["raw_fem_p"] + 76.0
    result = metrics.summarize_quadrature_case(**fields).all_points
    decomposition = result.decomposition
    assert decomposition is not None
    assert decomposition.c_normalized_component_rms == pytest.approx(0.1)
    assert decomposition.material_fem_relative_l2 > 0.0
    assert decomposition.cross_ac_normalized_mean > 0.0
    assert decomposition.cross_bc_normalized_mean < 0.0
    assert decomposition.identity_absolute_error < 1e-16


def test_independent_scalar_weighted_energy_and_cross_term_reconstruction(
    linear_material: None,
) -> None:
    fields = _fields()
    factors = torch.tensor(((1.2, 0.8), (1.7, 1.1)), dtype=torch.float64)
    fields["grad_model"] = fields["grad_fem"] * factors[..., None, None]
    offsets = torch.arange(16, dtype=torch.float64).reshape(2, 2, 2, 2) * 0.01
    delta = torch.tensor(((0.02, -0.03), (-0.01, 0.04)), dtype=torch.float64).expand(
        2, 2, 2, 2
    )
    fields["raw_fem_p"] = 760.0 * (fields["grad_fem"] + offsets)
    fields["served_p"] = fields["raw_fem_p"] + 760.0 * delta
    result = metrics.summarize_quadrature_case(**fields)

    for population, cells in (
        (result.all_points, (0, 1)),
        (result.root_incident, (0,)),
        (result.nonroot, (1,)),
    ):
        terms: list[tuple[float, float, float, float, float]] = []
        denominator = 0.0
        area = 0.0
        for cell in cells:
            for point in range(2):
                weight = float(fields["area_weights"][cell, point])
                area += weight
                for row in range(2):
                    for column in range(2):
                        fem = float(fields["grad_fem"][cell, point, row, column])
                        model = float(fields["grad_model"][cell, point, row, column])
                        raw = (
                            float(fields["raw_fem_p"][cell, point, row, column]) / 760.0
                        )
                        served = (
                            float(fields["served_p"][cell, point, row, column]) / 760.0
                        )
                        terms.append(
                            (
                                weight,
                                served - model,
                                model - fem,
                                fem - raw,
                                served - raw,
                            )
                        )
                        denominator += weight * raw**2
        norm_a, norm_b, norm_c = (
            math.sqrt(sum(term[0] * term[index] ** 2 for term in terms) / (4 * area))
            for index in (1, 2, 3)
        )
        cross_ab, cross_ac, cross_bc = (
            sum(2 * term[0] * term[left] * term[right] for term in terms) / (4 * area)
            for left, right in ((1, 2), (1, 3), (2, 3))
        )
        total = sum(term[0] * term[4] ** 2 for term in terms) / (4 * area)
        derived_stress_relative = math.sqrt(
            sum(term[0] * (term[2] + term[3]) ** 2 for term in terms) / denominator
        )
        decomposition = population.decomposition
        assert decomposition is not None
        assert decomposition.a_normalized_component_rms == pytest.approx(norm_a)
        assert decomposition.b_normalized_component_rms == pytest.approx(norm_b)
        assert decomposition.c_normalized_component_rms == pytest.approx(norm_c)
        assert decomposition.cross_ab_normalized_mean == pytest.approx(cross_ab)
        assert decomposition.cross_ac_normalized_mean == pytest.approx(cross_ac)
        assert decomposition.cross_bc_normalized_mean == pytest.approx(cross_bc)
        assert decomposition.total_normalized_mean_squared == pytest.approx(total)
        assert decomposition.material_model_relative_l2 == pytest.approx(
            derived_stress_relative
        )
        assert decomposition.reconstructed_normalized_mean_squared == pytest.approx(
            norm_a**2 + norm_b**2 + norm_c**2 + cross_ab + cross_ac + cross_bc
        )


def test_invalid_model_points_are_retained_and_material_is_conditional(
    linear_material: None,
) -> None:
    fields = _fields()
    fields["grad_model"][0, :, 0, 0] = -2.0
    fields["grad_model"][1, 0, 0, 0] = -1.0

    result = metrics.summarize_quadrature_case(**fields)

    assert result.root_incident.decomposition is None
    assert result.root_incident.material_fem_relative_l2 == 0.0
    assert result.root_incident.material_fem_normalized_component_rms == 0.0
    assert result.root_incident.admissible_count == 0
    assert result.root_incident.admissible_area_fraction == 0.0
    assert result.all_points.invalid_count == 3
    assert result.all_points.admissible_count == 1
    assert result.all_points.admissible_area_fraction == pytest.approx(0.4)
    assert result.all_points.gradient_relative_h1 > 1.0
    assert result.all_points.p_relative_l2 == 0.0
    invalid = result.all_points.invalid_points
    assert tuple((point.cell_index, point.quadrature_index) for point in invalid) == (
        (0, 0),
        (0, 1),
        (1, 0),
    )
    assert invalid[0].xy_mm == (0.0, 1.0)
    assert invalid[0].area_weight_mm2 == 1.0
    assert invalid[0].determinant < 0.0
    conditional = result.all_points.decomposition
    assert conditional is not None
    assert conditional.admissible_count == 1
    assert conditional.admissible_area_mm2 == 4.0


def test_actual_s4_material_matches_affine_fem_stress() -> None:
    fields = _fields()
    fields["grad_model"] = fields["grad_fem"].clone()
    deformation = torch.eye(2, dtype=torch.float64) + fields["grad_fem"]
    deformation3 = torch.zeros((2, 2, 3, 3), dtype=torch.float64)
    deformation3[..., :2, :2] = deformation
    deformation3[..., 2, 2] = plane_stress_closure(deformation, fields["material"])
    fields["raw_fem_p"] = first_piola(deformation3, fields["material"])[..., :2, :2]
    fields["served_p"] = fields["raw_fem_p"].clone()

    result = metrics.summarize_quadrature_case(**fields)

    assert result.all_points.gradient_relative_h1 == 0.0
    assert result.all_points.p_relative_l2 == 0.0
    assert result.all_points.u_relative_l2 is None
    decomposition = result.all_points.decomposition
    assert decomposition is not None
    assert decomposition.c_normalized_component_rms < 1e-15
    assert decomposition.b_normalized_component_rms == 0.0
    assert decomposition.a_normalized_component_rms < 1e-15


@pytest.mark.parametrize(
    ("name", "replacement", "message"),
    (
        ("grad_model", torch.zeros((2, 2, 2)), "aligned"),
        ("grad_model", torch.ones((2, 2, 2, 2)), "CPU float64"),
        ("served_p", torch.full((2, 2, 2, 2), math.nan, dtype=torch.float64), "finite"),
        ("area_weights", torch.zeros((2, 2), dtype=torch.float64), "positive"),
        ("root_cell_mask", torch.tensor((True, True)), "complement"),
        ("root_cell_mask", torch.ones(2), "boolean"),
        ("grad_fem", torch.zeros((2, 2, 2, 2), dtype=torch.float64), "gradient norm"),
        ("raw_fem_p", torch.zeros((2, 2, 2, 2), dtype=torch.float64), "stress norm"),
    ),
)
def test_bad_inputs_fail_closed(
    linear_material: None, name: str, replacement: torch.Tensor, message: str
) -> None:
    fields = _fields()
    fields[name] = replacement
    with pytest.raises(ValueError, match=message):
        metrics.summarize_quadrature_case(**fields)


def test_invalid_fem_or_displacement_inputs_fail_closed(linear_material: None) -> None:
    fields = _fields()
    fields["grad_fem"][0, 0, 0, 0] = -2.0
    with pytest.raises(ValueError, match="FEM deformation"):
        metrics.summarize_quadrature_case(**fields)
    fields = _fields()
    fields["u_model"] = torch.ones((2, 2, 2), dtype=torch.float64)
    with pytest.raises(ValueError, match="together"):
        metrics.summarize_quadrature_case(**fields)
    fields["u_fem"] = torch.zeros((2, 2, 2), dtype=torch.float64)
    with pytest.raises(ValueError, match="displacement norm"):
        metrics.summarize_quadrature_case(**fields)


def test_all_invalid_model_points_keep_unconditional_fem_crosscheck(
    linear_material: None,
) -> None:
    fields = _fields()
    fields["grad_model"][..., 0, 0] = -2.0
    fields["raw_fem_p"] = fields["raw_fem_p"] + 76.0
    result = metrics.summarize_quadrature_case(**fields)
    assert result.all_points.decomposition is None
    assert result.all_points.invalid_count == 4
    assert result.all_points.material_fem_normalized_component_rms == pytest.approx(0.1)
    assert result.all_points.material_fem_relative_l2 > 0.0


def test_material_failure_never_drops_an_admissible_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_kernel(*args: object) -> torch.Tensor:
        raise NewtonNonConvergenceError("Newton failed")

    monkeypatch.setattr(metrics, "_material_stress", failed_kernel)
    with pytest.raises(ValueError, match="FEM constitutive.*no points dropped"):
        metrics.summarize_quadrature_case(**_fields())


def test_sparse_fields_and_nonfinite_determinants_are_rejected(
    linear_material: None,
) -> None:
    fields = _fields()
    fields["served_p"] = fields["served_p"].to_sparse()
    with pytest.raises(ValueError, match="CPU float64"):
        metrics.summarize_quadrature_case(**fields)
    fields = _fields()
    fields["grad_model"] = torch.eye(2, dtype=torch.float64).expand(2, 2, 2, 2) * 1e308
    with pytest.raises(ValueError, match="determinants must be finite"):
        metrics.summarize_quadrature_case(**fields)
    fields = _fields()
    fields["material"] = object()
    with pytest.raises(TypeError, match="exact S13"):
        metrics.summarize_quadrature_case(**fields)
