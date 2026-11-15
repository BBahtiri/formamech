"""Known finite-element fields; no reference PDE solves or optimizer steps."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from piginot.cad.refinement import refine_fixed_mesh
from piginot.compiler.reference_convergence import _relative, compare_fem_pair
from piginot.contracts import Unit
from piginot.fem.assembly import recover_stress
from piginot.fem.backend import _build_basis, _kernel_material
from piginot.fem.models import FemResult
from piginot.geometry import CellType, MeshPayload, canonicalize_mesh
from piginot.physics import first_piola, plane_stress_closure
from tests.compiler.test_execution_references import fixture

TOLERANCES = {
    "u_relative_l2": 1e-9,
    "p_relative_l2": 1e-9,
    "absolute_peak_error": 1e-9,
}


def pair(tmp_path, *, order=2):
    plan, request = fixture(tmp_path)
    source = request.mesh
    mesh = canonicalize_mesh(
        MeshPayload(
            source.nodes,
            source.cells,
            CellType(source.cell_type),
            source.boundary_facets,
            source.semantic_tags,
            source.region_ids,
            Unit.MM,
            source.source_provenance,
        )
    )
    coarse = replace(
        request,
        mesh=mesh,
        request_id="coarse",
        config=replace(request.config, element_order=order),
    )
    fine = replace(coarse, request_id="fine", mesh=refine_fixed_mesh(mesh, 2))
    return plan.problem, coarse, fine


def field(points, *, quadratic=False, factor=1.0):
    x, y = points[..., 0], points[..., 1]
    values = np.stack((0.03 * x, -0.01 * y), axis=-1)
    gradient = np.broadcast_to(np.diag([0.03, -0.01]), (*x.shape, 2, 2)).copy()
    if quadratic:
        values = values + np.stack((0.001 * x * y, 0.001 * y**2), axis=-1)
        gradient[..., 0, 0] += 0.001 * y
        gradient[..., 0, 1] += 0.001 * x
        gradient[..., 1, 1] += 0.002 * y
    return factor * values, factor * gradient


def result_for(request, *, quadratic=False, factor=1.0):
    _, basis, scalar = _build_basis(request)
    xy = scalar.doflocs.T
    points = np.moveaxis(np.asarray(basis.global_coordinates()), 0, -1)
    displacement, _ = field(xy, quadratic=quadratic, factor=factor)
    _, gradient = field(points, quadratic=quadratic, factor=factor)
    deformation = torch.from_numpy(gradient + np.eye(2))
    material = _kernel_material(request.material)
    closure = plane_stress_closure(deformation, material)
    full = torch.zeros((*points.shape[:2], 3, 3), dtype=torch.float64)
    full[..., :2, :2] = deformation
    full[..., 2, 2] = closure
    stress = first_piola(full, material).detach().numpy()
    recovered = recover_stress(scalar, stress)
    return FemResult(
        request.request_id,
        request.mesh.mesh_id,
        request.config.element_order,
        xy.tolist(),
        displacement.tolist(),
        points.tolist(),
        basis.dx.tolist(),
        stress.tolist(),
        full.detach().numpy().tolist(),
        recovered.tolist(),
        {},
        (),
        0.0,
        0.0,
        0.0,
        (),
        {},
    )


@pytest.mark.parametrize("order", [1, 2])
def test_affine_field_is_identical_on_both_layouts(tmp_path, order):
    spec, coarse, fine = pair(tmp_path, order=order)
    result = compare_fem_pair(
        spec,
        coarse,
        result_for(coarse),
        fine,
        result_for(fine),
        tolerances=TOLERANCES,
    )
    assert result["status"] == "PASS"
    assert result["geometry_status"] == "SAME_POLYGON"
    assert result["qualification"] == "UNCALIBRATED"
    assert result["total_error_bound"] is None
    assert result["reference_terms_qualified"] is False
    for value in result["quantities"].values():
        assert value == pytest.approx(0, abs=2e-13)


def test_exact_quadratic_field_uses_p2_and_its_gradient(tmp_path):
    spec, coarse, fine = pair(tmp_path)
    result = compare_fem_pair(
        spec,
        coarse,
        result_for(coarse, quadratic=True),
        fine,
        result_for(fine, quadratic=True),
        tolerances={**TOLERANCES, "absolute_peak_error": 1},
    )
    assert result["quantities"]["u_relative_l2"] == pytest.approx(0, abs=2e-14)
    assert result["quantities"]["p_relative_l2"] == pytest.approx(0, abs=2e-13)
    assert result["status"] == "PASS"


def test_quadratic_field_can_skip_the_middle_reference_level(tmp_path):
    spec, coarse, _ = pair(tmp_path)
    fine = replace(coarse, request_id="fine", mesh=refine_fixed_mesh(coarse.mesh, 4))
    result = compare_fem_pair(
        spec,
        coarse,
        result_for(coarse, quadratic=True),
        fine,
        result_for(fine, quadratic=True),
        tolerances={**TOLERANCES, "absolute_peak_error": 1},
    )
    assert result["quantities"]["u_relative_l2"] == pytest.approx(0, abs=2e-14)
    assert result["quantities"]["p_relative_l2"] == pytest.approx(0, abs=2e-13)
    assert result["status"] == "PASS"


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_selected_peak_norm_scales_before_squaring(tmp_path, scale):
    spec, coarse, fine = pair(tmp_path)
    first, second = result_for(coarse), result_for(fine)
    first = replace(
        first,
        recovered_first_piola_stress=(
            np.asarray(first.recovered_first_piola_stress) * scale
        ).tolist(),
    )
    second = replace(
        second,
        recovered_first_piola_stress=(
            np.asarray(second.recovered_first_piola_stress) * (2 * scale)
        ).tolist(),
    )
    result = compare_fem_pair(spec, coarse, first, fine, second, tolerances=None)
    peak = result["qois"]["peak"]
    assert 0 < peak["coarse"] < peak["fine"] < float("inf")
    assert peak["relative_change"] == pytest.approx(0.5, rel=1e-12)


def test_second_pair_of_base_ladder_retains_the_same_polygon(tmp_path):
    spec, base, medium = pair(tmp_path)
    fine = replace(base, request_id="fine", mesh=refine_fixed_mesh(base.mesh, 4))
    result = compare_fem_pair(
        spec,
        medium,
        result_for(medium),
        fine,
        result_for(fine),
        tolerances=TOLERANCES,
    )
    assert result["status"] == "PASS"


def test_measurement_only_does_not_invent_acceptance_thresholds(tmp_path):
    spec, coarse, fine = pair(tmp_path)
    result = compare_fem_pair(
        spec,
        coarse,
        result_for(coarse),
        fine,
        result_for(fine, factor=2),
        tolerances=None,
    )
    assert result["status"] == "MEASURED_CHANGES_ONLY"
    assert result["tolerances"] is None
    assert result["quantities"]["u_relative_l2"] == pytest.approx(0.5)
    assert result["quantities"]["p_relative_l2"] > 0.1
    assert result["qualification"] == "UNCALIBRATED"
    assert result["total_error_bound"] is None


def test_reports_nonzero_field_and_selected_peak_changes(tmp_path):
    spec, coarse, fine = pair(tmp_path)
    result = compare_fem_pair(
        spec,
        coarse,
        result_for(coarse),
        fine,
        result_for(fine, factor=2),
        tolerances=TOLERANCES,
    )
    assert result["status"] == "FAIL"
    assert result["quantities"]["u_relative_l2"] == pytest.approx(0.5)
    assert result["quantities"]["p_relative_l2"] > 0.1
    qoi = result["qois"]["peak"]
    assert qoi["coarse"] < qoi["fine"]
    assert qoi["relative_change"] == pytest.approx(
        (qoi["fine"] - qoi["coarse"]) / qoi["fine"]
    )


def test_zero_integral_delta_does_not_hide_stress_field_change(tmp_path):
    spec, coarse, fine = pair(tmp_path)
    coarse_result, fine_result = result_for(coarse), result_for(fine)
    points = np.asarray(fine_result.quadrature_points)
    weights = np.asarray(fine_result.quadrature_weights)
    stress = np.asarray(fine_result.first_piola_stress)
    oscillation = points[..., 0] - np.sum(weights * points[..., 0]) / weights.sum()
    stress[..., 0, 0] += oscillation
    changed = replace(fine_result, first_piola_stress=stress.tolist())
    assert abs(float(np.sum(oscillation * weights))) < 2e-13
    result = compare_fem_pair(
        spec, coarse, coarse_result, fine, changed, tolerances=TOLERANCES
    )
    assert result["quantities"]["p_relative_l2"] > 0.1
    assert result["status"] == "FAIL"


def test_sigma_peaks_use_declared_stress_conversion(tmp_path):
    spec, coarse, fine = pair(tmp_path)
    spec = replace(spec, qois=(replace(spec.qois[0], field="sigma"),))
    result = compare_fem_pair(
        spec,
        coarse,
        result_for(coarse),
        fine,
        result_for(fine),
        tolerances=TOLERANCES,
    )
    assert result["status"] == "PASS"
    assert result["qois"]["peak"]["field"] == "sigma"


@pytest.mark.parametrize("change", ["material", "loads", "solver", "tags"])
def test_refuses_different_physical_problem(tmp_path, change):
    spec, coarse, fine = pair(tmp_path)
    coarse_result, fine_result = result_for(coarse), result_for(fine)
    if change == "material":
        fine = replace(fine, material=replace(fine.material, name="other"))
    elif change == "loads":
        fine = replace(fine, boundary_conditions=())
    elif change == "solver":
        fine = replace(fine, config=replace(fine.config, load_steps=1))
    else:
        fine = replace(fine, mesh=replace(fine.mesh, semantic_tags={}))
    with pytest.raises(ValueError):
        compare_fem_pair(
            spec, coarse, coarse_result, fine, fine_result, tolerances=TOLERANCES
        )


def test_refuses_equal_mesh(tmp_path):
    spec, coarse, _ = pair(tmp_path)
    fine = replace(coarse, request_id="fine", mesh=refine_fixed_mesh(coarse.mesh, 1))
    with pytest.raises(ValueError, match="midpoint"):
        compare_fem_pair(
            spec,
            coarse,
            result_for(coarse),
            fine,
            result_for(fine),
            tolerances=TOLERANCES,
        )


def test_refuses_non_nested_mesh_with_supported_cell_count(tmp_path):
    spec, coarse, _ = pair(tmp_path)
    original = refine_fixed_mesh(coarse.mesh, 4)
    width = max(x for x, y in original.nodes)
    height = max(y for x, y in original.nodes)
    index = next(
        i for i, (x, y) in enumerate(original.nodes) if 0 < x < width and 0 < y < height
    )
    nodes = tuple(
        (x + 1e-3, y) if i == index else (x, y)
        for i, (x, y) in enumerate(original.nodes)
    )
    mesh = canonicalize_mesh(
        MeshPayload(
            nodes,
            original.cells,
            CellType(original.cell_type),
            original.boundary_facets,
            original.semantic_tags,
            original.region_ids,
            Unit.MM,
            original.source_provenance,
        )
    )
    fine = replace(coarse, request_id="fine", mesh=mesh)
    with pytest.raises(ValueError, match="midpoint"):
        compare_fem_pair(
            spec,
            coarse,
            result_for(coarse),
            fine,
            result_for(fine),
            tolerances=TOLERANCES,
        )


@pytest.mark.parametrize("attribute", ["request_id", "mesh_id", "element_order"])
def test_refuses_result_identity_mismatch(tmp_path, attribute):
    spec, coarse, fine = pair(tmp_path)
    changed = replace(
        result_for(fine), **{attribute: 1 if attribute == "element_order" else "other"}
    )
    with pytest.raises(ValueError, match="identity"):
        compare_fem_pair(
            spec, coarse, result_for(coarse), fine, changed, tolerances=TOLERANCES
        )


@pytest.mark.parametrize(
    "attribute",
    [
        "displacement",
        "displacement_dof_coordinates",
        "quadrature_points",
        "quadrature_weights",
        "first_piola_stress",
        "recovered_first_piola_stress",
    ],
)
def test_refuses_malformed_result_layout(tmp_path, attribute):
    spec, coarse, fine = pair(tmp_path)
    changed = replace(result_for(fine), **{attribute: [[float("nan")]]})
    with pytest.raises(ValueError):
        compare_fem_pair(
            spec, coarse, result_for(coarse), fine, changed, tolerances=TOLERANCES
        )


@pytest.mark.parametrize(
    "attribute",
    ["displacement_dof_coordinates", "quadrature_points", "quadrature_weights"],
)
def test_refuses_finite_but_wrong_layout(tmp_path, attribute):
    spec, coarse, fine = pair(tmp_path)
    result = result_for(fine)
    values = np.asarray(getattr(result, attribute)) + 0.01
    changed = replace(result, **{attribute: values.tolist()})
    with pytest.raises(ValueError, match="layout"):
        compare_fem_pair(
            spec, coarse, result_for(coarse), fine, changed, tolerances=TOLERANCES
        )


@pytest.mark.parametrize(
    "tolerances",
    [
        {},
        {**TOLERANCES, "other": 1},
        {**TOLERANCES, "u_relative_l2": -1},
        {**TOLERANCES, "u_relative_l2": float("inf")},
        {**TOLERANCES, "u_relative_l2": True},
    ],
)
def test_requires_explicit_finite_tolerances(tmp_path, tolerances):
    spec, coarse, fine = pair(tmp_path)
    with pytest.raises(ValueError, match="tolerances"):
        compare_fem_pair(
            spec,
            coarse,
            result_for(coarse),
            fine,
            result_for(fine),
            tolerances=tolerances,
        )


def test_refuses_empty_selected_region(tmp_path):
    spec, coarse, fine = pair(tmp_path)
    spec = replace(
        spec, qois=(replace(spec.qois[0], exclude_regions=(spec.qois[0].region,)),)
    )
    with pytest.raises(ValueError, match="no vertices"):
        compare_fem_pair(
            spec,
            coarse,
            result_for(coarse),
            fine,
            result_for(fine),
            tolerances=TOLERANCES,
        )


def test_refuses_nonpositive_cauchy_determinant(tmp_path):
    spec, coarse, fine = pair(tmp_path)
    spec = replace(spec, qois=(replace(spec.qois[0], field="sigma"),))
    result = result_for(fine)
    deformation = np.asarray(result.deformation_gradient)
    deformation[..., 2, 2] = 0
    with pytest.raises(ValueError, match="positive reference determinant"):
        compare_fem_pair(
            spec,
            coarse,
            result_for(coarse),
            fine,
            replace(result, deformation_gradient=deformation.tolist()),
            tolerances=TOLERANCES,
        )


def test_refuses_inadmissible_coarse_gradient(tmp_path):
    spec, coarse, fine = pair(tmp_path)
    result = result_for(coarse)
    displacement = np.asarray(result.displacement)
    displacement[:, 0] = -2 * np.asarray(result.displacement_dof_coordinates)[:, 0]
    with pytest.raises(ValueError, match="constitutive evaluation failed"):
        compare_fem_pair(
            spec,
            coarse,
            replace(result, displacement=displacement.tolist()),
            fine,
            result_for(fine),
            tolerances=TOLERANCES,
        )


def test_zero_reference_and_nonfinite_norms_are_explicit():
    zero = np.zeros((1, 1))
    weights = np.ones(1)
    assert _relative(zero, zero, weights) == 0
    with pytest.raises(ValueError, match="zero fine-reference norm"):
        _relative(np.ones((1, 1)), zero, weights)
    assert _relative(np.array([[1e200]]), np.ones((1, 1)), weights) == pytest.approx(
        1e200
    )
    with pytest.raises(ValueError, match="nonfinite"):
        _relative(np.array([[float("nan")]]), np.ones((1, 1)), weights)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
@pytest.mark.parametrize("weight", [1e-200, 1e200])
def test_relative_norm_scales_before_squaring(scale, weight):
    assert _relative(
        np.array([[scale]]), np.array([[2 * scale]]), np.array([weight])
    ) == pytest.approx(0.5, rel=1e-12)


def test_relative_norm_preserves_extreme_finite_weighted_ratio():
    weights = np.array([1e-308, 1e308])
    difference = np.array([[1e308], [0.0]])
    reference = np.array([[0.0], [1.0]])
    assert _relative(difference, reference, weights) == pytest.approx(1.0, rel=1e-12)
    with pytest.raises(ValueError, match="nonfinite"):
        _relative(np.array([[1e308]]), np.array([[1e-308]]), np.ones(1))


def test_import_does_not_require_numerical_dependencies():
    source = Path(__file__).resolve().parents[2] / "src"
    code = f"""
import builtins, sys
sys.path.insert(0, {str(source)!r})
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split('.')[0] in {{'numpy', 'torch', 'scipy', 'skfem', 'gmsh'}}:
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
import piginot.compiler.reference_convergence
"""
    subprocess.run([sys.executable, "-I", "-c", code], check=True)
