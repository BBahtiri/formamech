from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from campaigns.dogbone_v1.family import compiled_family, parameter_point
from campaigns.dogbone_v1.s10_requests import _load_public_sources
from campaigns.dogbone_v1.s14_reference import (
    RECOVERY_SAFETY_FACTOR,
    RICHARDSON_ORDER,
    RICHARDSON_SAFETY_FACTOR,
    FemReferenceTerms,
    level_differences,
    level_mesh,
    nested_vertex_map,
    richardson_bound,
    root_vertices,
)
from piginot.fem import s8_gate
from piginot.morphing import apply_morph


def _case():  # type: ignore[no-untyped-def]
    _, manifest = _load_public_sources()
    return next(case for case in manifest.cases if case.status.value == "ACTIVE")


def test_safety_factors_and_orders_are_the_s8_values() -> None:
    assert RECOVERY_SAFETY_FACTOR == s8_gate._RECOVERY_SAFETY_FACTOR
    assert RICHARDSON_SAFETY_FACTOR == s8_gate._RICHARDSON_SAFETY_FACTOR
    assert RICHARDSON_ORDER == 1.0


def test_unit_level_mesh_is_the_s10_mesh_and_coarse_vertices_nest() -> None:
    case = _case()
    family = compiled_family()
    theta = parameter_point(
        *(
            float(case.primitive_coordinates[n])
            for n in ("L_total", "W_grip", "W_gauge", "R_fillet")
        )
    )
    s10 = apply_morph(family.reference_mesh, family.morph_map, family.membership, theta)
    assert s10.mesh is not None
    unit = level_mesh(case, 1.0)
    assert unit.nodes == s10.mesh.nodes and unit.cells == s10.mesh.cells
    assert root_vertices(unit) == root_vertices(s10.mesh)

    coarse = level_mesh(case, 0.5)
    mapped = nested_vertex_map(coarse, unit)
    assert (len(coarse.nodes), len(unit.nodes)) == (261, 969)
    assert np.array_equal(np.asarray(unit.nodes)[mapped], np.asarray(coarse.nodes))
    assert set(mapped[list(root_vertices(coarse))]) <= set(root_vertices(unit))
    with pytest.raises(ValueError):
        nested_vertex_map(unit, unit)
    with pytest.raises(ValueError):
        level_mesh(case, 0.3)


def test_richardson_bound_uses_the_s8_formula() -> None:
    assert richardson_bound(0.0) == 0.0
    assert richardson_bound(0.03) == pytest.approx(1.97 * 0.03)
    assert richardson_bound(0.03, 2.0) == pytest.approx(1.97 * 0.03 / 3.0)
    assert richardson_bound(0.07, 3.0) == pytest.approx(1.97 * 0.07 / 7.0)


def test_components_follow_the_documented_rules() -> None:
    terms = FemReferenceTerms("c", 0.001, 0.004, 0.0007, 0.006, 0.0003, 261)
    parts = terms.components()

    assert parts["p_relative_l2"] == {
        "discretisation": pytest.approx(max(2.61 * 0.001, 1.97 * 0.006)),
        "projection": 0.001,
    }
    assert parts["absolute_peak_error"] == {
        "discretisation": pytest.approx(2.61 * 0.004),
        "projection": 0.004,
    }
    assert parts["u_relative_l2"] == {
        "discretisation": pytest.approx(1.97 * 0.0007),
        "projection": 0.0,
    }
    with pytest.raises(ValueError):
        FemReferenceTerms("c", -1.0, 0.0, 0.0, 0.0, 0.0, 1)
    with pytest.raises(ValueError):
        FemReferenceTerms("c", float("nan"), 0.0, 0.0, 0.0, 0.0, 1)


def _fake(mesh, u, p):  # type: ignore[no-untyped-def]
    stress = np.zeros((len(mesh.nodes), 3, 3))
    stress[:, :2, :2] = p
    request = SimpleNamespace(mesh=mesh)
    result = SimpleNamespace(
        mesh_id=mesh.mesh_id,
        displacement_dof_coordinates=mesh.nodes,
        displacement=u,
        recovered_first_piola_stress=stress,
    )
    return request, result


def test_level_differences_on_known_fields() -> None:
    case = _case()
    coarse, unit = level_mesh(case, 0.5), level_mesh(case, 1.0)
    xy_c, xy_u = np.asarray(coarse.nodes), np.asarray(unit.nodes)
    stress_u = np.stack([np.eye(2) * (1.0 + x) for x, _ in xy_u])
    stress_c = np.stack([np.eye(2) * (1.0 + x) for x, _ in xy_c])
    fine = _fake(unit, 0.01 * xy_u, stress_u)
    same = _fake(coarse, 0.01 * xy_c, stress_c)
    assert level_differences(*same, *fine)[:3] == (0.0, 0.0, 0.0)  # type: ignore[arg-type]

    shifted = _fake(coarse, 0.0101 * xy_c, 1.02 * stress_c)
    du, dp, dpeak, count = level_differences(*shifted, *fine)  # type: ignore[arg-type]
    assert (du, dp, dpeak) == pytest.approx((0.01, 0.02, 0.02))
    assert count == 261
    with pytest.raises(ValueError):
        level_differences(shifted[0], fine[1], *fine)  # type: ignore[arg-type]
