"""RED tests for the S1 closeout Phase A ground-truth repair."""

from __future__ import annotations

import numpy as np

from experiments.spike.fem import (
    DEFAULT_ELEMENT_ORDER,
    area_weighted_nodal_average,
    solve_case,
    solve_uniaxial_square,
)
from experiments.spike.geometry import generate_geometry

# two triangles sharing node 1, with areas 0.5 and 3.5
_PATCH_POINTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [4.0, 4.0]])
_PATCH_CELLS = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)


def _patch_values(first: float, second: float) -> np.ndarray:
    values = np.zeros((2, 2, 2))
    values[0] = first
    values[1] = second
    return values


def test_area_weighted_average_reproduces_a_constant_field_exactly() -> None:
    recovered = area_weighted_nodal_average(
        _patch_values(3.0, 3.0), _PATCH_POINTS, _PATCH_CELLS
    )
    np.testing.assert_allclose(recovered, 3.0, atol=1e-14)


def test_area_weighted_average_differs_from_the_count_average() -> None:
    recovered = area_weighted_nodal_average(
        _patch_values(0.0, 8.0), _PATCH_POINTS, _PATCH_CELLS
    )
    shared = float(recovered[1, 0, 0])
    assert shared == 8.0 * 3.5 / 4.0  # area weighted
    assert shared != 4.0  # the superseded count average


def test_area_weighted_average_leaves_single_element_nodes_untouched() -> None:
    recovered = area_weighted_nodal_average(
        _patch_values(0.0, 8.0), _PATCH_POINTS, _PATCH_CELLS
    )
    assert float(recovered[0, 0, 0]) == 0.0
    assert float(recovered[3, 0, 0]) == 8.0


def test_default_element_order_is_quadratic() -> None:
    assert DEFAULT_ELEMENT_ORDER == 2


def test_uniaxial_square_is_exact_for_both_element_orders() -> None:
    for order in (1, 2):
        result = solve_uniaxial_square(
            axial_stretch=1.10, nx=6, ny=6, element_order=order
        )
        assert result.converged, (order, result.failure_reason)
        assert result.max_error <= 1e-6, (order, result.max_error)


def test_solve_case_records_the_element_order_it_used() -> None:
    mesh = generate_geometry(
        "centre_notched",
        {"width": 20.6, "notch_radius": 4.67, "notch_depth": 2.57},
        1,
        nx=16,
        ny=5,
    )
    for order in (1, 2):
        result = solve_case(mesh, element_order=order)
        assert result.converged, result.failure_reason
        assert result.element_order == order
        assert result.first_piola.shape == (len(mesh.points), 2, 2)


def test_vertex_average_reproduces_a_constant_field_exactly() -> None:
    from experiments.spike.fem import area_weighted_vertex_average

    values = np.full((2, 3, 2, 2), 5.0)
    recovered = area_weighted_vertex_average(values, _PATCH_POINTS, _PATCH_CELLS)
    np.testing.assert_allclose(recovered, 5.0, atol=1e-14)


def test_vertex_average_weights_by_area_at_a_shared_node() -> None:
    from experiments.spike.fem import area_weighted_vertex_average

    values = np.zeros((2, 3, 2, 2))
    values[0, 1] = 0.0  # element 0, local vertex 1 -> global node 1
    values[1, 0] = 8.0  # element 1, local vertex 0 -> global node 1
    recovered = area_weighted_vertex_average(values, _PATCH_POINTS, _PATCH_CELLS)
    assert float(recovered[1, 0, 0]) == 8.0 * 3.5 / 4.0


def test_vertex_recovery_makes_quadratic_elements_beat_linear_at_a_fillet() -> None:
    """The dominant ground-truth peak error was the recovery, not the element order.

    With element-averaged stress, P2 looked worse than P1. With stress evaluated at
    the vertices, P2 on the production mesh drops from ~11% remaining error to ~1%.
    """
    from experiments.spike.refinement import richardson

    parameters = {
        "L_total": 40.96,
        "R_fillet": 8.17,
        "W_gauge": 11.30,
        "W_grip": 25.91,
    }

    def peak(nx: int, ny: int, order: int) -> float:
        mesh = generate_geometry("dogbone", parameters, 1, nx=nx, ny=ny)
        result = solve_case(mesh, element_order=order)
        assert result.converged, result.failure_reason
        magnitude = np.linalg.norm(result.first_piola[mesh.qoi_node_ids], axis=(-2, -1))
        return float(magnitude.max())

    levels = [(16, 5), (32, 10), (64, 20)]
    remaining = {}
    for order in (1, 2):
        estimate = richardson([peak(nx, ny, order) for nx, ny in levels])
        assert estimate.status == "OK", (order, estimate.reason)
        remaining[order] = estimate.production_remaining_error()
    assert remaining[2] < 0.02, remaining
    assert remaining[2] < remaining[1] / 5.0, remaining
