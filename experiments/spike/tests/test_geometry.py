from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.spike.geometry import (
    CENTRE_NOTCHED_RANGES,
    DOGBONE_RANGES,
    generate_geometry,
    sample_geometry_cases,
)


@pytest.mark.parametrize(
    ("family", "parameters"),
    [
        (
            "dogbone",
            {"L_total": 55.0, "W_grip": 21.0, "W_gauge": 10.0, "R_fillet": 14.0},
        ),
        (
            "centre_notched",
            {"width": 24.0, "notch_radius": 4.0, "notch_depth": 2.0},
        ),
    ],
)
def test_geometry_has_valid_fixed_topology(
    family: str, parameters: dict[str, float]
) -> None:
    mesh = generate_geometry(family, parameters, seed=101, nx=12, ny=5)

    assert mesh.family == family
    assert mesh.points.shape == (13 * 6, 2)
    assert mesh.cells.shape == (12 * 5 * 2, 3)
    assert mesh.boundary_points.shape == (320, 2)
    assert mesh.topology_signature == "quarter-single-component-no-holes"
    assert len(mesh.node_ids) == len(set(mesh.node_ids)) == mesh.points.shape[0]
    assert mesh.qoi_node_ids.size > 1
    assert np.isfinite(mesh.points).all()

    triangles = mesh.points[mesh.cells]
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    signed_twice_area = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
    assert np.min(signed_twice_area) > 0.0


def test_declared_parameter_bounds_are_enforced() -> None:
    dogbone_mid = {name: sum(bounds) / 2.0 for name, bounds in DOGBONE_RANGES.items()}
    notch_mid = {
        name: sum(bounds) / 2.0 for name, bounds in CENTRE_NOTCHED_RANGES.items()
    }

    generate_geometry("dogbone", dogbone_mid, seed=1, nx=6, ny=3)
    generate_geometry("centre_notched", notch_mid, seed=2, nx=6, ny=3)

    for family, ranges, valid in (
        ("dogbone", DOGBONE_RANGES, dogbone_mid),
        ("centre_notched", CENTRE_NOTCHED_RANGES, notch_mid),
    ):
        for name, (lower, upper) in ranges.items():
            for invalid in (
                math.nextafter(lower, -math.inf),
                math.nextafter(upper, math.inf),
            ):
                bad = {**valid, name: invalid}
                with pytest.raises(ValueError, match=name):
                    generate_geometry(family, bad, seed=3, nx=6, ny=3)

            with pytest.raises(ValueError, match=name):
                generate_geometry(family, {**valid, name: math.nan}, seed=4, nx=6, ny=3)


def test_sixty_sampled_cases_are_reproducible_and_unique() -> None:
    first = sample_geometry_cases(master_seed=20260809, count_per_family=30, nx=6, ny=3)
    second = sample_geometry_cases(
        master_seed=20260809, count_per_family=30, nx=6, ny=3
    )

    assert len(first) == 60
    assert [case.family for case in first].count("dogbone") == 30
    assert [case.family for case in first].count("centre_notched") == 30
    assert [case.geometry_id for case in first] == [case.geometry_id for case in second]
    assert [case.mesh_id for case in first] == [case.mesh_id for case in second]
    assert len({case.seed for case in first}) == 60
    assert len({case.geometry_id for case in first}) == 60
    assert len({case.mesh_id for case in first}) == 60

    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left.points, right.points)
        np.testing.assert_array_equal(left.cells, right.cells)
