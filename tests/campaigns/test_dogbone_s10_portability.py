from __future__ import annotations

import math
from dataclasses import replace

import pytest

from campaigns.dogbone_v1.family import compiled_family
from campaigns.dogbone_v1.s11_contingency_source import (
    _repair_quality_angle_ulps,
    _ulp_neighbourhood,
)
from piginot.contracts import CanonicalMesh, hash_spec


def _shifted(mesh: CanonicalMesh, name: str, ulps: int) -> CanonicalMesh:
    value = float(mesh.quality_metrics[name])
    direction = math.inf if ulps > 0 else -math.inf
    for _ in range(abs(ulps)):
        value = math.nextafter(value, direction)
    return replace(mesh, quality_metrics={**mesh.quality_metrics, name: value})


def test_ulp_neighbourhood_is_symmetric_and_exact() -> None:
    values = _ulp_neighbourhood(1.0, 2)

    assert len(values) == 5
    assert values[0] == 1.0
    assert sorted(values) == [
        math.nextafter(math.nextafter(1.0, 0.0), 0.0),
        math.nextafter(1.0, 0.0),
        1.0,
        math.nextafter(1.0, 2.0),
        math.nextafter(math.nextafter(1.0, 2.0), 2.0),
    ]


@pytest.mark.parametrize(
    ("minimum_ulps", "maximum_ulps"),
    [(0, 0), (1, 0), (0, -3), (-2, 4)],
)
def test_repair_recovers_platform_angle_bits_exactly(
    minimum_ulps: int, maximum_ulps: int
) -> None:
    reference = compiled_family().reference_mesh
    planned = _shifted(
        _shifted(reference, "minimum_angle_degrees", minimum_ulps),
        "maximum_angle_degrees",
        maximum_ulps,
    )

    repaired = _repair_quality_angle_ulps(reference, hash_spec(planned, "content"))

    assert repaired is not None
    assert repaired == planned
    assert hash_spec(repaired, "content") == hash_spec(planned, "content")


def test_repair_refuses_differences_outside_the_angle_diagnostics() -> None:
    reference = compiled_family().reference_mesh
    far = _shifted(reference, "minimum_angle_degrees", 6)
    other_metric = _shifted(reference, "maximum_aspect_ratio", 1)
    nodes = list(reference.nodes)
    nodes[0] = (nodes[0][0], math.nextafter(nodes[0][1], 1.0))
    moved = replace(reference, nodes=tuple(nodes))

    for planned in (far, other_metric, moved):
        assert _repair_quality_angle_ulps(reference, hash_spec(planned, "content")) is None
    missing = replace(reference, quality_metrics={"maximum_aspect_ratio": 1.0})
    assert _repair_quality_angle_ulps(missing, "0" * 64) is None
