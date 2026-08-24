from __future__ import annotations

import math
from typing import Any

import pytest

from piginot.doe.leakage import LeakageDescriptor, bind_leakage_groups


def _descriptor(
    case_id: str,
    *,
    lineage: str,
    seed: int,
    x: float,
    y: float,
) -> LeakageDescriptor:
    return LeakageDescriptor(
        case_id=case_id,
        geometry_lineage_id=lineage,
        morph_seed=seed,
        normalized_coordinates={"x": x, "governing_ratio": y},
    )


def test_groups_bind_lineage_seed_and_near_duplicate_relations_transitively() -> None:
    descriptors = (
        _descriptor("a", lineage="lineage-a", seed=10, x=0.10, y=0.20),
        _descriptor("b", lineage="lineage-a", seed=11, x=0.80, y=0.80),
        _descriptor("c", lineage="lineage-c", seed=11, x=0.50, y=0.50),
        _descriptor("d", lineage="lineage-d", seed=99, x=0.505, y=0.495),
        _descriptor("isolated", lineage="lineage-z", seed=100, x=0.95, y=0.05),
    )

    groups = bind_leakage_groups(descriptors, near_duplicate_tolerance=0.01)

    assert groups["a"] == groups["b"] == groups["c"] == groups["d"]
    assert groups["isolated"] != groups["a"]
    assert all(value.startswith("leakage-") for value in groups.values())


def test_group_binding_is_order_independent_and_byte_stable() -> None:
    descriptors = (
        _descriptor("a", lineage="same", seed=1, x=0.1, y=0.2),
        _descriptor("b", lineage="same", seed=2, x=0.9, y=0.8),
    )

    forward = bind_leakage_groups(descriptors, near_duplicate_tolerance=0.0)
    reverse = bind_leakage_groups(
        tuple(reversed(descriptors)), near_duplicate_tolerance=0.0
    )

    assert dict(forward) == dict(reverse)


@pytest.mark.parametrize(
    ("descriptors", "tolerance", "message"),
    (
        ((), 0.0, "at least one"),
        (
            (
                _descriptor("duplicate", lineage="a", seed=1, x=0.1, y=0.2),
                _descriptor("duplicate", lineage="b", seed=2, x=0.2, y=0.3),
            ),
            0.0,
            "case_id",
        ),
        (
            (
                _descriptor("a", lineage="a", seed=1, x=0.1, y=0.2),
                LeakageDescriptor(
                    case_id="b",
                    geometry_lineage_id="b",
                    morph_seed=2,
                    normalized_coordinates={"different": 0.2},
                ),
            ),
            0.0,
            "schema",
        ),
        (
            (_descriptor("a", lineage="a", seed=1, x=0.1, y=0.2),),
            -0.1,
            "tolerance",
        ),
    ),
)
def test_group_binding_rejects_ambiguous_or_invalid_populations(
    descriptors: tuple[LeakageDescriptor, ...],
    tolerance: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        bind_leakage_groups(
            descriptors,
            near_duplicate_tolerance=tolerance,
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (
            {
                "case_id": "",
                "geometry_lineage_id": "lineage",
                "morph_seed": 1,
                "normalized_coordinates": {"x": 0.5},
            },
            "case_id",
        ),
        (
            {
                "case_id": "case",
                "geometry_lineage_id": "",
                "morph_seed": 1,
                "normalized_coordinates": {"x": 0.5},
            },
            "geometry_lineage_id",
        ),
        (
            {
                "case_id": "case",
                "geometry_lineage_id": "lineage",
                "morph_seed": True,
                "normalized_coordinates": {"x": 0.5},
            },
            "morph_seed",
        ),
        (
            {
                "case_id": "case",
                "geometry_lineage_id": "lineage",
                "morph_seed": -1,
                "normalized_coordinates": {"x": 0.5},
            },
            "morph_seed",
        ),
        (
            {
                "case_id": "case",
                "geometry_lineage_id": "lineage",
                "morph_seed": 1,
                "normalized_coordinates": [],
            },
            "must be a mapping",
        ),
        (
            {
                "case_id": "case",
                "geometry_lineage_id": "lineage",
                "morph_seed": 1,
                "normalized_coordinates": {},
            },
            "must not be empty",
        ),
        (
            {
                "case_id": "case",
                "geometry_lineage_id": "lineage",
                "morph_seed": 1,
                "normalized_coordinates": {"": 0.5},
            },
            "coordinate name",
        ),
        (
            {
                "case_id": "case",
                "geometry_lineage_id": "lineage",
                "morph_seed": 1,
                "normalized_coordinates": {"x": True},
            },
            "finite normalized number",
        ),
        (
            {
                "case_id": "case",
                "geometry_lineage_id": "lineage",
                "morph_seed": 1,
                "normalized_coordinates": {"x": math.nan},
            },
            "must lie in",
        ),
        (
            {
                "case_id": "case",
                "geometry_lineage_id": "lineage",
                "morph_seed": 1,
                "normalized_coordinates": {"x": 1.01},
            },
            "must lie in",
        ),
    ),
)
def test_leakage_descriptor_rejects_invalid_correlation_evidence(
    arguments: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LeakageDescriptor(**arguments)


def test_group_binding_rejects_non_descriptor_and_non_numeric_tolerance() -> None:
    descriptor = _descriptor("case", lineage="lineage", seed=1, x=0.5, y=0.5)

    with pytest.raises(ValueError, match="LeakageDescriptor"):
        bind_leakage_groups(
            (descriptor, object()),  # type: ignore[arg-type]
            near_duplicate_tolerance=0.0,
        )
    with pytest.raises(ValueError, match="must be numeric"):
        bind_leakage_groups(
            (descriptor,),
            near_duplicate_tolerance=True,
        )


def test_redundant_correlation_edges_preserve_one_component() -> None:
    descriptors = tuple(
        _descriptor(
            f"case-{index}",
            lineage="same-lineage",
            seed=index,
            x=0.1 * index,
            y=0.1 * index,
        )
        for index in range(3)
    )

    groups = bind_leakage_groups(descriptors, near_duplicate_tolerance=0.0)

    assert len(set(groups.values())) == 1


def test_group_mapping_and_descriptor_coordinates_are_immutable() -> None:
    descriptor = _descriptor("case", lineage="lineage", seed=1, x=0.5, y=0.5)
    groups = bind_leakage_groups((descriptor,), near_duplicate_tolerance=0.0)

    with pytest.raises(TypeError):
        descriptor.normalized_coordinates["x"] = 0.8  # type: ignore[index]
    with pytest.raises(TypeError):
        groups["case"] = "rewritten"  # type: ignore[index]
