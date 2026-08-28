from __future__ import annotations

from pathlib import Path

from campaigns.dogbone_v1.family import (
    REFERENCE_REPOSITORY_COMMIT,
    REFERENCE_REPOSITORY_URL,
    compiled_family,
    parameter_point,
)
from campaigns.dogbone_v1.pilot import run_family_validation_sweep
from piginot.contracts import (
    Dimension,
    Unit,
    canonical_json_bytes,
    hash_spec,
)
from piginot.morphing import apply_morph

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT_ROOT = _REPOSITORY_ROOT / "artifacts" / "campaigns" / "dogbone-v1"


def test_family_is_pinned_to_the_user_approved_reference_implementation() -> None:
    family = compiled_family()

    assert REFERENCE_REPOSITORY_URL == ("https://github.com/BBahtiri/PI-GINOT-DogBone")
    assert REFERENCE_REPOSITORY_COMMIT == ("9f6794141beca9d3e66433465c3b07039a7e41d2")
    assert family.recipe_id == "dogbone-quarter-no-hole-v1"
    assert tuple(bound.name for bound in family.membership.primitive_bounds) == (
        "L_total",
        "W_grip",
        "W_gauge",
        "R_fillet",
    )
    assert {bound.name for bound in family.membership.derived_bounds} == {
        "grip_to_gauge_ratio",
        "fillet_to_gauge_ratio",
        "length_to_gauge_ratio",
        "arc_feasibility_ratio",
    }


def test_compiled_family_binds_reference_map_membership_and_policy() -> None:
    first = compiled_family()
    second = compiled_family()

    assert first == second
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert hash_spec(first, "content") == hash_spec(second, "content")
    assert first.reference_mesh.mesh_id == first.membership.reference_mesh_id
    assert first.membership.validity_policy_hash == hash_spec(
        first.membership.criteria, "content"
    )
    assert set(first.morph_map.parameter_names) == set(first.membership.parameter_names)


def test_committed_compiled_family_is_the_exact_canonical_preimage() -> None:
    assert (
        _ARTIFACT_ROOT / "compiled-family.json"
    ).read_bytes() == canonical_json_bytes(compiled_family())


def test_reference_and_an_interior_family_point_pass_the_full_s6_path() -> None:
    family = compiled_family()
    points = (
        parameter_point(54.0, 20.0, 10.0, 12.0),
        parameter_point(60.0, 22.0, 9.0, 16.0),
    )

    for theta in points:
        result = apply_morph(
            family.reference_mesh,
            family.morph_map,
            family.membership,
            theta,
        )
        assert result.accepted, result.reason_codes
        assert result.mesh is not None
        assert (
            result.mesh.topology_signature == family.reference_mesh.topology_signature
        )
        assert tuple(result.membership.derived_values) == tuple(
            bound.name for bound in family.membership.derived_bounds
        )
        assert all(
            quantity.unit is Unit.ONE and quantity.dimension is Dimension.DIMENSIONLESS
            for quantity in result.membership.derived_values.values()
        )


def test_coupled_invalid_corner_is_rejected_before_morphing() -> None:
    family = compiled_family()
    theta = parameter_point(54.0, 26.0, 6.0, 8.0)

    decision = family.membership.evaluate(theta)

    assert decision.accepted is False
    assert decision.derived_values["arc_feasibility_ratio"].value < 1.35


def test_disjoint_public_sweep_validates_500_membership_accepted_shapes() -> None:
    summary = run_family_validation_sweep()

    assert summary.public_seed == "dogbone-family-validation-20260828-v1"
    assert summary.accepted == 500
    assert summary.morph_rejections == {}
    assert summary.membership_rejections == {
        "DERIVED_QUANTITY_OUT_OF_BOUNDS": summary.attempted - 500
    }
    assert summary.minimum_jacobian_determinant > 0.01
    assert summary.minimum_feature_size_mm >= 0.05
    assert summary.minimum_angle_degrees >= 5.0
    assert summary.maximum_aspect_ratio <= 20.0
    assert (_ARTIFACT_ROOT / "family-validation.json").read_bytes() == (
        canonical_json_bytes(summary) + b"\n"
    )
