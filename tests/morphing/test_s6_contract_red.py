from __future__ import annotations

import json

import pytest

from piginot.contracts import (
    GeometryFamilySpec,
    ParameterSpec,
    Quantity,
    Unit,
)
from piginot.geometry import canonicalize_mesh
from piginot.morphing import (
    FamilyMembershipPredicate,
    FFDControlPoint,
    FFDMap,
    MorphingError,
    MorphReasonCode,
    MorphValidityCriteria,
    RBFControlPoint,
    RBFMap,
    apply_morph,
)
from tests.morphing.helpers import rectangle_payload


def _length(value: float):
    return Quantity(value, Unit.MM).to_canonical()


def _finite_difference_jacobian(deformation, point, step=1.0e-7):
    x_value, y_value = point
    plus_x = deformation.map_point((x_value + step, y_value))
    minus_x = deformation.map_point((x_value - step, y_value))
    plus_y = deformation.map_point((x_value, y_value + step))
    minus_y = deformation.map_point((x_value, y_value - step))
    return (
        (
            (plus_x[0] - minus_x[0]) / (2.0 * step),
            (plus_y[0] - minus_y[0]) / (2.0 * step),
        ),
        (
            (plus_x[1] - minus_x[1]) / (2.0 * step),
            (plus_y[1] - minus_y[1]) / (2.0 * step),
        ),
    )


def test_rbf_is_interpolatory_and_exposes_analytic_jacobian() -> None:
    morph = RBFMap(
        control_points=(
            RBFControlPoint((0.0, 0.0), None, None),
            RBFControlPoint((2.0, 0.0), "right_dx", None),
            RBFControlPoint((2.0, 1.0), "right_dx", "top_dy"),
            RBFControlPoint((0.0, 1.0), None, "top_dy"),
        ),
        radius_mm=2.0,
    )
    deformation = morph.bind({"right_dx": _length(0.2), "top_dy": _length(0.1)})

    assert deformation.map_point((2.0, 1.0)) == pytest.approx((2.2, 1.1))
    analytic = deformation.jacobian((0.9, 0.4))
    finite_difference = _finite_difference_jacobian(deformation, (0.9, 0.4))
    assert analytic[0] == pytest.approx(finite_difference[0], rel=2.0e-8, abs=2.0e-8)
    assert analytic[1] == pytest.approx(finite_difference[1], rel=2.0e-8, abs=2.0e-8)


def test_ffd_supports_a_tensor_product_bernstein_lattice() -> None:
    controls = tuple(
        FFDControlPoint("bulge" if (i, j) == (1, 1) else None, None)
        for i in range(3)
        for j in range(3)
    )
    morph = FFDMap(
        bounds=((0.0, 0.0), (2.0, 1.0)),
        control_shape=(3, 3),
        control_points=controls,
    )
    deformation = morph.bind({"bulge": _length(0.4)})

    assert deformation.map_point((1.0, 0.5))[0] == pytest.approx(1.1)
    analytic = deformation.jacobian((0.7, 0.4))
    finite_difference = _finite_difference_jacobian(deformation, (0.7, 0.4))
    assert analytic[0] == pytest.approx(finite_difference[0], rel=2.0e-8, abs=2.0e-8)
    assert analytic[1] == pytest.approx(finite_difference[1], rel=2.0e-8, abs=2.0e-8)


@pytest.mark.parametrize(
    "builder",
    [
        lambda: RBFMap(
            control_points=(
                RBFControlPoint((0.0, 0.0), None, None),
                RBFControlPoint((0.0, 0.0), "dx", None),
                RBFControlPoint((1.0, 1.0), None, "dy"),
            ),
            radius_mm=1.0,
        ),
        lambda: RBFMap(
            control_points=(
                RBFControlPoint((0.0, 0.0), None, None),
                RBFControlPoint((1.0, 1.0), "dx", None),
                RBFControlPoint((2.0, 2.0), None, "dy"),
            ),
            radius_mm=1.0,
        ),
        lambda: RBFMap(
            control_points=(
                RBFControlPoint((0.0, 0.0), None, None),
                RBFControlPoint((1.0, 0.0), "dx", None),
                RBFControlPoint((1.0, 1.0), None, "dy"),
            ),
            radius_mm=0.0,
        ),
        lambda: FFDMap(
            bounds=((0.0, 0.0), (0.0, 1.0)),
            control_shape=(2, 2),
            control_points=(
                FFDControlPoint(None, None),
                FFDControlPoint(None, None),
                FFDControlPoint(None, None),
                FFDControlPoint(None, None),
            ),
        ),
        lambda: FFDMap(
            bounds=((0.0, 0.0), (1.0, 1.0)),
            control_shape=(1, 2),
            control_points=(FFDControlPoint(None, None), FFDControlPoint(None, None)),
        ),
        lambda: FFDMap(
            bounds=((0.0, 0.0), (1.0, 1.0)),
            control_shape=(2, 2),
            control_points=(FFDControlPoint(None, None),),
        ),
    ],
)
def test_morph_map_configuration_fails_closed(builder) -> None:
    with pytest.raises(MorphingError) as error:
        builder()

    assert error.value.reason_code is MorphReasonCode.INVALID_MAP_CONFIGURATION


@pytest.mark.parametrize(
    "builder",
    [
        lambda: RBFMap(control_points=(), radius_mm=1.0),
        lambda: RBFMap(
            control_points=(
                RBFControlPoint((0.0, 0.0), None, None),
                RBFControlPoint((1.0, 0.0), "dx", None),
                RBFControlPoint((1.0, 1.0), None, "dy"),
            ),
            radius_mm=float("inf"),
        ),
        lambda: FFDMap(
            bounds=((0.0, 0.0), (1.0, 1.0)),
            control_shape=(2, True),
            control_points=(
                FFDControlPoint(None, None),
                FFDControlPoint(None, None),
            ),
        ),
        lambda: FFDMap(
            bounds=((0.0, 0.0), (1.0, 1.0)),
            control_shape=(2, 2),
            control_points=(
                FFDControlPoint(None, None),
                FFDControlPoint(None, None),
                FFDControlPoint(None, None),
                FFDControlPoint(None, None),
            ),
            unit=Unit.N,
        ),
    ],
)
def test_more_invalid_map_configurations_keep_typed_reasons(builder) -> None:
    with pytest.raises(MorphingError) as error:
        builder()

    assert error.value.reason_code in {
        MorphReasonCode.INVALID_MAP_CONFIGURATION,
        MorphReasonCode.PARAMETER_UNIT_MISMATCH,
    }


@pytest.mark.parametrize(
    ("theta", "reason"),
    [
        ((), MorphReasonCode.INVALID_PARAMETER_VALUE),
        ({}, MorphReasonCode.MISSING_PARAMETER),
        (
            {"shift": _length(0.1), "extra": _length(0.2)},
            MorphReasonCode.UNKNOWN_PARAMETER,
        ),
        ({"shift": 0.1}, MorphReasonCode.INVALID_PARAMETER_VALUE),
        (
            {"shift": Quantity(0.1, Unit.N).to_canonical()},
            MorphReasonCode.PARAMETER_UNIT_MISMATCH,
        ),
    ],
)
def test_map_binding_rejects_invalid_parameter_payloads(theta, reason) -> None:
    morph = FFDMap(
        bounds=((0.0, 0.0), (1.0, 1.0)),
        control_shape=(2, 2),
        control_points=(
            FFDControlPoint(None, None),
            FFDControlPoint(None, None),
            FFDControlPoint("shift", None),
            FFDControlPoint("shift", None),
        ),
    )

    with pytest.raises(MorphingError) as error:
        morph.bind(theta)

    assert error.value.reason_code is reason


def test_bound_ffd_rejects_points_outside_its_declared_box() -> None:
    morph = FFDMap(
        bounds=((0.0, 0.0), (1.0, 1.0)),
        control_shape=(2, 2),
        control_points=(
            FFDControlPoint(None, None),
            FFDControlPoint(None, None),
            FFDControlPoint("shift", None),
            FFDControlPoint("shift", None),
        ),
    )
    deformation = morph.bind({"shift": _length(0.1)})

    with pytest.raises(MorphingError) as error:
        deformation.map_point((1.0 + 1.0e-12, 0.5))

    assert error.value.reason_code is MorphReasonCode.POINT_OUTSIDE_FFD_BOX


def test_predicate_roundtrip_is_strict_and_reference_bound() -> None:
    reference = canonicalize_mesh(rectangle_payload())
    family = GeometryFamilySpec(
        name="translated_rectangle",
        parameters=(ParameterSpec("shift", _length(-0.25), _length(0.25)),),
        topology_signature=reference.topology_signature,
        semantic_tags=tuple(reference.semantic_tags),
    )
    criteria = MorphValidityCriteria(
        jacobian_margin=0.25,
        minimum_feature_size_mm=0.2,
        minimum_angle_degrees=10.0,
        maximum_aspect_ratio=12.0,
    )
    predicate = FamilyMembershipPredicate.compile(family, reference, criteria)

    encoded = predicate.to_bytes()
    restored = FamilyMembershipPredicate.from_bytes(encoded)

    assert restored == predicate
    assert restored.to_bytes() == encoded

    payload = json.loads(encoded)
    payload["undeclared"] = True
    with pytest.raises(ValueError):
        FamilyMembershipPredicate.from_bytes(
            json.dumps(payload, separators=(",", ":")).encode()
        )


def test_validity_report_records_quadrature_and_policy_identity() -> None:
    reference = canonicalize_mesh(rectangle_payload())
    family = GeometryFamilySpec(
        name="translated_rectangle",
        parameters=(ParameterSpec("shift", _length(-0.25), _length(0.25)),),
        topology_signature=reference.topology_signature,
        semantic_tags=tuple(reference.semantic_tags),
    )
    criteria = MorphValidityCriteria(
        jacobian_margin=0.25,
        minimum_feature_size_mm=0.2,
        minimum_angle_degrees=10.0,
        maximum_aspect_ratio=12.0,
    )
    predicate = FamilyMembershipPredicate.compile(family, reference, criteria)
    morph = FFDMap(
        bounds=((0.0, 0.0), (4.0, 2.0)),
        control_shape=(2, 2),
        control_points=(
            FFDControlPoint("shift", None),
            FFDControlPoint("shift", None),
            FFDControlPoint("shift", None),
            FFDControlPoint("shift", None),
        ),
    )

    result = apply_morph(reference, morph, predicate, {"shift": _length(0.1)})

    assert result.accepted
    assert result.report.quadrature_rule == "triangle-degree-5-dunavant-v1"
    assert result.report.quadrature_points_checked == 7 * len(reference.cells)
    assert result.report.reference_mesh_id == reference.mesh_id
    assert result.report.validity_policy_hash == predicate.validity_policy_hash
    assert result.report.tags_preserved
    assert MorphReasonCode.NONINJECTIVE_MORPH not in result.reason_codes
