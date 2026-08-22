from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from piginot.contracts import GeometryFamilySpec, ParameterSpec, Quantity, Unit
from piginot.geometry import canonicalize_mesh
from piginot.morphing import (
    FamilyMembershipPredicate,
    FFDControlPoint,
    FFDMap,
    MembershipDecision,
    MorphAttempt,
    MorphingError,
    MorphLedger,
    MorphReasonCode,
    MorphRecord,
    MorphStatus,
    MorphValidityCriteria,
    MorphValidityReport,
    RBFControlPoint,
    RBFMap,
    apply_morph,
)
from piginot.morphing.models import Matrix2, Point
from piginot.morphing.validation import validate_deformation
from tests.morphing.helpers import (
    gate_criteria,
    length,
    rectangle_payload,
    stretch_family,
)


def _valid_rbf_controls() -> tuple[RBFControlPoint, ...]:
    return (
        RBFControlPoint((0.0, 0.0), "shift", None),
        RBFControlPoint((1.0, 0.0), None, None),
        RBFControlPoint((0.0, 1.0), None, None),
    )


def _ffd(parameter: str | None = "stretch") -> FFDMap:
    return FFDMap(
        bounds=((0.0, 0.0), (4.0, 2.0)),
        control_shape=(2, 2),
        control_points=(
            FFDControlPoint(None, None),
            FFDControlPoint(None, None),
            FFDControlPoint(parameter, None),
            FFDControlPoint(parameter, None),
        ),
    )


def _setup() -> tuple[object, FamilyMembershipPredicate, FFDMap]:
    reference = canonicalize_mesh(rectangle_payload())
    predicate = FamilyMembershipPredicate.compile(
        stretch_family(reference), reference, gate_criteria()
    )
    return reference, predicate, _ffd()


def _report(reference: object) -> MorphValidityReport:
    return MorphValidityReport(
        reference_mesh_id=reference.mesh_id,
        validity_policy_hash="policy-hash",
        quadrature_rule="triangle-degree-2-v1",
        quadrature_points_checked=0,
        minimum_jacobian_determinant=None,
        topology_signature=None,
        semantic_tag_counts={},
        tags_preserved=False,
        minimum_feature_size_mm=None,
        minimum_angle_degrees=None,
        maximum_aspect_ratio=None,
    )


class _IdentityDeformation:
    kind = "test-identity"
    parameter_hash = "identity-parameters"

    @staticmethod
    def map_point(point: Point) -> Point:
        return point

    @staticmethod
    def jacobian(point: Point) -> Matrix2:
        del point
        return ((1.0, 0.0), (0.0, 1.0))


class _NonfiniteJacobianDeformation(_IdentityDeformation):
    @staticmethod
    def jacobian(point: Point) -> Matrix2:
        del point
        return ((float("nan"), 0.0), (0.0, 1.0))


class _CollapsedNodesDeformation(_IdentityDeformation):
    @staticmethod
    def map_point(point: Point) -> Point:
        del point
        return (0.0, 0.0)


class _MissingMetadataDeformation(_IdentityDeformation):
    kind = ""


def _raise_singular_system(*_args: object, **_kwargs: object) -> None:
    raise np.linalg.LinAlgError("synthetic singular interpolation system")


@pytest.mark.parametrize(
    ("controls", "radius"),
    (
        (_valid_rbf_controls()[:2], 1.0),
        (
            (
                RBFControlPoint((0.0, 0.0), "shift", None),
                RBFControlPoint((0.0, 0.0), None, None),
                RBFControlPoint((0.0, 1.0), None, None),
            ),
            1.0,
        ),
        (
            (
                RBFControlPoint((0.0, 0.0), "shift", None),
                RBFControlPoint((1.0, 0.0), None, None),
                RBFControlPoint((2.0, 0.0), None, None),
            ),
            1.0,
        ),
        (_valid_rbf_controls(), 0.0),
    ),
    ids=("too-few", "duplicate", "collinear", "non-positive-radius"),
)
def test_rbf_rejects_invalid_interpolation_configurations(
    controls: tuple[RBFControlPoint, ...], radius: float
) -> None:
    with pytest.raises(MorphingError) as captured:
        RBFMap(controls, radius)

    assert captured.value.reason_code is MorphReasonCode.INVALID_MAP_CONFIGURATION


def test_rbf_reports_a_numerically_singular_interpolation_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(np.linalg, "solve", _raise_singular_system)
    morph = RBFMap(_valid_rbf_controls(), radius_mm=1.0)

    with pytest.raises(MorphingError) as captured:
        morph.bind({"shift": length(0.1)})

    assert captured.value.reason_code is MorphReasonCode.SINGULAR_RBF_SYSTEM


@pytest.mark.parametrize(
    ("bounds", "shape", "controls", "unit"),
    (
        (
            ((1.0, 0.0), (0.0, 2.0)),
            (2, 2),
            tuple(FFDControlPoint(None, None) for _ in range(4)),
            Unit.MM,
        ),
        (
            ((0.0, 0.0), (4.0, 2.0)),
            (1, 2),
            tuple(FFDControlPoint(None, None) for _ in range(2)),
            Unit.MM,
        ),
        (
            ((0.0, 0.0), (4.0, 2.0)),
            (2, 2),
            tuple(FFDControlPoint(None, None) for _ in range(3)),
            Unit.MM,
        ),
        (
            ((0.0, 0.0), (4.0, 2.0)),
            (2, 2),
            tuple(FFDControlPoint(None, None) for _ in range(4)),
            Unit.CM,
        ),
    ),
    ids=("reversed-bounds", "small-shape", "wrong-count", "noncanonical-unit"),
)
def test_ffd_rejects_invalid_lattice_configurations(
    bounds: tuple[Point, Point],
    shape: tuple[int, int],
    controls: tuple[FFDControlPoint, ...],
    unit: Unit,
) -> None:
    with pytest.raises(MorphingError) as captured:
        FFDMap(bounds, shape, controls, unit)

    expected = (
        MorphReasonCode.PARAMETER_UNIT_MISMATCH
        if unit is Unit.CM
        else MorphReasonCode.INVALID_MAP_CONFIGURATION
    )
    assert captured.value.reason_code is expected


@pytest.mark.parametrize(
    ("theta", "reason"),
    (
        ([], MorphReasonCode.INVALID_PARAMETER_VALUE),
        ({}, MorphReasonCode.MISSING_PARAMETER),
        (
            {"stretch": length(0.0), "extra": length(0.0)},
            MorphReasonCode.UNKNOWN_PARAMETER,
        ),
        ({"stretch": 0.0}, MorphReasonCode.INVALID_PARAMETER_VALUE),
        (
            {"stretch": Quantity(1.0, Unit.N).to_canonical()},
            MorphReasonCode.PARAMETER_UNIT_MISMATCH,
        ),
    ),
    ids=("not-mapping", "missing", "unknown", "bare", "wrong-unit"),
)
def test_map_binding_rejects_noncanonical_parameter_sets(
    theta: object, reason: MorphReasonCode
) -> None:
    with pytest.raises(MorphingError) as captured:
        _ffd().bind(theta)

    assert captured.value.reason_code is reason


@pytest.mark.parametrize("operation", ("map_point", "jacobian"))
def test_bound_ffd_rejects_points_outside_its_declared_box(operation: str) -> None:
    bound = _ffd().bind({"stretch": length(0.0)})

    with pytest.raises(MorphingError) as captured:
        getattr(bound, operation)((4.0 + 1.0e-12, 1.0))

    assert captured.value.reason_code is MorphReasonCode.POINT_OUTSIDE_FFD_BOX


@pytest.mark.parametrize(
    "factory",
    (
        lambda: RBFControlPoint((0.0,), None, None),
        lambda: RBFControlPoint((float("inf"), 0.0), None, None),
        lambda: FFDControlPoint("not-a-python-identifier", None),
    ),
    ids=("wrong-point-shape", "nonfinite-point", "invalid-parameter-name"),
)
def test_control_point_contracts_reject_invalid_values(factory: object) -> None:
    with pytest.raises(MorphingError) as captured:
        factory()

    assert captured.value.reason_code is MorphReasonCode.INVALID_MAP_CONFIGURATION


@pytest.mark.parametrize(
    "overrides",
    (
        {"jacobian_margin": float("nan")},
        {"minimum_feature_size_mm": -1.0},
        {"minimum_angle_degrees": 60.0 + 1.0e-12},
        {"maximum_aspect_ratio": 1.0 - 1.0e-12},
    ),
    ids=("nonfinite", "negative-feature", "angle", "aspect"),
)
def test_validity_criteria_reject_invalid_numerical_policies(
    overrides: dict[str, float],
) -> None:
    with pytest.raises(MorphingError) as captured:
        MorphValidityCriteria(**overrides)

    assert captured.value.reason_code is MorphReasonCode.INVALID_MAP_CONFIGURATION


@pytest.mark.parametrize(
    ("accepted", "reasons"),
    (
        (True, (MorphReasonCode.INVALID_PARAMETER_VALUE,)),
        (False, ()),
    ),
)
def test_membership_decision_requires_consistent_status_and_reasons(
    accepted: bool, reasons: tuple[MorphReasonCode, ...]
) -> None:
    with pytest.raises(ValueError, match="accepted decisions"):
        MembershipDecision(accepted, reasons)


def test_attempt_freezes_theta_and_requires_an_identity() -> None:
    with pytest.raises(ValueError, match="attempt_id"):
        MorphAttempt("", {})

    source = {"stretch": length(0.0)}
    attempt = MorphAttempt("attempt-1", source)
    source["stretch"] = length(0.5)

    assert attempt.theta["stretch"] == length(0.0)
    with pytest.raises(TypeError):
        attempt.theta["stretch"] = length(1.0)


def test_records_and_ledgers_enforce_status_and_unique_identity() -> None:
    reference = canonicalize_mesh(rectangle_payload())
    report = _report(reference)

    with pytest.raises(ValueError, match="accepted records"):
        MorphRecord("bad-accepted", MorphStatus.ACCEPTED, (), None, report)
    with pytest.raises(ValueError, match="rejected records"):
        MorphRecord("bad-rejected", MorphStatus.REJECTED, (), None, report)

    accepted = MorphRecord(
        "accepted", MorphStatus.ACCEPTED, (), reference, report
    )
    assert accepted.mesh_id == reference.mesh_id
    with pytest.raises(ValueError, match="unique"):
        MorphLedger((accepted, accepted))


def test_apply_morph_rejects_a_reference_from_another_validation_context() -> None:
    _, predicate, morph = _setup()
    other_reference = canonicalize_mesh(rectangle_payload(width=5.0))

    result = apply_morph(
        other_reference, morph, predicate, {"stretch": length(0.0)}
    )

    assert result.reason_codes == (MorphReasonCode.VALIDATION_CONTEXT_MISMATCH,)
    assert result.report.quadrature_points_checked == 0


def test_apply_morph_rejects_map_and_predicate_parameter_mismatch() -> None:
    reference, predicate, _ = _setup()

    result = apply_morph(
        reference, _ffd("offset"), predicate, {"stretch": length(0.0)}
    )

    assert result.reason_codes == (MorphReasonCode.INVALID_MAP_CONFIGURATION,)
    assert result.report.quadrature_points_checked == 0


def test_apply_morph_records_a_typed_map_binding_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = canonicalize_mesh(rectangle_payload())
    family = GeometryFamilySpec(
        name="rbf_family",
        parameters=(ParameterSpec("shift", length(-1.0), length(1.0)),),
        topology_signature=reference.topology_signature,
        semantic_tags=tuple(reference.semantic_tags),
    )
    predicate = FamilyMembershipPredicate.compile(family, reference, gate_criteria())
    monkeypatch.setattr(np.linalg, "solve", _raise_singular_system)

    result = apply_morph(
        reference,
        RBFMap(_valid_rbf_controls(), radius_mm=1.0),
        predicate,
        {"shift": length(0.1)},
    )

    assert result.membership.accepted
    assert result.reason_codes == (MorphReasonCode.SINGULAR_RBF_SYSTEM,)
    assert result.report.quadrature_points_checked == 0


def test_validation_propagates_an_ffd_outside_box_reason() -> None:
    reference = canonicalize_mesh(rectangle_payload())
    morph = FFDMap(
        bounds=((0.0, 0.0), (3.0, 2.0)),
        control_shape=(2, 2),
        control_points=tuple(FFDControlPoint(None, None) for _ in range(4)),
    ).bind({})

    mesh, report = validate_deformation(reference, morph, gate_criteria())

    assert mesh is None
    assert report.reason_codes == (MorphReasonCode.POINT_OUTSIDE_FFD_BOX,)
    assert 0 < report.quadrature_points_checked < 3 * len(reference.cells)


def test_validation_fails_closed_on_nonfinite_jacobians() -> None:
    reference = canonicalize_mesh(rectangle_payload())

    mesh, report = validate_deformation(
        reference, _NonfiniteJacobianDeformation(), gate_criteria()
    )

    assert mesh is None
    assert report.reason_codes == (MorphReasonCode.INVALID_MORPHED_MESH,)
    assert report.minimum_jacobian_determinant is None


def test_validation_rejects_collapsed_geometry_even_with_identity_jacobians() -> None:
    reference = canonicalize_mesh(rectangle_payload())

    mesh, report = validate_deformation(
        reference, _CollapsedNodesDeformation(), gate_criteria()
    )

    assert mesh is None
    assert MorphReasonCode.NONINJECTIVE_MORPH in report.reason_codes


def test_validation_rejects_missing_deformation_provenance_metadata() -> None:
    reference = canonicalize_mesh(rectangle_payload())

    mesh, report = validate_deformation(
        reference, _MissingMetadataDeformation(), gate_criteria()
    )

    assert mesh is None
    assert report.reason_codes == (MorphReasonCode.INVALID_MAP_CONFIGURATION,)


def test_validation_detects_a_mismatched_declared_topology_signature() -> None:
    reference = canonicalize_mesh(rectangle_payload())
    inconsistent_reference = replace(
        reference, topology_signature="deliberately-inconsistent-signature"
    )

    mesh, report = validate_deformation(
        inconsistent_reference, _IdentityDeformation(), gate_criteria()
    )

    assert mesh is None
    assert report.tags_preserved
    assert report.reason_codes == (MorphReasonCode.TOPOLOGY_CHANGED,)
