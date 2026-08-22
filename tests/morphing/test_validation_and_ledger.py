from __future__ import annotations

from dataclasses import replace

import pytest

from piginot.contracts import CanonicalQuantity, Unit, canonical_json_bytes
from piginot.geometry import CellType, MeshPayload, canonicalize_mesh
from piginot.morphing import (
    FamilyMembershipPredicate,
    FFDControlPoint,
    FFDMap,
    MorphAttempt,
    MorphReasonCode,
    MorphValidityCriteria,
    Point,
    RBFControlPoint,
    RBFMap,
    apply_morph,
    run_morph_sweep,
)
from tests.morphing.helpers import (
    gate_criteria,
    length,
    rectangle_payload,
    stretch_controls,
    stretch_family,
)


class _InvalidJacobianDeformation:
    kind = "bad-jacobian"
    parameter_hash = "sha256:bad"

    def map_point(self, point):
        return point

    def jacobian(self, point):
        return ((float("nan"), 0.0), (0.0, 1.0))


class _InvalidPointDeformation:
    kind = "bad-point"
    parameter_hash = "sha256:bad"

    def map_point(self, point):
        return (point[0], object())

    def jacobian(self, point):
        return ((1.0, 0.0), (0.0, 1.0))


class _InvalidIdentityDeformation:
    kind = ""
    parameter_hash = ""

    def map_point(self, point):
        return point

    def jacobian(self, point):
        return ((1.0, 0.0), (0.0, 1.0))


class _InteriorFoldDeformation:
    """Identity at triangle vertices with a negative Jacobian at its centroid."""

    kind = "interior-fold"
    parameter_hash = "interior-fold-v1"

    @staticmethod
    def map_point(point: Point) -> Point:
        x_value, y_value = point
        displacement = (
            72.0
            * x_value
            * (1.0 - x_value)
            * (y_value - 1.0 / 6.0)
            * (y_value - 2.0 / 3.0)
        )
        return x_value + displacement, y_value

    @staticmethod
    def jacobian(point: Point):
        x_value, y_value = point
        y_polynomial = (y_value - 1.0 / 6.0) * (y_value - 2.0 / 3.0)
        displacement_dx = 72.0 * (1.0 - 2.0 * x_value) * y_polynomial
        displacement_dy = 72.0 * x_value * (1.0 - x_value) * (2.0 * y_value - 5.0 / 6.0)
        return ((1.0 + displacement_dx, displacement_dy), (0.0, 1.0))


class _RejectingMap:
    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("stretch",)

    def bind(self, theta: dict[str, CanonicalQuantity]):
        raise ValueError("not a typed morphing failure")


def _setup(criteria: MorphValidityCriteria | None = None):
    reference = canonicalize_mesh(rectangle_payload())
    selected = criteria or gate_criteria()
    predicate = FamilyMembershipPredicate.compile(
        stretch_family(reference), reference, selected
    )
    morph = FFDMap(
        bounds=((0.0, 0.0), (4.0, 2.0)),
        control_shape=(2, 2),
        control_points=stretch_controls(),
    )
    return reference, predicate, morph


def test_valid_morph_transports_tags_topology_and_is_byte_deterministic() -> None:
    reference, predicate, morph = _setup()
    theta = {"stretch": length(0.2)}

    first = apply_morph(reference, morph, predicate, theta)
    second = apply_morph(reference, morph, predicate, theta)

    assert first.accepted
    assert first.mesh is not None
    assert first.report.topology_signature == reference.topology_signature
    assert first.report.tags_preserved
    assert dict(first.report.semantic_tag_counts) == {
        name: len(facets) for name, facets in reference.semantic_tags.items()
    }
    assert canonical_json_bytes(first.mesh) == canonical_json_bytes(second.mesh)


def test_rbf_morph_uses_the_same_tagged_validation_boundary() -> None:
    reference, predicate, _ = _setup()
    morph = RBFMap(
        control_points=(
            RBFControlPoint((0.0, 0.0), None, None),
            RBFControlPoint((4.0, 0.0), "stretch", None),
            RBFControlPoint((4.0, 2.0), "stretch", None),
            RBFControlPoint((0.0, 2.0), None, None),
        ),
        radius_mm=4.0,
    )

    result = apply_morph(
        reference,
        morph,
        predicate,
        {"stretch": length(0.2)},
    )

    assert result.accepted
    assert result.mesh is not None
    assert result.report.tags_preserved
    assert result.report.topology_signature == reference.topology_signature


def test_deliberately_noninjective_map_fires_the_required_mutant() -> None:
    reference, predicate, morph = _setup()

    result = apply_morph(
        reference,
        morph,
        predicate,
        {"stretch": length(-5.0)},
    )

    assert not result.accepted
    assert result.mesh is None
    assert MorphReasonCode.NONINJECTIVE_MORPH in result.reason_codes
    assert result.report.minimum_jacobian_determinant is not None
    assert result.report.minimum_jacobian_determinant <= 0.0


def test_higher_order_rule_detects_an_interior_fold_with_valid_mesh_vertices() -> None:
    reference = canonicalize_mesh(
        MeshPayload(
            nodes=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
            cells=((0, 1, 2),),
            cell_type=CellType.TRIANGLE,
            boundary_facets=((0, 1), (1, 2), (2, 0)),
            boundary_tags={"base": (0,), "diagonal": (1,), "left": (2,)},
            region_ids=(1,),
            unit=Unit.MM,
            source_provenance={"fixture": "interior-fold"},
        )
    )
    deformation = _InteriorFoldDeformation()
    old_degree_2_points = (
        (1.0 / 6.0, 1.0 / 6.0),
        (2.0 / 3.0, 1.0 / 6.0),
        (1.0 / 6.0, 2.0 / 3.0),
    )

    assert all(deformation.map_point(node) == node for node in reference.nodes)
    assert all(deformation.jacobian(point)[0][0] > 0.0 for point in old_degree_2_points)

    from piginot.morphing.validation import validate_deformation

    mesh, report = validate_deformation(
        reference,
        deformation,
        MorphValidityCriteria(),
    )

    assert mesh is None
    assert report.reason_codes == (MorphReasonCode.NONINJECTIVE_MORPH,)
    assert report.quadrature_rule == "triangle-degree-5-dunavant-v1"
    assert report.quadrature_points_checked == 7
    assert report.minimum_jacobian_determinant is not None
    assert report.minimum_jacobian_determinant < 0.0


def test_feature_size_and_mesh_quality_are_independent_validity_reasons() -> None:
    feature_criteria = MorphValidityCriteria(
        jacobian_margin=0.01,
        minimum_feature_size_mm=0.1,
        minimum_angle_degrees=0.0,
        maximum_aspect_ratio=1.0e6,
    )
    reference, predicate, morph = _setup(feature_criteria)

    feature_result = apply_morph(reference, morph, predicate, {"stretch": length(-3.8)})

    assert MorphReasonCode.FEATURE_SIZE_BELOW_MARGIN in feature_result.reason_codes

    skew_family = stretch_family(reference, lower=0.0, upper=8.0)
    quality_criteria = MorphValidityCriteria(
        jacobian_margin=0.01,
        minimum_feature_size_mm=0.0,
        minimum_angle_degrees=20.0,
        maximum_aspect_ratio=2.0,
    )
    quality_predicate = FamilyMembershipPredicate.compile(
        skew_family, reference, quality_criteria
    )
    skew_map = FFDMap(
        bounds=((0.0, 0.0), (4.0, 2.0)),
        control_shape=(2, 2),
        control_points=(
            FFDControlPoint(None, None),
            FFDControlPoint(None, None),
            FFDControlPoint(None, None),
            FFDControlPoint("stretch", None),
        ),
    )

    quality_result = apply_morph(
        reference, skew_map, quality_predicate, {"stretch": length(8.0)}
    )

    assert MorphReasonCode.MESH_QUALITY_OUT_OF_BOUNDS in quality_result.reason_codes


def test_ledger_accounts_for_every_attempt_and_every_rejection_has_a_reason() -> None:
    reference, predicate, morph = _setup()
    attempts = (
        MorphAttempt("valid", {"stretch": length(0.1)}),
        MorphAttempt("folded", {"stretch": length(-5.0)}),
        MorphAttempt("outside", {"stretch": length(1.0 + 1.0e-12)}),
    )

    ledger = run_morph_sweep(reference, morph, predicate, attempts)

    assert ledger.attempted_count == len(attempts)
    assert ledger.accepted_count + ledger.rejected_count == ledger.attempted_count
    assert ledger.accepted_count == 1
    assert ledger.rejected_count == 2
    assert all(record.reason_codes for record in ledger.rejected)
    assert {record.reason_code for record in ledger.rejected} == {
        MorphReasonCode.NONINJECTIVE_MORPH,
        MorphReasonCode.PARAMETER_OUT_OF_BOUNDS,
    }
    assert ledger.to_bytes() == ledger.to_bytes()


def test_ledger_rejects_duplicate_attempt_identity_before_execution() -> None:
    reference, predicate, morph = _setup()
    attempts = (
        MorphAttempt("duplicate", {"stretch": length(0.0)}),
        MorphAttempt("duplicate", {"stretch": length(0.1)}),
    )

    with pytest.raises(ValueError, match="unique"):
        run_morph_sweep(reference, morph, predicate, attempts)


def test_apply_morph_rejects_reference_and_map_context_mismatches() -> None:
    reference, predicate, _ = _setup()
    wrong_reference_predicate = replace(
        predicate,
        reference_mesh_id="canonical-mesh-v1:sha256:different",
    )
    wrong_map = FFDMap(
        bounds=((0.0, 0.0), (4.0, 2.0)),
        control_shape=(2, 2),
        control_points=(
            FFDControlPoint(None, None),
            FFDControlPoint(None, None),
            FFDControlPoint("other", None),
            FFDControlPoint("other", None),
        ),
    )

    wrong_reference = apply_morph(
        reference,
        FFDMap(((0.0, 0.0), (4.0, 2.0)), (2, 2), stretch_controls()),
        wrong_reference_predicate,
        {"stretch": length(0.0)},
    )
    wrong_parameter_map = apply_morph(
        reference, wrong_map, predicate, {"stretch": length(0.0)}
    )

    assert (
        wrong_reference.reason_codes[0] is MorphReasonCode.VALIDATION_CONTEXT_MISMATCH
    )
    assert (
        wrong_parameter_map.reason_codes[0] is MorphReasonCode.INVALID_MAP_CONFIGURATION
    )


def test_apply_morph_surfaces_untyped_bind_failures() -> None:
    reference, predicate, _ = _setup()

    result = apply_morph(
        reference, _RejectingMap(), predicate, {"stretch": length(0.0)}
    )

    assert result.reason_codes == (MorphReasonCode.INVALID_MAP_CONFIGURATION,)


def test_validation_rejects_nonfinite_jacobians_and_invalid_mapped_points() -> None:
    from piginot.morphing.validation import validate_deformation

    reference = canonicalize_mesh(rectangle_payload())
    jacobian_result = validate_deformation(
        reference, _InvalidJacobianDeformation(), gate_criteria()
    )
    point_result = validate_deformation(
        reference, _InvalidPointDeformation(), gate_criteria()
    )

    assert jacobian_result[0] is None
    assert jacobian_result[1].reason_codes[0] is MorphReasonCode.INVALID_MORPHED_MESH
    assert point_result[0] is None
    assert point_result[1].reason_codes[0] is MorphReasonCode.INVALID_MORPHED_MESH


def test_validation_rejects_bad_context_and_deformation_identity() -> None:
    from piginot.morphing.validation import validate_deformation

    reference = canonicalize_mesh(rectangle_payload())
    identity_result = validate_deformation(
        reference, _InvalidIdentityDeformation(), gate_criteria()
    )

    with pytest.raises(TypeError, match="CanonicalMesh"):
        validate_deformation(object(), _InvalidIdentityDeformation(), gate_criteria())
    with pytest.raises(TypeError, match="MorphValidityCriteria"):
        validate_deformation(reference, _InvalidIdentityDeformation(), object())

    assert identity_result[0] is None
    assert (
        identity_result[1].reason_codes[0] is MorphReasonCode.INVALID_MAP_CONFIGURATION
    )
