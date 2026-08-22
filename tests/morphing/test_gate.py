from __future__ import annotations

from collections import Counter

from piginot.contracts import canonical_json_bytes
from piginot.geometry import canonicalize_mesh
from piginot.morphing import (
    FamilyMembershipPredicate,
    FFDMap,
    MorphAttempt,
    MorphReasonCode,
    run_morph_sweep,
)
from tests.morphing.helpers import (
    gate_criteria,
    length,
    rectangle_payload,
    stretch_controls,
    stretch_family,
)


def test_g6_500_sample_stress_has_accepted_and_reasoned_rejected_instances() -> None:
    reference = canonicalize_mesh(rectangle_payload())
    predicate = FamilyMembershipPredicate.compile(
        stretch_family(reference, lower=-5.0, upper=1.0),
        reference,
        gate_criteria(),
    )
    morph = FFDMap(
        bounds=((0.0, 0.0), (4.0, 2.0)),
        control_shape=(2, 2),
        control_points=stretch_controls(),
    )
    attempts = tuple(
        MorphAttempt(
            f"stress-{index:03d}",
            {"stretch": length(-5.0 + 6.0 * index / 499.0)},
        )
        for index in range(500)
    )

    ledger = run_morph_sweep(reference, morph, predicate, attempts)
    rejection_reasons = Counter(
        reason for record in ledger.rejected for reason in record.reason_codes
    )

    assert ledger.attempted_count == 500
    assert ledger.accepted_count == 387
    assert ledger.rejected_count == 113
    assert ledger.accepted_count + ledger.rejected_count == ledger.attempted_count
    assert all(record.report.accepted for record in ledger.accepted)
    assert all(record.reason_codes for record in ledger.rejected)
    assert all(not record.report.accepted for record in ledger.rejected)
    assert rejection_reasons == Counter(
        {
            MorphReasonCode.NONINJECTIVE_MORPH: 87,
            MorphReasonCode.MESH_QUALITY_OUT_OF_BOUNDS: 29,
            MorphReasonCode.FEATURE_SIZE_BELOW_MARGIN: 6,
        }
    )


def test_g6_500_accepted_tag_sweep_and_determinism() -> None:
    reference = canonicalize_mesh(rectangle_payload())
    predicate = FamilyMembershipPredicate.compile(
        stretch_family(reference, lower=-0.25, upper=0.25),
        reference,
        gate_criteria(),
    )
    morph = FFDMap(
        bounds=((0.0, 0.0), (4.0, 2.0)),
        control_shape=(2, 2),
        control_points=stretch_controls(),
    )
    attempts = tuple(
        MorphAttempt(
            f"sample-{index:03d}",
            {"stretch": length(-0.25 + 0.5 * index / 499.0)},
        )
        for index in range(500)
    )
    criteria = predicate.validity
    expected_tag_counts = {
        name: len(facet_ids) for name, facet_ids in reference.semantic_tags.items()
    }

    ledger = run_morph_sweep(reference, morph, predicate, attempts)
    repeated = run_morph_sweep(reference, morph, predicate, (attempts[173],))

    assert ledger.attempted_count == 500
    assert ledger.accepted_count == 500
    assert ledger.rejected_count == 0
    assert ledger.accepted_count + ledger.rejected_count == ledger.attempted_count
    assert all(record.report.accepted for record in ledger.records)
    assert all(record.report.tags_preserved for record in ledger.records)
    assert all(
        record.report.topology_signature == reference.topology_signature
        for record in ledger.records
    )
    assert all(
        dict(record.report.semantic_tag_counts) == expected_tag_counts
        for record in ledger.records
    )
    assert all(
        record.report.minimum_jacobian_determinant is not None
        and record.report.minimum_jacobian_determinant > criteria.jacobian_margin
        for record in ledger.records
    )
    assert all(
        record.report.minimum_feature_size_mm is not None
        and record.report.minimum_feature_size_mm >= criteria.minimum_feature_size_mm
        for record in ledger.records
    )
    assert all(
        record.report.minimum_angle_degrees is not None
        and record.report.minimum_angle_degrees >= criteria.minimum_angle_degrees
        for record in ledger.records
    )
    assert all(
        record.report.maximum_aspect_ratio is not None
        and record.report.maximum_aspect_ratio <= criteria.maximum_aspect_ratio
        for record in ledger.records
    )
    assert all(
        record.report.quadrature_points_checked == 7 * len(reference.cells)
        for record in ledger.records
    )
    assert ledger.records[173].mesh is not None
    assert repeated.records[0].mesh is not None
    assert canonical_json_bytes(ledger.records[173].mesh) == canonical_json_bytes(
        repeated.records[0].mesh
    )
