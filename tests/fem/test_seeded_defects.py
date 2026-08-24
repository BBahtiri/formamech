"""Executable seeded-defect detection required by G8."""

from __future__ import annotations

from piginot.fem.seeded_defects import run_seeded_defect_suite


def test_each_required_seeded_defect_fires_at_least_one_real_check() -> None:
    report = run_seeded_defect_suite()

    assert tuple(item.defect_id for item in report.detections) == (
        "body-force sign",
        "boundary normal",
        "tag swap",
        "closure wrong root",
        "inconsistent tangent",
        "anisotropic scaling",
    )
    assert report.detected_count == 6
    assert report.required_count == 6
    for detection in report.detections:
        assert detection.control_value <= detection.threshold
        assert detection.mutant_value > detection.threshold
        assert detection.detected
        assert detection.fired_checks
