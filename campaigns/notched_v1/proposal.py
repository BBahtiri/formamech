"""Executable public S9 proposal for the notched campaign (ADR 0025).

The envelope, design policy and budget mirror the DogBone proposal: the
material is fixed and carried as two degenerate axes, the load is the nominal
axial strain, and the derived ratios steer coverage towards the deep, sharp
notches where the stress concentration is largest.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from campaigns.dogbone_v1.campaign_proposal import PublicCampaignDryRun
from campaigns.notched_v1.family import BOUNDS, compiled_family
from campaigns.notched_v1.pilot import ARTIFACTS, write_exact
from piginot.contracts import (
    CanonicalQuantity,
    Dimension,
    Unit,
    canonical_json_bytes,
    hash_spec,
)
from piginot.doe.checkpoint import CampaignBudget
from piginot.doe.coverage import coverage_gate_checks, measure_design_coverage
from piginot.doe.design import (
    AxisSource,
    AxisSpec,
    CoupledDesignSpec,
    DifficultRegion,
    InteractionSpec,
    generate_coupled_design,
)
from piginot.doe.envelope import ExecutableEnvelope
from piginot.doe.leakage import LeakagePolicy
from piginot.morphing.membership import PrimitiveBound

ENVELOPE_ID = "notched-v1-strain-controlled-10-20-v1"
LOAD_APPLICATION_POLICY_ID = "nominal-axial-strain-times-quarter-length-v1"
PUBLIC_DRY_RUN_ID = "public-notched-s9-proposal-dry-run-v1"
PUBLIC_DRY_RUN_SEED_ID = "public-notched-s9-proposal-dry-run-seed-v1"
DRY_RUN_PATH = ARTIFACTS / "s9-proposal-dry-run.json"
COVERAGE_THRESHOLDS = {
    "minimum_fraction": 1.0,
    "maximum_fill_distance": 0.10,
    "minimum_stratum_count": 8,
}


def _bound(
    name: str, lower: float, upper: float, unit: Unit, dimension: Dimension
) -> PrimitiveBound:
    return PrimitiveBound(
        name,
        CanonicalQuantity(lower, unit, dimension),
        CanonicalQuantity(upper, unit, dimension),
    )


def executable_envelope() -> ExecutableEnvelope:
    """The exact, secret-free envelope of the notched campaign."""

    family = compiled_family()
    strain = BOUNDS["nominal_axial_strain"]
    return ExecutableEnvelope(
        envelope_id=ENVELOPE_ID,
        compiled_family_hash=hash_spec(family, "content"),
        geometry_membership=family.membership,
        geometry_membership_hash=hash_spec(family.membership, "content"),
        material_bounds=(
            _bound("elastic_modulus", 760.0, 760.0, Unit.MPA, Dimension.STRESS),
            _bound("poisson_ratio", 0.23, 0.23, Unit.ONE, Dimension.DIMENSIONLESS),
        ),
        load_bounds=(
            _bound(
                "nominal_axial_strain",
                strain[0],
                strain[1],
                Unit.ONE,
                Dimension.DIMENSIONLESS,
            ),
        ),
    )


def _axis(
    name: str,
    source: AxisSource,
    *,
    weight: float = 1.0,
    strata: int = 4,
    minimum_per_stratum: int = 8,
    boundary_width: float = 0.10,
    minimum_per_boundary: int = 8,
) -> AxisSpec:
    return AxisSpec(
        name,
        source,
        weight=weight,
        strata=strata,
        minimum_per_stratum=minimum_per_stratum,
        boundary_width=boundary_width,
        minimum_per_boundary=minimum_per_boundary,
    )


def _fixed(name: str) -> AxisSpec:
    return _axis(
        name,
        AxisSource.PRIMITIVE,
        weight=0.01,
        strata=1,
        minimum_per_stratum=1,
        minimum_per_boundary=0,
    )


def design_spec(*, seed: bytes | str) -> CoupledDesignSpec:
    """The coupled DOE policy: 264 active cases and 16 ordered reserves."""

    return CoupledDesignSpec(
        seed=seed,
        sample_count=264,
        reserve_count=16,
        candidate_count=8192,
        validation_count=512,
        validation_candidate_count=4096,
        guard_band_count=32,
        guard_band_fraction=0.05,
        axes=(
            _axis("width", AxisSource.PRIMITIVE),
            _axis("notch_radius", AxisSource.PRIMITIVE),
            _axis("notch_depth", AxisSource.PRIMITIVE),
            _fixed("elastic_modulus"),
            _fixed("poisson_ratio"),
            _axis("nominal_axial_strain", AxisSource.PRIMITIVE, weight=0.05),
            _axis(
                "net_section_factor",
                AxisSource.DERIVED,
                weight=1.5,
                boundary_width=0.15,
            ),
            _axis(
                "notch_sharpness",
                AxisSource.DERIVED,
                weight=1.5,
                boundary_width=0.15,
            ),
        ),
        leakage_policy=LeakagePolicy(near_duplicate_tolerance=0.015),
        interactions=(
            InteractionSpec(
                "width-by-depth", ("width", "notch_depth"), minimum_per_cell=2
            ),
            InteractionSpec(
                "radius-by-depth", ("notch_radius", "notch_depth"), minimum_per_cell=2
            ),
            InteractionSpec(
                "strain-by-sharpness",
                ("nominal_axial_strain", "notch_sharpness"),
                minimum_per_cell=1,
            ),
        ),
        difficult_regions=(
            DifficultRegion(
                "deep-sharp-notch",
                {"net_section_factor": (0.5, 1.0), "notch_sharpness": (0.5, 1.0)},
                minimum_count=16,
            ),
            DifficultRegion(
                "high-strain-sharp-notch",
                {"nominal_axial_strain": (0.75, 1.0), "notch_sharpness": (0.5, 1.0)},
                minimum_count=8,
            ),
            DifficultRegion(
                "narrow-ligament",
                {"net_section_factor": (0.75, 1.0)},
                minimum_count=8,
            ),
        ),
    )


def budget() -> CampaignBudget:
    """ADR 0025 decision 4: 8 PILOT, 128 TRAIN, 64 DEVELOPMENT, 64 HOLDOUT."""

    return CampaignBudget(
        pilot=8, train=128, development=64, holdout=64, legacy=0, reserve=16
    )


def run_public_dry_run() -> PublicCampaignDryRun:
    """Generate and measure the proposal under a public seed."""

    envelope = executable_envelope()
    spec = design_spec(seed=PUBLIC_DRY_RUN_SEED_ID)
    design = generate_coupled_design(envelope, spec)
    coverage = measure_design_coverage(envelope, spec, design)
    checks = coverage_gate_checks(coverage, **COVERAGE_THRESHOLDS)  # type: ignore[arg-type]
    points = (*design.selected_points, *design.reserve_points)
    return PublicCampaignDryRun(
        dry_run_id=PUBLIC_DRY_RUN_ID,
        design_seed_id=PUBLIC_DRY_RUN_SEED_ID,
        load_application_policy_id=LOAD_APPLICATION_POLICY_ID,
        envelope_hash=hash_spec(envelope, "content"),
        design_spec_hash=hash_spec(spec, "content"),
        design_hash=hash_spec(design, "content"),
        budget=budget(),
        candidate_count=spec.candidate_count,
        candidate_accepted_count=design.candidate_accepted_count,
        candidate_rejection_count=len(design.candidate_rejections),
        validation_candidate_count=spec.validation_candidate_count,
        validation_accepted_count=design.validation_accepted_count,
        validation_rejection_count=len(design.validation_rejections),
        active_count=len(design.selected_points),
        reserve_count=len(design.reserve_points),
        validation_count=len(design.validation_points),
        guard_band_count=len(design.guard_band_points),
        leakage_group_count=len({point.leakage_group_id for point in points}),
        unresolved_quota_ids=design.unresolved_quota_ids,
        coverage=coverage,
        coverage_checks=checks,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m campaigns.notched_v1.proposal")
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args(argv)
    report = run_public_dry_run()
    print(f"content hash: {hash_spec(report, 'content')}")
    print(
        f"fill={report.coverage.empirical_fill_distance} "
        f"minimum_stratum={report.coverage.minimum_stratum_count} "
        f"unresolved={list(report.unresolved_quota_ids)} "
        f"checks={dict(report.coverage_checks)}"
    )
    if arguments.write:
        root = Path(__file__).resolve().parents[2]
        write_exact(root / DRY_RUN_PATH, canonical_json_bytes(report) + b"\n")
    return 0 if all(report.coverage_checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "ENVELOPE_ID",
    "LOAD_APPLICATION_POLICY_ID",
    "budget",
    "design_spec",
    "executable_envelope",
    "run_public_dry_run",
)
