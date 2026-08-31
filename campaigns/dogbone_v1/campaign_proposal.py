"""Executable public S9 proposal for the first DogBone campaign."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

from campaigns.dogbone_v1.family import compiled_family
from campaigns.dogbone_v1.strain_pilot import LOAD_POLICY_ID
from piginot.contracts import (
    CanonicalQuantity,
    Dimension,
    Unit,
    canonical_json_bytes,
    hash_spec,
)
from piginot.doe.checkpoint import CampaignBudget
from piginot.doe.coverage import (
    DesignCoverage,
    coverage_gate_checks,
    measure_design_coverage,
)
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

PUBLIC_DRY_RUN_ID = "public-dogbone-s9-strain-proposal-dry-run-v1"
PAIRED_PUBLIC_DRY_RUN_SEED_ID = "public-dogbone-s9-proposal-dry-run-v2"
LOAD_APPLICATION_POLICY_ID = LOAD_POLICY_ID
PROPOSED_ENVELOPE_ID = "dogbone-v1-strain-controlled-10-20-v1"
_DRY_RUN_ARTIFACT = Path(
    "artifacts/campaigns/dogbone-v1/s9-strain-load-proposal-dry-run.json"
)


def _quantity(
    value: float,
    unit: Unit,
    dimension: Dimension,
) -> CanonicalQuantity:
    return CanonicalQuantity(value, unit, dimension)


def _bound(
    name: str,
    lower: float,
    upper: float,
    unit: Unit,
    dimension: Dimension,
) -> PrimitiveBound:
    return PrimitiveBound(
        name,
        _quantity(lower, unit, dimension),
        _quantity(upper, unit, dimension),
    )


def proposed_executable_envelope() -> ExecutableEnvelope:
    """Build the exact, secret-free envelope awaiting load ratification."""

    family = compiled_family()
    return ExecutableEnvelope(
        envelope_id=PROPOSED_ENVELOPE_ID,
        compiled_family_hash=hash_spec(family, "content"),
        geometry_membership=family.membership,
        geometry_membership_hash=hash_spec(family.membership, "content"),
        material_bounds=(
            _bound(
                "elastic_modulus",
                760.0,
                760.0,
                Unit.MPA,
                Dimension.STRESS,
            ),
            _bound(
                "poisson_ratio",
                0.23,
                0.23,
                Unit.ONE,
                Dimension.DIMENSIONLESS,
            ),
        ),
        load_bounds=(
            _bound(
                "nominal_axial_strain",
                0.10,
                0.20,
                Unit.ONE,
                Dimension.DIMENSIONLESS,
            ),
        ),
    )


def _axis(
    name: str,
    source: AxisSource,
    *,
    weight: float,
    strata: int,
    minimum_per_stratum: int,
    boundary_width: float,
    minimum_per_boundary: int,
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


def proposed_design_spec(*, seed: bytes | str) -> CoupledDesignSpec:
    """Build the exact coupled DOE policy with an explicit load axis."""

    return CoupledDesignSpec(
        seed=seed,
        sample_count=408,
        reserve_count=32,
        candidate_count=4096,
        validation_count=512,
        validation_candidate_count=4096,
        guard_band_count=32,
        guard_band_fraction=0.05,
        axes=(
            _axis(
                "L_total",
                AxisSource.PRIMITIVE,
                weight=1.0,
                strata=4,
                minimum_per_stratum=8,
                boundary_width=0.10,
                minimum_per_boundary=8,
            ),
            _axis(
                "W_grip",
                AxisSource.PRIMITIVE,
                weight=1.0,
                strata=4,
                minimum_per_stratum=8,
                boundary_width=0.10,
                minimum_per_boundary=8,
            ),
            _axis(
                "W_gauge",
                AxisSource.PRIMITIVE,
                weight=1.0,
                strata=4,
                minimum_per_stratum=8,
                boundary_width=0.10,
                minimum_per_boundary=8,
            ),
            _axis(
                "R_fillet",
                AxisSource.PRIMITIVE,
                weight=1.0,
                strata=4,
                minimum_per_stratum=8,
                boundary_width=0.10,
                minimum_per_boundary=8,
            ),
            _axis(
                "elastic_modulus",
                AxisSource.PRIMITIVE,
                weight=0.01,
                strata=1,
                minimum_per_stratum=1,
                boundary_width=0.10,
                minimum_per_boundary=0,
            ),
            _axis(
                "poisson_ratio",
                AxisSource.PRIMITIVE,
                weight=0.01,
                strata=1,
                minimum_per_stratum=1,
                boundary_width=0.10,
                minimum_per_boundary=0,
            ),
            _axis(
                "nominal_axial_strain",
                AxisSource.PRIMITIVE,
                weight=0.05,
                strata=4,
                minimum_per_stratum=8,
                boundary_width=0.10,
                minimum_per_boundary=8,
            ),
            _axis(
                "arc_feasibility_ratio",
                AxisSource.DERIVED,
                weight=1.5,
                strata=2,
                minimum_per_stratum=8,
                boundary_width=0.10,
                minimum_per_boundary=0,
            ),
            _axis(
                "fillet_to_gauge_ratio",
                AxisSource.DERIVED,
                weight=1.5,
                strata=4,
                minimum_per_stratum=8,
                boundary_width=0.15,
                minimum_per_boundary=8,
            ),
            _axis(
                "grip_to_gauge_ratio",
                AxisSource.DERIVED,
                weight=2.0,
                strata=4,
                minimum_per_stratum=8,
                boundary_width=0.15,
                minimum_per_boundary=8,
            ),
            _axis(
                "length_to_gauge_ratio",
                AxisSource.DERIVED,
                weight=1.25,
                strata=4,
                minimum_per_stratum=8,
                boundary_width=0.15,
                minimum_per_boundary=8,
            ),
        ),
        leakage_policy=LeakagePolicy(near_duplicate_tolerance=0.015),
        interactions=(
            InteractionSpec(
                "grip-by-gauge",
                ("W_grip", "W_gauge"),
                minimum_per_cell=2,
            ),
            InteractionSpec(
                "gauge-by-fillet",
                ("W_gauge", "R_fillet"),
                minimum_per_cell=2,
            ),
            InteractionSpec(
                "length-by-grip-ratio",
                ("L_total", "grip_to_gauge_ratio"),
                minimum_per_cell=2,
            ),
            InteractionSpec(
                "strain-by-grip-ratio",
                ("nominal_axial_strain", "grip_to_gauge_ratio"),
                minimum_per_cell=2,
            ),
        ),
        difficult_regions=(
            DifficultRegion(
                "high-concentration",
                {
                    "grip_to_gauge_ratio": (0.75, 1.0),
                    "arc_feasibility_ratio": (0.0, 0.10),
                },
                minimum_count=32,
            ),
            DifficultRegion(
                "near-arc-margin",
                {"arc_feasibility_ratio": (0.0, 0.025)},
                minimum_count=16,
            ),
            DifficultRegion(
                "short-high-concentration",
                {
                    "L_total": (0.0, 0.25),
                    "grip_to_gauge_ratio": (0.75, 1.0),
                    "arc_feasibility_ratio": (0.0, 0.10),
                },
                minimum_count=8,
            ),
            DifficultRegion(
                "high-strain-high-concentration",
                {
                    "nominal_axial_strain": (0.75, 1.0),
                    "grip_to_gauge_ratio": (0.75, 1.0),
                    "arc_feasibility_ratio": (0.0, 0.10),
                },
                minimum_count=8,
            ),
        ),
    )


def proposed_budget() -> CampaignBudget:
    """Return the already-ratified first-campaign case budget."""

    return CampaignBudget(
        pilot=24,
        train=256,
        development=64,
        holdout=64,
        legacy=0,
        reserve=32,
    )


@dataclass(frozen=True, slots=True)
class PublicCampaignDryRun:
    """Secret-free summary of one deterministic campaign-proposal replay."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    dry_run_id: str
    design_seed_id: str
    load_application_policy_id: str
    envelope_hash: str
    design_spec_hash: str
    design_hash: str
    budget: CampaignBudget
    candidate_count: int
    candidate_accepted_count: int
    candidate_rejection_count: int
    validation_candidate_count: int
    validation_accepted_count: int
    validation_rejection_count: int
    active_count: int
    reserve_count: int
    validation_count: int
    guard_band_count: int
    leakage_group_count: int
    unresolved_quota_ids: tuple[str, ...]
    coverage: DesignCoverage
    coverage_checks: Mapping[str, bool]
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        checks = dict(sorted(self.coverage_checks.items()))
        if any(not isinstance(name, str) or not name for name in checks):
            raise ValueError("coverage check names must be non-empty text")
        if any(not isinstance(value, bool) for value in checks.values()):
            raise ValueError("coverage checks must be boolean")
        unresolved = tuple(self.unresolved_quota_ids)
        if any(not isinstance(value, str) or not value for value in unresolved):
            raise ValueError("unresolved quota IDs must be non-empty text")
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported campaign dry run {self.schema_version!r}")
        object.__setattr__(self, "coverage_checks", MappingProxyType(checks))
        object.__setattr__(self, "unresolved_quota_ids", unresolved)


def run_public_campaign_dry_run() -> PublicCampaignDryRun:
    """Generate and measure the exact public load-aware DOE proposal."""

    envelope = proposed_executable_envelope()
    spec = proposed_design_spec(seed=PAIRED_PUBLIC_DRY_RUN_SEED_ID)
    design = generate_coupled_design(envelope, spec)
    coverage = measure_design_coverage(envelope, spec, design)
    checks = coverage_gate_checks(
        coverage,
        minimum_fraction=1.0,
        maximum_fill_distance=0.10,
        minimum_stratum_count=8,
    )
    campaign_points = (*design.selected_points, *design.reserve_points)
    return PublicCampaignDryRun(
        dry_run_id=PUBLIC_DRY_RUN_ID,
        design_seed_id=PAIRED_PUBLIC_DRY_RUN_SEED_ID,
        load_application_policy_id=LOAD_APPLICATION_POLICY_ID,
        envelope_hash=hash_spec(envelope, "content"),
        design_spec_hash=hash_spec(spec, "content"),
        design_hash=hash_spec(design, "content"),
        budget=proposed_budget(),
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
        leakage_group_count=len({point.leakage_group_id for point in campaign_points}),
        unresolved_quota_ids=design.unresolved_quota_ids,
        coverage=coverage,
        coverage_checks=checks,
    )


def write_public_campaign_dry_run(
    report: PublicCampaignDryRun,
    target: Path,
) -> None:
    """Write a canonical dry-run artifact without replacing different evidence."""

    payload = canonical_json_bytes(report) + b"\n"
    if target.exists():
        if target.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace existing evidence: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def main() -> int:
    """Create the deterministic public proposal artifact once."""

    report = run_public_campaign_dry_run()
    repository_root = Path(__file__).resolve().parents[2]
    target = repository_root / _DRY_RUN_ARTIFACT
    write_public_campaign_dry_run(report, target)
    print(f"wrote {target}")
    print(f"content hash: {hash_spec(report, 'content')}")
    print(
        "coverage: "
        f"fill={report.coverage.empirical_fill_distance}, "
        f"minimum_stratum={report.coverage.minimum_stratum_count}, "
        f"unresolved={len(report.unresolved_quota_ids)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "LOAD_APPLICATION_POLICY_ID",
    "PAIRED_PUBLIC_DRY_RUN_SEED_ID",
    "PROPOSED_ENVELOPE_ID",
    "PUBLIC_DRY_RUN_ID",
    "PublicCampaignDryRun",
    "proposed_budget",
    "proposed_design_spec",
    "proposed_executable_envelope",
    "run_public_campaign_dry_run",
    "write_public_campaign_dry_run",
)
