from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from campaigns.dogbone_v1.campaign_proposal import (
    LOAD_APPLICATION_POLICY_ID,
    PAIRED_PUBLIC_DRY_RUN_SEED_ID,
    PUBLIC_DRY_RUN_ID,
    PublicCampaignDryRun,
    proposed_design_spec,
    proposed_executable_envelope,
    run_public_campaign_dry_run,
    write_public_campaign_dry_run,
)
from campaigns.dogbone_v1.strain_pilot import LOAD_POLICY_ID
from piginot.contracts import Unit, canonical_json_bytes, strict_json_loads
from piginot.doe.design import AxisSource

_DRY_RUN_ARTIFACT = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "campaigns"
    / "dogbone-v1"
    / "s9-strain-load-proposal-dry-run.json"
)


@pytest.fixture(scope="module")
def dry_run() -> PublicCampaignDryRun:
    return run_public_campaign_dry_run()


def test_proposal_binds_ratified_geometry_material_and_strain_load() -> None:
    envelope = proposed_executable_envelope()

    assert LOAD_APPLICATION_POLICY_ID == LOAD_POLICY_ID
    assert envelope.envelope_id == "dogbone-v1-strain-controlled-10-20-v1"
    assert tuple(bound.name for bound in envelope.material_bounds) == (
        "elastic_modulus",
        "poisson_ratio",
    )
    assert envelope.material_bounds[0].lower_bound.value == 760.0
    assert envelope.material_bounds[0].upper_bound.value == 760.0
    assert envelope.material_bounds[0].unit is Unit.MPA
    assert envelope.material_bounds[1].lower_bound.value == 0.23
    assert envelope.material_bounds[1].upper_bound.value == 0.23
    assert envelope.load_bounds[0].name == "nominal_axial_strain"
    assert envelope.load_bounds[0].lower_bound.value == 0.10
    assert envelope.load_bounds[0].upper_bound.value == 0.20
    assert envelope.load_bounds[0].unit is Unit.ONE


def test_load_axis_is_stratified_and_interacts_with_concentration() -> None:
    spec = proposed_design_spec(seed=b"public-test-seed")
    axes = {axis.name: axis for axis in spec.axes}
    load_axis = axes["nominal_axial_strain"]

    assert load_axis.source is AxisSource.PRIMITIVE
    assert load_axis.weight == 0.05
    assert load_axis.strata == 4
    assert load_axis.minimum_per_stratum == 8
    assert load_axis.boundary_width == 0.10
    assert load_axis.minimum_per_boundary == 8
    assert any(
        interaction.axes == ("nominal_axial_strain", "grip_to_gauge_ratio")
        and interaction.minimum_per_cell == 2
        for interaction in spec.interactions
    )
    assert any(
        region.region_id == "high-strain-high-concentration"
        and region.minimum_count == 8
        for region in spec.difficult_regions
    )


def test_public_dry_run_resolves_every_quota_and_threshold(
    dry_run: PublicCampaignDryRun,
) -> None:
    assert dry_run.dry_run_id == PUBLIC_DRY_RUN_ID
    assert dry_run.design_seed_id == PAIRED_PUBLIC_DRY_RUN_SEED_ID
    assert dry_run.active_count == 408
    assert dry_run.reserve_count == 32
    assert dry_run.budget.production_cases == dry_run.active_count
    assert dry_run.budget.reserve == dry_run.reserve_count
    assert (
        dry_run.candidate_accepted_count + dry_run.candidate_rejection_count
        == dry_run.candidate_count
    )
    assert (
        dry_run.validation_accepted_count + dry_run.validation_rejection_count
        == dry_run.validation_candidate_count
    )
    assert dry_run.validation_count == 512
    assert dry_run.guard_band_count == 32
    assert dry_run.unresolved_quota_ids == ()
    assert dry_run.leakage_group_count == 440
    assert all(dry_run.coverage_checks.values())
    assert dry_run.coverage.minimum_stratum_count >= 8
    assert dry_run.coverage.empirical_fill_distance <= 0.10


def test_public_dry_run_rejects_invalid_summary_fields(
    dry_run: PublicCampaignDryRun,
) -> None:
    with pytest.raises(ValueError, match="check names"):
        replace(dry_run, coverage_checks={"": True})
    with pytest.raises(ValueError, match="boolean"):
        replace(dry_run, coverage_checks=cast(Any, {"fill": 1}))
    with pytest.raises(ValueError, match="unresolved quota"):
        replace(dry_run, unresolved_quota_ids=("",))
    with pytest.raises(ValueError, match="unsupported campaign dry run"):
        replace(dry_run, schema_version="2.0")


def test_public_dry_run_writer_is_canonical_and_refuses_replacement(
    dry_run: PublicCampaignDryRun,
    tmp_path: Path,
) -> None:
    target = tmp_path / "nested" / "dry-run.json"

    write_public_campaign_dry_run(dry_run, target)
    assert target.read_bytes() == canonical_json_bytes(dry_run) + b"\n"
    write_public_campaign_dry_run(dry_run, target)

    target.write_bytes(b"different evidence\n")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_public_campaign_dry_run(dry_run, target)


def test_recorded_public_dry_run_is_exact_evidence() -> None:
    payload = _DRY_RUN_ARTIFACT.read_bytes()
    assert payload.endswith(b"\n")
    canonical_payload = payload[:-1]
    document = strict_json_loads(canonical_payload)
    assert isinstance(document, dict)
    assert document["dry_run_id"] == PUBLIC_DRY_RUN_ID
    assert document["design_seed_id"] == PAIRED_PUBLIC_DRY_RUN_SEED_ID
    assert document["active_count"] == 408
    assert document["reserve_count"] == 32
    assert document["validation_count"] == 512
    assert document["guard_band_count"] == 32
    assert document["unresolved_quota_ids"] == []
    framed = b'{"domain":"content","payload":' + canonical_payload + b"}"
    digest = hashlib.blake2b(framed, digest_size=32).hexdigest()

    assert digest == "42cb8f7ee8a16f89663289ecf39cc959d656f3fbba152c9826c469f0a920be34"
