"""Synthetic public records cannot close a mismatched campaign binding."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from piginot.contracts import hash_spec
from piginot.doe import preregistration as registration
from piginot.doe.design import DesignPoint, generate_coupled_design
from piginot.doe.preregistration import (
    LegacyCaseLedger,
    PreregistrationError,
    PreregistrationResult,
    build_preregistration,
)
from tests.doe.test_preregistration import _guarded_design, _inputs


@pytest.fixture(scope="module")
def values() -> dict[str, Any]:
    # This helper generates its tiny deterministic test roster from scratch.
    return _inputs()


@pytest.fixture(scope="module")
def result(values: dict[str, Any]) -> PreregistrationResult:
    return build_preregistration(**values)


@pytest.mark.parametrize("case_id", [None, "", "case-001:quadrature"])
def test_legacy_identity_must_name_one_complete_case(case_id: object) -> None:
    with pytest.raises(ValueError, match="case_id"):
        LegacyCaseLedger((case_id,))


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("spec", "CoupledDesignSpec"),
        ("design", "CoupledDesign"),
        ("role_allocation", "RoleAllocation"),
        ("legacy_ledger", "LegacyCaseLedger"),
        ("replacement_policy", "OrderedReservePolicy"),
        ("budget", "CampaignBudget"),
    ],
)
def test_untyped_campaign_parts_refuse_before_hashing_or_commitment(
    values: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    message: str,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("untyped campaign data reached hashing or commitment")

    monkeypatch.setattr(registration, "hash_spec", forbidden)
    monkeypatch.setattr(registration, "compute_holdout_commitment", forbidden)
    with pytest.raises(PreregistrationError, match=message):
        build_preregistration(**{**values, field: object()})


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("production_allocation", "RoleAllocation"),
        ("legacy_ledger", "LegacyCaseLedger"),
        ("public_manifest", "PublishedCaseManifest"),
        ("binding", "CampaignBinding"),
        ("checkpoint", "CPEnvelope"),
    ],
)
def test_closed_result_requires_typed_public_parts(
    result: PreregistrationResult, field: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(result, **{field: object()})


@pytest.mark.parametrize(
    "change", ["unverified", "roles", "manifest", "campaign", "commitment"]
)
def test_result_cannot_mix_otherwise_valid_record_identities(
    result: PreregistrationResult, change: str
) -> None:
    if change == "unverified":
        changes = {"reveal_verified": False}
    elif change == "roles":
        changes = {"binding": replace(result.binding, role_ledger_hash="f" * 64)}
    elif change == "manifest":
        changes = {
            "public_manifest": replace(
                result.public_manifest, manifest_id="another-manifest"
            )
        }
    elif change == "campaign":
        changes = {"campaign_hash": "f" * 64}
    else:
        changes = {
            "holdout_commitment": replace(
                result.holdout_commitment, commitment="f" * 64
            )
        }
    with pytest.raises(ValueError, match="(verify|does not match)"):
        replace(result, **changes)


def test_checkpoint_cannot_transfer_between_individually_valid_campaigns(
    values: dict[str, Any], result: PreregistrationResult
) -> None:
    other = build_preregistration(
        **{**values, "manifest_id": "different-public-manifest"}
    )
    with pytest.raises(
        ValueError, match="checkpoint does not match the campaign binding"
    ):
        replace(result, checkpoint=other.checkpoint)


def test_compiled_family_must_expose_the_executable_membership(
    values: dict[str, Any],
) -> None:
    family = replace(values["compiled_family"], membership=None)
    envelope = replace(
        values["executable_envelope"], compiled_family_hash=hash_spec(family, "content")
    )
    with pytest.raises(PreregistrationError, match="exact FamilyMembershipPredicate"):
        build_preregistration(
            **{**values, "compiled_family": family, "executable_envelope": envelope}
        )


def test_complete_design_population_cannot_exceed_the_declared_budget(
    values: dict[str, Any],
) -> None:
    spec = replace(values["spec"], sample_count=68)
    design = generate_coupled_design(values["executable_envelope"], spec)
    with pytest.raises(PreregistrationError, match="fixed production case budget"):
        build_preregistration(**{**values, "spec": spec, "design": design})


def test_selected_cases_cannot_be_relabelled_as_guard_negatives(
    values: dict[str, Any],
) -> None:
    design = values["design"]
    first = replace(design.selected_points[0], guard_band=True)
    changed = replace(design, selected_points=(first, *design.selected_points[1:]))
    with pytest.raises(PreregistrationError, match="invalid guard flag"):
        build_preregistration(**{**values, "design": changed})


def test_reserve_case_cannot_duplicate_an_active_case_identity(
    values: dict[str, Any],
) -> None:
    design = values["design"]
    removed = design.selected_points[0].case_id
    changed = replace(
        design,
        selected_points=(design.reserve_points[0], *design.selected_points[1:]),
        leakage_descriptors=tuple(
            item for item in design.leakage_descriptors if item.case_id != removed
        ),
    )
    with pytest.raises(PreregistrationError, match="globally unique"):
        build_preregistration(**{**values, "design": changed})


def test_positive_derived_coordinates_must_replay_from_the_same_inputs(
    values: dict[str, Any],
) -> None:
    design = values["design"]
    first = replace(
        design.selected_points[0], derived_coordinates={"net_section_ratio": 0.12345}
    )
    changed = replace(design, selected_points=(first, *design.selected_points[1:]))
    with pytest.raises(PreregistrationError, match="derived coordinates do not replay"):
        build_preregistration(**{**values, "design": changed})


def test_doe_derived_axes_cannot_change_the_envelope_vocabulary(
    values: dict[str, Any],
) -> None:
    old, new = "net_section_ratio", "renamed_ratio"

    def rename(point: DesignPoint) -> DesignPoint:
        return replace(
            point,
            derived_coordinates={
                new if name == old else name: value
                for name, value in point.derived_coordinates.items()
            },
            strata={
                new if name == old else name: value
                for name, value in point.strata.items()
            },
        )

    spec = replace(
        values["spec"],
        axes=tuple(
            replace(axis, name=new) if axis.name == old else axis
            for axis in values["spec"].axes
        ),
    )
    design = values["design"]
    changed = replace(
        design,
        selected_points=tuple(map(rename, design.selected_points)),
        reserve_points=tuple(map(rename, design.reserve_points)),
        validation_points=tuple(map(rename, design.validation_points)),
    )
    with pytest.raises(PreregistrationError, match="derived axes do not match"):
        build_preregistration(**{**values, "spec": spec, "design": changed})


@pytest.mark.parametrize("change", ["accepted", "derived"])
def test_guard_negative_must_replay_its_rejection_and_derived_values(
    values: dict[str, Any], change: str
) -> None:
    spec, design = _guarded_design(values)
    guard = design.guard_band_points[0]
    if change == "accepted":
        guard = replace(
            design.selected_points[0],
            case_id=guard.case_id,
            leakage_group_id=guard.leakage_group_id,
            guard_band=True,
        )
    else:
        guard = replace(guard, derived_coordinates={"net_section_ratio": 999.0})
    rejection = replace(design.guard_band_rejections[0], coordinates=guard.coordinates)
    changed = replace(
        design, guard_band_points=(guard,), guard_band_rejections=(rejection,)
    )
    with pytest.raises(
        PreregistrationError, match="(is accepted|derived coordinates do not replay)"
    ):
        build_preregistration(**{**values, "spec": spec, "design": changed})


def test_reserve_group_cannot_be_substituted_in_the_policy(
    values: dict[str, Any],
) -> None:
    policy = values["replacement_policy"]
    changed = replace(
        policy,
        candidates=(
            replace(policy.candidates[0], leakage_group_id="unrelated-group"),
            *policy.candidates[1:],
        ),
    )
    with pytest.raises(PreregistrationError, match="leakage_group_id differs"):
        build_preregistration(**{**values, "replacement_policy": changed})


@pytest.mark.parametrize("failure", ["reveal", "doe-replay"])
def test_failed_final_verification_blocks_checkpoint_publication(
    values: dict[str, Any], monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("failed verification reached checkpoint publication")

    monkeypatch.setattr(registration, "CPEnvelope", forbidden)
    if failure == "reveal":
        monkeypatch.setattr(
            registration, "verify_holdout_reveal", lambda *args, **kwargs: False
        )
        message = "reveal did not verify"
    else:

        def fail(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("declared replay unavailable")

        monkeypatch.setattr(registration, "generate_coupled_design", fail)
        message = "complete DOE replay failed"
    with pytest.raises(PreregistrationError, match=message):
        build_preregistration(**values)
