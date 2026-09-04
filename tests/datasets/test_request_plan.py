from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    CanonicalQuantity,
    ExecutionProfile,
    MaterialSpec,
    Unit,
    hash_spec,
    strict_json_loads,
)
from piginot.datasets.request_plan import (
    PublicFemRequestPlan,
    RequestPlanValidationError,
    build_public_fem_request_plan,
)
from piginot.doe.checkpoint import (
    CampaignBinding,
    CampaignBudget,
    CPEnvelope,
    EvidenceReference,
    compute_campaign_hash,
)
from piginot.doe.commitment import HoldoutCommitment
from piginot.doe.publication import (
    PublishedCase,
    PublishedCaseManifest,
    PublishedCaseStatus,
)
from piginot.doe.roles import EvidenceRole
from piginot.doe.secret_material import CAMPAIGN_SECRET_GENERATOR_ID
from piginot.fem import FemRequest, FemSolverConfig


def _request(case_id: str, *, profile_seed: int = 0) -> FemRequest:
    mesh = CanonicalMesh(
        mesh_id=f"mesh-{case_id}",
        nodes=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        cells=((0, 1, 2),),
        cell_type="triangle",
        boundary_facets=((0, 1), (1, 2), (2, 0)),
        semantic_tags={"bottom": (0,), "side": (1, 2)},
        region_ids=(1,),
        unit=Unit.MM,
        orientation="ccw",
        topology_signature="triangle-v1",
        quality_metrics={"min_scaled_jacobian": 0.5},
        source_provenance={"fixture": "s10-request-plan"},
    )
    return FemRequest(
        request_id=case_id,
        mesh=mesh,
        material=MaterialSpec(
            name="rubber",
            model="compressible_neo_hookean_plane_stress",
            properties={
                "elastic_modulus": CanonicalQuantity(100.0, Unit.MPA),
                "poisson_ratio": CanonicalQuantity(0.3, Unit.ONE),
            },
        ),
        boundary_conditions=(
            BoundaryConditionSpec(
                "fixed",
                "bottom",
                "dirichlet",
                ("x", "y"),
                (CanonicalQuantity(0.0, Unit.MM), CanonicalQuantity(0.0, Unit.MM)),
            ),
        ),
        loads=(),
        qois=(),
        config=FemSolverConfig(element_order=1),
        execution_profile=ExecutionProfile(
            python_version="3.12.0",
            library_versions={"numpy": "2.2.6"},
            blas_threads=1,
            device="cpu",
            dtype="float64",
            rng_algorithm="none",
            seed=profile_seed,
        ),
    )


def _case(
    case_id: str,
    role: EvidenceRole,
    *,
    status: PublishedCaseStatus = PublishedCaseStatus.ACTIVE,
) -> PublishedCase:
    return PublishedCase(
        case_id=case_id,
        leakage_group_id=f"group-{case_id}",
        role=role,
        primitive_coordinates={"width": 1.0},
        derived_coordinates={"net_section_ratio": 0.5},
        strata={"width": 0, "net_section_ratio": 1},
        status=status,
    )


def _manifest() -> PublishedCaseManifest:
    return PublishedCaseManifest(
        manifest_id="public-cases",
        primitive_axes=("width",),
        derived_axes=("net_section_ratio",),
        stratum_axes=("width", "net_section_ratio"),
        cases=(
            _case("train-1", EvidenceRole.TRAIN),
            _case("dev-1", EvidenceRole.DEVELOPMENT),
            _case(
                "reserve-1",
                EvidenceRole.TRAIN,
                status=PublishedCaseStatus.RESERVE,
            ),
        ),
    )


def _checkpoint(manifest: PublishedCaseManifest) -> CPEnvelope:
    budget = CampaignBudget(
        pilot=0,
        train=1,
        development=1,
        holdout=64,
        legacy=0,
        reserve=1,
    )
    binding = CampaignBinding(
        compiled_family_hash="a" * 64,
        pilot_evidence=(
            EvidenceReference(
                evidence_id="s1-closeout",
                content_hash="9" * 64,
                role=EvidenceRole.LEGACY,
            ),
        ),
        executable_envelope_hash="b" * 64,
        private_doe_plan_hash="c" * 64,
        role_ledger_hash="d" * 64,
        replacement_policy_hash="e" * 64,
        public_case_manifest_hash=manifest.content_hash,
        doe_seed_commitment="f" * 64,
        secret_generator_id=CAMPAIGN_SECRET_GENERATOR_ID,
        budget=budget,
        derived_axes=("net_section_ratio",),
    )
    campaign_hash = compute_campaign_hash(binding)
    return CPEnvelope(
        campaign_hash=campaign_hash,
        compiled_family_hash=binding.compiled_family_hash,
        pilot_evidence=binding.pilot_evidence,
        executable_envelope_hash=binding.executable_envelope_hash,
        private_doe_plan_hash=binding.private_doe_plan_hash,
        role_ledger_hash=binding.role_ledger_hash,
        replacement_policy_hash=binding.replacement_policy_hash,
        public_case_manifest_hash=binding.public_case_manifest_hash,
        doe_seed_commitment=binding.doe_seed_commitment,
        secret_generator_id=binding.secret_generator_id,
        holdout_commitment=HoldoutCommitment(
            scheme="blake2b-256",
            campaign_hash=campaign_hash,
            n_holdout_points=64,
            commitment="1" * 64,
            created_utc="2026-08-31T12:00:00Z",
        ),
        budget=budget,
        derived_axes=binding.derived_axes,
        created_utc="2026-08-31T12:01:00Z",
    )


def _build() -> PublicFemRequestPlan:
    manifest = _manifest()
    return build_public_fem_request_plan(
        _checkpoint(manifest),
        manifest,
        {"dev-1": _request("dev-1"), "train-1": _request("train-1")},
    )


def test_plan_is_cp_bound_complete_immutable_and_content_addressed() -> None:
    plan = _build()

    assert tuple(entry.case_id for entry in plan.entries) == ("dev-1", "train-1")
    assert plan.public_case_manifest_hash == _manifest().content_hash
    assert plan.execution_profile_hash == hash_spec(
        _request("train-1").execution_profile, "content"
    )
    train = plan.entry_by_case_id("train-1")
    request = _request("train-1")
    assert train.leakage_group_id == "group-train-1"
    assert train.role is EvidenceRole.TRAIN
    assert train.request_recipe_hash == request.recipe_hash
    assert train.request_content_hash == hash_spec(request, "content")
    assert train.mesh_id == request.mesh.mesh_id
    assert train.mesh_content_hash == hash_spec(request.mesh, "content")
    assert plan.content_hash == hash_spec(plan, "content")
    with pytest.raises(FrozenInstanceError):
        plan.campaign_hash = "0" * 64  # type: ignore[misc]


def test_plan_round_trip_is_strict_and_canonical() -> None:
    plan = _build()

    assert PublicFemRequestPlan.from_bytes(plan.to_bytes()) == plan
    payload = strict_json_loads(plan.to_bytes())
    assert isinstance(payload, dict)
    payload["unexpected"] = True
    altered = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    with pytest.raises(RequestPlanValidationError, match="unexpected fields"):
        PublicFemRequestPlan.from_bytes(altered)
    with pytest.raises(RequestPlanValidationError, match="canonical JSON"):
        PublicFemRequestPlan.from_bytes(plan.to_bytes() + b"\n")


@pytest.mark.parametrize(
    "requests",
    (
        {"train-1": _request("train-1")},
        {
            "train-1": _request("train-1"),
            "dev-1": _request("dev-1"),
            "reserve-1": _request("reserve-1"),
        },
    ),
)
def test_builder_requires_exactly_the_active_public_case_ids(
    requests: dict[str, FemRequest],
) -> None:
    manifest = _manifest()

    with pytest.raises(RequestPlanValidationError, match="request case IDs"):
        build_public_fem_request_plan(_checkpoint(manifest), manifest, requests)


def test_builder_rejects_request_case_or_execution_profile_mismatch() -> None:
    manifest = _manifest()
    checkpoint = _checkpoint(manifest)
    wrong_id = {
        "train-1": _request("not-train-1"),
        "dev-1": _request("dev-1"),
    }
    with pytest.raises(RequestPlanValidationError, match="request identity"):
        build_public_fem_request_plan(checkpoint, manifest, wrong_id)

    mixed_profiles = {
        "train-1": _request("train-1"),
        "dev-1": _request("dev-1", profile_seed=1),
    }
    with pytest.raises(RequestPlanValidationError, match="execution profile"):
        build_public_fem_request_plan(checkpoint, manifest, mixed_profiles)


def test_plan_constructor_and_parser_reject_incoherent_entry_provenance() -> None:
    plan = _build()
    bad_entry = replace(
        plan.entries[0],
        leakage_group_id=plan.entries[1].leakage_group_id,
    )

    with pytest.raises(RequestPlanValidationError, match="leakage group|role"):
        PublicFemRequestPlan(
            campaign_hash=plan.campaign_hash,
            public_case_manifest_hash=plan.public_case_manifest_hash,
            execution_profile_hash=plan.execution_profile_hash,
            entries=(bad_entry, *plan.entries[1:]),
        )


def test_plan_rejects_missing_lookup_empty_population_and_invalid_documents() -> None:
    plan = _build()

    with pytest.raises(KeyError):
        plan.entry_by_case_id("absent-case")
    with pytest.raises(RequestPlanValidationError, match="must not be empty"):
        PublicFemRequestPlan(
            campaign_hash=plan.campaign_hash,
            public_case_manifest_hash=plan.public_case_manifest_hash,
            execution_profile_hash=plan.execution_profile_hash,
            entries=(),
        )
    with pytest.raises(RequestPlanValidationError, match="text or bytes"):
        PublicFemRequestPlan.from_bytes(object())  # type: ignore[arg-type]
    with pytest.raises(RequestPlanValidationError, match="invalid request plan JSON"):
        PublicFemRequestPlan.from_bytes(b"not-json")
