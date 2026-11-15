"""A compiler plan validates public metadata and does no numerical work."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from piginot.compiler import planning
from piginot.contracts import canonical_json_bytes
from piginot.doe import EvidenceRole, PublishedCaseStatus
from tests.compiler.plan_fixtures import inputs


def test_plan_is_deterministic_and_preserves_pending_numerical_checks(tmp_path):
    spec, config, manifest, adapter = inputs(tmp_path)
    output = tmp_path / "run"
    first = planning.build_plan(spec, config, output, descriptor=adapter)
    second = planning.build_plan(spec, config, output, descriptor=adapter)
    assert first.document() == second.document()
    assert first.plan_sha256 == second.plan_sha256
    assert not output.exists()
    document = first.document()
    assert document["status"] == "PLANNED"
    assert document["claim_level"] == "NOT_VERIFIED"
    assert document["source"]["manifest_content_hash"] == manifest.content_hash
    assert document["stages"]["plan"] == "PASS"
    assert document["stages"]["eligibility"] == "PASS"
    assert document["stages"]["material"] == "PENDING"
    assert document["stages"]["public-development"] == "NOT_REQUESTED"
    assert "primitive_coordinates" not in document["source"]
    assert len(document["source"]["fit_case_ids"]) == 2
    assert document["execution_profile_status"] == "DECLARED_NOT_NUMERICALLY_PROBED"
    assert len(document["implementation"]["package_sha256"]) == 64


def test_requested_public_development_stays_pending_until_screen_and_freeze(tmp_path):
    spec, config, _, adapter = inputs(tmp_path, development=True)
    plan = planning.build_plan(spec, config, tmp_path / "run", descriptor=adapter)
    assert plan.document()["stages"]["public-development"] == "PENDING"


@pytest.mark.parametrize(
    "change",
    [
        {"implementation_sha256": "b" * 64},
        {"adapter_id": "other"},
        {"materials": ("gent",)},
        {"liftings": ("adf",)},
        {"reference_initializations": ("neo-hookean-full-load-hgo",)},
    ],
)
def test_capability_mismatch_refuses_before_output_creation(tmp_path, change):
    spec, config, _, adapter = inputs(tmp_path)
    output = tmp_path / "run"
    with pytest.raises(planning.PlanError):
        planning.build_plan(spec, config, output, descriptor=replace(adapter, **change))
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["budget", "conditioning", "neumann", "reference"])
def test_spec_authority_and_unsupported_paths_are_checked_before_sources(
    tmp_path, monkeypatch, mutation
):
    spec, config, _, adapter = inputs(tmp_path)
    problem = json.loads(spec.read_text())
    campaign = json.loads(config.read_text())
    if mutation == "budget":
        problem["split"]["train"] = 4
    elif mutation == "conditioning":
        campaign["model"]["parameter_names"] = ["width"]
    elif mutation == "neumann":
        problem["boundary_conditions"][-1] = {
            "name": "top_neumann",
            "region": "top",
            "kind": "neumann",
            "components": ["x"],
            "values": [{"expression": "0", "unit": "MPa"}],
        }
    else:
        campaign["reference"]["initialization"] = "neo-hookean-full-load-hgo"
    spec.write_text(json.dumps(problem))
    config.write_text(json.dumps(campaign))
    with pytest.raises(planning.PlanError):
        planning.build_plan(spec, config, tmp_path / "run", descriptor=adapter)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "mutation",
    ["missing", "inactive", "legacy", "development", "leakage", "coordinates"],
)
def test_public_roster_is_fully_guarded(tmp_path, mutation):
    spec, config, manifest, adapter = inputs(tmp_path)
    cases = list(manifest.cases)
    index = next(i for i, case in enumerate(cases) if case.case_id == "fit-001")
    if mutation == "missing":
        cases.pop(index)
    elif mutation == "inactive":
        cases[index] = replace(cases[index], status=PublishedCaseStatus.RESERVE)
    elif mutation == "legacy":
        cases[index] = replace(cases[index], role=EvidenceRole.LEGACY)
    elif mutation == "development":
        cases[index] = replace(cases[index], role=EvidenceRole.DEVELOPMENT)
    elif mutation == "leakage":
        cases[index] = replace(cases[index], leakage_group_id="group-screen-001")
    else:
        cases[index] = replace(
            cases[index],
            primitive_coordinates={
                **cases[index].primitive_coordinates,
                "width": 100.0,
            },
        )
    payload = replace(manifest, cases=tuple(cases)).to_bytes()
    (tmp_path / "manifest.json").write_bytes(payload)
    campaign = json.loads(config.read_text())
    campaign["source"]["sha256"] = hashlib.sha256(payload).hexdigest()
    config.write_text(json.dumps(campaign))
    with pytest.raises(planning.PlanError):
        planning.build_plan(spec, config, tmp_path / "run", descriptor=adapter)


def test_manifest_digest_is_verified_and_inputs_are_not_replaced(tmp_path):
    spec, config, _, adapter = inputs(tmp_path)
    original = spec.read_bytes()
    with pytest.raises(planning.PlanError):
        planning.build_plan(spec, config, spec, descriptor=adapter)
    assert spec.read_bytes() == original
    with pytest.raises(planning.PlanError):
        planning.build_plan(spec, config, tmp_path, descriptor=adapter)
    (tmp_path / "manifest.json").write_bytes(b"invalid changed source")
    with pytest.raises(planning.PlanError, match="digest"):
        planning.build_plan(spec, config, tmp_path / "run", descriptor=adapter)


def test_changed_inputs_produce_a_new_plan_binding(tmp_path):
    spec, config, _, adapter = inputs(tmp_path)
    first = planning.build_plan(spec, config, tmp_path / "run", descriptor=adapter)
    document = json.loads(config.read_text())
    document["training"]["optimizer_steps"] = 5
    config.write_text(json.dumps(document))
    second = planning.build_plan(spec, config, tmp_path / "run", descriptor=adapter)
    assert first.plan_sha256 != second.plan_sha256
    raw = first.document()
    digest = raw.pop("plan_sha256")
    assert digest == hashlib.sha256(canonical_json_bytes(raw)).hexdigest()
