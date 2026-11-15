"""Candidate exports bind inputs and preserve an explicitly unverified claim."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from piginot.compiler import exporting
from piginot.compiler.execution_records import digest
from piginot.compiler.planning import build_plan
from piginot.operators import EnergyFormOperator, EnergyOperatorConfig
from piginot.problem.verification import LiftingEvidence
from piginot.training.checkpoint import model_state_hash
from tests.compiler.plan_fixtures import inputs
from tests.training.test_checkpoint import _binding


@pytest.fixture
def context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    spec, config, _, descriptor = inputs(tmp_path)
    plan = build_plan(spec, config, tmp_path / "output", descriptor=descriptor)
    shape = replace(
        plan.config.model,
        hidden_dim=8,
        parameter_hidden_dim=8,
        latent_token_count=2,
        num_attention_heads=2,
        decoder_layers=1,
        positional_frequencies=1,
    )
    plan = replace(plan, config=replace(plan.config, model=shape))
    model_config = asdict(shape)
    model_config.pop("architecture_id")
    with torch.random.fork_rng(devices=[]):
        model = EnergyFormOperator(EnergyOperatorConfig(**model_config))
    names = list(shape.parameter_names)
    normalization = {
        "parameter_names": names,
        "fit_case_ids": ["fit-001", "fit-002"],
        "mean": [0.0] * len(names),
        "standard_deviation": [1.0] * len(names),
    }
    binding = replace(
        _binding(),
        problem_spec_hash=plan.problem.problem_hash,
        public_manifest_hash=plan.manifest.content_hash,
        normalization_hash=digest(normalization),
        execution_profile=plan.config.execution.training,
        seed=plan.config.training.seed,
        architecture_id=model.architecture_id,
        architecture_module_digest=exporting.problem_energy_module_digest(
            model, plan.problem.problem_hash
        ),
    )
    evidence = LiftingEvidence(
        plan.problem.problem_hash,
        "quarter",
        65,
        0,
        (320, 4800),
        0.0,
        None,
        None,
        {},
        1.0,
    )
    checkpoint = plan.output_path / "checkpoints" / "step-000000004"
    checkpoint.mkdir(parents=True)
    monkeypatch.setattr(exporting, "_verify_inputs", lambda plan: None)

    def load(path, *, model: EnergyFormOperator, **kwargs):
        assert path == checkpoint
        return SimpleNamespace(
            manifest=SimpleNamespace(
                step=4, manifest_hash="e" * 64, model_state_hash=model_state_hash(model)
            )
        )

    monkeypatch.setattr(exporting, "load_training_checkpoint", load)
    return plan, {
        "checkpoint_path": checkpoint,
        "run_binding": binding,
        "normalization": normalization,
        "material_admission": {
            "status": "PASS",
            "all_eight_checks_complete": True,
            "evidence": {
                "problem_hash": plan.problem.problem_hash,
                "model": plan.problem.material.model,
                "passed": True,
            },
        },
        "lifting_verification": evidence.verification(plan.problem),
        "lifting_evidence": evidence,
        "model": model,
    }


@pytest.mark.parametrize(
    "change",
    [
        "problem",
        "normalization",
        "fit",
        "material",
        "steps",
        "shape",
        "path",
        "digest",
        "lifting",
    ],
)
def test_export_refuses_mixed_evidence_before_publication(context, monkeypatch, change):
    plan, arguments = context
    if change == "problem":
        arguments["run_binding"] = replace(
            arguments["run_binding"], problem_spec_hash="f" * 64
        )
    elif change == "normalization":
        arguments["normalization"] = {**arguments["normalization"], "mean": [99.0] * 5}
    elif change == "fit":
        arguments["normalization"]["fit_case_ids"] = ["screen-001"]
        arguments["run_binding"] = replace(
            arguments["run_binding"],
            normalization_hash=digest(arguments["normalization"]),
        )
    elif change == "material":
        arguments["material_admission"]["evidence"]["passed"] = False
    elif change == "steps":
        monkeypatch.setattr(
            exporting,
            "load_training_checkpoint",
            lambda *args, **kwargs: SimpleNamespace(manifest=SimpleNamespace(step=3)),
        )
    elif change == "shape":
        arguments["model"] = EnergyFormOperator(
            EnergyOperatorConfig(parameter_names=("other",))
        )
    elif change == "path":
        arguments["checkpoint_path"] = arguments["checkpoint_path"].with_name(
            "step-000000003"
        )
    elif change == "digest":
        arguments["run_binding"] = replace(
            arguments["run_binding"],
            architecture_module_digest="blake2b256:" + "f" * 64,
        )
    else:
        evidence = replace(arguments["lifting_evidence"], refused_geometries=1)
        arguments["lifting_evidence"] = evidence
        arguments["lifting_verification"] = evidence.verification(plan.problem)
    with pytest.raises(ValueError):
        exporting.export_candidate(plan, **arguments)
    assert not (plan.output_path / "candidate-bundle").exists()


def test_envelope_contains_only_geometry_constraints_and_retains_all_bounds(context):
    from piginot.morphing import MorphValidityCriteria
    from tests.problem_helpers import plate_mesh

    plan, _ = context
    mesh = plate_mesh(mesh_id=plan.problem.geometry.reference_mesh_id)
    envelope = exporting.problem_envelope(plan.problem, mesh, MorphValidityCriteria())
    assert {bound.name for bound in envelope.primitive_bounds} == {
        parameter.name for parameter in plan.problem.parameters
    }
    assert envelope.geometry_membership.reference_mesh_id == mesh.mesh_id
    assert envelope.compiled_family_hash == plan.problem.problem_hash
    assert [bound.name for bound in envelope.derived_bounds] == ["aspect"]


def test_envelope_substitutes_constants_and_leaves_cross_role_constraints_embedded(
    context,
):
    from piginot.contracts import ProblemSpec
    from piginot.morphing import MorphValidityCriteria
    from tests.problem_helpers import plate_mesh

    plan, _ = context
    document = plan.problem.document()
    document["constants"] = [
        {"name": "length_scale", "value": 2.0, "unit": "mm"},
        {"name": "factor", "value": 2.0, "unit": "1"},
    ]
    document["constraints"].extend(
        [
            {
                "name": "scaled_width",
                "expression": "width/factor",
                "lo": 1.0,
                "hi": 10.0,
                "unit": "mm",
            },
            {
                "name": "scaled_length",
                "expression": "width/length_scale",
                "lo": 1.0,
                "hi": 10.0,
                "unit": "1",
            },
            {
                "name": "material_check",
                "expression": "modulus",
                "lo": 1.0,
                "hi": 1000.0,
                "unit": "MPa",
            },
        ]
    )
    spec = ProblemSpec.from_document(document)
    mesh = plate_mesh(mesh_id=spec.geometry.reference_mesh_id)
    envelope = exporting.problem_envelope(spec, mesh, MorphValidityCriteria())
    assert [bound.name for bound in envelope.derived_bounds] == [
        "aspect",
        "scaled_width",
    ]
    assert envelope.derived_bounds[1].canonical_ast == (
        "binary",
        "/",
        ("name", "width"),
        ("number", 2.0),
    )


@pytest.fixture
def public_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from piginot.compiler.config import CampaignConfig
    from piginot.contracts import ProblemSpec
    from piginot.doe import (
        EvidenceRole,
        PublishedCase,
        PublishedCaseManifest,
        PublishedCaseStatus,
    )
    from piginot_campaigns.adapters import (
        DOGBONE,
        load_geometry_templates,
        load_problem,
    )
    from tests.compiler.config_fixtures import config_document

    document = load_problem("dogbone-v1").document()
    document["geometry"]["reference_mesh_id"] = load_geometry_templates("dogbone-v1")[
        1
    ].identity.reference_mesh_id
    document["split"] = {
        "pilot": 0,
        "train": 3,
        "development": 1,
        "holdout": 1,
        "reserve": 0,
    }
    problem = ProblemSpec.from_document(document)
    values = {p.name: (p.lo + p.hi) / 2 for p in problem.parameters}
    derived = {
        "arc_feasibility_ratio": 2
        * values["R_fillet"]
        / (values["W_grip"] - values["W_gauge"])
    }
    cases = tuple(
        PublishedCase(
            case_id,
            "group-" + case_id,
            EvidenceRole.TRAIN,
            values,
            derived,
            {"L_total": 0},
            PublishedCaseStatus.ACTIVE,
        )
        for case_id in ("fit-001", "fit-002", "screen-001")
    )
    manifest = PublishedCaseManifest(
        "export-fixed-weights", tuple(values), tuple(derived), ("L_total",), cases
    )
    source = tmp_path / "manifest.json"
    source.write_bytes(manifest.to_bytes())
    config = config_document()
    config["adapter"] = {
        "id": DOGBONE.adapter_id,
        "implementation_sha256": DOGBONE.descriptor.implementation_sha256,
    }
    config["source"]["path"] = source.name
    config["source"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    names = tuple(values)
    config["model"].update(
        parameter_names=list(names),
        hidden_dim=8,
        parameter_hidden_dim=8,
        latent_token_count=2,
        num_attention_heads=2,
        decoder_layers=1,
        positional_frequencies=1,
    )
    spec_path, config_path = tmp_path / "problem.json", tmp_path / "config.json"
    spec_path.write_text(json.dumps(document))
    config_path.write_text(json.dumps(config))
    CampaignConfig.load(config_path)
    plan = build_plan(
        spec_path, config_path, tmp_path / "output", descriptor=DOGBONE.descriptor
    )
    shape = asdict(plan.config.model)
    shape.pop("architecture_id")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9)
        model = EnergyFormOperator(EnergyOperatorConfig(**shape))
    state = {name: value.clone() for name, value in model.state_dict().items()}
    normalization = {
        "parameter_names": list(names),
        "fit_case_ids": ["fit-001", "fit-002"],
        "mean": list(values.values()),
        "standard_deviation": [1.0] * len(names),
    }
    binding = replace(
        _binding(),
        problem_spec_hash=problem.problem_hash,
        public_manifest_hash=manifest.content_hash,
        normalization_hash=digest(normalization),
        execution_profile=plan.config.execution.training,
        seed=plan.config.training.seed,
        architecture_id=model.architecture_id,
        architecture_module_digest=exporting.problem_energy_module_digest(
            model, problem.problem_hash
        ),
    )
    evidence = LiftingEvidence(
        problem.problem_hash, "quarter", 65, 0, (320, 4800), 0.0, None, None, {}, 1.0
    )
    checkpoint = plan.output_path / "checkpoints" / "step-000000004"
    checkpoint.mkdir(parents=True)

    def load(path, *, model: EnergyFormOperator, **kwargs):
        model.load_state_dict(state)
        return SimpleNamespace(
            manifest=SimpleNamespace(
                step=4, manifest_hash="e" * 64, model_state_hash=model_state_hash(model)
            )
        )

    monkeypatch.setattr(exporting, "load_training_checkpoint", load)
    monkeypatch.setattr(exporting, "resolve_adapter", lambda adapter_id: DOGBONE)
    return plan, {
        "checkpoint_path": checkpoint,
        "run_binding": binding,
        "normalization": normalization,
        "material_admission": {
            "status": "PASS",
            "all_eight_checks_complete": True,
            "evidence": {
                "problem_hash": problem.problem_hash,
                "model": problem.material.model,
                "passed": True,
            },
        },
        "lifting_verification": evidence.verification(problem),
        "lifting_evidence": evidence,
        "model": model,
    }


def test_export_roundtrip_is_unverified_and_resume_keeps_identical_bytes(
    public_context,
):
    from piginot.bundle import load_bundle

    plan, arguments = public_context
    result = exporting.export_candidate(plan, **arguments)
    path = Path(result["bundle_path"])
    before = {
        file.relative_to(path): file.read_bytes()
        for file in path.rglob("*")
        if file.is_file()
    }
    assert result["claim_level"] == "NOT_VERIFIED"
    assert load_bundle(path).model.problem.spec == plan.problem
    assert exporting.export_candidate(plan, **arguments) == result
    assert {
        file.relative_to(path): file.read_bytes()
        for file in path.rglob("*")
        if file.is_file()
    } == before


@pytest.mark.parametrize(
    "change", ["adapter", "state", "bundle", "unbound", "symlink", "timestamp"]
)
def test_export_refuses_corrupt_or_mixed_existing_candidate(
    public_context, monkeypatch, change
):
    plan, arguments = public_context
    if change == "adapter":
        monkeypatch.setattr(
            exporting,
            "resolve_adapter",
            lambda adapter_id: SimpleNamespace(descriptor=None),
        )
    else:
        result = exporting.export_candidate(plan, **arguments)
        path = Path(result["bundle_path"])
        if change == "state":
            (plan.output_path / "export-state.json").write_text("{}")
        elif change == "bundle":
            (path / "weights/model.safetensors").write_bytes(b"corrupt")
        elif change == "unbound":
            (plan.output_path / "export-state.json").unlink()
        elif change == "symlink":
            target = path.with_name("elsewhere")
            path.rename(target)
            path.symlink_to(target, target_is_directory=True)
        else:
            state_path = plan.output_path / "export-state.json"
            state = json.loads(state_path.read_bytes())
            state["payload"]["created_utc"] = "2020-01-01T00:00:00Z"
            state["content_sha256"] = digest(
                {key: state[key] for key in ("binding_sha256", "payload")}
            )
            state_path.write_text(json.dumps(state))
    with pytest.raises((ValueError, RuntimeError)):
        exporting.export_candidate(plan, **arguments)


@pytest.mark.parametrize("change", ["envelope", "name", "limitations"])
def test_export_refuses_self_consistent_changed_manifest(public_context, change):
    import rfc8785

    from piginot.bundle.manifest import MANIFEST_NAME, write_bundle
    from piginot.contracts import canonical_json_bytes

    plan, arguments = public_context
    result = exporting.export_candidate(plan, **arguments)
    directory = Path(result["bundle_path"])
    manifest = json.loads((directory / MANIFEST_NAME).read_bytes())
    files = {name: (directory / name).read_bytes() for name in manifest["files"]}
    if change == "envelope":
        document = json.loads(files["spec/envelope.json"])
        document["load_bounds"][0]["lower_bound"]["value"] += 0.01
        files["spec/envelope.json"] = rfc8785.dumps(document)
    elif change == "name":
        manifest["name"] = "another-candidate"
    else:
        document = json.loads(files["report/limitations.json"])
        document["items"][0]["statement"] = "different limitations"
        files["report/limitations.json"] = canonical_json_bytes(document)
    directory.rename(directory.with_name("original"))
    write_bundle(
        directory,
        files,
        name=manifest["name"],
        created_utc=manifest["created_utc"],
        claim_level=manifest["claim_level"],
        identities=manifest["identities"],
    )
    with pytest.raises(ValueError, match="differs from the bound export"):
        exporting.export_candidate(plan, **arguments)
