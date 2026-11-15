"""Portable exports retain exactly bound CAD preparation evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from piginot.bundle import load_bundle
from piginot.bundle.manifest import MANIFEST_NAME, write_bundle
from piginot.cad.adapter import FIXED_CAD
from piginot.cad.configuration import prepare_campaign
from piginot.cad.mesh_policy import MeshPolicy
from piginot.cad.refinement import refine_fixed_mesh
from piginot.cad.resource import load_geometry_resource
from piginot.compiler import exporting
from piginot.compiler.execution_records import digest
from piginot.compiler.planning import build_plan
from piginot.contracts import canonical_json_bytes
from piginot.operators import EnergyFormOperator, EnergyOperatorConfig
from piginot.problem.verification import LiftingEvidence
from tests.cad.test_configuration import save_settings, settings
from tests.cad.test_resource import write_resource
from tests.compiler.test_exporting import public_context as public_context
from tests.training.test_checkpoint import _binding


@pytest.fixture(params=("sized", "adapted", "legacy"))
def cad_context(tmp_path, monkeypatch, request):
    path, sha = write_resource(tmp_path / "geometry")
    resource = load_geometry_resource(path, sha)
    base = resource.mesh(1)
    document = json.loads(path.read_bytes())
    for level in (1, 2, 4):
        mesh = refine_fixed_mesh(base, level)
        payload = canonical_json_bytes(mesh)
        (path.parent / f"mesh-{level}.json").write_bytes(payload)
        document["meshes"][str(level)] = {
            "path": f"mesh-{level}.json",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "mesh_id": mesh.mesh_id,
        }
    metadata = {
        "mesh_policy": MeshPolicy(
            1.0, 0.1, 0.01, 24, (), (), 20.0, 5.0, 1000, 10000
        ).document(),
        "mesh_evidence": {
            "geometry_status": "CHECKED_LINEAR",
            "maximum_curve_deviation_mm": 0.0,
            "base_triangles": 2,
            "reference_triangles": 32,
            "reference_p2_vector_dofs": 162,
            "claim_level": "NOT_VERIFIED",
        },
    }
    if request.param == "adapted":
        report = {
            "status": "ACCEPTED",
            "attempts": [{"mesh_id": base.mesh_id, "indicator": 0.0}],
            "new_optimizer_steps": 0,
        }
        metadata["adaptation"] = {
            "sha256": hashlib.sha256(canonical_json_bytes(report)).hexdigest(),
            "report": report,
        }
    if request.param == "legacy":
        metadata = {}
    document["provenance"].update(metadata)
    path.write_bytes(canonical_json_bytes(document))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    adapter = FIXED_CAD.bind_resource(path, sha)
    monkeypatch.setattr(
        "piginot.compiler.adapters.entry_points",
        lambda **kwargs: (
            SimpleNamespace(name="cad-fixed-v1", load=lambda: FIXED_CAD),
        ),
    )
    physics = settings()
    physics["qois"] = [{**physics["qois"][0], "exclude_regions": []}]
    physics["model"] = {
        "hidden_dim": 8,
        "parameter_hidden_dim": 8,
        "latent_token_count": 2,
        "num_attention_heads": 2,
        "decoder_layers": 1,
        "positional_frequencies": 1,
    }
    settings_path = save_settings(tmp_path / "physics.json", physics)
    inputs = tmp_path / "inputs"
    prepare_campaign(path.parent, settings_path, inputs)
    plan = build_plan(
        inputs / "problem.json",
        inputs / "config.json",
        tmp_path / "run",
        descriptor=adapter.descriptor,
    )
    shape = asdict(plan.config.model)
    shape.pop("architecture_id")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9)
        model = EnergyFormOperator(EnergyOperatorConfig(**shape))
    state = {name: value.clone() for name, value in model.state_dict().items()}
    normalization = {
        "parameter_names": list(plan.config.model.parameter_names),
        "fit_case_ids": list(plan.config.source.fit_case_ids),
        "mean": [(p.lo + p.hi) / 2 for p in plan.problem.parameters],
        "standard_deviation": [1.0] * len(plan.problem.parameters),
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
        "adf",
        65,
        0,
        (320, 4800),
        0.0,
        0.0,
        0.0,
        {320: 0.0, 4800: 0.0},
        1.0,
    )
    checkpoint = plan.output_path / "checkpoints" / "step-000000001"
    checkpoint.mkdir(parents=True)

    def load(path, *, model, **kwargs):
        assert path == checkpoint
        model.load_state_dict(state)
        return SimpleNamespace(manifest=SimpleNamespace(step=1, manifest_hash="e" * 64))

    def no_optimizer_step(*args, **kwargs):
        pytest.fail("candidate export must not perform optimizer steps")

    monkeypatch.setattr(exporting, "load_training_checkpoint", load)
    monkeypatch.setattr(torch.optim.Adam, "step", no_optimizer_step)
    arguments = {
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
    expected = (
        {
            "geometry_resource_sha256": sha,
            "source_sha256": resource.source_sha256,
            **metadata,
        }
        if metadata
        else None
    )
    return plan, arguments, expected, path


def _files(path):
    return {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def test_cad_export_embeds_preparation_without_expanding_bundle_or_claim(cad_context):
    plan, arguments, expected, _ = cad_context
    result = exporting.export_candidate(plan, **arguments)
    path = Path(result["bundle_path"])
    bundle = load_bundle(path)
    assert bundle.verification.get("geometry_preparation") == expected
    assert bundle.manifest.claim_level == "NOT_VERIFIED"
    assert bundle.verification["verdict"] == "NOT_VERIFIED"
    assert len(_files(path)) == 8
    before = _files(path)
    assert exporting.export_candidate(plan, **arguments) == result
    assert _files(path) == before


def test_cad_preparation_survives_fresh_runtime_without_cad_or_fem(cad_context):
    plan, arguments, expected, _ = cad_context
    result = exporting.export_candidate(plan, **arguments)
    root = Path(__file__).resolve().parents[2]
    script = f"""
import importlib.abc
import json
import sys
from pathlib import Path
sys.path.insert(0, {str(root / "src")!r})
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'gmsh', 'ezdxf', 'scipy', 'skfem'}}:
            raise AssertionError('CAD/FEM import during portable replay: ' + fullname)
sys.meta_path.insert(0, Deny())
from piginot.bundle import load_bundle
bundle = load_bundle(Path({result["bundle_path"]!r}))
assert bundle.manifest.claim_level == 'NOT_VERIFIED'
print(json.dumps(bundle.verification.get('geometry_preparation'), sort_keys=True))
"""
    checked = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        cwd=Path(result["bundle_path"]).parent,
    )
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout) == expected


def test_changed_preparation_is_refused_without_altering_the_export(cad_context):
    plan, arguments, _, resource_path = cad_context
    result = exporting.export_candidate(plan, **arguments)
    path = Path(result["bundle_path"])
    before = _files(path)
    document = json.loads(resource_path.read_bytes())
    document["provenance"]["domain"] = "changed-input"
    resource_path.write_bytes(canonical_json_bytes(document))
    with pytest.raises(ValueError, match="geometry-resource changed"):
        exporting.export_candidate(plan, **arguments)
    assert _files(path) == before


@pytest.mark.parametrize("cad_context", ["sized"], indirect=True)
def test_structured_mesh_policy_requires_achieved_evidence(cad_context):
    plan, arguments, _, resource_path = cad_context
    document = json.loads(resource_path.read_bytes())
    document["provenance"].pop("mesh_evidence")
    resource_path.write_bytes(canonical_json_bytes(document))
    sha = hashlib.sha256(resource_path.read_bytes()).hexdigest()
    config_path = plan.input_paths[1]
    config = json.loads(config_path.read_bytes())
    config["adapter"]["geometry_resource"]["sha256"] = sha
    config_path.write_bytes(canonical_json_bytes(config))
    adapter = FIXED_CAD.bind_resource(resource_path, sha)
    updated = build_plan(
        plan.input_paths[0],
        config_path,
        plan.output_path,
        descriptor=adapter.descriptor,
    )
    with pytest.raises(ValueError, match="no achieved mesh evidence"):
        exporting.export_candidate(updated, **arguments)
    assert not (plan.output_path / "candidate-bundle").exists()


@pytest.mark.parametrize("cad_context", ["sized", "adapted"], indirect=True)
def test_self_consistent_changed_bundle_preparation_does_not_match_bound_export(
    cad_context,
):
    plan, arguments, expected, _ = cad_context
    result = exporting.export_candidate(plan, **arguments)
    directory = Path(result["bundle_path"])
    manifest = json.loads((directory / MANIFEST_NAME).read_bytes())
    files = {name: (directory / name).read_bytes() for name in manifest["files"]}
    verification = json.loads(files["report/verification.json"])
    verification["geometry_preparation"]["mesh_evidence"]["base_triangles"] = 999
    files["report/verification.json"] = canonical_json_bytes(verification)
    directory.rename(directory.with_name("original"))
    write_bundle(
        directory,
        files,
        name=manifest["name"],
        created_utc=manifest["created_utc"],
        claim_level=manifest["claim_level"],
        identities=manifest["identities"],
    )
    assert load_bundle(directory).verification["geometry_preparation"] != expected
    with pytest.raises(ValueError, match="differs from the bound export"):
        exporting.export_candidate(plan, **arguments)


def test_specimen_exports_retain_the_legacy_verification_vocabulary(public_context):
    plan, arguments = public_context
    result = exporting.export_candidate(plan, **arguments)
    assert set(load_bundle(Path(result["bundle_path"])).verification) == {
        "gate_id",
        "verdict",
        "claim_level",
        "material_admission",
        "lifting_evidence",
        "outstanding_work",
    }
