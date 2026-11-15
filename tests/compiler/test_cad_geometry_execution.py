"""Prepared CAD inputs must pass the real compiler lifting handoff."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from piginot.cad.adapter import FIXED_CAD
from piginot.cad.configuration import prepare_campaign
from piginot.cad.refinement import refine_fixed_mesh
from piginot.cad.resource import load_geometry_resource
from piginot.compiler.execution_numerics import NumericalOperations
from piginot.compiler.planning import build_plan
from piginot.contracts import canonical_json_bytes
from tests.cad.test_configuration import save_settings, settings
from tests.cad.test_resource import write_resource


@pytest.fixture
def prepared_cad(tmp_path, monkeypatch):
    resource, digest = write_resource(tmp_path / "geometry")
    document = json.loads(resource.read_bytes())
    base = load_geometry_resource(resource, digest).mesh(1)
    meshes = {}
    for level in (1, 2, 4):
        mesh = refine_fixed_mesh(base, level)
        payload = canonical_json_bytes(mesh)
        (resource.parent / f"mesh-{level}.json").write_bytes(payload)
        meshes[str(level)] = {
            "path": f"mesh-{level}.json",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "mesh_id": mesh.mesh_id,
        }
    payload = canonical_json_bytes({**document, "meshes": meshes})
    resource.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    physics = settings()
    physics = {
        **physics,
        "qois": [{**physics["qois"][0], "exclude_regions": []}],
    }
    settings_path = save_settings(tmp_path / "physics.json", physics)
    inputs = tmp_path / "inputs"
    prepare_campaign(resource.parent, settings_path, inputs)
    adapter = FIXED_CAD.bind_resource(resource, digest)
    monkeypatch.setattr(
        "piginot.compiler.adapters.entry_points",
        lambda **kwargs: (
            SimpleNamespace(name="cad-fixed-v1", load=lambda: FIXED_CAD),
        ),
    )
    plan = build_plan(
        inputs / "problem.json",
        inputs / "config.json",
        tmp_path / "run",
        descriptor=adapter.descriptor,
    )
    return plan, adapter


def test_prepared_cad_lifting_keeps_constants_out_of_adapter_parameters(prepared_cad):
    plan, adapter = prepared_cad
    assert {constant.name for constant in plan.problem.constants} == {
        "cad_origin_x",
        "cad_origin_y",
        "cad_extent_x",
        "cad_extent_y",
    }

    result = NumericalOperations(plan, adapter, "a" * 40).verify_geometry()

    assert result["status"] == "PASS"
    assert result["refused_geometries"] == []
    assert (
        result["evidence"]["parameter_vectors"] == result["requested_parameter_vectors"]
    )
    assert result["verification"]["full_parameter_range_verified"] is True


@pytest.mark.parametrize("level", [2, 4])
def test_registered_cad_refinements_validate_without_changing_the_problem(
    prepared_cad, level
):
    plan, adapter = prepared_cad
    spec = plan.problem
    before = canonical_json_bytes(spec.document())
    values = {p.name: (p.lo + p.hi) / 2 for p in spec.parameters}

    mesh = adapter.geometry_values(spec, values, level)

    assert mesh.mesh_id == adapter.resource.meshes[level].mesh_id
    assert mesh.mesh_id != spec.geometry.reference_mesh_id
    assert len(mesh.cells) == 2 * level**2
    assert canonical_json_bytes(spec.document()) == before
    with pytest.raises(ValueError, match="mesh-id"):
        spec.check_mesh(mesh)


def test_cad_refinement_still_requires_the_declared_base_identity(prepared_cad):
    plan, adapter = prepared_cad
    spec = replace(
        plan.problem,
        geometry=replace(plan.problem.geometry, reference_mesh_id="a" * 64),
    )
    values = {p.name: (p.lo + p.hi) / 2 for p in spec.parameters}
    with pytest.raises(ValueError, match="differs from the prepared CAD resource"):
        adapter.geometry_values(spec, values, 2)
