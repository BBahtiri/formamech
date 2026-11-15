from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import rfc8785

from piginot.cad.adapter import FIXED_CAD, implementation_sha256, morph_map_sha256
from piginot.compiler import adapters
from piginot.compiler.adapters import (
    AdapterError,
    resolve_adapter,
    resolve_configured_adapter,
)
from piginot.compiler.config import (
    CampaignConfig,
    CompilerConfigError,
    GeometryResourceRef,
)
from piginot.compiler.config_types import AdapterRef
from piginot.contracts import canonical_json_bytes
from tests.cad.test_resource import save_resource, write_resource
from tests.compiler.config_fixtures import config_document

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def bound(tmp_path: Path, monkeypatch):
    path, sha = write_resource(tmp_path)
    entry = SimpleNamespace(name="cad-fixed-v1", load=lambda: FIXED_CAD)
    monkeypatch.setattr(adapters, "entry_points", lambda **kwargs: (entry,))
    reference = AdapterRef(
        "cad-fixed-v1",
        implementation_sha256(),
        GeometryResourceRef("geometry.json", sha),
    )
    adapter = resolve_configured_adapter(reference, tmp_path / "config.json")
    return adapter, reference


def spec_for(adapter):
    return SimpleNamespace(
        geometry=SimpleNamespace(
            reference_mesh_id=adapter.resource.meshes[1].mesh_id,
            morph_map_sha256=morph_map_sha256(),
        ),
        parameters=(SimpleNamespace(name="load", role="load", lo=0.01, hi=0.02),),
        lifting=SimpleNamespace(kind="adf"),
        boundary_conditions=(
            SimpleNamespace(kind="dirichlet", region="left"),
            SimpleNamespace(kind="traction_free", region="right"),
        ),
        check_mesh=lambda mesh: None,
    )


@pytest.mark.parametrize("adapter_id", ["dogbone-v1", "notched-v1", "unit-adapter"])
def test_optional_geometry_reference_roundtrip_and_legacy_bytes(
    tmp_path: Path, adapter_id: str
) -> None:
    document = config_document()
    document["adapter"]["id"] = adapter_id
    config = CampaignConfig.from_document(document)
    assert "geometry_resource" not in config.document()["adapter"]
    legacy_path = tmp_path / "legacy-config.json"
    legacy_path.write_text(json.dumps(document))
    assert CampaignConfig.load(legacy_path).document() == config.document()
    path, sha = write_resource(tmp_path)
    current = {
        **document,
        "adapter": {
            "id": "cad-fixed-v1",
            "implementation_sha256": implementation_sha256(),
            "geometry_resource": {"path": path.name, "sha256": sha},
        },
    }
    configured = CampaignConfig.from_document(current)
    assert configured.adapter.geometry_resource == GeometryResourceRef(path.name, sha)
    assert configured.document()["adapter"] == current["adapter"]
    assert configured == CampaignConfig.from_document(configured.document())


@pytest.mark.parametrize(
    "value",
    [
        None,
        "geometry.json",
        {},
        {"path": "geometry.json", "sha256": "bad"},
        {"path": "", "sha256": "a" * 64},
        {"path": "geometry.json", "sha256": "a" * 64, "extra": 1},
    ],
)
def test_invalid_or_incomplete_reference_refuses(value) -> None:
    document = config_document()
    document["adapter"] = {**document["adapter"], "geometry_resource": value}
    with pytest.raises(CompilerConfigError):
        CampaignConfig.from_document(document)


def test_reference_constructor_rejects_an_untyped_resource() -> None:
    with pytest.raises(CompilerConfigError):
        AdapterRef("cad-fixed-v1", "a" * 64, {})


def test_legacy_resolution_keeps_the_existing_single_argument(
    monkeypatch, tmp_path: Path
) -> None:
    expected = object()
    calls = []
    monkeypatch.setattr(
        adapters, "resolve_adapter", lambda name: calls.append(name) or expected
    )
    assert (
        resolve_configured_adapter(
            AdapterRef("old-adapter", "a" * 64), tmp_path / "config.json"
        )
        is expected
    )
    assert calls == ["old-adapter"]


def test_bound_descriptor_and_every_input_hash_match(bound) -> None:
    adapter, reference = bound
    descriptor = adapter.descriptor
    assert descriptor.adapter_id == reference.id
    assert descriptor.implementation_sha256 == reference.implementation_sha256
    assert (
        descriptor.geometries[0].reference_mesh_id == adapter.resource.meshes[1].mesh_id
    )
    assert descriptor.geometries[0].morph_map_sha256 == morph_map_sha256()
    assert descriptor.liftings == ("adf",)
    assert len(adapter.geometry_inputs) == 5
    for (label, path), (expected_label, sha) in zip(
        adapter.geometry_inputs, adapter.geometry_input_sha256, strict=True
    ):
        assert label == expected_label
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha


def test_absolute_reference_is_explicit_not_cwd_dependent(
    bound, tmp_path: Path
) -> None:
    adapter, reference = bound
    absolute = AdapterRef(
        reference.id,
        reference.implementation_sha256,
        GeometryResourceRef(str(adapter.resource.path), adapter.resource.sha256),
    )
    assert (
        resolve_configured_adapter(
            absolute, tmp_path / "elsewhere/config.json"
        ).resource.path
        == adapter.resource.path
    )


def test_unbound_provider_incomplete_binding_and_plain_adapter_refuse(
    bound, monkeypatch
) -> None:
    adapter, _ = bound
    with pytest.raises(AdapterError, match="requires a bound"):
        resolve_adapter("cad-fixed-v1")
    with pytest.raises(AdapterError):
        resolve_adapter("cad-fixed-v1", geometry_sha256="a" * 64)
    with pytest.raises(AdapterError):
        resolve_adapter("cad-fixed-v1", geometry_resource=adapter.resource.path)
    with pytest.raises(AdapterError):
        resolve_adapter(
            "cad-fixed-v1",
            geometry_resource=adapter.resource.path,
            geometry_sha256="bad",
        )
    with pytest.raises(AdapterError, match="bytes differ"):
        resolve_adapter(
            "cad-fixed-v1",
            geometry_resource=adapter.resource.path,
            geometry_sha256="a" * 64,
        )
    monkeypatch.setattr(
        adapters,
        "entry_points",
        lambda **kwargs: (SimpleNamespace(name="cad-fixed-v1", load=lambda: adapter),),
    )
    with pytest.raises(AdapterError, match="cannot bind"):
        resolve_adapter(
            "cad-fixed-v1",
            geometry_resource=adapter.resource.path,
            geometry_sha256=adapter.resource.sha256,
        )


def test_fixed_geometry_all_levels_export_and_case_forwarding(bound) -> None:
    adapter, _ = bound
    spec = spec_for(adapter)
    for level in (1, 2, 4):
        mesh = adapter.geometry_values(spec, {"load": 0.015}, level)
        assert mesh.mesh_id == adapter.resource.meshes[level].mesh_id
    assert (
        adapter.geometry(
            spec, SimpleNamespace(primitive_coordinates={"load": 0.015})
        ).mesh_id
        == adapter.resource.meshes[1].mesh_id
    )
    mesh, morph = adapter.export_geometry(spec)
    assert mesh.mesh_id == adapter.resource.meshes[1].mesh_id
    assert morph.parameter_names == ()
    assert hashlib.sha256(canonical_json_bytes(morph)).hexdigest() == morph_map_sha256()
    assert adapter.validity_criteria().jacobian_margin == 0


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"load": 0.015, "other": 1},
        {"load": True},
        {"load": "0.015"},
        {"load": float("nan")},
        {"load": float("inf")},
        {"load": 0.1},
    ],
)
def test_unknown_or_outside_parameters_refuse(bound, values) -> None:
    adapter, _ = bound
    with pytest.raises(ValueError):
        adapter.geometry_values(spec_for(adapter), values)


@pytest.mark.parametrize(
    "change",
    ["mesh", "morph", "geometry-parameter", "lifting", "neumann", "curved-essential"],
)
def test_unsupported_geometry_and_boundary_physics_refuse(bound, change) -> None:
    adapter, _ = bound
    spec = spec_for(adapter)
    if change == "mesh":
        spec.geometry.reference_mesh_id = "a" * 64
    elif change == "morph":
        spec.geometry.morph_map_sha256 = "a" * 64
    elif change == "geometry-parameter":
        spec.parameters[0].role = "geometry"
    elif change == "lifting":
        spec.lifting.kind = "quarter"
    elif change == "neumann":
        spec.boundary_conditions[0].kind = "neumann"
    else:
        spec.boundary_conditions[0].region = "missing"
    with pytest.raises(ValueError):
        adapter.geometry_values(spec, {"load": 0.015})


def test_execution_forwards_only_the_explicit_resume_flag(bound, monkeypatch) -> None:
    from piginot.compiler import execution

    adapter, _ = bound
    calls = []
    monkeypatch.setattr(
        execution,
        "execute_campaign",
        lambda plan, chosen, *, resume: calls.append((plan, chosen, resume)) or 3,
    )
    marker = object()
    assert adapter.execute(marker, resume=True) == 3
    assert calls == [(marker, adapter, True)]


def test_provider_and_resolver_work_with_numerical_imports_denied(
    tmp_path: Path,
) -> None:
    path, sha = write_resource(tmp_path)
    source = """
import importlib.abc,sys
from pathlib import Path
from types import SimpleNamespace
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy','torch','scipy','sympy','skfem','gmsh','ezdxf'}:
            raise AssertionError('numerical import: '+fullname)
sys.meta_path.insert(0,Deny())
from piginot.cad.adapter import FIXED_CAD,implementation_sha256
from piginot.compiler import adapters
from piginot.compiler.config_types import AdapterRef,GeometryResourceRef
adapters.entry_points=lambda **kwargs:(SimpleNamespace(name='cad-fixed-v1',load=lambda:FIXED_CAD),)
ref=AdapterRef('cad-fixed-v1',implementation_sha256(),GeometryResourceRef(sys.argv[1],sys.argv[2]))
bound=adapters.resolve_configured_adapter(ref,Path('ignored.json'))
assert len(bound.geometry_inputs)==5
assert bound.descriptor.adapter_id=='cad-fixed-v1'
"""
    completed = subprocess.run(
        [sys.executable, "-c", source, str(path), sha],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert completed.returncode == 0, completed.stderr


def test_geometry_validation_rejects_false_topology_metadata(tmp_path: Path) -> None:
    path, _ = write_resource(tmp_path)
    document = json.loads(path.read_bytes())
    target = tmp_path / "mesh-1.json"
    mesh = json.loads(target.read_bytes())
    mesh["topology_signature"] = "false-topology"
    payload = rfc8785.dumps(mesh)
    target.write_bytes(payload)
    document["meshes"]["1"]["sha256"] = hashlib.sha256(payload).hexdigest()
    path, sha = save_resource(path, document)
    adapter = FIXED_CAD.bind_resource(path, sha)
    with pytest.raises(ValueError, match="validated geometry identity"):
        adapter.geometry_values(spec_for(adapter), {"load": 0.015})
