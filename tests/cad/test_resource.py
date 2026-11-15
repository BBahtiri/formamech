from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import rfc8785

from piginot.cad.resource import CadResourceError, load_geometry_resource
from piginot.contracts import canonical_json_bytes
from piginot.geometry import canonicalize_mesh
from tests.geometry.helpers import square_payload


def write_resource(root: Path) -> tuple[Path, str]:
    root.mkdir(exist_ok=True)
    source = b"ISO-10303-21; fixture; END-ISO-10303-21;"
    (root / "source.step").write_bytes(source)
    mesh = canonicalize_mesh(square_payload())
    payload = canonical_json_bytes(mesh)
    for level in (1, 2, 4):
        (root / f"mesh-{level}.json").write_bytes(payload)
    document = {
        "schema": "piginot.cad-geometry-v1",
        "source": {
            "path": "source.step",
            "sha256": hashlib.sha256(source).hexdigest(),
        },
        "meshes": {
            str(level): {
                "path": f"mesh-{level}.json",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "mesh_id": mesh.mesh_id,
            }
            for level in (1, 2, 4)
        },
        "boundary_kinds": {name: "line" for name in mesh.semantic_tags},
        "provenance": {"format": "step", "mesh_policy": "fixed-polygon"},
    }
    return save_resource(root / "geometry.json", document)


def save_resource(path: Path, document: dict) -> tuple[Path, str]:
    payload = canonical_json_bytes(document)
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def test_resource_binds_every_input_and_decodes_the_exact_mesh(tmp_path: Path) -> None:
    path, sha = write_resource(tmp_path)
    resource = load_geometry_resource(path, sha)
    assert resource.path == path.resolve()
    assert resource.sha256 == sha
    assert resource.mesh(1).mesh_id == resource.meshes[1].mesh_id
    assert [label for label, _ in resource.geometry_inputs] == [
        "geometry-resource",
        "geometry-source",
        "geometry-mesh-1",
        "geometry-mesh-2",
        "geometry-mesh-4",
    ]
    assert len({item for _, item in resource.geometry_inputs}) == 5
    assert resource.boundary_kinds["left"] == "line"
    with pytest.raises(TypeError):
        resource.boundary_kinds["left"] = "arc"


@pytest.mark.parametrize(
    "field", ["schema", "source", "meshes", "boundary_kinds", "provenance"]
)
def test_manifest_required_fields_are_not_optional(tmp_path: Path, field: str) -> None:
    path, _ = write_resource(tmp_path)
    document = json.loads(path.read_bytes())
    del document[field]
    path, sha = save_resource(path, document)
    with pytest.raises(CadResourceError):
        load_geometry_resource(path, sha)


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "other"},
        {"unexpected": True},
        {"source": []},
        {"meshes": {}},
        {"boundary_kinds": {"left": "line"}},
        {
            "boundary_kinds": {
                name: "spline" for name in ("left", "right", "top", "bottom")
            }
        },
        {"provenance": []},
    ],
)
def test_invalid_manifest_contracts_refuse(tmp_path: Path, change: dict) -> None:
    path, _ = write_resource(tmp_path)
    document = {**json.loads(path.read_bytes()), **change}
    path, sha = save_resource(path, document)
    with pytest.raises(CadResourceError):
        load_geometry_resource(path, sha)


@pytest.mark.parametrize(
    "name",
    [
        "../escape.step",
        "/tmp/source.step",
        "C:/source.step",
        "a\\source.step",
        "./source.step",
        "a/../source.step",
    ],
)
def test_member_paths_are_canonical_and_contained(tmp_path: Path, name: str) -> None:
    path, _ = write_resource(tmp_path)
    document = json.loads(path.read_bytes())
    document["source"]["path"] = name
    path, sha = save_resource(path, document)
    with pytest.raises(CadResourceError):
        load_geometry_resource(path, sha)


def test_symlinked_manifest_and_members_refuse(tmp_path: Path) -> None:
    path, sha = write_resource(tmp_path / "resource")
    link = tmp_path / "linked.json"
    try:
        link.symlink_to(path)
    except OSError:
        pytest.skip("OS cannot create symlinks")
    with pytest.raises(CadResourceError):
        load_geometry_resource(link, sha)
    source = path.parent / "source.step"
    payload = source.read_bytes()
    source.unlink()
    outside = tmp_path / "outside.step"
    outside.write_bytes(payload)
    source.symlink_to(outside)
    with pytest.raises(CadResourceError):
        load_geometry_resource(path, sha)


@pytest.mark.parametrize(
    "member",
    ["geometry.json", "source.step", "mesh-1.json", "mesh-2.json", "mesh-4.json"],
)
def test_changed_bytes_are_never_reused(tmp_path: Path, member: str) -> None:
    path, sha = write_resource(tmp_path)
    resource = load_geometry_resource(path, sha)
    (tmp_path / member).write_bytes(b"changed")
    with pytest.raises(CadResourceError):
        resource.mesh(1)


def test_duplicate_paths_and_wrong_mesh_identity_refuse(tmp_path: Path) -> None:
    path, _ = write_resource(tmp_path)
    original = json.loads(path.read_bytes())
    duplicate = {
        **original,
        "meshes": {**original["meshes"], "2": original["meshes"]["1"]},
    }
    path, sha = save_resource(path, duplicate)
    with pytest.raises(CadResourceError):
        load_geometry_resource(path, sha)
    original["meshes"]["1"]["mesh_id"] = "a" * 64
    path, sha = save_resource(path, original)
    with pytest.raises(CadResourceError):
        load_geometry_resource(path, sha)


def test_mesh_bytes_must_be_canonical_and_structurally_bound(tmp_path: Path) -> None:
    path, _ = write_resource(tmp_path)
    document = json.loads(path.read_bytes())
    mesh_path = tmp_path / "mesh-1.json"
    mesh = json.loads(mesh_path.read_bytes())
    mesh["mesh_id"] = "b" * 64
    payload = rfc8785.dumps(mesh)
    mesh_path.write_bytes(payload)
    document["meshes"]["1"] = {
        **document["meshes"]["1"],
        "sha256": hashlib.sha256(payload).hexdigest(),
        "mesh_id": "b" * 64,
    }
    path, sha = save_resource(path, document)
    with pytest.raises(CadResourceError):
        load_geometry_resource(path, sha)


def test_wrong_expected_digest_invalid_level_and_missing_file_refuse(
    tmp_path: Path,
) -> None:
    path, sha = write_resource(tmp_path)
    with pytest.raises(CadResourceError):
        load_geometry_resource(path, "bad")
    resource = load_geometry_resource(path, sha)
    for level in (True, 0, 3, "1"):
        with pytest.raises(CadResourceError):
            resource.mesh(level)
    (tmp_path / "mesh-2.json").unlink()
    with pytest.raises(CadResourceError):
        load_geometry_resource(path, sha)


def test_resource_planning_imports_no_numerical_stack(tmp_path: Path) -> None:
    path, sha = write_resource(tmp_path)
    source = """
import importlib.abc,sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy','torch','scipy','sympy','skfem','gmsh','ezdxf'}:
            raise AssertionError('numerical import: '+fullname)
sys.meta_path.insert(0,Deny())
from piginot.cad.resource import load_geometry_resource
from pathlib import Path
r=load_geometry_resource(Path(sys.argv[1]),sys.argv[2])
assert len(r.geometry_inputs)==5
"""
    completed = subprocess.run(
        [sys.executable, "-c", source, str(path), sha],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert completed.returncode == 0, completed.stderr


def test_nontext_paths_and_invalid_json_refuse(tmp_path: Path) -> None:
    path, _ = write_resource(tmp_path)
    document = json.loads(path.read_bytes())
    document["source"]["path"] = None
    path, sha = save_resource(path, document)
    with pytest.raises(CadResourceError, match="path must be text"):
        load_geometry_resource(path, sha)
    payload = b'{"schema": invalid}'
    path.write_bytes(payload)
    with pytest.raises(CadResourceError, match="invalid prepared CAD"):
        load_geometry_resource(path, hashlib.sha256(payload).hexdigest())


def test_nested_provenance_is_frozen_without_changing_source_bytes(
    tmp_path: Path,
) -> None:
    path, _ = write_resource(tmp_path)
    document = json.loads(path.read_bytes())
    document["provenance"]["profile"] = [{"values": [True, None, 1.5]}]
    path, sha = save_resource(path, document)
    resource = load_geometry_resource(path, sha)
    assert resource.provenance["profile"][0]["values"] == (True, None, 1.5)
    with pytest.raises(TypeError):
        resource.provenance["profile"][0]["values"] = ()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha


@pytest.mark.parametrize(
    "change", [{"$type": "unknown"}, {"cell_type": "quad"}, {"cells": [[0, 2, 3, 1]]}]
)
def test_noncanonical_mesh_types_and_cell_shapes_refuse(
    tmp_path: Path, change: dict
) -> None:
    path, _ = write_resource(tmp_path)
    document = json.loads(path.read_bytes())
    target = tmp_path / "mesh-1.json"
    mesh = {**json.loads(target.read_bytes()), **change}
    payload = rfc8785.dumps(mesh)
    target.write_bytes(payload)
    document["meshes"]["1"]["sha256"] = hashlib.sha256(payload).hexdigest()
    path, sha = save_resource(path, document)
    with pytest.raises(CadResourceError):
        load_geometry_resource(path, sha)
