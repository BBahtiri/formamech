"""CAD projects publish complete geometry bindings and inspect them as data."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import piginot
from piginot.cad import project
from piginot.cad.resource import CadResourceError, load_geometry_resource
from piginot.contracts import Unit, canonical_json_bytes
from piginot.geometry import CellType, MeshPayload, canonicalize_mesh


def _mesh():
    return canonicalize_mesh(
        MeshPayload(
            nodes=((2.0, 3.0), (6.0, 3.0), (6.0, 5.0), (2.0, 5.0)),
            cells=((0, 1, 2), (0, 2, 3)),
            cell_type=CellType.TRIANGLE,
            boundary_facets=((0, 1), (1, 2), (2, 3), (3, 0)),
            boundary_tags={"bottom": (0,), "right": (1,), "top": (2,), "left": (3,)},
            region_ids=(1, 1),
            unit=Unit.MM,
            source_provenance={"fixture": "own-offset-profile"},
        )
    )


@pytest.fixture
def imported(tmp_path, monkeypatch):
    source = tmp_path / "profile.StEp"
    source.write_bytes(b"fixed original CAD bytes\r\n")
    result = SimpleNamespace(
        mesh=_mesh(),
        boundaries=tuple(
            SimpleNamespace(name=name, kind="line", source_id=index, source_label=name)
            for index, name in enumerate(("bottom", "right", "top", "left"), 1)
        ),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_format="step",
        provenance={
            "backend": "test-only",
            "backend_version": "1.0",
            "input_unit": "mm",
        },
    )
    calls = []

    def importer(path, *, unit, mesh_size_mm):
        calls.append((path, unit, mesh_size_mm))
        return result

    monkeypatch.setattr(project, "_import_cad", importer)
    return source, result, calls


def _prepare(imported, tmp_path):
    source, _, _ = imported
    output = tmp_path / "project"
    result = project.prepare_geometry(source, output, unit=Unit.MM, mesh_size_mm=1.5)
    return output, result


def test_project_preserves_original_and_publishes_three_exact_canonical_meshes(
    imported, tmp_path
):
    source, original, calls = imported
    output, result = _prepare(imported, tmp_path)
    assert calls == [(source.resolve(), Unit.MM, 1.5)]
    assert (output / "source.step").read_bytes() == source.read_bytes()
    manifest = json.loads((output / "geometry.json").read_bytes())
    assert set(manifest) == {
        "schema",
        "source",
        "meshes",
        "boundary_kinds",
        "provenance",
    }
    assert manifest["schema"] == "piginot.cad-geometry-v1"
    assert manifest["source"] == {
        "path": "source.step",
        "sha256": original.source_sha256,
    }
    resource = load_geometry_resource(
        output / "geometry.json",
        hashlib.sha256((output / "geometry.json").read_bytes()).hexdigest(),
    )
    for level in (1, 2, 4):
        mesh = resource.mesh(level)
        assert len(mesh.cells) == 2 * level * level
        assert (output / f"mesh-{level}.json").read_bytes() == canonical_json_bytes(
            mesh
        )
        assert set(mesh.semantic_tags) == {
            boundary.name for boundary in original.boundaries
        }
    assert resource.provenance["domain"] == "fixed-polygonal-approximation"
    assert resource.provenance["refinement"] == "midpoint"
    assert resource.provenance["geometry_accuracy_bound"] == "NOT_ESTABLISHED"
    assert resource.provenance["unsupported_essential_boundary_kinds"] == (
        "arc",
        "other",
    )
    assert result["status"] == "IMPORTED"
    assert result["claim_level"] == "NOT_VERIFIED"
    assert result["geometry_resource_sha256"] == resource.sha256


def test_inspection_derives_boundaries_and_coordinates_from_verified_members(
    imported, tmp_path
):
    output, _ = _prepare(imported, tmp_path)
    preview = project.inspect_geometry(output)
    assert preview == json.loads((output / "preview.json").read_bytes())
    assert preview["bounding_box_mm"] == {"min": [2.0, 3.0], "max": [6.0, 5.0]}
    assert preview["mesh"]["node_count"] == 4
    assert preview["mesh"]["triangle_count"] == 2
    assert preview["source"]["format"] == "step"
    assert preview["units"] == "mm"
    assert {entry["name"] for entry in preview["boundaries"]} == {
        "bottom",
        "right",
        "top",
        "left",
    }
    left = next(entry for entry in preview["boundaries"] if entry["name"] == "left")
    assert left["source_id"] == 4 and left["source_label"] == "left"
    assert left["segments_mm"] == [[[2.0, 5.0], [2.0, 3.0]]]
    assert preview["provenance"]["geometry_accuracy_bound"] == "NOT_ESTABLISHED"
    assert preview["claim_level"] == "NOT_VERIFIED"
    (output / "preview.json").write_text('{"untrusted":"edited preview"}')
    assert project.inspect_geometry(output) == preview


def test_project_can_be_relocated_without_source_checkout(imported, tmp_path):
    output, _ = _prepare(imported, tmp_path)
    destination = tmp_path / "relocated"
    shutil.copytree(output, destination)
    assert project.inspect_geometry(destination) == project.inspect_geometry(output)


def test_import_backend_is_resolved_only_when_requested(monkeypatch, tmp_path):
    source = tmp_path / "profile.dxf"
    observed = []
    module = ModuleType("piginot.cad.importing")

    def importer(path, *, unit, mesh_size_mm):
        observed.append((path, unit, mesh_size_mm))
        return "backend-result"

    module.import_cad = importer
    monkeypatch.setitem(sys.modules, "piginot.cad.importing", module)
    assert project._import_cad(source, unit=Unit.CM, mesh_size_mm=2) == "backend-result"
    assert observed == [(source, Unit.CM, 2)]


@pytest.mark.parametrize("invalid", ([], {"left": []}))
def test_malformed_informational_boundary_sources_are_refused(
    imported, tmp_path, invalid
):
    output, _ = _prepare(imported, tmp_path)
    path = output / "geometry.json"
    document = json.loads(path.read_bytes())
    document["provenance"]["boundary_sources"] = invalid
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="boundary source metadata"):
        project.inspect_geometry(output)


def test_older_resources_without_optional_source_labels_remain_inspectable(
    imported, tmp_path
):
    output, _ = _prepare(imported, tmp_path)
    path = output / "geometry.json"
    document = json.loads(path.read_bytes())
    document["provenance"].pop("boundary_sources")
    path.write_text(json.dumps(document))
    assert all(
        entry["source_id"] is None
        for entry in project.inspect_geometry(output)["boundaries"]
    )


@pytest.mark.parametrize("level", (1, 2, 4))
def test_inspection_revalidates_even_unselected_reference_meshes(
    imported, tmp_path, level
):
    output, _ = _prepare(imported, tmp_path)
    (output / f"mesh-{level}.json").write_bytes(b"changed")
    with pytest.raises(CadResourceError, match="bytes differ"):
        project.inspect_geometry(output)


def test_source_copy_tampering_is_refused_by_inspection(imported, tmp_path):
    output, _ = _prepare(imported, tmp_path)
    (output / "source.step").write_bytes(b"changed")
    with pytest.raises(CadResourceError, match="bytes differ"):
        project.inspect_geometry(output)


@pytest.mark.parametrize(
    "mesh_size", (True, False, 0, -1, float("nan"), float("inf"), "2", None)
)
def test_invalid_mesh_size_refuses_before_import(imported, tmp_path, mesh_size):
    source, _, calls = imported
    with pytest.raises(ValueError, match="mesh size"):
        project.prepare_geometry(
            source, tmp_path / "project", unit=Unit.MM, mesh_size_mm=mesh_size
        )
    assert not calls and not (tmp_path / "project").exists()


@pytest.mark.parametrize("unit", (Unit.N, "mm", None))
def test_invalid_source_unit_refuses_before_import(imported, tmp_path, unit):
    source, _, calls = imported
    with pytest.raises(ValueError, match="unit"):
        project.prepare_geometry(
            source, tmp_path / "project", unit=unit, mesh_size_mm=1
        )
    assert not calls


def test_existing_output_is_not_modified_or_imported(imported, tmp_path):
    source, _, calls = imported
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "preserve"
    marker.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        project.prepare_geometry(source, output, unit=Unit.MM, mesh_size_mm=1)
    assert marker.read_bytes() == b"keep" and not calls


def test_output_cannot_be_source_or_its_ancestor(imported, tmp_path):
    source, _, calls = imported
    for output in (source, source.parent):
        with pytest.raises(ValueError, match="overlap"):
            project.prepare_geometry(source, output, unit=Unit.MM, mesh_size_mm=1)
    assert not calls


def test_output_cannot_be_inside_or_above_installed_package(
    imported, tmp_path, monkeypatch
):
    source, _, calls = imported
    package = tmp_path / "package"
    package.mkdir()
    monkeypatch.setattr(piginot, "__file__", str(package / "__init__.py"))
    with pytest.raises(ValueError, match="package"):
        project.prepare_geometry(
            source, package / "output", unit=Unit.MM, mesh_size_mm=1
        )
    assert not calls


@pytest.mark.parametrize("target", ("source", "output", "output_parent"))
def test_symlinks_are_refused_before_import(imported, tmp_path, monkeypatch, target):
    source, _, calls = imported
    output = tmp_path / "parent" / "project"
    linked = {"source": source, "output": output, "output_parent": output.parent}[
        target
    ]
    original = Path.is_symlink
    monkeypatch.setattr(
        Path, "is_symlink", lambda path: path == linked or original(path)
    )
    with pytest.raises(ValueError, match="symlink"):
        project.prepare_geometry(source, output, unit=Unit.MM, mesh_size_mm=1)
    assert not calls


@pytest.mark.parametrize("target", ("missing", "directory"))
def test_source_must_be_a_regular_file(imported, tmp_path, target):
    _, _, calls = imported
    source = tmp_path / target
    if target == "directory":
        source.mkdir()
    with pytest.raises(ValueError, match="regular"):
        project.prepare_geometry(
            source, tmp_path / "project", unit=Unit.MM, mesh_size_mm=1
        )
    assert not calls


def test_importer_hash_must_match_preserved_original_bytes(imported, tmp_path):
    source, result, _ = imported
    result.source_sha256 = "0" * 64
    with pytest.raises(ValueError, match="source"):
        project.prepare_geometry(
            source, tmp_path / "project", unit=Unit.MM, mesh_size_mm=1
        )
    assert not (tmp_path / "project").exists()


def test_source_mutation_during_import_never_publishes_a_project(
    imported, tmp_path, monkeypatch
):
    source, result, _ = imported

    def importer(*args, **kwargs):
        source.write_bytes(b"changed during import")
        return result

    monkeypatch.setattr(project, "_import_cad", importer)
    with pytest.raises(ValueError, match="source"):
        project.prepare_geometry(
            source, tmp_path / "project", unit=Unit.MM, mesh_size_mm=1
        )
    assert not (tmp_path / "project").exists()


def test_source_mutation_during_refinement_never_publishes_a_project(
    imported, tmp_path, monkeypatch
):
    from piginot.cad import refinement

    source, _, _ = imported
    original = refinement.refine_fixed_mesh

    def changed(mesh, level):
        result = original(mesh, level)
        source.write_bytes(b"changed during refinement")
        return result

    monkeypatch.setattr(refinement, "refine_fixed_mesh", changed)
    with pytest.raises(ValueError, match="source"):
        project.prepare_geometry(
            source, tmp_path / "project", unit=Unit.MM, mesh_size_mm=1
        )
    assert not (tmp_path / "project").exists()


@pytest.mark.parametrize("invalid", ("duplicate", "missing", "kind"))
def test_boundary_metadata_must_match_all_mesh_tags(imported, tmp_path, invalid):
    source, result, _ = imported
    if invalid == "duplicate":
        result.boundaries = (*result.boundaries, result.boundaries[0])
    elif invalid == "missing":
        result.boundaries = result.boundaries[:-1]
    else:
        result.boundaries[0].kind = "unknown"
    with pytest.raises(ValueError, match="boundar"):
        project.prepare_geometry(
            source, tmp_path / "project", unit=Unit.MM, mesh_size_mm=1
        )
    assert not (tmp_path / "project").exists()


def test_invalid_imported_mesh_does_not_leave_output(imported, tmp_path):
    source, result, _ = imported
    result.mesh = replace(result.mesh, mesh_id="0" * 64)
    with pytest.raises(ValueError, match="identity"):
        project.prepare_geometry(
            source, tmp_path / "project", unit=Unit.MM, mesh_size_mm=1
        )
    assert not (tmp_path / "project").exists()


def test_output_creation_race_does_not_overwrite_existing_data(
    imported, tmp_path, monkeypatch
):
    source, result, _ = imported
    output = tmp_path / "project"

    def importer(*args, **kwargs):
        output.mkdir()
        (output / "racer").write_bytes(b"preserve")
        return result

    monkeypatch.setattr(project, "_import_cad", importer)
    with pytest.raises(FileExistsError):
        project.prepare_geometry(source, output, unit=Unit.MM, mesh_size_mm=1)
    assert (output / "racer").read_bytes() == b"preserve"
    assert not (output / "geometry.json").exists()


def test_write_failure_leaves_no_published_manifest(imported, tmp_path, monkeypatch):
    source, _, _ = imported
    output = tmp_path / "project"
    original = Path.open

    def failing_open(path, *args, **kwargs):
        if path == output / "mesh-2.json":
            raise PermissionError("controlled write failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(PermissionError, match="controlled"):
        project.prepare_geometry(source, output, unit=Unit.MM, mesh_size_mm=1)
    assert not (output / "geometry.json").exists()


def test_geometry_manifest_is_the_last_exclusively_created_file(
    imported, tmp_path, monkeypatch
):
    source, _, _ = imported
    output = tmp_path / "project"
    original = Path.open
    writes = []

    def observed(path, *args, **kwargs):
        if path.parent == output and args and args[0] == "xb":
            writes.append(path.name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observed)
    project.prepare_geometry(source, output, unit=Unit.MM, mesh_size_mm=1)
    assert writes[-1] == "geometry.json"
    assert set(writes[:-1]) == {
        "source.step",
        "mesh-1.json",
        "mesh-2.json",
        "mesh-4.json",
        "preview.json",
    }


def test_inspection_of_missing_or_symlink_manifest_is_refused(
    imported, tmp_path, monkeypatch
):
    with pytest.raises(CadResourceError):
        project.inspect_geometry(tmp_path / "missing")
    output, _ = _prepare(imported, tmp_path)
    original = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: path == output / "geometry.json" or original(path),
    )
    with pytest.raises(CadResourceError, match="symlink"):
        project.inspect_geometry(output)


def test_inspect_imports_no_cad_or_numerical_stack_in_a_fresh_process(
    imported, tmp_path
):
    output, _ = _prepare(imported, tmp_path)
    root = Path(__file__).resolve().parents[2]
    code = """
import importlib.abc, json, sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'gmsh','ezdxf','numpy','torch','scipy','skfem','sympy','meshio'}:
            raise AssertionError('numerical import: '+fullname)
sys.meta_path.insert(0,Deny())
from piginot.cad.project import inspect_geometry
print(json.dumps(inspect_geometry(__import__('pathlib').Path(sys.argv[1]))))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(output)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root))),
        },
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["mesh"]["node_count"] == 4
