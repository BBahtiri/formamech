"""Real CAD profiles become tagged canonical meshes without specimen recipes."""

from __future__ import annotations

import hashlib
import math
import subprocess
import sys
from pathlib import Path

import pytest

from piginot.cad.importing import CadImportError, import_cad
from piginot.contracts import Unit

ezdxf = pytest.importorskip(
    "ezdxf", reason="real CAD import tests require the cad extra"
)
gmsh = pytest.importorskip("gmsh", reason="real CAD import tests require the cad extra")


def _dxf(path: Path, *, unit: int = 4, hole: bool = False) -> Path:
    drawing = ezdxf.new()
    drawing.units = unit
    drawing.modelspace().add_lwpolyline(
        [(0, 0), (8, 0), (8, 4), (0, 4)],
        close=True,
        dxfattribs={"layer": "outline"},
    )
    if hole:
        drawing.modelspace().add_circle((3, 2), 0.5, dxfattribs={"layer": "hole"})
    drawing.saveas(path)
    return path


def _step(path: Path, *, kind: str = "face") -> Path:
    gmsh.initialize([], readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        if kind == "solid":
            gmsh.model.occ.addBox(0, 0, 0, 8, 4, 1)
        elif kind == "curved":
            gmsh.model.occ.addSphere(0, 0, 0, 2)
        else:
            face = gmsh.model.occ.addRectangle(0, 0, 0, 8, 4)
            if kind == "hole":
                hole = gmsh.model.occ.addDisk(3, 2, 0, 0.5, 0.5)
                gmsh.model.occ.cut([(2, face)], [(2, hole)])
            if kind == "disconnected":
                gmsh.model.occ.addRectangle(10, 0, 0, 3, 2)
            if kind == "raised":
                gmsh.model.occ.translate([(2, face)], 0, 0, 3)
            if kind == "tilted":
                gmsh.model.occ.rotate([(2, face)], 0, 0, 0, 1, 0, 0, math.pi / 2)
            if kind == "unused-curve":
                a = gmsh.model.occ.addPoint(20, 0, 0)
                b = gmsh.model.occ.addPoint(24, 0, 0)
                gmsh.model.occ.addLine(a, b)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


@pytest.mark.parametrize("suffix", [".step", ".stp", ".dxf", ".DXF"])
def test_imports_real_closed_profile_and_preserves_identity(tmp_path, suffix):
    source = tmp_path / ("profile" + suffix)
    _dxf(source) if suffix.lower() == ".dxf" else _step(source)
    result = import_cad(source, unit=Unit.MM, mesh_size_mm=1)
    assert result.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result.source_format == ("dxf" if suffix.lower() == ".dxf" else "step")
    assert result.mesh.unit is Unit.MM
    assert len(result.mesh.cells) > 2
    assert result.mesh.topology_signature.endswith("components=1;holes=0")
    assert len(result.boundaries) == 4
    assert {boundary.name for boundary in result.boundaries} == set(
        result.mesh.semantic_tags
    )
    assert all(boundary.kind == "line" for boundary in result.boundaries)
    assert all(
        boundary.document()["name"] == boundary.name for boundary in result.boundaries
    )
    assert max(x for x, _ in result.mesh.nodes) == pytest.approx(8)
    assert max(y for _, y in result.mesh.nodes) == pytest.approx(4)
    assert "linearized" in result.provenance["geometry_policy"]
    assert not gmsh.isInitialized()


@pytest.mark.parametrize("format", ["step", "dxf"])
def test_imports_a_hole_and_names_its_curved_boundary(tmp_path, format):
    source = tmp_path / ("hole." + format)
    _dxf(source, hole=True) if format == "dxf" else _step(source, kind="hole")
    result = import_cad(source, unit=Unit.MM, mesh_size_mm=0.6)
    assert result.mesh.topology_signature.endswith("components=1;holes=1")
    assert len(result.boundaries) == 5
    # STEP can represent a circular wire using an OCC ellipse. Keep unknown
    # curve kinds conservative instead of pretending they are verified arcs.
    assert any(boundary.kind != "line" for boundary in result.boundaries)
    assert all(math.hypot(x - 3, y - 2) >= 0.499999 for x, y in result.mesh.nodes)


def test_dxf_declared_units_scale_once_and_unitless_is_explicit(tmp_path):
    cm = import_cad(_dxf(tmp_path / "cm.dxf", unit=5), unit=Unit.CM, mesh_size_mm=10)
    assert max(x for x, _ in cm.mesh.nodes) == pytest.approx(80)
    unitless = import_cad(
        _dxf(tmp_path / "unitless.dxf", unit=0), unit=Unit.CM, mesh_size_mm=10
    )
    assert unitless.mesh.mesh_id == cm.mesh.mesh_id


def test_step_embedded_units_are_converted_once(tmp_path):
    source = _step(tmp_path / "units.step")
    mm = import_cad(source, unit=Unit.MM, mesh_size_mm=1)
    cm = import_cad(source, unit=Unit.CM, mesh_size_mm=1)
    assert max(x for x, _ in cm.mesh.nodes) == pytest.approx(8)
    assert max(y for _, y in cm.mesh.nodes) == pytest.approx(4)
    assert cm.mesh.mesh_id == mm.mesh.mesh_id


def test_repeated_import_is_deterministic(tmp_path):
    source = _dxf(tmp_path / "repeat.dxf", hole=True)
    first = import_cad(source, unit=Unit.MM, mesh_size_mm=0.7)
    second = import_cad(source, unit=Unit.MM, mesh_size_mm=0.7)
    assert first.mesh == second.mesh
    assert first.boundaries == second.boundaries
    with pytest.raises(TypeError):
        first.provenance["geometry_policy"] = "changed"


@pytest.mark.parametrize("size", [0, -1, float("nan"), float("inf"), True, "1"])
def test_rejects_invalid_mesh_size_without_loading_cad(tmp_path, size):
    with pytest.raises(CadImportError, match="mesh_size"):
        import_cad(tmp_path / "missing.step", unit=Unit.MM, mesh_size_mm=size)


@pytest.mark.parametrize("unit", [Unit.ONE, "mm", None])
def test_rejects_missing_or_nonlength_units(tmp_path, unit):
    with pytest.raises(CadImportError, match="unit"):
        import_cad(tmp_path / "missing.step", unit=unit, mesh_size_mm=1)


def test_rejects_missing_unknown_and_symlink_sources(tmp_path):
    with pytest.raises(CadImportError, match="regular"):
        import_cad(tmp_path / "missing.step", unit=Unit.MM, mesh_size_mm=1)
    unknown = tmp_path / "unsupported.stl"
    unknown.write_bytes(b"not a profile")
    with pytest.raises(CadImportError, match="format"):
        import_cad(unknown, unit=Unit.MM, mesh_size_mm=1)
    link = tmp_path / "link.dxf"
    try:
        link.symlink_to(_dxf(tmp_path / "real.dxf"))
    except OSError:
        pytest.skip("OS cannot create symlinks")
    with pytest.raises(CadImportError, match="regular"):
        import_cad(link, unit=Unit.MM, mesh_size_mm=1)


@pytest.mark.parametrize(
    "kind", ["solid", "curved", "disconnected", "tilted", "unused-curve"]
)
def test_rejects_nonprofile_step_and_cleans_its_session(tmp_path, kind):
    source = _step(tmp_path / (kind + ".step"), kind=kind)
    with pytest.raises(CadImportError):
        import_cad(source, unit=Unit.MM, mesh_size_mm=1)
    assert not gmsh.isInitialized()


def test_rejects_a_conflicting_dxf_unit_header(tmp_path):
    with pytest.raises(CadImportError, match="unit"):
        import_cad(_dxf(tmp_path / "wrong.dxf", unit=6), unit=Unit.MM, mesh_size_mm=1)


@pytest.mark.parametrize("entity", ["open", "line", "text", "nonplanar", "width"])
def test_rejects_unsupported_or_open_dxf_profiles(tmp_path, entity):
    source = tmp_path / (entity + ".dxf")
    drawing = ezdxf.new()
    drawing.units = 4
    model = drawing.modelspace()
    if entity == "line":
        model.add_line((0, 0), (2, 0))
    elif entity == "text":
        model.add_text("not geometry")
    else:
        line = model.add_lwpolyline(
            [(0, 0), (2, 0), (2, 2), (0, 2)], close=entity != "open"
        )
        if entity == "nonplanar":
            line.dxf.extrusion = (1, 0, 0)
        if entity == "width":
            line.dxf.const_width = 1
    drawing.saveas(source)
    with pytest.raises(CadImportError):
        import_cad(source, unit=Unit.MM, mesh_size_mm=1)
    assert not gmsh.isInitialized()


def test_does_not_clear_a_callers_gmsh_session(tmp_path):
    source = _dxf(tmp_path / "busy.dxf")
    gmsh.initialize([], readConfigFiles=False)
    try:
        gmsh.model.add("owned-by-caller")
        with pytest.raises(CadImportError, match="session"):
            import_cad(source, unit=Unit.MM, mesh_size_mm=1)
        assert gmsh.isInitialized()
        assert gmsh.model.getCurrent() == "owned-by-caller"
    finally:
        gmsh.finalize()


def test_translated_xy_face_records_its_frame_offset(tmp_path):
    result = import_cad(
        _step(tmp_path / "raised.step", kind="raised"), unit=Unit.MM, mesh_size_mm=1
    )
    assert float(result.provenance["source_plane_z_offset_mm"]) == pytest.approx(3)
    assert max(x for x, _ in result.mesh.nodes) == pytest.approx(8)


def test_reports_missing_native_cad_library_without_claiming_input_support(
    tmp_path, monkeypatch
):
    import builtins

    source = _dxf(tmp_path / "dependency.dxf")
    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "gmsh":
            raise OSError("native library missing")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(CadImportError, match="dependency"):
        import_cad(source, unit=Unit.MM, mesh_size_mm=1)


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "zero-line",
        "radius",
        "arc",
        "z",
        "cross",
        "separate",
        "touch",
        "nested",
    ],
)
def test_refuses_invalid_dxf_domain_before_publishing_a_mesh(tmp_path, kind):
    source = tmp_path / (kind + ".dxf")
    drawing = ezdxf.new()
    drawing.units = 4
    model = drawing.modelspace()
    if kind == "zero-line":
        model.add_line((0, 0), (0, 0))
    elif kind == "radius":
        model.add_circle((0, 0), 0)
    elif kind == "arc":
        model.add_arc((0, 0), 1, 0, 0)
    elif kind == "z":
        model.add_line((0, 0, 1), (1, 0, 1))
    elif kind == "cross":
        model.add_lwpolyline([(0, 0), (4, 4), (0, 4), (4, 0)], close=True)
    elif kind in ("separate", "touch", "nested"):
        model.add_lwpolyline([(0, 0), (8, 0), (8, 4), (0, 4)], close=True)
        if kind == "separate":
            model.add_circle((12, 2), 1)
        elif kind == "touch":
            model.add_circle((0.5, 2), 0.5)
        else:
            model.add_circle((3, 2), 1)
            model.add_circle((3, 2), 0.5)
    drawing.saveas(source)
    with pytest.raises(CadImportError):
        import_cad(source, unit=Unit.MM, mesh_size_mm=1)
    assert not gmsh.isInitialized()


@pytest.mark.parametrize("bulge", [1.0, -0.5])
def test_real_arc_and_bulged_polyline_are_preserved_as_curves(tmp_path, bulge):
    source = tmp_path / "arc.dxf"
    drawing = ezdxf.new()
    drawing.units = 4
    drawing.modelspace().add_lwpolyline(
        [(0, 0, 0), (4, 0, bulge), (4, 4, 0), (0, 4, 0)], format="xyb", close=True
    )
    drawing.saveas(source)
    result = import_cad(source, unit=Unit.MM, mesh_size_mm=0.5)
    assert "arc" in {boundary.kind for boundary in result.boundaries}
    assert result.mesh.topology_signature.endswith("components=1;holes=0")


def test_generated_clockwise_elements_are_oriented_to_xy(tmp_path, monkeypatch):
    source = _dxf(tmp_path / "clockwise.dxf")
    original = gmsh.model.mesh.getElements

    def clockwise(dim, tag):
        types, ids, elements = original(dim, tag)
        if dim == 2:
            elements = [
                block.reshape(-1, 3)[:, [0, 2, 1]].reshape(-1) for block in elements
            ]
        return types, ids, elements

    monkeypatch.setattr(gmsh.model.mesh, "getElements", clockwise)
    mesh = import_cad(source, unit=Unit.MM, mesh_size_mm=1).mesh
    assert mesh.orientation == "counterclockwise"


@pytest.mark.parametrize("kind", ["triangles", "segments", "off-plane", "zero-area"])
def test_validates_backend_mesh_and_domain_contracts(tmp_path, monkeypatch, kind):
    source = _dxf(tmp_path / (kind + ".dxf"))
    elements = gmsh.model.mesh.getElements
    nodes = gmsh.model.mesh.getNodes
    if kind in ("triangles", "segments"):

        def wrong_type(dim, tag):
            types, ids, blocks = elements(dim, tag)
            return (
                (
                    [3]
                    if (kind == "triangles" and dim == 2)
                    or (kind == "segments" and dim == 1)
                    else types
                ),
                ids,
                blocks,
            )

        monkeypatch.setattr(gmsh.model.mesh, "getElements", wrong_type)
    elif kind == "off-plane":

        def off_plane():
            tags, coords, params = nodes()
            coords[2] = 1
            return tags, coords, params

        monkeypatch.setattr(gmsh.model.mesh, "getNodes", off_plane)
    else:
        monkeypatch.setattr(gmsh.model.occ, "getMass", lambda dim, tag: 0)
    with pytest.raises(CadImportError):
        import_cad(source, unit=Unit.MM, mesh_size_mm=1)


def test_detects_source_change_during_meshing(tmp_path, monkeypatch):
    source = _dxf(tmp_path / "changed.dxf")
    generate = gmsh.model.mesh.generate

    def changing(dim):
        generate(dim)
        source.write_bytes(source.read_bytes() + b"\n")

    monkeypatch.setattr(gmsh.model.mesh, "generate", changing)
    with pytest.raises(CadImportError, match="changed"):
        import_cad(source, unit=Unit.MM, mesh_size_mm=1)


def test_invalid_cad_backend_error_has_a_refusal_and_releases_session(tmp_path):
    source = tmp_path / "corrupt.step"
    source.write_text("not a STEP file")
    with pytest.raises(CadImportError, match="CAD backend rejected"):
        import_cad(source, unit=Unit.MM, mesh_size_mm=1)
    assert not gmsh.isInitialized()


def test_import_module_needs_no_cad_or_numerical_stack():
    root = Path(__file__).resolve().parents[2]
    script = f"""
import importlib.abc
import sys
sys.path.insert(0, {str(root / "src")!r})
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'gmsh', 'ezdxf', 'numpy', 'torch', 'scipy', 'skfem'}}:
            raise AssertionError('Numerical import during metadata: ' + fullname)
sys.meta_path.insert(0, Deny())
from piginot.cad.importing import import_cad, BoundaryInfo, ImportedCad
assert BoundaryInfo('edge-000001', 'line', 1, 'outer').document()['source_id'] == 1
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_independent_arc_line_loop_and_major_arc(tmp_path):
    drawing = ezdxf.new()
    drawing.units = 4
    model = drawing.modelspace()
    model.add_arc((0, 0), 2, 0, 270)
    model.add_line((0, -2), (0, 0))
    model.add_line((0, 0), (2, 0))
    source = tmp_path / "major-arc.dxf"
    drawing.saveas(source)
    mesh = import_cad(source, unit=Unit.MM, mesh_size_mm=0.4).mesh
    assert mesh.topology_signature.endswith("components=1;holes=0")
    assert min(x for x, _ in mesh.nodes) == pytest.approx(-2, abs=0.02)


def test_circle_outer_profile_and_circular_hole(tmp_path):
    drawing = ezdxf.new()
    drawing.units = 4
    model = drawing.modelspace()
    model.add_circle((0, 0), 4)
    model.add_circle((0, 0), 1)
    source = tmp_path / "annulus.dxf"
    drawing.saveas(source)
    result = import_cad(source, unit=Unit.MM, mesh_size_mm=0.5)
    assert result.mesh.topology_signature.endswith("components=1;holes=1")
    assert len(result.boundaries) == 2
    assert all(boundary.kind == "arc" for boundary in result.boundaries)


def test_missing_dxf_reader_is_an_explicit_capability_refusal(tmp_path, monkeypatch):
    import builtins

    source = _dxf(tmp_path / "missing-reader.dxf")
    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "ezdxf.filemanagement":
            raise ImportError("reader missing")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(CadImportError, match="dependency"):
        import_cad(source, unit=Unit.MM, mesh_size_mm=1)
    assert not gmsh.isInitialized()


@pytest.mark.parametrize("suffix", ["step", "dxf"])
def test_checked_in_profile_fixtures_work_without_a_geometry_generator(suffix):
    source = (
        Path(__file__).resolve().parents[1]
        / "fixtures/cad"
        / ("plate-with-hole." + suffix)
    )
    result = import_cad(source, unit=Unit.MM, mesh_size_mm=1)
    assert result.mesh.topology_signature.endswith("components=1;holes=1")
    assert len(result.boundaries) == 5
