"""Real CAD acceptance checks for explicit sizing and curve accuracy."""

import json
from dataclasses import replace

import pytest

from piginot.cad.importing import CadImportError, import_cad
from piginot.cad.mesh_policy import FeatureSize, MeshPolicy, PointSize
from piginot.cad.project import inspect_geometry, prepare_geometry
from piginot.contracts import Unit

gmsh = pytest.importorskip("gmsh")
ezdxf = pytest.importorskip("ezdxf")


def _profile(path):
    if path.suffix == ".dxf":
        drawing = ezdxf.new()
        drawing.units = 4
        model = drawing.modelspace()
        model.add_lwpolyline([(0, 0), (12, 0), (12, 8), (0, 8)], close=True)
        model.add_circle((4, 4), 1.5)
        drawing.saveas(path)
    else:
        gmsh.initialize([], readConfigFiles=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            face = gmsh.model.occ.addRectangle(0, 0, 0, 12, 8)
            circle = gmsh.model.occ.addCircle(4, 4, 0, 1.5)
            hole = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([circle])]
            )
            gmsh.model.occ.cut([(2, face)], [(2, hole)])
            gmsh.model.occ.synchronize()
            gmsh.write(str(path))
        finally:
            gmsh.finalize()
    return path


def _policy(**changes):
    return replace(
        MeshPolicy(
            global_size_mm=1.2,
            minimum_size_mm=0.05,
            curve_tolerance_mm=0.003,
            curvature_points_per_circle=24,
            feature_sizes=(),
            point_sizes=(),
            minimum_angle_deg=15,
            maximum_aspect_ratio=6,
            maximum_triangles=100000,
            maximum_p2_dofs=500000,
        ),
        **changes,
    )


@pytest.mark.parametrize("suffix", [".step", ".dxf"])
def test_policy_preserves_tags_hole_units_and_checks_circle_deviation(tmp_path, suffix):
    source = _profile(tmp_path / ("hole" + suffix))
    old = import_cad(source, unit=Unit.MM, mesh_size_mm=1.2)
    hole = next(b.name for b in old.boundaries if b.kind == "arc")
    policy = _policy(feature_sizes=(FeatureSize(hole, 0.15, 0.3, 1.8),))
    result = import_cad(source, unit=Unit.MM, mesh_policy=policy)
    assert result.mesh.topology_signature == old.mesh.topology_signature
    assert result.boundaries == old.boundaries
    assert result.mesh.unit == Unit.MM
    assert result.mesh_policy == policy
    evidence = result.mesh_evidence
    assert evidence["policy_sha256"] == policy.sha256
    assert evidence["geometry"]["status"] == "PASS"
    curves = evidence["geometry"]["boundaries"]
    assert curves[hole]["max_deviation_mm"] <= policy.curve_tolerance_mm
    assert curves[hole]["radius_mm"] == pytest.approx(1.5)
    straight = [v for k, v in curves.items() if k != hole]
    assert curves[hole]["max_chord_mm"] < min(v["max_chord_mm"] for v in straight) / 2
    assert evidence["quality"]["minimum_angle_deg"] >= policy.minimum_angle_deg
    assert evidence["quality"]["maximum_aspect_ratio"] <= policy.maximum_aspect_ratio
    assert evidence["limits_status"] == "PASS"
    assert not gmsh.isInitialized()


def test_policy_and_measured_evidence_are_bound_to_prepared_project(tmp_path):
    source = _profile(tmp_path / "profile.dxf")
    policy = _policy()
    first = prepare_geometry(
        source, tmp_path / "first", unit=Unit.MM, mesh_policy=policy
    )
    second = prepare_geometry(
        source, tmp_path / "second", unit=Unit.MM, mesh_policy=policy
    )
    assert first["geometry_resource_sha256"] == second["geometry_resource_sha256"]
    inspected = inspect_geometry(tmp_path / "first")
    assert inspected["provenance"]["mesh_policy"] == policy.document()
    assert inspected["provenance"]["mesh_evidence"]["geometry"]["status"] == "PASS"
    assert inspected["provenance"]["geometry_accuracy_bound"] == "NOT_ESTABLISHED"


@pytest.mark.parametrize("limit", ["maximum_triangles", "maximum_p2_dofs"])
def test_reference_ladder_budget_refuses_before_project_publication(tmp_path, limit):
    source = _profile(tmp_path / "profile.dxf")
    output = tmp_path / "refused"
    with pytest.raises(CadImportError, match="budget"):
        prepare_geometry(
            source, output, unit=Unit.MM, mesh_policy=_policy(**{limit: 10})
        )
    assert not output.exists()
    assert not gmsh.isInitialized()


def test_unknown_feature_and_impossible_curve_floor_refuse(tmp_path):
    source = _profile(tmp_path / "profile.dxf")
    with pytest.raises(CadImportError, match="boundary"):
        import_cad(
            source,
            unit=Unit.MM,
            mesh_policy=_policy(feature_sizes=(FeatureSize("edge-999999", 0.2, 0, 1),)),
        )
    with pytest.raises(CadImportError, match="minimum_size"):
        import_cad(source, unit=Unit.MM, mesh_policy=_policy(curve_tolerance_mm=1e-8))
    assert not gmsh.isInitialized()


def test_point_size_field_refines_interior_without_new_boundary_entities(tmp_path):
    source = _profile(tmp_path / "profile.dxf")
    base = import_cad(source, unit=Unit.MM, mesh_policy=_policy())
    refined = import_cad(
        source,
        unit=Unit.MM,
        mesh_policy=_policy(point_sizes=(PointSize((9, 4), 0.15, 0.6, 1.5),)),
    )
    assert refined.boundaries == base.boundaries
    assert len(refined.mesh.cells) > len(base.mesh.cells)
    assert refined.mesh.mesh_id != base.mesh.mesh_id


def test_mesh_quality_limit_is_enforced(tmp_path):
    with pytest.raises(CadImportError, match="quality"):
        import_cad(
            _profile(tmp_path / "p.dxf"),
            unit=Unit.MM,
            mesh_policy=_policy(minimum_angle_deg=59.9),
        )


def test_legacy_and_explicit_policy_are_exclusive(tmp_path):
    source = _profile(tmp_path / "profile.dxf")
    with pytest.raises(CadImportError, match="exactly one"):
        import_cad(source, unit=Unit.MM, mesh_size_mm=1.2, mesh_policy=_policy())
    with pytest.raises(CadImportError, match="exactly one"):
        import_cad(source, unit=Unit.MM)


def test_unknown_curves_cannot_satisfy_an_analytic_geometry_check(
    tmp_path, monkeypatch
):
    source = _profile(tmp_path / "profile.dxf")
    get_type = gmsh.model.getType
    monkeypatch.setattr(
        gmsh.model,
        "getType",
        lambda dim, tag: (
            "BSpline" if get_type(dim, tag) == "Circle" else get_type(dim, tag)
        ),
    )
    with pytest.raises(CadImportError, match="unqualified"):
        import_cad(source, unit=Unit.MM, mesh_policy=_policy())


def test_cli_exposes_explicit_policy_and_refuses_two_size_sources(tmp_path, capsys):
    from piginot.cli import main

    source = _profile(tmp_path / "profile.dxf")
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(_policy().document()))
    args = [
        "cad",
        "import",
        str(source),
        "--unit",
        "mm",
        "--mesh-policy",
        str(path),
        "--output",
        str(tmp_path / "project"),
    ]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "IMPORTED"
    with pytest.raises(SystemExit) as error:
        main([*args, "--mesh-size-mm", "1"])
    assert error.value.code == 2
    path.write_text('{"unknown":1}')
    assert main(args) == 2
    assert json.loads(capsys.readouterr().err.splitlines()[-1])["status"] == "REFUSED"


@pytest.mark.parametrize("level", [1, 2, 4])
def test_projected_resource_counts_match_actual_refinements(tmp_path, level):
    from piginot.cad.mesh_controls import projected_counts
    from piginot.cad.refinement import refine_fixed_mesh

    base = import_cad(_profile(tmp_path / "p.dxf"), unit=Unit.MM, mesh_size_mm=1).mesh
    refined = refine_fixed_mesh(base, level)
    edges = {
        tuple(sorted((cell[i], cell[(i + 1) % 3])))
        for cell in refined.cells
        for i in range(3)
    }
    counts = projected_counts(base, level)
    assert counts["reference_triangles"] == len(refined.cells)
    assert counts["reference_p2_dofs"] == 2 * (len(refined.nodes) + len(edges))


def test_extreme_line_sizing_refuses_before_gmsh_generation(tmp_path, monkeypatch):
    source = _profile(tmp_path / "line.dxf")
    old = import_cad(source, unit=Unit.MM, mesh_size_mm=1.2)
    line = next(b.name for b in old.boundaries if b.kind == "line")
    called = []
    monkeypatch.setattr(gmsh.model.mesh, "generate", lambda dim: called.append(dim))
    policy = _policy(
        minimum_size_mm=1e-12, feature_sizes=(FeatureSize(line, 1e-12, 0.1, 0.5),)
    )
    with pytest.raises(CadImportError, match="budget"):
        import_cad(source, unit=Unit.MM, mesh_policy=policy)
    assert called == []
    assert not gmsh.isInitialized()


def test_preparation_binds_adaptation_and_rejects_changed_mesh_or_failure(tmp_path):
    from piginot.cad.adaptation import PilotCase, PilotStress, adapt_mesh

    source = _profile(tmp_path / "profile.dxf")
    policy = _policy()

    def remesh(selected):
        return import_cad(source, unit=Unit.MM, mesh_policy=selected).mesh

    def manufactured(mesh, case):
        return PilotStress(
            case.case_id, mesh.mesh_id, (((1.0, 0.0), (0.0, 1.0)),) * len(mesh.cells)
        )

    report = adapt_mesh(
        policy,
        (PilotCase("pilot-1", "TRAIN-pilot", {"ux": 0.1}),),
        ("screen-1",),
        remesh=remesh,
        evaluate=manufactured,
        max_rounds=2,
        max_solves=2,
        indicator_target=0,
        mark_fraction=0.1,
        refinement_factor=2,
        max_markers=4,
    )
    assert report.status == "HEURISTIC_TARGET_MET"
    output = tmp_path / "accepted"
    prepare_geometry(
        source, output, unit=Unit.MM, mesh_policy=policy, adaptation_report=report
    )
    evidence = inspect_geometry(output)["provenance"]["adaptation"]
    assert evidence["sha256"] == report.sha256
    assert evidence["report"]["error_bound"] == "NOT_ESTABLISHED"
    assert evidence["report"]["solves_used"] == 1
    with pytest.raises(ValueError, match="differs"):
        prepare_geometry(
            source,
            tmp_path / "changed",
            unit=Unit.MM,
            mesh_policy=replace(policy, global_size_mm=1),
            adaptation_report=report,
        )
    with pytest.raises(ValueError, match="completed"):
        prepare_geometry(
            source,
            tmp_path / "failed",
            unit=Unit.MM,
            mesh_policy=policy,
            adaptation_report=replace(report, status="ROUND_LIMIT"),
        )
    assert not (tmp_path / "changed").exists()
    assert not (tmp_path / "failed").exists()


def test_pilot_markers_drive_real_cad_remeshing_and_stop_at_declared_round_limit(
    tmp_path,
):
    from piginot.cad.adaptation import PilotCase, PilotStress, adapt_mesh

    source = _profile(tmp_path / "profile.dxf")

    def remesh(selected):
        return import_cad(source, unit=Unit.MM, mesh_policy=selected).mesh

    def smooth_manufactured_field(mesh, case):
        # Exact cell averages of P_xx = 1+x; no PDE solve or NN evaluation.
        values = tuple(
            ((1 + sum(mesh.nodes[n][0] for n in c) / 3, 0.0), (0.0, 0.0))
            for c in mesh.cells
        )
        return PilotStress(case.case_id, mesh.mesh_id, values)

    report = adapt_mesh(
        _policy(),
        (PilotCase("pilot-1", "TRAIN-pilot", {"ux": 0.1}),),
        ("screen-1",),
        remesh=remesh,
        evaluate=smooth_manufactured_field,
        max_rounds=2,
        max_solves=2,
        indicator_target=0,
        mark_fraction=0.05,
        refinement_factor=2,
        max_markers=4,
    )
    assert report.status == "ROUND_LIMIT"
    assert report.solves_used == 2
    first, last = report.attempts
    assert last.mesh.mesh_id != first.mesh.mesh_id
    assert len(last.mesh.cells) > len(first.mesh.cells)
    assert len(last.policy.point_sizes) == 4
    assert set(last.mesh.semantic_tags) == set(first.mesh.semantic_tags)


def test_changed_policy_budget_invalidates_binding_even_for_identical_geometry(
    tmp_path,
):
    from piginot.cad.resource import CadResourceError, load_geometry_resource

    source = _profile(tmp_path / "profile.dxf")
    policy = _policy()
    first = prepare_geometry(
        source, tmp_path / "first", unit=Unit.MM, mesh_policy=policy
    )
    second = prepare_geometry(
        source,
        tmp_path / "second",
        unit=Unit.MM,
        mesh_policy=replace(policy, maximum_triangles=120000),
    )
    assert first["mesh_id"] == second["mesh_id"]
    assert first["geometry_resource_sha256"] != second["geometry_resource_sha256"]
    with pytest.raises(CadResourceError, match="bytes differ"):
        load_geometry_resource(
            tmp_path / "second/geometry.json", first["geometry_resource_sha256"]
        )


@pytest.mark.parametrize("precision,tolerance", [("float32", 1e-6), ("float64", 1e-13)])
def test_locally_dense_training_quadrature_preserves_physical_area(
    tmp_path, precision, tolerance
):
    import torch

    from piginot.objectives.energy import TriangleDomain, stratified_triangle_quadrature

    mesh = import_cad(
        _profile(tmp_path / "p.dxf"),
        unit=Unit.MM,
        mesh_policy=_policy(point_sizes=(PointSize((9, 4), 0.15, 0.6, 1.5),)),
    ).mesh
    areas = []
    for cell in mesh.cells:
        a, b, c = (mesh.nodes[n] for n in cell)
        areas.append(
            ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2
        )
    assert max(areas) > 10 * min(areas)
    domain = TriangleDomain(
        torch.tensor(mesh.nodes, dtype=getattr(torch, precision)),
        torch.tensor(mesh.cells, dtype=torch.int64),
    )
    _, weights = stratified_triangle_quadrature(
        domain, 3, generator=torch.Generator().manual_seed(2901)
    )
    assert float(weights.sum()) == pytest.approx(sum(areas), rel=tolerance)
    assert float((weights * 7.25).sum()) == pytest.approx(
        7.25 * sum(areas), rel=tolerance
    )
