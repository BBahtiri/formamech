"""Tests for Feature A: nested mesh hierarchy, cross-mesh transfer, and convergence."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from piginot.cad.hierarchy import (
    CrossMeshTransfer,
    GeometryConvergenceSample,
    GeometryConvergenceStatus,
    ReferenceConvergenceStatus,
    build_mesh_hierarchy,
    build_reference_convergence_report,
    evaluate_geometry_convergence,
)
from piginot.cad.refinement import refine_fixed_mesh
from piginot.compiler.reference_convergence import (
    compare_fem_pair,
    compare_remeshed_cad_fem_pair,
)
from piginot.contracts import Unit
from piginot.geometry import CellType, MeshPayload, canonicalize_mesh
from tests.compiler.test_reference_convergence import (
    TOLERANCES,
    pair,
    result_for,
)

SOURCE_HASH = "a" * 64


def _unit_square_mesh():
    return canonicalize_mesh(
        MeshPayload(
            ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)),
            ((0, 1, 2), (0, 2, 3)),
            CellType.TRIANGLE,
            ((0, 1), (1, 2), (2, 3), (3, 0)),
            {"bottom": (0,), "right": (1,), "top": (2,), "left": (3,)},
            (1, 1),
            Unit.MM,
            {"source": "unit-square"},
        )
    )


def _quarter_ring_polygon(n_arc: int, r_in: float = 1.0, r_out: float = 2.0):
    """Polygonal approximation of a quarter-annulus with n_arc chords per arc."""
    nodes = []
    for k in range(n_arc + 1):
        theta = (0.5 * math.pi * k) / n_arc
        nodes.append((r_in * math.cos(theta), r_in * math.sin(theta)))
    for k in range(n_arc + 1):
        theta = (0.5 * math.pi * k) / n_arc
        nodes.append((r_out * math.cos(theta), r_out * math.sin(theta)))
    cells = []
    for k in range(n_arc):
        i0, i1 = k, k + 1
        o0, o1 = (n_arc + 1) + k, (n_arc + 1) + k + 1
        cells.append((i0, o0, o1))
        cells.append((i0, o1, i1))
    facets = []
    inner_idx, outer_idx = [], []
    for k in range(n_arc):
        inner_idx.append(len(facets))
        facets.append((k + 1, k))
        outer_idx.append(len(facets))
        facets.append(((n_arc + 1) + k, (n_arc + 1) + k + 1))
    bottom_idx = len(facets)
    facets.append((0, n_arc + 1))
    left_idx = len(facets)
    facets.append((2 * n_arc + 1, n_arc))
    return canonicalize_mesh(
        MeshPayload(
            tuple(nodes),
            tuple(cells),
            CellType.TRIANGLE,
            tuple(facets),
            {
                "inner_arc": tuple(inner_idx),
                "outer_arc": tuple(outer_idx),
                "bottom": (bottom_idx,),
                "left": (left_idx,),
            },
            (1,) * len(cells),
            Unit.MM,
            {"n_arc": str(n_arc)},
        )
    )


def test_build_mesh_hierarchy_preserves_area_tags_and_hash() -> None:
    base = _unit_square_mesh()
    h = build_mesh_hierarchy(
        base,
        cad_source_hash=SOURCE_HASH,
        geometry_tolerance_mm=1e-2,
        refinement_history_hash="b" * 64,
    )
    assert len(h.m0_coarse.cells) == 2
    assert len(h.m1_training.cells) == 8
    assert len(h.m2_reference.cells) == 32
    assert h.area_mm2 == pytest.approx(2.0)
    doc = h.document()
    assert doc["levels"]["M0"]["cell_count"] == 2
    assert doc["levels"]["M1"]["cell_count"] == 8
    assert doc["levels"]["M2"]["cell_count"] == 32
    assert len(h.hierarchy_hash) == 64
    # Deterministic hash on repeated build.
    h2 = build_mesh_hierarchy(
        base,
        cad_source_hash=SOURCE_HASH,
        geometry_tolerance_mm=1e-2,
        refinement_history_hash="b" * 64,
    )
    assert h.hierarchy_hash == h2.hierarchy_hash


def test_cross_mesh_transfer_is_exact_on_affine_and_conservative_on_cells() -> None:
    base = _unit_square_mesh()
    h = build_mesh_hierarchy(
        base, cad_source_hash=SOURCE_HASH, geometry_tolerance_mm=1e-2
    )
    transfer = CrossMeshTransfer.between(h.m0_coarse, h.m1_training)

    # Exact prolongation on an affine vector field u(x, y) = (2x - 3y + 1, -x + 4y).
    coarse_u = [(2.0 * x - 3.0 * y + 1.0, -x + 4.0 * y) for x, y in h.m0_coarse.nodes]
    prolonged = transfer.prolong_nodal(coarse_u)
    expected = [(2.0 * x - 3.0 * y + 1.0, -x + 4.0 * y) for x, y in h.m1_training.nodes]
    for got, exp in zip(prolonged, expected, strict=True):
        assert got[0] == pytest.approx(exp[0], abs=1e-13)
        assert got[1] == pytest.approx(exp[1], abs=1e-13)

    # Injection recovers coarse nodal values identically.
    injected = transfer.restrict_nodal_injection(prolonged)
    assert injected == tuple(coarse_u)

    # Conservative cell restriction preserves total domain integral identically.
    fine_densities = [1.5 + 0.7 * i for i in range(len(h.m1_training.cells))]
    coarse_densities = transfer.restrict_cell_conservative(fine_densities)
    fine_integral = sum(
        d * a for d, a in zip(fine_densities, transfer.fine_cell_areas, strict=True)
    )
    coarse_integral = sum(
        d * a for d, a in zip(coarse_densities, transfer.coarse_cell_areas, strict=True)
    )
    assert coarse_integral == pytest.approx(fine_integral, rel=1e-13)

    # Prolonging piecewise-constant densities and restricting is the exact identity.
    roundtrip = transfer.restrict_cell_conservative(
        transfer.prolong_cell_piecewise_constant(coarse_densities)
    )
    assert roundtrip == pytest.approx(coarse_densities, rel=1e-13)


def test_geometry_convergence_measures_curved_arc_remesh_refinement() -> None:
    # Distinct CAD remeshes of a quarter-annulus: n_arc = 8 -> 16 -> 32.
    samples = []
    for n_arc in (8, 16, 32):
        mesh = _quarter_ring_polygon(n_arc, r_in=1.0, r_out=2.0)
        dtheta = (0.5 * math.pi) / n_arc
        # Exact sagitta / chord error on outer radius R=2.
        chord_err = 2.0 * (1.0 - math.cos(0.5 * dtheta))
        samples.append(
            GeometryConvergenceSample.from_mesh(
                f"n_arc_{n_arc}",
                mesh,
                cad_source_hash=SOURCE_HASH,
                declared_curve_tolerance_mm=0.05,
                achieved_max_chord_error_mm=chord_err,
            )
        )

    measured = evaluate_geometry_convergence(samples)
    assert measured.status is GeometryConvergenceStatus.MEASURED_CHANGES_ONLY
    assert measured.document()["is_continuum_solution_bound"] is False

    passed = evaluate_geometry_convergence(
        samples,
        max_relative_area_change=1e-2,
        max_relative_perimeter_change=1e-2,
        max_chord_error_mm=0.01,
    )
    assert passed.status is GeometryConvergenceStatus.CONVERGED

    failed = evaluate_geometry_convergence(
        samples,
        max_relative_area_change=1e-6,
        max_relative_perimeter_change=1e-6,
        max_chord_error_mm=1e-6,
    )
    assert failed.status is GeometryConvergenceStatus.NOT_CONVERGED

    # Reject duplicate polygon pretending to be geometry convergence.
    with pytest.raises(ValueError, match="distinct polygonal remeshes"):
        evaluate_geometry_convergence([samples[0], samples[0]])


def test_reference_convergence_report_marks_not_converged_when_floor_fails(
    tmp_path,
) -> None:
    spec, m0_req, m1_req = pair(tmp_path, order=2)
    m2_req = replace(m0_req, request_id="m2", mesh=refine_fixed_mesh(m0_req.mesh, 4))
    hierarchy = build_mesh_hierarchy(
        m0_req.mesh, cad_source_hash=SOURCE_HASH, geometry_tolerance_mm=0.01
    )

    m0_res = result_for(m0_req, quadratic=True, factor=1.0)
    m1_res = result_for(m1_req, quadratic=True, factor=1.02)
    m2_res = result_for(m2_req, quadratic=True, factor=1.05)

    p01 = compare_fem_pair(spec, m0_req, m0_res, m1_req, m1_res, tolerances=None)
    p12 = compare_fem_pair(spec, m1_req, m1_res, m2_req, m2_res, tolerances=None)

    # Tight floor -> REFERENCE_NOT_CONVERGED and accuracy conclusions withheld.
    not_conv = build_reference_convergence_report(
        hierarchy,
        p01,
        p12,
        reference_floor={
            "u_relative_l2": 1e-4,
            "p_relative_l2": 1e-4,
            "absolute_peak_error": 1e-4,
        },
    )
    assert not_conv.status is ReferenceConvergenceStatus.REFERENCE_NOT_CONVERGED
    assert not_conv.withhold_accuracy_conclusions is True
    assert not_conv.document()["total_error_bound"] is None

    # Generous floor -> CONVERGED.
    conv = build_reference_convergence_report(
        hierarchy,
        p01,
        p12,
        reference_floor={
            "u_relative_l2": 0.1,
            "p_relative_l2": 0.1,
            "absolute_peak_error": 0.1,
        },
    )
    assert conv.status is ReferenceConvergenceStatus.CONVERGED
    assert conv.withhold_accuracy_conclusions is False


def test_compare_remeshed_cad_fem_pair_evaluates_common_interior(tmp_path) -> None:
    spec, coarse_req, _ = pair(tmp_path, order=2)
    # Build a distinct mesh that extends slightly beyond coarse_req on the right
    # edge (simulating a curved boundary remesh sliver outside coarse polygon).
    base = coarse_req.mesh
    max_x = max(x for x, _ in base.nodes)
    shifted_nodes = tuple(
        (x * 1.15 if abs(x - max_x) < 1e-9 else x, y) for x, y in base.nodes
    )
    remeshed = canonicalize_mesh(
        MeshPayload(
            shifted_nodes,
            base.cells,
            CellType(base.cell_type),
            base.boundary_facets,
            base.semantic_tags,
            base.region_ids,
            Unit.MM,
            {**base.source_provenance, "remesh": "expanded-1pct"},
        )
    )
    fine_req = replace(coarse_req, request_id="remeshed", mesh=remeshed)
    coarse_res = result_for(coarse_req, quadratic=True, factor=1.0)
    fine_res = result_for(fine_req, quadratic=True, factor=1.0)

    report = compare_remeshed_cad_fem_pair(
        spec, coarse_req, coarse_res, fine_req, fine_res, tolerances=TOLERANCES
    )
    assert report["geometry_status"] == "REMESHED_CAD_COMMON_INTERIOR"
    assert report["excluded_weight_fraction"] > 0.0
    assert report["common_area_mm2"] < report["fine_area_mm2"]
    # Exact same quadratic field evaluated on the common interior matches to machine precision.
    assert report["quantities"]["u_relative_l2"] < 1e-11
    assert report["quantities"]["p_relative_l2"] < 1e-11


def test_hierarchy_and_convergence_validation_guards(tmp_path) -> None:
    base = _unit_square_mesh()
    h = build_mesh_hierarchy(
        base, cad_source_hash=SOURCE_HASH, geometry_tolerance_mm=1e-2
    )
    transfer = CrossMeshTransfer.between(h.m0_coarse, h.m1_training)
    with pytest.raises(ValueError):
        transfer.prolong_nodal([(0.0,)])
    with pytest.raises(ValueError):
        transfer.restrict_nodal_injection([(0.0,)])
    with pytest.raises(ValueError):
        transfer.restrict_cell_conservative([1.0])
    with pytest.raises(ValueError):
        transfer.prolong_cell_piecewise_constant([1.0])

    s1 = GeometryConvergenceSample.from_mesh(
        "a",
        _quarter_ring_polygon(6),
        cad_source_hash=SOURCE_HASH,
        declared_curve_tolerance_mm=0.2,
        achieved_max_chord_error_mm=0.05,
    )
    s2 = GeometryConvergenceSample.from_mesh(
        "b",
        _quarter_ring_polygon(12),
        cad_source_hash=SOURCE_HASH,
        declared_curve_tolerance_mm=0.2,
        achieved_max_chord_error_mm=0.01,
    )
    with pytest.raises(ValueError, match="at least two"):
        evaluate_geometry_convergence([s1])
    with pytest.raises(ValueError, match="all three"):
        evaluate_geometry_convergence([s1, s2], max_relative_area_change=0.1)

    spec, m0_req, m1_req = pair(tmp_path, order=1)
    m2_req = replace(m0_req, request_id="m2", mesh=refine_fixed_mesh(m0_req.mesh, 4))
    h_ref = build_mesh_hierarchy(
        m0_req.mesh, cad_source_hash=SOURCE_HASH, geometry_tolerance_mm=0.01
    )
    p01 = compare_fem_pair(
        spec,
        m0_req,
        result_for(m0_req),
        m1_req,
        result_for(m1_req),
        tolerances=None,
    )
    p12 = compare_fem_pair(
        spec,
        m1_req,
        result_for(m1_req),
        m2_req,
        result_for(m2_req),
        tolerances=None,
    )
    measured = build_reference_convergence_report(h_ref, p01, p12, reference_floor=None)
    assert measured.status is ReferenceConvergenceStatus.MEASURED_CHANGES_ONLY
    assert measured.withhold_accuracy_conclusions is True
    assert len(measured.report_hash) == 64
    with pytest.raises(ValueError, match="M0 -> M1"):
        build_reference_convergence_report(h_ref, p12, p12, reference_floor=None)
    with pytest.raises(ValueError, match="M1 -> M2"):
        build_reference_convergence_report(h_ref, p01, p01, reference_floor=None)
