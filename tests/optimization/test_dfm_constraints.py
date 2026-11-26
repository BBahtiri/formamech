"""Gate G-V5-4a (DfM) tests: Keep-out zones, min-thickness, overhang & pull-draft constraints."""

from __future__ import annotations

import numpy as np

from piginot.fem.backend_3d import DirichletBC3D, TractionBC3D
from piginot.geometry.mesh3d import make_brick_tet4_mesh
from piginot.optimization.dfm_constraints import (
    DfmConstraintSet,
    KeepOutZoneSpec,
    MinThicknessConstraint,
    OverhangConstraint3D,
    PullDirectionDraftConstraint,
)
from piginot.optimization.topology_opt_3d import (
    TopologyOpt3DConfig,
    TopologyOpt3DState,
    optimize_3d_topology_simp,
)


def test_dfm_constraints_project_and_audit_cleanly() -> None:
    """Keep-out bosses/bores, min-thickness, overhang & pull-draft project densities safely."""
    mesh = make_brick_tet4_mesh(20.0, 10.0, 8.0, nx=4, ny=2, nz=2)
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64)
    centroids = np.mean(nodes[cells], axis=1)

    pad_zone = KeepOutZoneSpec(
        name="mount-pad",
        kind="box",
        lock_solid=True,
        bbox_min_mm=(0.0, 0.0, 0.0),
        bbox_max_mm=(5.0, 10.0, 8.0),
    )
    bore_zone = KeepOutZoneSpec(
        name="center-bore",
        kind="cylinder",
        lock_solid=False,
        origin_mm=(10.0, 5.0, 0.0),
        axis=(0.0, 0.0, 1.0),
        radius_mm=2.0,
        length_mm=8.0,
    )
    dfm = DfmConstraintSet(
        keep_out_zones=(pad_zone, bore_zone),
        min_thickness=MinThicknessConstraint(min_thickness_mm=3.0),
        overhang=OverhangConstraint3D(
            build_direction=(0.0, 0.0, 1.0),
            max_overhang_angle_deg=45.0,
            layer_search_radius_mm=6.0,
        ),
        pull_draft=PullDirectionDraftConstraint(
            pull_direction=(0.0, 0.0, 1.0),
            column_radius_mm=4.0,
        ),
    )

    rho = np.full(len(cells), 0.7, dtype=np.float64)
    projected = dfm.apply_to_densities(rho, centroids)
    assert np.all(projected[pad_zone.element_mask(centroids)] == 1.0)
    assert np.all(projected[bore_zone.element_mask(centroids)] <= 0.01)

    solid_mask = projected >= 0.45
    audit = dfm.audit_solid_mask(solid_mask, centroids)
    assert audit.keep_out_violations == 0
    assert audit.keep_out_zone_count == 2
    assert audit.min_thickness_enabled is True
    assert audit.overhang_enabled is True
    assert audit.pull_draft_enabled is True


def test_topology_opt_with_dfm_keep_out_preserves_mounting_pad_and_certifies() -> None:
    """SIMP optimization with DfmConstraintSet preserves locked zones and passes exact 3D FEM."""
    mesh = make_brick_tet4_mesh(20.0, 8.0, 8.0, nx=4, ny=2, nz=2)
    dfm = DfmConstraintSet(
        keep_out_zones=(
            KeepOutZoneSpec(
                name="clamp-pad",
                kind="box",
                lock_solid=True,
                bbox_min_mm=(0.0, 0.0, 0.0),
                bbox_max_mm=(5.0, 8.0, 8.0),
            ),
            KeepOutZoneSpec(
                name="load-pad",
                kind="box",
                lock_solid=True,
                bbox_min_mm=(15.0, 0.0, 0.0),
                bbox_max_mm=(20.0, 8.0, 8.0),
            ),
        ),
    )
    res = optimize_3d_topology_simp(
        mesh,
        dirichlet_bcs=(
            DirichletBC3D("fix", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
        ),
        traction_bcs=(TractionBC3D("load", ("x_max",), (0.0, -1.0, 0.0)),),
        config=TopologyOpt3DConfig(
            volume_fraction=0.65, max_iterations=6, threshold_density=0.35
        ),
        dfm_constraints=dfm,
    )
    assert res.state is TopologyOpt3DState.CERTIFIED
    assert res.dfm_audit is not None
    assert res.dfm_audit.keep_out_violations == 0
    assert res.dfm_audit.passed is True
