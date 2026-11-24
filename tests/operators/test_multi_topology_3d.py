"""Verification tests for Phase 4 (V4-4) multi-topology 3D operators & multi-genus OOD."""

from __future__ import annotations

from pathlib import Path

import torch

from piginot.cad.importing_3d import (
    classify_faces_by_bbox,
    generate_3d_mounting_lug_step,
    import_cad_3d,
)
from piginot.cad.parametric_3d import generate_3d_holed_block_step
from piginot.geometry.mesh3d import CellType3D, make_brick_tet4_mesh
from piginot.operators.geometry_3d import (
    GeometryOOD3DStatus,
    compute_geometry_descriptor_3d,
)
from piginot.operators.multi_topology_3d import (
    MULTI_TOPOLOGY_FEATURE_DIM,
    MultiTopologyOODDetector3D,
    MultiTopologyOperator3D,
    build_multi_topology_surface_tensors,
)


def test_multi_topology_surface_features_and_two_tier_ood_across_genera(
    tmp_path: Path,
) -> None:
    """One 20-channel operator spans genus-0 + genus-1 while refusing unseen genus-3 & OOD."""
    # Genus-0 bricks (chi = 2)
    b1 = make_brick_tet4_mesh(24.0, 12.0, 10.0, nx=2, ny=1, nz=1)
    b2 = make_brick_tet4_mesh(28.0, 13.0, 10.0, nx=2, ny=1, nz=1)
    b_unseen_in = make_brick_tet4_mesh(26.0, 12.5, 10.0, nx=2, ny=1, nz=1)
    b_extreme_ood = make_brick_tet4_mesh(220.0, 3.0, 3.0, nx=2, ny=1, nz=1)

    # Genus-1 holed blocks (chi = 0)
    h1_p = generate_3d_holed_block_step(
        tmp_path / "h1.step", lx=26.0, ly=14.0, lz=10.0, hole_radius=2.8
    )
    h2_p = generate_3d_holed_block_step(
        tmp_path / "h2.step", lx=30.0, ly=15.0, lz=10.0, hole_radius=3.2
    )
    h1 = import_cad_3d(h1_p, mesh_size_mm=9.5, element_type=CellType3D.TET4)
    h2 = import_cad_3d(h2_p, mesh_size_mm=9.5, element_type=CellType3D.TET4)
    f1 = classify_faces_by_bbox(h1)
    f2 = classify_faces_by_bbox(h2)

    # Genus-3 mounting lug (chi = -4): unseen topology!
    lug_p = generate_3d_mounting_lug_step(tmp_path / "lug.step")
    lug = import_cad_3d(lug_p, mesh_size_mm=8.0, element_type=CellType3D.TET4)
    f_lug = classify_faces_by_bbox(lug)

    # 1) Verify 20-channel intrinsic tensors across genus-0, genus-1, genus-3
    for m, c_tags, l_tags in (
        (b1, ("x_min",), ("x_max",)),
        (h1.mesh, tuple(f1["x_min"]), tuple(f1["x_max"])),
        (lug.mesh, tuple(f_lug["z_min"]), tuple(f_lug["cylindrical"])),
    ):
        c_t, n_t, a_t, f_t = build_multi_topology_surface_tensors(
            m, clamped_face_tags=c_tags, loaded_face_tags=l_tags
        )
        assert f_t.shape[-1] == MULTI_TOPOLOGY_FEATURE_DIM == 20
        assert torch.isfinite(f_t).all()
        assert c_t.shape[1] == n_t.shape[1] == a_t.shape[1] == f_t.shape[1]

    # 2) Multi-genus OOD detector trained on genus-0 (chi=2) + genus-1 (chi=0)
    descs_train = [
        compute_geometry_descriptor_3d(
            b1, clamped_face_tags=("x_min",), loaded_face_tags=("x_max",)
        ),
        compute_geometry_descriptor_3d(
            b2, clamped_face_tags=("x_min",), loaded_face_tags=("x_max",)
        ),
        compute_geometry_descriptor_3d(
            h1.mesh,
            clamped_face_tags=tuple(f1["x_min"]),
            loaded_face_tags=tuple(f1["x_max"]),
        ),
        compute_geometry_descriptor_3d(
            h2.mesh,
            clamped_face_tags=tuple(f2["x_min"]),
            loaded_face_tags=tuple(f2["x_max"]),
        ),
    ]
    det = MultiTopologyOODDetector3D.fit(descs_train)
    assert det.allowed_eulers == (0, 2)

    # Accepts unseen genus-0 inside envelope
    rep_in = det.score(
        compute_geometry_descriptor_3d(
            b_unseen_in, clamped_face_tags=("x_min",), loaded_face_tags=("x_max",)
        )
    )
    assert rep_in.status is GeometryOOD3DStatus.IN_DISTRIBUTION

    # Refuses extreme aspect-ratio OOD brick even though chi=2 is allowed
    rep_ood = det.score(
        compute_geometry_descriptor_3d(
            b_extreme_ood, clamped_face_tags=("x_min",), loaded_face_tags=("x_max",)
        )
    )
    assert rep_ood.status is GeometryOOD3DStatus.OUT_OF_DISTRIBUTION

    # Refuses unseen genus-3 mounting lug (chi=-4) as TOPOLOGY_REFUSED
    rep_lug = det.score(
        compute_geometry_descriptor_3d(
            lug.mesh,
            clamped_face_tags=tuple(f_lug["z_min"]),
            loaded_face_tags=tuple(f_lug["cylindrical"]),
        )
    )
    assert rep_lug.status is GeometryOOD3DStatus.TOPOLOGY_REFUSED

    # 3) Shared MultiTopologyOperator3D predicts on genus-0 & genus-1, refuses genus-3
    op = MultiTopologyOperator3D(
        parameter_dim=2, hidden_dim=16, slice_count=4, num_blocks=1
    )
    op.fit_ood(
        [
            (b1, ("x_min",), ("x_max",)),
            (b2, ("x_min",), ("x_max",)),
            (h1.mesh, tuple(f1["x_min"]), tuple(f1["x_max"])),
            (h2.mesh, tuple(f2["x_min"]), tuple(f2["x_max"])),
        ]
    )
    u_g0, ood_g0 = op.predict_nodal_displacement(
        b_unseen_in,
        parameters=(0.0, 0.0),
        clamped_face_tags=("x_min",),
        loaded_face_tags=("x_max",),
    )
    assert u_g0 is not None and ood_g0 is not None and ood_g0.accepted
    assert u_g0.shape == (len(b_unseen_in.nodes), 3)

    u_g1, ood_g1 = op.predict_nodal_displacement(
        h1.mesh,
        parameters=(0.0, 0.0),
        clamped_face_tags=tuple(f1["x_min"]),
        loaded_face_tags=tuple(f1["x_max"]),
    )
    assert u_g1 is not None and ood_g1 is not None and ood_g1.accepted

    u_refused, ood_refused = op.predict_nodal_displacement(
        lug.mesh,
        parameters=(0.0, 0.0),
        clamped_face_tags=tuple(f_lug["z_min"]),
        loaded_face_tags=tuple(f_lug["cylindrical"]),
    )
    assert u_refused is None and ood_refused is not None
    assert ood_refused.status is GeometryOOD3DStatus.TOPOLOGY_REFUSED
