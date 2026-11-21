"""Unit tests for 3D canonical meshes and TET4 volumetric-locking refusal."""

from __future__ import annotations

import pytest

from piginot.geometry import (
    CanonicalMesh3D,
    CellType3D,
    VolumetricLockingError,
    make_brick_hex8_mesh,
    make_brick_tet4_mesh,
    make_brick_tet10_mesh,
)


def test_brick_meshes_geometry_ids_and_locking_guard() -> None:
    hex8 = make_brick_hex8_mesh(10.0, 5.0, 4.0, nx=2, ny=1, nz=1)
    tet4 = make_brick_tet4_mesh(10.0, 5.0, 4.0, nx=2, ny=1, nz=1)
    tet10 = make_brick_tet10_mesh(10.0, 5.0, 4.0, nx=2, ny=1, nz=1)

    assert hex8.cell_type is CellType3D.HEX8
    assert tet4.cell_type is CellType3D.TET4
    assert tet10.cell_type is CellType3D.TET10
    assert len(hex8.cells) == 2
    assert len(tet4.cells) == 12
    assert len(tet10.cells) == 12
    assert len(tet10.cells[0]) == 10

    # Deterministic content-addressed mesh_id
    assert len(tet10.mesh_id) == 64
    assert (
        tet10.mesh_id == make_brick_tet10_mesh(10.0, 5.0, 4.0, nx=2, ny=1, nz=1).mesh_id
    )
    assert hex8.mesh_id != tet4.mesh_id

    # Surface geometry: total surface area of 10x5x4 brick = 2*(50 + 40 + 20) = 220
    centroids, normals, areas = tet10.surface_geometry()
    assert len(centroids) == len(tet10.surface_triangles)
    assert len(normals) == len(tet10.surface_triangles)
    assert sum(areas) == pytest.approx(220.0, rel=1.0e-12)

    # Locking refusal: TET4 allowed for compressible, refused for nu >= 0.45
    tet4.validate_incompressibility_locking(0.30)
    with pytest.raises(VolumetricLockingError, match="TET4"):
        tet4.validate_incompressibility_locking(0.45)
    with pytest.raises(VolumetricLockingError, match="TET4"):
        tet4.validate_incompressibility_locking(0.49)
    tet10.validate_incompressibility_locking(0.49)
    hex8.validate_incompressibility_locking(0.49)
    with pytest.raises(ValueError, match="poisson_ratio"):
        tet10.validate_incompressibility_locking(0.5)


def test_canonical_mesh_3d_validation_guards() -> None:
    unit_nodes = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    unit_tet = ((0, 1, 2, 3),)
    unit_tris = ((0, 1, 2),)

    # Degenerate zero-area surface triangle normal fallback
    m_deg = CanonicalMesh3D(
        nodes=unit_nodes,
        cells=unit_tet,
        cell_type=CellType3D.TET4,
        surface_triangles=((0, 0, 1),),
        semantic_face_tags={"face": (0,)},
    )
    _, norms, areas = m_deg.surface_geometry()
    assert areas == (0.0,)
    assert norms == ((0.0, 0.0, 1.0),)

    with pytest.raises(ValueError, match="at least one node"):
        CanonicalMesh3D(
            nodes=(),
            cells=unit_tet,
            cell_type=CellType3D.TET4,
            surface_triangles=(),
            semantic_face_tags={},
        )
    with pytest.raises(ValueError, match="at least one cell"):
        CanonicalMesh3D(
            nodes=unit_nodes,
            cells=(),
            cell_type=CellType3D.TET4,
            surface_triangles=(),
            semantic_face_tags={},
        )
    with pytest.raises(ValueError, match="3 finite coordinates"):
        CanonicalMesh3D(
            nodes=((0.0, float("nan"), 0.0), *unit_nodes[1:]),
            cells=unit_tet,
            cell_type=CellType3D.TET4,
            surface_triangles=unit_tris,
            semantic_face_tags={},
        )
    with pytest.raises(ValueError, match="must have 4 nodes"):
        CanonicalMesh3D(
            nodes=unit_nodes,
            cells=((0, 1, 2),),
            cell_type=CellType3D.TET4,
            surface_triangles=unit_tris,
            semantic_face_tags={},
        )
    with pytest.raises(ValueError, match="out of bounds"):
        CanonicalMesh3D(
            nodes=unit_nodes,
            cells=((0, 1, 2, 99),),
            cell_type=CellType3D.TET4,
            surface_triangles=unit_tris,
            semantic_face_tags={},
        )
    with pytest.raises(ValueError, match="non-positive signed volume"):
        CanonicalMesh3D(
            nodes=unit_nodes,
            cells=((0, 2, 1, 3),),
            cell_type=CellType3D.TET4,
            surface_triangles=unit_tris,
            semantic_face_tags={},
        )
    with pytest.raises(ValueError, match="surface triangle"):
        CanonicalMesh3D(
            nodes=unit_nodes,
            cells=unit_tet,
            cell_type=CellType3D.TET4,
            surface_triangles=((0, 1, 99),),
            semantic_face_tags={},
        )
    with pytest.raises(ValueError, match="out-of-range triangle index"):
        CanonicalMesh3D(
            nodes=unit_nodes,
            cells=unit_tet,
            cell_type=CellType3D.TET4,
            surface_triangles=unit_tris,
            semantic_face_tags={"bad": (5,)},
        )
