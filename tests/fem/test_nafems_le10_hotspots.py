"""Gate G-V5-1c tests: Element-bound hotspot preservation & NAFEMS LE10 thick elliptic plate."""

from __future__ import annotations

import math

import numpy as np
import torch

from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.fem.backend_3d import (
    DirichletBC3D,
    FemRequest3D,
    ScikitFemBackend3D,
    TorchFemBackend3D,
    TractionBC3D,
    _tet_shape_grads_ref,
)
from piginot.geometry.mesh3d import (
    CanonicalMesh3D,
    CellType3D,
    _split_hex_to_6_tets,
    make_brick_tet4_mesh,
)
from piginot.physics import first_piola


def _build_nafems_le10_tet10_mesh(
    nu: int = 5,
    nv: int = 8,
    z_levels: tuple[float, ...] = (0.0, 0.27, 0.33, 0.6),
) -> CanonicalMesh3D:
    """Build a quarter-symmetry curved TET10 mesh for NAFEMS LE10 thick elliptic plate.

    Geometry (scaled identically in mm; pressure = 1.0 MPa -> invariant stress in MPa):
    - Inner ellipse: a_i = 2.0, b_i = 1.0
    - Outer ellipse: a_o = 3.25, b_o = 2.75
    - Thickness: z in [0.0, 0.6], midplane at z = 0.3
    - Point D = (2.0, 0.0, 0.6) -> target sigma_yy(D) = -5.38 MPa.
    """
    nz = len(z_levels) - 1

    def _map_xyz(u: float, v: float, z_val: float) -> tuple[float, float, float]:
        theta = 0.5 * math.pi * v
        a = 2.0 + u * (3.25 - 2.0)
        b = 1.0 + u * (2.75 - 1.0)
        return (a * math.cos(theta), b * math.sin(theta), z_val)

    nodes: list[tuple[float, float, float]] = []
    uvz_param: list[tuple[float, float, float]] = []

    def _idx(i: int, j: int, k: int) -> int:
        return (i * (nv + 1) + j) * (nz + 1) + k

    for i in range(nu + 1):
        u = i / nu
        for j in range(nv + 1):
            v = j / nv
            for k in range(nz + 1):
                z_val = z_levels[k]
                nodes.append(_map_xyz(u, v, z_val))
                uvz_param.append((u, v, z_val))

    hexes: list[tuple[int, int, int, int, int, int, int, int]] = []
    for i in range(nu):
        for j in range(nv):
            for k in range(nz):
                hexes.append(
                    (
                        _idx(i, j, k),
                        _idx(i + 1, j, k),
                        _idx(i + 1, j + 1, k),
                        _idx(i, j + 1, k),
                        _idx(i, j, k + 1),
                        _idx(i + 1, j, k + 1),
                        _idx(i + 1, j + 1, k + 1),
                        _idx(i, j + 1, k + 1),
                    )
                )

    tris: list[tuple[int, int, int]] = []
    tags: dict[str, list[int]] = {
        "sym_y0": [],
        "sym_x0": [],
        "outer_xy": [],
        "outer_mid_z": [],
        "top_pressure": [],
    }

    def _add_quad(tag: str, a: int, b: int, c: int, d: int) -> None:
        t0 = len(tris)
        tris.append((a, b, c))
        tris.append((a, c, d))
        tags[tag].extend([t0, t0 + 1])

    # j = 0 -> y = 0 symmetry plane
    for i in range(nu):
        for k in range(nz):
            _add_quad(
                "sym_y0",
                _idx(i, 0, k),
                _idx(i + 1, 0, k),
                _idx(i + 1, 0, k + 1),
                _idx(i, 0, k + 1),
            )
    # j = nv -> x = 0 symmetry plane
    for i in range(nu):
        for k in range(nz):
            _add_quad(
                "sym_x0",
                _idx(i, nv, k),
                _idx(i, nv, k + 1),
                _idx(i + 1, nv, k + 1),
                _idx(i + 1, nv, k),
            )
    # i = nu -> outer ellipse boundary (u_x=u_y=0 everywhere; u_z=0 on midplane strip k=1)
    mid_k = nz // 2
    for j in range(nv):
        for k in range(nz):
            t0 = len(tris)
            a, b, c, d = (
                _idx(nu, j, k),
                _idx(nu, j + 1, k),
                _idx(nu, j + 1, k + 1),
                _idx(nu, j, k + 1),
            )
            tris.append((a, b, c))
            tris.append((a, c, d))
            tags["outer_xy"].extend([t0, t0 + 1])
            if k == mid_k:
                tags["outer_mid_z"].extend([t0, t0 + 1])
    # k = nz -> top surface z = 0.6 under 1.0 MPa downward pressure
    for i in range(nu):
        for j in range(nv):
            _add_quad(
                "top_pressure",
                _idx(i, j, nz),
                _idx(i + 1, j, nz),
                _idx(i + 1, j + 1, nz),
                _idx(i, j + 1, nz),
            )

    tets4: list[tuple[int, int, int, int]] = []
    for h in hexes:
        tets4.extend(_split_hex_to_6_tets(h, nodes))

    edge_mid: dict[tuple[int, int], int] = {}
    tet10s: list[tuple[int, ...]] = []

    def _mid(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        existing = edge_mid.get(key)
        if existing is not None:
            return existing
        ua, va, za = uvz_param[a]
        ub, vb, zb = uvz_param[b]
        idx = len(nodes)
        # Map midpoint on curved elliptic parameter space for quadratic exactness
        nodes.append(_map_xyz(0.5 * (ua + ub), 0.5 * (va + vb), 0.5 * (za + zb)))
        uvz_param.append((0.5 * (ua + ub), 0.5 * (va + vb), 0.5 * (za + zb)))
        edge_mid[key] = idx
        return idx

    for v0, v1, v2, v3 in tets4:
        tet10s.append(
            (
                v0,
                v1,
                v2,
                v3,
                _mid(v0, v1),
                _mid(v1, v2),
                _mid(v2, v0),
                _mid(v0, v3),
                _mid(v1, v3),
                _mid(v2, v3),
            )
        )

    return CanonicalMesh3D(
        nodes=tuple(nodes),
        cells=tuple(tet10s),
        cell_type=CellType3D.TET10,
        surface_triangles=tuple(tris),
        semantic_face_tags={k: tuple(v) for k, v in tags.items()},
        metadata={"kind": "nafems-le10-tet10"},
    )


def _evaluate_cauchy_stress_at_vertex(
    mesh: CanonicalMesh3D,
    u_nodal: np.ndarray,
    material: MaterialSpec,
    target_xyz: tuple[float, float, float],
) -> np.ndarray:
    """Evaluate averaged element Cauchy stress directly at the target corner vertex."""
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64)
    target = np.asarray(target_xyz, dtype=np.float64)
    v_idx = int(np.argmin(np.linalg.norm(nodes - target[None, :], axis=1)))
    assert float(np.linalg.norm(nodes[v_idx] - target)) < 1.0e-9

    corner_xi = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    d_ref = _tet_shape_grads_ref(mesh.cell_type, corner_xi)
    sigmas: list[np.ndarray] = []
    for c_row in cells:
        matches = np.where(c_row[:4] == v_idx)[0]
        if len(matches) == 0:
            continue
        local_c = int(matches[0])
        x_e = nodes[c_row]
        u_e = u_nodal[c_row]
        jac = x_e.T @ d_ref[local_c]
        d_phys = d_ref[local_c] @ np.linalg.inv(jac)
        grad_u = u_e.T @ d_phys
        f_mat = np.eye(3, dtype=np.float64) + grad_u
        det_f = float(np.linalg.det(f_mat))
        with torch.no_grad():
            p_t = first_piola(torch.as_tensor(f_mat, dtype=torch.float64), material)
        p_mat = p_t.numpy()
        sigma = (p_mat @ f_mat.T) / det_f
        sigmas.append(sigma)
    assert sigmas, "target vertex not attached to any corner cell"
    return np.mean(np.stack(sigmas, axis=0), axis=0)


def test_element_bound_hotspots_prevent_nodal_smoothing_attenuation() -> None:
    """Gate G-V5-1c: Unsmoothed Gauss-point peak stress strictly >= nodal averaged peak."""
    mesh = make_brick_tet4_mesh(30.0, 6.0, 6.0, nx=6, ny=2, nz=2)
    mat = MaterialSpec(
        name="cantilever-nh",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(1000.0, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(0.30, Unit.ONE).to_canonical(),
        },
    )
    req = FemRequest3D(
        request_id="hotspot-cantilever",
        mesh=mesh,
        material=mat,
        dirichlet_bcs=(
            DirichletBC3D("fix", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
        ),
        traction_bcs=(TractionBC3D("shear", ("x_max",), (0.0, -0.5, 0.0)),),
        load_steps=2,
    )
    res_t = TorchFemBackend3D().solve(req)
    res_s = ScikitFemBackend3D().solve(req)
    assert res_t.converged and res_s.converged

    for res in (res_t, res_s):
        assert res.element_peak_von_mises_mpa is not None
        assert res.quadrature_peak_von_mises_mpa is not None
        assert res.element_max_principal_strain is not None
        assert res.element_peak_von_mises_mpa.shape == (len(mesh.cells),)
        assert res.element_max_principal_strain.shape == (len(mesh.cells),)
        assert res.quadrature_peak_von_mises_mpa >= res.peak_von_mises_mpa - 1.0e-12
        assert res.quadrature_peak_von_mises_mpa > res.peak_von_mises_mpa
        assert res.summary_document()["quadrature_peak_von_mises_mpa"] == (
            res.quadrature_peak_von_mises_mpa
        )

    np.testing.assert_allclose(
        res_t.element_peak_von_mises_mpa,
        res_s.element_peak_von_mises_mpa,
        rtol=1.0e-6,
        atol=1.0e-8,
    )


def test_nafems_le10_thick_elliptic_plate_sigma_yy_within_2_percent() -> None:
    """Gate G-V5-1c: NAFEMS LE10 sigma_yy(D) within <= 2.0% of -5.38 MPa on TET10."""
    mesh = _build_nafems_le10_tet10_mesh(nu=5, nv=8)
    # Dimensionless stress scaling: p/E = 1/210_000 stays strictly small-strain
    # while respecting compressible_neo_hookean's canonical [1, 10_000] MPa envelope.
    stress_scale = 21.0
    mat = MaterialSpec(
        name="nafems-le10-steel",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(
                210_000.0 / stress_scale, Unit.MPA
            ).to_canonical(),
            "poisson_ratio": Quantity(0.30, Unit.ONE).to_canonical(),
        },
    )
    req = FemRequest3D(
        request_id="nafems-le10",
        mesh=mesh,
        material=mat,
        dirichlet_bcs=(
            DirichletBC3D("sym_y", ("sym_y0",), ("y",), (0.0,)),
            DirichletBC3D("sym_x", ("sym_x0",), ("x",), (0.0,)),
            DirichletBC3D("outer_xy", ("outer_xy",), ("x", "y"), (0.0, 0.0)),
            DirichletBC3D("outer_z", ("outer_mid_z",), ("z",), (0.0,)),
        ),
        traction_bcs=(
            TractionBC3D("top_p", ("top_pressure",), (0.0, 0.0, -1.0 / stress_scale)),
        ),
        load_steps=1,
    )
    res = TorchFemBackend3D().solve(req)
    assert res.converged
    assert res.quadrature_peak_von_mises_mpa is not None
    assert res.quadrature_peak_von_mises_mpa >= res.peak_von_mises_mpa - 1.0e-12

    sigma_d = (
        _evaluate_cauchy_stress_at_vertex(
            mesh, res.nodal_displacement, mat, (2.0, 0.0, 0.6)
        )
        * stress_scale
    )
    sigma_yy_d = float(sigma_d[1, 1])
    rel_err = abs(sigma_yy_d - (-5.38)) / 5.38
    assert rel_err <= 0.02, (
        f"NAFEMS LE10 sigma_yy(D) = {sigma_yy_d:.4f} MPa, target -5.38 MPa "
        f"(rel_err={rel_err:.4%})"
    )
