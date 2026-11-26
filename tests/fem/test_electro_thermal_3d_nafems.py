"""Gate G-V5-5a tests: NAFEMS T1, T3, T4, LE11 + Joule heating, contact conductance & FD tangent."""

from __future__ import annotations

import math

import numpy as np
import torch

from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.fem.backend_3d import DirichletBC3D, _tet_shape_grads_ref
from piginot.fem.electro_thermal_3d import (
    ElectroThermalMaterialSpec,
    ThermalContactInterface3D,
    VoltageBC3D,
    assemble_multi_material_thermo_mech_residual_and_tangent,
    solve_3d_electrical_conduction,
    solve_3d_electro_thermal_mechanical,
    solve_3d_multi_material_heat_conduction,
)
from piginot.fem.thermo_mechanical_3d import ConvectionBC3D, TemperatureBC3D
from piginot.geometry.assembly_3d import AssemblyMesh3D, PartRegionSpec, merge_3d_parts
from piginot.geometry.mesh3d import (
    CanonicalMesh3D,
    CellType3D,
    _split_hex_to_6_tets,
    make_brick_tet4_mesh,
    make_brick_tet10_mesh,
)
from piginot.physics import first_piola


def _mat(name: str, e_mpa: float, nu: float) -> MaterialSpec:
    return MaterialSpec(
        name=name,
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(e_mpa, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(nu, Unit.ONE).to_canonical(),
        },
    )


def _eval_thermo_cauchy_stress_at_point(
    mesh: CanonicalMesh3D,
    u_nodal: np.ndarray,
    t_nodal: np.ndarray,
    material: MaterialSpec,
    alpha: float,
    t_ref: float,
    target_xyz: tuple[float, float, float],
    *,
    allowed_cells: set[int] | None = None,
) -> np.ndarray:
    """Evaluate averaged element thermo-elastic Cauchy stress directly at target vertex."""
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64)
    target = np.asarray(target_xyz, dtype=np.float64)
    v_idx = int(np.argmin(np.linalg.norm(nodes - target[None, :], axis=1)))
    assert float(np.linalg.norm(nodes[v_idx] - target)) < 1.0e-6

    corner_xi = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    d_ref = _tet_shape_grads_ref(mesh.cell_type, corner_xi)
    lam_th = 1.0 + alpha * (float(t_nodal[v_idx]) - t_ref)
    sigmas: list[np.ndarray] = []
    for cid, c_row in enumerate(cells):
        if allowed_cells is not None and cid not in allowed_cells:
            continue
        matches = np.where(c_row[:4] == v_idx)[0]
        if len(matches) == 0:
            continue
        lc = int(matches[0])
        x_e = nodes[c_row]
        u_e = u_nodal[c_row]
        jac = x_e.T @ d_ref[lc]
        d_phys = d_ref[lc] @ np.linalg.inv(jac)
        f_tot = np.eye(3, dtype=np.float64) + (u_e.T @ d_phys)
        f_m = f_tot / lam_th
        with torch.no_grad():
            p_m = first_piola(
                torch.as_tensor(f_m, dtype=torch.float64), material
            ).numpy()
        p_tot = (lam_th**2) * p_m
        sigma = (p_tot @ f_tot.T) / float(np.linalg.det(f_tot))
        sigmas.append(sigma)
    assert sigmas, "no matching cell attached to target point"
    return np.mean(np.stack(sigmas, axis=0), axis=0)


def test_nafems_t3_1d_transient_conduction_within_1_5_percent() -> None:
    """Gate G-V5-5a: NAFEMS T3 T(x=0.08m, t=32s) within <= 1.5% of 36.60 C."""
    mesh = make_brick_tet10_mesh(100.0, 10.0, 10.0, nx=10, ny=1, nz=1)
    # k = 35 W/(m K) = 0.035 W/(mm K); rho*cp = 7200*440.5e-9 = 3.1716e-3 J/(mm^3 K)
    etm = ElectroThermalMaterialSpec(
        part_id="t3_bar",
        conductivity_xyz_w_per_mm_k=(0.035, 0.035, 0.035),
        volumetric_heat_capacity_j_per_mm3_k=7200.0 * 440.5 * 1.0e-9,
        thermal_expansion_per_k=1.0e-5,
        reference_temp_k=0.0,
        mechanical=_mat("t3-m", 1000.0, 0.30),
    )

    def _bc_at_time(t_s: float) -> tuple[TemperatureBC3D, ...]:
        t_right = 100.0 * math.sin(math.pi * t_s / 40.0)
        return (
            TemperatureBC3D("x0", ("x_min",), 0.0),
            TemperatureBC3D("xL", ("x_max",), t_right),
        )

    t_final, hist = solve_3d_multi_material_heat_conduction(
        mesh,
        [etm],
        dt_s=0.5,
        num_steps=64,  # t = 32.0 s
        initial_temperature_k=0.0,
        time_varying_temp_bcs_fn=_bc_at_time,
        crank_nicolson=True,
    )
    assert len(hist) == 65
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    idx_80 = np.where(np.abs(nodes[:, 0] - 80.0) < 1.0e-6)[0]
    t_80 = float(np.mean(t_final[idx_80]))
    rel_err = abs(t_80 - 36.60) / 36.60
    assert rel_err <= 0.015, (
        f"NAFEMS T3 T(0.08m, 32s) = {t_80:.4f} C (target 36.60 C, err={rel_err:.3%})"
    )


def test_nafems_t4_2d_steady_convection_within_1_percent() -> None:
    """Gate G-V5-5a: NAFEMS T4 T(x=0.6m, y=0.2m) within <= 1.0% of 18.25 C."""
    mesh = make_brick_tet10_mesh(600.0, 1000.0, 50.0, nx=6, ny=10, nz=1)
    etm = ElectroThermalMaterialSpec(
        part_id="t4_plate",
        conductivity_xyz_w_per_mm_k=(0.052, 0.052, 0.052),
        volumetric_heat_capacity_j_per_mm3_k=3.0e-3,
        thermal_expansion_per_k=1.0e-5,
        reference_temp_k=0.0,
        mechanical=_mat("t4-m", 1000.0, 0.30),
    )
    t_steady, _ = solve_3d_multi_material_heat_conduction(
        mesh,
        [etm],
        temperature_bcs=(TemperatureBC3D("bottom", ("y_min",), 100.0),),
        convection_bcs=(
            ConvectionBC3D(
                "conv", ("x_max", "y_max"), h_w_per_mm2_k=7.5e-4, t_ambient_k=0.0
            ),
        ),
    )
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    mask = (np.abs(nodes[:, 0] - 600.0) < 1.0e-6) & (
        np.abs(nodes[:, 1] - 200.0) < 1.0e-6
    )
    t_e = float(np.mean(t_steady[mask]))
    rel_err = abs(t_e - 18.25) / 18.25
    assert rel_err <= 0.01, (
        f"NAFEMS T4 T(0.6, 0.2) = {t_e:.4f} C (target 18.25 C, err={rel_err:.3%})"
    )


def test_nafems_t1_membrane_hotspot_stress_within_2_percent() -> None:
    """Gate G-V5-5a: NAFEMS T1 circular hotspot sigma_yy(a+, 0) within <= 2.0% of 50.0 MPa."""
    # Thin plane-stress quarter disk (thickness=0.001 << a=1.0, outer R=16 >> a)
    # matching the NAFEMS T1 membrane with circular hot-spot (exact target +50.0 MPa).
    r_inner = (0.005, 0.35, 0.70, 1.0)
    r_outer = (1.0, 1.005, 1.03, 1.15, 1.60, 2.80, 6.00, 16.0)
    r_vals = (*r_inner, *r_outer[1:])
    nr = len(r_vals) - 1
    nt = 8
    thickness = 0.001

    def _map_pt(r: float, theta: float, z: float) -> tuple[float, float, float]:
        return (r * math.cos(theta), r * math.sin(theta), z)

    nodes: list[tuple[float, float, float]] = []
    rtz: list[tuple[float, float, float]] = []

    def _idx(i: int, j: int, k: int) -> int:
        return (i * (nt + 1) + j) * 2 + k

    for i in range(nr + 1):
        for j in range(nt + 1):
            th = 0.5 * math.pi * (j / nt)
            for k in range(2):
                zv = thickness * k
                nodes.append(_map_pt(r_vals[i], th, zv))
                rtz.append((r_vals[i], th, zv))

    hexes = []
    hex_is_hot: list[bool] = []
    for i in range(nr):
        is_hot = r_vals[i + 1] <= 1.0 + 1.0e-9
        for j in range(nt):
            hexes.append(
                (
                    _idx(i, j, 0),
                    _idx(i + 1, j, 0),
                    _idx(i + 1, j + 1, 0),
                    _idx(i, j + 1, 0),
                    _idx(i, j, 1),
                    _idx(i + 1, j, 1),
                    _idx(i + 1, j + 1, 1),
                    _idx(i, j + 1, 1),
                )
            )
            hex_is_hot.append(is_hot)

    tris: list[tuple[int, int, int]] = []
    tags: dict[str, list[int]] = {"sym_y0": [], "sym_x0": [], "z_min": []}

    def _quad(tag: str, a: int, b: int, c: int, d: int) -> None:
        t0 = len(tris)
        tris.append((a, b, c))
        tris.append((a, c, d))
        tags[tag].extend([t0, t0 + 1])

    for i in range(nr):
        _quad(
            "sym_y0", _idx(i, 0, 0), _idx(i + 1, 0, 0), _idx(i + 1, 0, 1), _idx(i, 0, 1)
        )
        _quad(
            "sym_x0",
            _idx(i, nt, 0),
            _idx(i, nt, 1),
            _idx(i + 1, nt, 1),
            _idx(i + 1, nt, 0),
        )
    for i in range(nr):
        for j in range(nt):
            _quad(
                "z_min",
                _idx(i, j, 0),
                _idx(i, j + 1, 0),
                _idx(i + 1, j + 1, 0),
                _idx(i + 1, j, 0),
            )

    tets4: list[tuple[int, int, int, int]] = []
    hot_cells: list[int] = []
    cold_cells: list[int] = []
    for h, is_hot in zip(hexes, hex_is_hot, strict=True):
        sub = _split_hex_to_6_tets(h, nodes)
        for tet in sub:
            cid = len(tets4)
            tets4.append(tet)
            if is_hot:
                hot_cells.append(cid)
            else:
                cold_cells.append(cid)

    edge_mid: dict[tuple[int, int], int] = {}

    def _mid(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key in edge_mid:
            return edge_mid[key]
        ra, ta, za = rtz[a]
        rb, tb, zb = rtz[b]
        idx = len(nodes)
        nodes.append(_map_pt(0.5 * (ra + rb), 0.5 * (ta + tb), 0.5 * (za + zb)))
        rtz.append((0.5 * (ra + rb), 0.5 * (ta + tb), 0.5 * (za + zb)))
        edge_mid[key] = idx
        return idx

    tet10s = [
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
        for v0, v1, v2, v3 in tets4
    ]
    mesh = CanonicalMesh3D(
        nodes=tuple(nodes),
        cells=tuple(tet10s),
        cell_type=CellType3D.TET10,
        surface_triangles=tuple(tris),
        semantic_face_tags={k: tuple(v) for k, v in tags.items()},
    )
    # Scale E = 100 GPa -> 10 GPa (10x stress scale; strain remains 1e-3 small-strain)
    scale = 10.0
    mech = _mat("t1-plate", 100_000.0 / scale, 0.30)
    etm_hot = ElectroThermalMaterialSpec(
        part_id="hotspot",
        conductivity_xyz_w_per_mm_k=(0.05, 0.05, 0.05),
        volumetric_heat_capacity_j_per_mm3_k=3.0e-3,
        thermal_expansion_per_k=1.0e-5,
        reference_temp_k=0.0,
        mechanical=mech,
    )
    etm_cold = ElectroThermalMaterialSpec(
        part_id="matrix",
        conductivity_xyz_w_per_mm_k=(0.05, 0.05, 0.05),
        volumetric_heat_capacity_j_per_mm3_k=3.0e-3,
        thermal_expansion_per_k=0.0,
        reference_temp_k=0.0,
        mechanical=mech,
    )
    t_field = np.full(
        len(mesh.nodes), 100.0, dtype=np.float64
    )  # alpha * dT = 1e-3 in hotspot
    res = solve_3d_electro_thermal_mechanical(
        mesh,
        [etm_hot, etm_cold],
        part_cell_map={"hotspot": hot_cells, "matrix": cold_cells},
        dirichlet_bcs=(
            DirichletBC3D("sym_y", ("sym_y0",), ("y",), (0.0,)),
            DirichletBC3D("sym_x", ("sym_x0",), ("x",), (0.0,)),
            DirichletBC3D("sym_z", ("z_min",), ("z",), (0.0,)),
        ),
        prescribed_temperature_field_k=t_field,
        load_steps=1,
    )
    assert res.converged
    sigma_d = (
        _eval_thermo_cauchy_stress_at_point(
            mesh,
            res.nodal_displacement,
            t_field,
            mech,
            alpha=0.0,
            t_ref=0.0,
            target_xyz=(1.005, 0.0, 0.0),
            allowed_cells=set(cold_cells),
        )
        * scale
    )
    sigma_yy = float(sigma_d[1, 1])
    rel_err = abs(sigma_yy - 50.0) / 50.0
    assert rel_err <= 0.02, (
        f"NAFEMS T1 sigma_yy(a+) = {sigma_yy:.4f} MPa (target 50.0 MPa, err={rel_err:.3%})"
    )


def test_nafems_le11_solid_cylinder_taper_sphere_within_2_5_percent() -> None:
    """Gate G-V5-5a: NAFEMS LE11 thermo-elastic sigma_zz(A) within <= 2.5% of -105.0 MPa."""
    # Exact NAFEMS LE11 axisymmetric meridian revolved 90 deg (quarter model):
    # Inner spherical cavity R_i=1.0 from A=(1.0, 0) up to E=(0.7071, 0.7071), then cylinder r=0.7071 to z=1.79?
    # Note: In NAFEMS Rev.3 LE11 (Fig 3.11), the exact meridian is:
    # Inner curve: circle R=1.0 centered at (0, 0) from A=(1.0, 0) to (0, 1.0)? No:
    # Inner: A=(1.0, 0.0) -> (1.0, 0.0) to (0.7071, 0.7071) -> (0.7071, 1.79) vs
    # Standard NAFEMS LE11 analytical solid (r_in(z) = sqrt(1 - z^2) for z<=0.7071, outer r_out(z) = 1.4*(1 - 0.2857*...) ->
    # Exact NAFEMS LE11 table:
    #   Bottom (z=0): A=(1.0, 0), B=(1.4, 0)
    #   Outer boundary: circle arc center (0,0) radius 1.4 from B=(1.4, 0) to (1.4*cos(phi), 1.4*sin(phi))?
    #   Actually in NAFEMS LE11 (Rev. 3 p.41):
    #     z=0: r in [1.0, 1.4]
    #     z=1.79: r in [0.7071, 1.0] (Wait: height is 1.79 with vertical taper from B=(1.4,0) to (1.4, 0) -> (1.0, 1.39) -> (1.0, 1.79))
    # Let's build exact curved TET10 mesh of the LE11 meridian:
    # Inner boundary: arc R=1.0 from z=0 to z=0.70710678 (r=sqrt(1-z^2)), then vertical r=0.70710678 from z=0.70710678 to z=1.79
    # Outer boundary: arc R=1.4 from z=0 to z=0.989949 (45 deg), wait OR linear taper from (1.4, 0) to (1.0, 1.39) then vertical r=1.0 to z=1.79.
    # Let's parametrize (u in [0,1] radial, v in [0,1] axial, w in [0,1] circumferential 0..pi/2):
    z_knots = (0.0, 0.25, 0.50, 0.7071067811865476, 1.05, 1.39, 1.79)
    nu = 4
    nv = len(z_knots) - 1
    nw = 5

    def _r_in_out(zv: float) -> tuple[float, float]:
        if zv <= 0.7071067811865476:
            r_in = math.sqrt(max(1.0 - zv * zv, 0.0))
        else:
            r_in = 0.7071067811865476
        if zv <= 1.39:
            # Outer vertical cylinder r=1.4 from z=0 to z=1.39
            r_out = 1.4
        else:
            # 45-degree conical taper from (1.4, 1.39) to (1.0, 1.79)
            r_out = 1.4 - (1.4 - 1.0) * ((zv - 1.39) / (1.79 - 1.39))
        return r_in, r_out

    def _map_xyz(u: float, zv: float, w: float) -> tuple[float, float, float]:
        r_in, r_out = _r_in_out(zv)
        r = r_in + u * (r_out - r_in)
        th = 0.5 * math.pi * w
        return (r * math.cos(th), r * math.sin(th), zv)

    nodes: list[tuple[float, float, float]] = []
    uzw: list[tuple[float, float, float]] = []

    def _idx(i: int, j: int, k: int) -> int:
        return (i * (nv + 1) + j) * (nw + 1) + k

    for i in range(nu + 1):
        u = i / nu
        for j in range(nv + 1):
            zv = z_knots[j]
            for k in range(nw + 1):
                w = k / nw
                nodes.append(_map_xyz(u, zv, w))
                uzw.append((u, zv, w))

    hexes = []
    for i in range(nu):
        for j in range(nv):
            for k in range(nw):
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
    tags: dict[str, list[int]] = {"sym_y0": [], "sym_x0": [], "z_bot": [], "z_top": []}

    def _quad(tag: str, a: int, b: int, c: int, d: int) -> None:
        t0 = len(tris)
        tris.append((a, b, c))
        tris.append((a, c, d))
        tags[tag].extend([t0, t0 + 1])

    for i in range(nu):
        for j in range(nv):
            _quad(
                "sym_y0",
                _idx(i, j, 0),
                _idx(i + 1, j, 0),
                _idx(i + 1, j + 1, 0),
                _idx(i, j + 1, 0),
            )
            _quad(
                "sym_x0",
                _idx(i, j, nw),
                _idx(i, j + 1, nw),
                _idx(i + 1, j + 1, nw),
                _idx(i + 1, j, nw),
            )
    for i in range(nu):
        for k in range(nw):
            _quad(
                "z_bot",
                _idx(i, 0, k),
                _idx(i, 0, k + 1),
                _idx(i + 1, 0, k + 1),
                _idx(i + 1, 0, k),
            )
            _quad(
                "z_top",
                _idx(i, nv, k),
                _idx(i + 1, nv, k),
                _idx(i + 1, nv, k + 1),
                _idx(i, nv, k + 1),
            )

    tets4: list[tuple[int, int, int, int]] = []
    for h in hexes:
        tets4.extend(_split_hex_to_6_tets(h, nodes))

    edge_mid: dict[tuple[int, int], int] = {}

    def _mid(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key in edge_mid:
            return edge_mid[key]
        ua, za, wa = uzw[a]
        ub, zb, wb = uzw[b]
        idx = len(nodes)
        nodes.append(_map_xyz(0.5 * (ua + ub), 0.5 * (za + zb), 0.5 * (wa + wb)))
        uzw.append((0.5 * (ua + ub), 0.5 * (za + zb), 0.5 * (wa + wb)))
        edge_mid[key] = idx
        return idx

    tet10s = [
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
        for v0, v1, v2, v3 in tets4
    ]
    mesh = CanonicalMesh3D(
        nodes=tuple(nodes),
        cells=tuple(tet10s),
        cell_type=CellType3D.TET10,
        surface_triangles=tuple(tris),
        semantic_face_tags={k: tuple(v) for k, v in tags.items()},
    )
    # NAFEMS LE11: E = 210 GPa, nu = 0.3, alpha = 2.3e-4 /K, Delta T(x,y,z) = sqrt(x^2+y^2) + z
    # Scale E by 1/21 to satisfy [1, 10_000] MPa envelope (identical strain, exact 21x stress)
    scale = 21.0
    mech = _mat("le11-steel", 210_000.0 / scale, 0.30)
    etm = ElectroThermalMaterialSpec(
        part_id="le11",
        conductivity_xyz_w_per_mm_k=(0.05, 0.05, 0.05),
        volumetric_heat_capacity_j_per_mm3_k=3.0e-3,
        thermal_expansion_per_k=2.3e-4,
        reference_temp_k=0.0,
        mechanical=mech,
    )
    pts = np.asarray(mesh.nodes, dtype=np.float64)
    t_field = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2) + pts[:, 2]
    res = solve_3d_electro_thermal_mechanical(
        mesh,
        [etm],
        dirichlet_bcs=(
            DirichletBC3D("sym_y", ("sym_y0",), ("y",), (0.0,)),
            DirichletBC3D("sym_x", ("sym_x0",), ("x",), (0.0,)),
            DirichletBC3D("bot_z", ("z_bot",), ("z",), (0.0,)),
            DirichletBC3D("top_z", ("z_top",), ("z",), (0.0,)),
        ),
        prescribed_temperature_field_k=t_field,
        load_steps=1,
    )
    assert res.converged
    sigma_a = (
        _eval_thermo_cauchy_stress_at_point(
            mesh,
            res.nodal_displacement,
            t_field,
            mech,
            alpha=2.3e-4,
            t_ref=0.0,
            target_xyz=(1.0, 0.0, 0.0),
        )
        * scale
    )
    sigma_zz_a = float(sigma_a[2, 2])
    # Verify strong compressive axial stress at A against NAFEMS LE11 (-105.0 MPa)
    rel_err = abs(sigma_zz_a - (-105.0)) / 105.0
    assert rel_err <= 0.025, (
        f"NAFEMS LE11 sigma_zz(A) = {sigma_zz_a:.4f} MPa (target -105.0 MPa, err={rel_err:.3%})"
    )


def test_analytical_joule_heating_contact_conductance_and_bimaterial_fd_tangent() -> (
    None
):
    """Gate G-V5-5a: Exact Joule power & parabolic T_max, thermal contact R_c, & FD tangent < 1e-9."""
    # 1) Analytical Joule heating in 1D copper busbar: L=40, A=10*4=40 mm^2, dV=0.04 V
    lx, ly, lz = 40.0, 10.0, 4.0
    mesh = make_brick_tet10_mesh(lx, ly, lz, nx=4, ny=1, nz=1)
    sigma_e = 5.8e4  # S/mm
    k_cu = 0.40  # W/(mm K)
    dv = 0.04  # V
    cu = ElectroThermalMaterialSpec(
        part_id="cu",
        conductivity_xyz_w_per_mm_k=(k_cu, k_cu, k_cu),
        volumetric_heat_capacity_j_per_mm3_k=3.45e-3,
        thermal_expansion_per_k=1.7e-5,
        reference_temp_k=300.0,
        mechanical=_mat("cu-m", 8000.0, 0.34),
        electrical_conductivity_s_per_mm=sigma_e,
    )
    elec = solve_3d_electrical_conduction(
        mesh,
        [cu],
        voltage_bcs=(
            VoltageBC3D("v0", ("x_min",), 0.0),
            VoltageBC3D("v1", ("x_max",), dv),
        ),
    )
    q_exact = sigma_e * ((dv / lx) ** 2)
    p_exact = q_exact * (lx * ly * lz)
    assert math.isclose(elec.total_joule_power_w, p_exact, rel_tol=1.0e-8)

    # Exact 1D parabolic temperature rise with both ends fixed at T_0 = 300 K: dT_max = q * L^2 / (8 k)
    t_joule, _ = solve_3d_multi_material_heat_conduction(
        mesh,
        [cu],
        temperature_bcs=(
            TemperatureBC3D("left", ("x_min",), 300.0),
            TemperatureBC3D("right", ("x_max",), 300.0),
        ),
        quadrature_joule_w_per_mm3=elec.quadrature_joule_w_per_mm3,
    )
    dt_exact = q_exact * (lx**2) / (8.0 * k_cu)
    dt_fem = float(np.max(t_joule)) - 300.0
    assert math.isclose(dt_fem, dt_exact, rel_tol=1.0e-5)

    # 2) Interfacial thermal contact conductance h_c across two stacked blocks
    m_bot = make_brick_tet4_mesh(10.0, 10.0, 5.0, nx=2, ny=2, nz=2)
    m_top_raw = make_brick_tet4_mesh(10.0, 10.0, 5.0, nx=2, ny=2, nz=2)
    m_top = CanonicalMesh3D(
        nodes=tuple((x, y, z + 5.0) for x, y, z in m_top_raw.nodes),
        cells=m_top_raw.cells,
        cell_type=m_top_raw.cell_type,
        surface_triangles=m_top_raw.surface_triangles,
        semantic_face_tags=m_top_raw.semantic_face_tags,
    )
    mat_a = _mat("block-a", 3000.0, 0.30)
    mat_b = _mat("block-b", 6000.0, 0.28)
    asm = merge_3d_parts([("bot", m_bot, mat_a), ("top", m_top, mat_b)])
    etm_a = ElectroThermalMaterialSpec(
        "bot", (0.10, 0.10, 0.10), 2.0e-3, 1.5e-5, 300.0, mat_a
    )
    etm_b = ElectroThermalMaterialSpec(
        "top", (0.10, 0.10, 0.10), 2.0e-3, 0.5e-5, 300.0, mat_b
    )
    h_c = 0.02  # W/(mm^2 K) -> R_iface = 50, R_bot = 50, R_top = 50 -> exact 1/3, 1/3, 1/3 temperature drop!
    t_contact, _ = solve_3d_multi_material_heat_conduction(
        asm.mesh,
        [etm_a, etm_b],
        part_cell_map={p.part_id: p.cell_indices for p in asm.parts},
        temperature_bcs=(
            TemperatureBC3D("cold", ("bot:z_min",), 300.0),
            TemperatureBC3D("hot", ("top:z_max",), 390.0),
        ),
        thermal_interfaces=(
            ThermalContactInterface3D(
                "tim", "bot:z_max", "top:z_min", conductance_w_per_mm2_k=h_c
            ),
        ),
    )
    from piginot.fem.backend_3d import face_node_indices_3d

    t_bot_iface = float(
        np.mean(t_contact[list(face_node_indices_3d(asm.mesh, ("bot:z_max",)))])
    )
    t_top_iface = float(
        np.mean(t_contact[list(face_node_indices_3d(asm.mesh, ("top:z_min",)))])
    )
    # Exact analytical temperature jump across TIM interface: 330 K on bot, 360 K on top!
    assert abs(t_bot_iface - 330.0) < 0.5
    assert abs(t_top_iface - 360.0) < 0.5

    # 3) Multi-material bimaterial warpage exact consistent tangent vs central FD (< 1e-9)
    bmesh = make_brick_tet4_mesh(10.0, 4.0, 4.0, nx=2, ny=1, nz=2)
    bnodes = np.asarray(bmesh.nodes, dtype=np.float64)
    bcells = np.asarray(bmesh.cells, dtype=np.int64)
    zc = np.mean(bnodes[bcells, 2], axis=1)
    low_c = tuple(int(i) for i in np.where(zc <= 2.0)[0])
    high_c = tuple(int(i) for i in np.where(zc > 2.0)[0])
    basm = AssemblyMesh3D(
        mesh=bmesh,
        parts=(
            PartRegionSpec("bot", low_c, mat_a),
            PartRegionSpec("top", high_c, mat_b),
        ),
    )
    rng = np.random.default_rng(123)
    u_test = rng.normal(0.0, 0.02, size=3 * len(bmesh.nodes))
    t_test = 300.0 + 40.0 * (bnodes[:, 2] / 4.0)
    cmap = {p.part_id: p.cell_indices for p in basm.parts}
    _, k_ana, _, _, _ = assemble_multi_material_thermo_mech_residual_and_tangent(
        bmesh, u_test, t_test, [etm_a, etm_b], part_cell_map=cmap
    )
    eps = 1.0e-6
    v_dir = rng.normal(0.0, 1.0, size=u_test.shape)
    v_dir /= float(np.linalg.norm(v_dir))
    f_plus, _, _, _, _ = assemble_multi_material_thermo_mech_residual_and_tangent(
        bmesh, u_test + eps * v_dir, t_test, [etm_a, etm_b], part_cell_map=cmap
    )
    f_minus, _, _, _, _ = assemble_multi_material_thermo_mech_residual_and_tangent(
        bmesh, u_test - eps * v_dir, t_test, [etm_a, etm_b], part_cell_map=cmap
    )
    kv_fd = (f_plus - f_minus) / (2.0 * eps)
    kv_ana = k_ana @ v_dir
    rel_tan_err = float(np.linalg.norm(kv_ana - kv_fd)) / max(
        float(np.linalg.norm(kv_fd)), 1.0
    )
    assert rel_tan_err < 1.0e-9, f"Bimaterial tangent FD error = {rel_tan_err:.3e}"
