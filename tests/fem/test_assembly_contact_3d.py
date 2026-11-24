"""Verification tests for Phase 4A (V3-4A) multi-part 3D assemblies & 3D contact."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from piginot.cad.importing_3d import (
    classify_faces_by_bbox,
    generate_3d_box_step,
    import_cad_3d,
)
from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.fem.backend_3d import DirichletBC3D, TractionBC3D
from piginot.fem.contact_3d import (
    RigidSphereIndenter3D,
    assemble_rigid_sphere_contact_3d,
    solve_3d_assembly_and_contact,
    third_medium_energy_density,
)
from piginot.geometry.assembly_3d import (
    AssemblyMesh3D,
    InterfaceKind,
    InterfacePairSpec,
    PartRegionSpec,
    classify_assembly_faces_by_bbox,
    merge_3d_parts,
)
from piginot.geometry.mesh3d import CanonicalMesh3D, CellType3D


def _mat(name: str, e_mpa: float, nu: float = 0.0) -> MaterialSpec:
    return MaterialSpec(
        name=name,
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(e_mpa, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(nu, Unit.ONE).to_canonical(),
        },
    )


def _mirror_mesh_x(mesh: CanonicalMesh3D, x_plane: float) -> CanonicalMesh3D:
    """Mirror a TET10 mesh across ``x == x_plane`` preserving positive orientation & conformal interface."""
    nodes = tuple((2.0 * x_plane - p[0], p[1], p[2]) for p in mesh.nodes)
    # Mirroring flips signed volume -> swap corner vertices 0 <-> 1 and corresponding midside edges:
    # TET10: (v0, v1, v2, v3, e01, e12, e20, e03, e13, e23) -> (v1, v0, v2, v3, e01, e20, e12, e13, e03, e23)
    cells = tuple(
        (c[1], c[0], c[2], c[3], c[4], c[6], c[5], c[8], c[7], c[9]) for c in mesh.cells
    )
    tris = tuple((a, c, b) for a, b, c in mesh.surface_triangles)
    return CanonicalMesh3D(
        nodes=nodes,
        cells=cells,
        cell_type=mesh.cell_type,
        surface_triangles=tris,
        semantic_face_tags=mesh.semantic_face_tags,
        unit=mesh.unit,
    )


def test_bimaterial_bonded_3d_assembly_matches_analytical_series_stiffness(
    tmp_path: Path,
) -> None:
    """Two-material bar in series (E1=10 MPa, E2=30 MPa, nu=0) under small tension."""
    step = generate_3d_box_step(tmp_path / "half.step", lx=10.0, ly=6.0, lz=6.0)
    imp_l = import_cad_3d(step, mesh_size_mm=6.0, element_type=CellType3D.TET10)
    mesh_r = _mirror_mesh_x(imp_l.mesh, x_plane=10.0)

    e1, e2 = 10.0, 30.0
    asm = merge_3d_parts(
        [
            ("left_soft", imp_l.mesh, _mat("soft", e1, nu=0.0)),
            ("right_stiff", mesh_r, _mat("stiff", e2, nu=0.0)),
        ],
        merge_coincident_tolerance_mm=1.0e-5,
    )
    assert len(asm.parts) == 2
    assert len(asm.assembly_id) == 64

    bbox = classify_assembly_faces_by_bbox(asm)
    sigma = 0.05  # small traction MPa -> near-exact linear series response
    res = solve_3d_assembly_and_contact(
        asm,
        dirichlet_bcs=(
            DirichletBC3D("fix_x0", bbox["x_min"], ("x",), (0.0,)),
            DirichletBC3D("fix_y0", bbox["y_min"], ("y",), (0.0,)),
            DirichletBC3D("fix_z0", bbox["z_min"], ("z",), (0.0,)),
        ),
        traction_bcs=(TractionBC3D("pull_x", bbox["x_max"], (sigma, 0.0, 0.0)),),
        load_steps=1,
    )
    assert res.converged
    # Exact small-strain total elongation: L1*(sigma/E1) + L2*(sigma/E2)
    u_exact_end = 10.0 * (sigma / e1) + 10.0 * (sigma / e2)
    nodes = np.asarray(asm.mesh.nodes, dtype=np.float64)
    end_nodes = np.where(np.abs(nodes[:, 0] - 20.0) < 1.0e-3)[0]
    u_x_end = float(np.mean(res.nodal_displacement[end_nodes, 0]))
    assert abs(u_x_end - u_exact_end) / u_exact_end < 0.015


def test_3d_rigid_sphere_contact_fd_tangent_and_augmented_lagrangian_solve(
    tmp_path: Path,
) -> None:
    step = generate_3d_box_step(tmp_path / "pad.step", lx=12.0, ly=12.0, lz=6.0)
    imp = import_cad_3d(step, mesh_size_mm=6.0, element_type=CellType3D.TET10)
    faces = classify_faces_by_bbox(imp)

    # 1. Exact central finite-difference verification of 3D contact consistent tangent
    nodes_np = np.asarray(imp.mesh.nodes, dtype=np.float64)
    indenter = RigidSphereIndenter3D(
        name="punch",
        contact_face_tags=faces["z_max"],
        center_mm=(6.0, 6.0, 10.2),  # R=5.0 -> 0.8 mm indentation into z=6 top face
        radius_mm=5.0,
        penalty_mpa_per_mm=150.0,
        friction_mu=0.0,
    )
    from piginot.fem.backend_3d import face_node_indices_3d

    c_nids = face_node_indices_3d(imp.mesh, faces["z_max"])
    u0 = np.zeros(3 * len(imp.mesh.nodes), dtype=np.float64)
    lam0 = np.full(len(c_nids), 1.5, dtype=np.float64)
    f0, k0, pen0, _, active0, _ = assemble_rigid_sphere_contact_3d(
        nodes_np, u0, c_nids, indenter, lam0
    )
    assert active0 > 0 and pen0 > 0.5
    rng = np.random.default_rng(11)
    du = rng.standard_normal(u0.shape)
    eps = 1.0e-7
    fp, _, _, _, _, _ = assemble_rigid_sphere_contact_3d(
        nodes_np, u0 + eps * du, c_nids, indenter, lam0
    )
    fm, _, _, _, _, _ = assemble_rigid_sphere_contact_3d(
        nodes_np, u0 - eps * du, c_nids, indenter, lam0
    )
    fd_dir = (fp - fm) / (2.0 * eps)
    an_dir = np.asarray(k0 @ du, dtype=np.float64)
    assert float(np.linalg.norm(fd_dir - an_dir) / np.linalg.norm(an_dir)) < 1.0e-8

    # 2. Full nonlinear 3D AL rigid-indenter contact solve with smooth indentation ramp
    indenter_solve = RigidSphereIndenter3D(
        name="punch-solve",
        contact_face_tags=faces["z_max"],
        initial_center_mm=(6.0, 6.0, 11.0),  # tangent to z=6 top face
        center_mm=(6.0, 6.0, 10.6),  # 0.4 mm smooth indentation
        radius_mm=5.0,
        penalty_mpa_per_mm=40.0,
        friction_mu=0.0,
    )
    asm = AssemblyMesh3D(
        mesh=imp.mesh,
        parts=(
            PartRegionSpec(
                "elastomer_pad",
                tuple(range(len(imp.mesh.cells))),
                _mat("pad-nh", 8.0, nu=0.35),
            ),
        ),
    )
    res = solve_3d_assembly_and_contact(
        asm,
        dirichlet_bcs=(
            DirichletBC3D(
                "clamp_bot", faces["z_min"], ("x", "y", "z"), (0.0, 0.0, 0.0)
            ),
        ),
        indenter=indenter_solve,
        load_steps=4,
        max_al_iterations=8,
        penetration_tol_mm=2.0e-3,
    )
    assert res.converged is True
    assert res.active_contact_nodes > 0
    assert res.max_penetration_mm < 5.0e-3
    assert res.contact_force_n[2] < 0.0  # downward push by indenter
    assert res.min_det_f > 0.5

    # 3. Third-Medium Contact void energy positivity & convexity sanity
    f_comp = torch.diag(torch.tensor([1.0, 1.0, 0.4], dtype=torch.float64)).unsqueeze(0)
    f_id = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    assert float(third_medium_energy_density(f_id).item()) == 0.0
    assert float(third_medium_energy_density(f_comp).item()) > 0.0

    # 4. Exact FD check & nonlinear AL solve for 3D Coulomb frictional contact (mu = 0.25)
    indenter_fric = RigidSphereIndenter3D(
        name="punch-fric",
        contact_face_tags=faces["z_max"],
        center_mm=(6.0, 6.0, 10.2),
        radius_mm=5.0,
        penalty_mpa_per_mm=150.0,
        friction_mu=0.25,
    )
    u_slide = np.zeros_like(u0)
    u_slide[0::3] = 0.03
    u_slide[1::3] = -0.02
    f_fr0, k_fr0, _, cforce_fr0, active_fr0, _ = assemble_rigid_sphere_contact_3d(
        nodes_np, u_slide, c_nids, indenter_fric, lam0
    )
    assert active_fr0 > 0
    # Net contact force exerted on body must equal -sum(f_c) identically (normal + friction)
    f_sum = -np.sum(f_fr0.reshape(-1, 3), axis=0)
    assert float(np.linalg.norm(np.asarray(cforce_fr0) - f_sum)) < 1.0e-12
    fp_fr, _, _, _, _, _ = assemble_rigid_sphere_contact_3d(
        nodes_np, u_slide + eps * du, c_nids, indenter_fric, lam0
    )
    fm_fr, _, _, _, _, _ = assemble_rigid_sphere_contact_3d(
        nodes_np, u_slide - eps * du, c_nids, indenter_fric, lam0
    )
    fd_fr = (fp_fr - fm_fr) / (2.0 * eps)
    an_fr = np.asarray(k_fr0 @ du, dtype=np.float64)
    assert float(np.linalg.norm(fd_fr - an_fr) / np.linalg.norm(an_fr)) < 1.0e-8

    indenter_fric_solve = RigidSphereIndenter3D(
        name="punch-fric-solve",
        contact_face_tags=faces["z_max"],
        initial_center_mm=(6.0, 6.0, 11.0),
        center_mm=(6.4, 6.0, 10.6),
        radius_mm=5.0,
        penalty_mpa_per_mm=40.0,
        friction_mu=0.25,
    )
    res_fric = solve_3d_assembly_and_contact(
        asm,
        dirichlet_bcs=(
            DirichletBC3D(
                "clamp_bot", faces["z_min"], ("x", "y", "z"), (0.0, 0.0, 0.0)
            ),
        ),
        indenter=indenter_fric_solve,
        load_steps=4,
        max_al_iterations=8,
        penetration_tol_mm=2.0e-3,
    )
    assert res_fric.converged is True
    assert res_fric.active_contact_nodes > 0
    assert res_fric.max_penetration_mm < 5.0e-3
    assert res_fric.contact_force_n[2] < 0.0

    # 5. Non-conformal tied interface contract sanity
    iface = InterfacePairSpec(
        name="tie-1",
        kind=InterfaceKind.TIED,
        master_face_tag=faces["z_min"][0],
        slave_face_tag=faces["z_max"][0],
    )
    assert iface.document()["kind"] == "tied"
