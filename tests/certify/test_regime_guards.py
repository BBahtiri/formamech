"""Unit & Gate G-V5-1e tests for 3D physics-regime transition guards."""

from __future__ import annotations

import numpy as np

from piginot.certify.regime_guards import (
    BcLayoutOODDetector3D,
    ContactGapPairSpec,
    PhysicsRegimeGuard3D,
    PhysicsRegimeGuardConfig3D,
    RegimeFlag,
    compute_bc_layout_descriptor_3d,
)
from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.fem.backend_3d import DirichletBC3D, TractionBC3D
from piginot.geometry.mesh3d import make_brick_tet4_mesh


def test_bc_layout_shift_guard_catches_aerozoo_style_load_shift() -> None:
    """Flags shifted loaded/clamped face patches even on identical geometry."""
    meshes = [
        make_brick_tet4_mesh(lx, 8.0, 8.0, nx=2, ny=1, nz=1)
        for lx in (18.0, 20.0, 22.0, 24.0)
    ]
    train_descs = [
        compute_bc_layout_descriptor_3d(
            m,
            clamped_face_tags=("x_min",),
            loaded_face_tags=("x_max",),
            traction_mpa=(1.0, 0.0, 0.0),
        )
        for m in meshes
    ]
    bc_det = BcLayoutOODDetector3D.fit(train_descs)
    guard = PhysicsRegimeGuard3D(PhysicsRegimeGuardConfig3D(bc_layout_detector=bc_det))

    m_test = make_brick_tet4_mesh(21.0, 8.0, 8.0, nx=2, ny=1, nz=1)
    # In-distribution BC layout
    v_in = guard.evaluate(
        m_test,
        clamped_face_tags=("x_min",),
        loaded_face_tags=("x_max",),
        traction_mpa=(1.0, 0.0, 0.0),
    )
    assert v_in.passed

    # Shifted BC layout: same geometry, load moved to y_max and transverse direction!
    v_shift = guard.evaluate(
        m_test,
        clamped_face_tags=("x_min",),
        loaded_face_tags=("y_max",),
        traction_mpa=(0.0, -1.0, 0.0),
    )
    assert not v_shift.passed
    assert RegimeFlag.BC_LAYOUT_SHIFT in v_shift.triggered_flags


def test_yield_buckling_contact_and_folding_regime_guards() -> None:
    """Gate G-V5-1e: Flags yield crossing, buckling proximity, contact closure & folding."""
    mesh = make_brick_tet4_mesh(20.0, 4.0, 4.0, nx=4, ny=1, nz=1)
    mat = MaterialSpec(
        name="guard-nh",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(1000.0, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(0.30, Unit.ONE).to_canonical(),
        },
    )
    guard = PhysicsRegimeGuard3D(
        PhysicsRegimeGuardConfig3D(
            yield_stress_mpa=250.0,
            yield_safety_factor=0.90,
            min_buckling_margin=1.5,
            contact_gap_pairs=(
                ContactGapPairSpec(
                    name="top_bottom",
                    face_tags_a=("y_min",),
                    face_tags_b=("y_max",),
                    min_clearance_mm=1.0,
                ),
            ),
        )
    )

    # 1) Safe elastic state
    u_safe = np.zeros((len(mesh.nodes), 3), dtype=np.float64)
    v_ok = guard.evaluate(
        mesh,
        nodal_displacement=u_safe,
        min_det_f=0.99,
        element_peak_von_mises_mpa=np.full(len(mesh.cells), 100.0),
        buckling_load_factor_override=3.5,
    )
    assert v_ok.passed

    # 2) Yield onset crossed using unsmoothed element peak
    v_yield = guard.evaluate(
        mesh,
        nodal_displacement=u_safe,
        min_det_f=0.98,
        element_peak_von_mises_mpa=np.array([100.0, 240.0]),
        buckling_load_factor_override=3.0,
    )
    assert not v_yield.passed
    assert RegimeFlag.YIELD_REGIME_CROSSED in v_yield.triggered_flags

    # 3) Live 3D linear buckling proximity check under heavy axial compression
    v_buck = guard.evaluate(
        mesh,
        nodal_displacement=u_safe,
        min_det_f=0.99,
        element_peak_von_mises_mpa=50.0,
        material=mat,
        dirichlet_bcs=(
            DirichletBC3D("fix", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
        ),
        traction_bcs=(TractionBC3D("compress", ("x_max",), (-80.0, 0.0, 0.0)),),
    )
    assert not v_buck.passed
    assert RegimeFlag.BUCKLING_PROXIMITY in v_buck.triggered_flags
    assert v_buck.buckling_load_factor is not None
    assert v_buck.buckling_load_factor < 1.5

    # 4) Contact gap closure + Jacobian folding
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    u_crush = np.zeros_like(nodes)
    u_crush[:, 1] = -0.85 * nodes[:, 1]  # squeezes y_max (4.0) down to 0.6 <= 1.0
    v_gap = guard.evaluate(
        mesh,
        nodal_displacement=u_crush,
        min_det_f=-0.1,
        element_peak_von_mises_mpa=50.0,
        buckling_load_factor_override=5.0,
    )
    assert not v_gap.passed
    assert RegimeFlag.CONTACT_GAP_CLOSED in v_gap.triggered_flags
    assert RegimeFlag.JACOBIAN_FOLDED in v_gap.triggered_flags
    assert len(v_gap.verdict_sha256) == 64
