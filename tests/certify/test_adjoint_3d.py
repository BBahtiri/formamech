"""Unit & Gate G-V5-1b tests for 3D linear DWR adjoint certification."""

from __future__ import annotations

import numpy as np

from piginot.certify.adjoint_3d import (
    ComplianceQoi3D,
    ReactionForceQoi3D,
    SmoothedPeakVonMisesQoi3D,
    TotalStrainEnergyQoi3D,
    estimate_qoi_dwr_3d,
    linearize_3d_state,
)
from piginot.certify.certificate import CertificateVerdict
from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.fem.backend_3d import (
    DirichletBC3D,
    FemRequest3D,
    TorchFemBackend3D,
    TractionBC3D,
)
from piginot.geometry.mesh3d import make_brick_tet4_mesh


def _brick_request() -> FemRequest3D:
    mesh = make_brick_tet4_mesh(20.0, 8.0, 8.0, nx=3, ny=2, nz=2)
    mat = MaterialSpec(
        name="dwr-3d-nh",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(100.0, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(0.30, Unit.ONE).to_canonical(),
        },
    )
    return FemRequest3D(
        request_id="dwr-3d-brick",
        mesh=mesh,
        material=mat,
        dirichlet_bcs=(
            DirichletBC3D("fix", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
        ),
        traction_bcs=(TractionBC3D("pull", ("x_max",), (1.5, 0.25, 0.0)),),
        load_steps=2,
    )


def test_3d_qoi_gradients_match_central_finite_differences() -> None:
    """Verify exact discrete gradients for all 4 3D QoIs including SmoothedPeakVonMisesQoi3D."""
    req = _brick_request()
    exact = TorchFemBackend3D().solve(req)
    assert exact.converged
    u_perturbed = exact.nodal_displacement * 0.92
    state = linearize_3d_state(req, u_perturbed)

    qois = (
        ComplianceQoi3D(),
        TotalStrainEnergyQoi3D(),
        ReactionForceQoi3D("rx_fix", "fix", component="x"),
        SmoothedPeakVonMisesQoi3D(p_norm=6.0),
    )
    rng = np.random.default_rng(42)
    direction = np.zeros_like(state.u_full)
    d_free = rng.normal(0.0, 1.0, size=state.free_dofs.size)
    d_free /= np.linalg.norm(d_free)
    direction[state.free_dofs] = d_free
    eps = 1.0e-6

    s_plus = linearize_3d_state(req, (state.u_full + eps * direction).reshape(-1, 3))
    s_minus = linearize_3d_state(req, (state.u_full - eps * direction).reshape(-1, 3))

    for qoi in qois:
        g_ana = float(np.dot(qoi.free_gradient(state), d_free))
        g_fd = (qoi.evaluate(s_plus) - qoi.evaluate(s_minus)) / (2.0 * eps)
        rel_err = abs(g_ana - g_fd) / max(abs(g_fd), 1.0e-9)
        assert rel_err < 1.0e-5, (
            f"{qoi.name} gradient mismatch: analytical={g_ana:.8e}, fd={g_fd:.8e}, rel={rel_err:.2e}"
        )


def test_3d_dwr_effectivity_and_escalation_trigger() -> None:
    """Gate G-V5-1b: DWR tracks true 3D FEM error & escalates bad/folded fields."""
    req = _brick_request()
    exact = TorchFemBackend3D().solve(req)
    assert exact.converged

    # Slightly perturbed surrogate field (5% scaling error)
    u_surr = exact.nodal_displacement * 0.95
    qois = (
        ComplianceQoi3D(),
        TotalStrainEnergyQoi3D(),
        SmoothedPeakVonMisesQoi3D(p_norm=6.0),
    )
    cert = estimate_qoi_dwr_3d(req, u_surr, qois, escalation_tolerance=0.20)
    assert cert.verdict is CertificateVerdict.OK
    assert cert.positive_definite is True
    assert cert.eta_energy is not None and cert.eta_energy > 0.0
    assert not cert.should_escalate

    # Compare DWR estimate against true discrete FEM QoI error
    exact_state = linearize_3d_state(req, exact.nodal_displacement)
    surr_state = linearize_3d_state(req, u_surr)
    for qoi, est in zip(qois, cert.estimates, strict=True):
        true_err = qoi.evaluate(exact_state) - qoi.evaluate(surr_state)
        assert est.dwr_estimate is not None
        eff = abs(est.dwr_estimate) / max(abs(true_err), 1.0e-12)
        assert 0.5 <= eff <= 2.0, f"{qoi.name} DWR effectivity out of range: {eff:.3f}"

    # Tight tolerance must trigger escalation
    cert_tight = estimate_qoi_dwr_3d(req, u_surr, qois, escalation_tolerance=1.0e-3)
    assert cert_tight.should_escalate is True

    # Folded (det F <= 0) field must be rejected as INADMISSIBLE & escalate
    nodes = np.asarray(req.mesh.nodes, dtype=np.float64)
    u_folded = np.zeros_like(nodes)
    u_folded[:, 0] = -2.0 * nodes[:, 0]
    cert_bad = estimate_qoi_dwr_3d(req, u_folded, qois)
    assert cert_bad.verdict is CertificateVerdict.INADMISSIBLE
    assert cert_bad.should_escalate is True
