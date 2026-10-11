"""One Newton step at the surrogate state: the error estimate (S21).

For a surrogate displacement ``u`` on the request's FEM space (nodal values,
Dirichlet DOFs set to their prescribed values), solve ``K(u) d = -r(u)`` and
report ``eta = sqrt(d^T K(u) d)``, the energy drop ``Pi(u) - Pi(u + d)`` and
relative field and QoI changes along ``d``. Near the discrete minimizer
``u + d`` is quadratically close to it, so ``eta`` estimates the energy-norm
distance to the converged FEM solution on the same mesh. FEM discretization
error is a separate term, outside this certificate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from piginot.certify.certificate import Certificate, CertificateVerdict
from piginot.certify.guards import symmetric_inertia
from piginot.fem.linearization import FemLinearization, LinearizedState
from piginot.fem.models import FemSolveFailure

FloatArray = NDArray[np.float64]
QoiFunction = Callable[[FemLinearization, LinearizedState], float]
_TINY = 1e-300


def field_l2(
    linearization: FemLinearization, state: LinearizedState
) -> tuple[float, float]:
    """L2 norms of displacement and first Piola stress at the quadrature points."""

    weights = linearization.quadrature_weights()
    displacement = np.asarray(
        linearization.basis.interpolate(state.displacement), dtype=np.float64
    )
    u_squared = float(np.sum(np.sum(displacement**2, axis=0) * weights))
    stress = np.asarray(state.assembly.first_piola, dtype=np.float64)
    p_squared = float(np.sum(np.sum(stress**2, axis=(-2, -1)) * weights))
    return float(np.sqrt(u_squared)), float(np.sqrt(p_squared))


def field_difference_l2(
    linearization: FemLinearization, first: LinearizedState, second: LinearizedState
) -> tuple[float, float]:
    """L2 norms of the displacement and stress differences between two states."""

    weights = linearization.quadrature_weights()
    delta = np.asarray(
        linearization.basis.interpolate(second.displacement - first.displacement),
        dtype=np.float64,
    )
    u_squared = float(np.sum(np.sum(delta**2, axis=0) * weights))
    stress = np.asarray(second.assembly.first_piola) - np.asarray(
        first.assembly.first_piola
    )
    p_squared = float(np.sum(np.sum(stress**2, axis=(-2, -1)) * weights))
    return float(np.sqrt(u_squared)), float(np.sqrt(p_squared))


def _relative(change: float, reference: float) -> float:
    return change / max(abs(reference), _TINY)


def certify(
    linearization: FemLinearization,
    nodal_displacement: FloatArray,
    *,
    qoi_functions: Mapping[str, QoiFunction] | None = None,
) -> Certificate:
    """The certificate of one surrogate displacement on the request's mesh."""

    raw = linearization.vector(nodal_displacement)
    deviation = linearization.dirichlet_deviation(raw)
    vector = linearization.with_exact_dirichlet(raw)
    dofs = int(linearization.basis.N)
    try:
        before = linearization.state(vector)
    except FemSolveFailure as exc:
        return Certificate(
            CertificateVerdict.INADMISSIBLE,
            None,
            None,
            None,
            guards={"admissible": False},
            dirichlet_deviation=deviation,
            dof_count=dofs,
            reason=f"{exc.code.value}: {exc}",
        )
    guards: dict[str, bool | None] = {"admissible": True}
    try:
        increment = linearization.newton_increment(before)
        guards["increment_solved"] = True
    except FemSolveFailure as exc:
        guards["increment_solved"] = False
        return Certificate(
            CertificateVerdict.INDEFINITE,
            None,
            None,
            before.potential_energy,
            guards=guards,
            dirichlet_deviation=deviation,
            dof_count=dofs,
            reason=f"{exc.code.value}: {exc}",
        )
    curvature = float(
        increment[linearization.constraints.free]
        @ (
            linearization.free_tangent(before)
            @ increment[linearization.constraints.free]
        )
    )
    guards["positive_curvature"] = curvature > 0.0
    if linearization.has_global_constraints:
        guards["positive_definite"] = None  # projected tangent: not tested
    else:
        guards["positive_definite"] = symmetric_inertia(
            linearization.free_tangent(before)
        ).positive_definite
    try:
        after = linearization.state(vector + increment)
        guards["step_admissible"] = True
    except FemSolveFailure as exc:
        guards["step_admissible"] = False
        return Certificate(
            CertificateVerdict.INDEFINITE,
            None,
            None,
            before.potential_energy,
            guards=guards,
            dirichlet_deviation=deviation,
            dof_count=dofs,
            reason=f"Newton step leaves the admissible set: {exc.code.value}",
        )
    energy_drop = before.potential_energy - after.potential_energy
    guards["energy_decrease"] = energy_drop > 0.0
    eta = float(np.sqrt(max(curvature, 0.0)))
    u_norm, p_norm = field_l2(linearization, after)
    u_change, p_change = field_difference_l2(linearization, before, after)
    indicators: dict[str, float] = {
        "u_l2_change": _relative(u_change, u_norm),
        "p_l2_change": _relative(p_change, p_norm),
        "energy_ratio": energy_drop / max(0.5 * eta * eta, _TINY),
    }
    request_qois_before = linearization.qois(before)
    request_qois_after = linearization.qois(after)
    for name, value in request_qois_after.items():
        indicators[f"qoi:{name}"] = _relative(
            abs(value - request_qois_before[name]), value
        )
    for name, function in (qoi_functions or {}).items():
        value_after = function(linearization, after)
        indicators[f"qoi:{name}"] = _relative(
            abs(value_after - function(linearization, before)), value_after
        )
    required = ("positive_curvature", "energy_decrease", "step_admissible")
    definite = guards["positive_definite"]
    ok = all(guards[name] for name in required) and definite is not False
    if definite is None and not linearization.has_global_constraints:
        ok = False  # pivoting hid the inertia: refuse rather than guess
    return Certificate(
        CertificateVerdict.OK if ok else CertificateVerdict.INDEFINITE,
        eta,
        energy_drop,
        before.potential_energy,
        indicators=indicators,
        guards=guards,
        dirichlet_deviation=deviation,
        dof_count=dofs,
        reason=""
        if ok
        else "a guard failed: "
        + ", ".join(
            name
            for name, value in guards.items()
            if value is False
            or (
                name == "positive_definite"
                and value is None
                and not linearization.has_global_constraints
            )
        ),
    )
