"""Own-CAD ``ProblemSpec`` trust evaluator producing versioned ``CADTrustReport``s.

Generalizes per-prediction certification beyond hardcoded specimen FEM recipes
to arbitrary fixed-CAD ``FemRequest`` problems using ``FemLinearization`` and
goal-oriented DWR adjoint estimators. Explicitly distinguishes solver failure,
inadmissible states, out-of-envelope queries, non-converged references, and
uncalibrated diagnostic indicators.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from piginot.certify.adjoint import DifferentiableQoi, estimate_qoi_dwr
from piginot.certify.trust_report import (
    CADTrustReport,
    DiscretizationErrorRecord,
    GeometryErrorRecord,
    QoITrustResult,
    StatisticalCalibrationRecord,
    SurrogateErrorRecord,
    build_cad_trust_report,
    reduce_qoi_trust_status,
)
from piginot.fem.backend import ScikitFemBackend
from piginot.fem.models import FemRequest, FemSolveFailure
from piginot.fem.protocol import DiscreteLinearization, SolverBackend
from piginot.verification.status import ClaimLevel

FloatArray = NDArray[np.float64]


def evaluate_cad_trust(
    request: FemRequest,
    nodal_displacement: FloatArray,
    qois: Sequence[DifferentiableQoi],
    *,
    problem_hash: str,
    geometry_error: GeometryErrorRecord,
    discretization_error: DiscretizationErrorRecord,
    in_parameter_envelope: bool = True,
    in_geometry_envelope: bool = True,
    claim_level: ClaimLevel = ClaimLevel.NOT_VERIFIED,
    bundle_digest: str | None = None,
    singular_qoi_names: frozenset[str] = frozenset(),
    statistical_by_qoi: Mapping[str, StatisticalCalibrationRecord] | None = None,
    indicators_qualified: bool = False,
    solve_failed: bool = False,
    backend: SolverBackend | None = None,
) -> CADTrustReport:
    """Evaluate per-QoI DWR + Newton indicators and assemble a ``CADTrustReport``."""
    if not qois:
        raise ValueError("evaluate_cad_trust requires at least one DifferentiableQoi")

    stat_map = {} if statistical_by_qoi is None else statistical_by_qoi
    linearization: DiscreteLinearization | None = None
    failure_reason = "exact FEM solve failed" if solve_failed else ""
    if not solve_failed:
        try:
            solver: SolverBackend = (
                backend if backend is not None else ScikitFemBackend()
            )
            linearization = solver.linearize(request)
        except FemSolveFailure as exc:
            solve_failed = True
            failure_reason = f"{exc.code.value}: {exc}"

    results: list[QoITrustResult] = []
    for qoi in qois:
        is_singular = qoi.name in singular_qoi_names or qoi.kind == "peak_stress"
        if solve_failed or linearization is None:
            surrogate = SurrogateErrorRecord(
                verdict="SOLVE_FAILED",
                eta_energy=None,
                dwr_estimate=None,
                newton_qoi_change=None,
                guards={"linearization_built": False},
            )
            status, reason = reduce_qoi_trust_status(
                in_parameter_envelope=in_parameter_envelope,
                in_geometry_envelope=in_geometry_envelope,
                solve_failed=True,
                surrogate=surrogate,
                discretization=discretization_error,
                singular_or_reentrant=is_singular,
                indicators_qualified=False,
            )
            results.append(
                QoITrustResult(
                    qoi_name=qoi.name,
                    qoi_kind=qoi.kind,
                    unit=qoi.unit,
                    predicted_value=None,
                    corrected_value=None,
                    status=status,
                    reason=f"{reason} ({failure_reason})" if failure_reason else reason,
                    geometry=geometry_error,
                    discretization=discretization_error,
                    surrogate=surrogate,
                    statistical=stat_map.get(qoi.name),
                    singular_or_reentrant=is_singular,
                )
            )
            continue

        dwr = estimate_qoi_dwr(
            linearization, np.asarray(nodal_displacement, dtype=np.float64), qoi
        )
        surrogate = SurrogateErrorRecord(
            verdict=dwr.verdict.value,
            eta_energy=dwr.eta_energy,
            dwr_estimate=dwr.dwr_estimate,
            newton_qoi_change=dwr.newton_qoi_change,
            dirichlet_deviation=dwr.dirichlet_deviation,
            guards=dwr.guards,
        )
        status, reason = reduce_qoi_trust_status(
            in_parameter_envelope=in_parameter_envelope,
            in_geometry_envelope=in_geometry_envelope,
            solve_failed=False,
            surrogate=surrogate,
            discretization=discretization_error,
            singular_or_reentrant=is_singular,
            indicators_qualified=indicators_qualified,
        )
        results.append(
            QoITrustResult(
                qoi_name=qoi.name,
                qoi_kind=qoi.kind,
                unit=qoi.unit,
                predicted_value=dwr.qoi_value,
                corrected_value=dwr.corrected_qoi_value,
                status=status,
                reason=reason,
                geometry=geometry_error,
                discretization=discretization_error,
                surrogate=surrogate,
                statistical=stat_map.get(qoi.name),
                singular_or_reentrant=is_singular,
            )
        )

    return build_cad_trust_report(
        problem_hash=problem_hash,
        mesh_id=request.mesh.mesh_id,
        cad_source_hash=geometry_error.cad_source_hash,
        bundle_digest=bundle_digest,
        claim_level=claim_level,
        qoi_results=results,
    )


__all__ = ("evaluate_cad_trust",)
