"""Tests for Feature D: CADTrustReport fail-closed rules and runtime-only isolation."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from piginot.certify import (
    CADTrustReport,
    DiscretizationErrorRecord,
    GeometryErrorRecord,
    QoITrustResult,
    ReactionForceQoi,
    RegionalAverageStressQoi,
    StatisticalCalibrationRecord,
    SurrogateErrorRecord,
    TrustStatus,
    build_cad_trust_report,
    evaluate_cad_trust,
    reduce_qoi_trust_status,
)
from piginot.fem import ScikitFemBackend
from piginot.verification.status import ClaimLevel
from tests.fem.test_nonlinear_solver import _refined_unit_square, _traction_request
from tests.platform_support import windows_safe_environment

ROOT = Path(__file__).resolve().parents[2]


def _geom_rec() -> GeometryErrorRecord:
    return GeometryErrorRecord(
        cad_source_hash="a" * 64,
        declared_curve_tolerance_mm=1e-2,
        achieved_max_chord_error_mm=2e-3,
        remesh_status="GEOMETRY_CONVERGED",
    )


def _disc_rec(status: str = "CONVERGED") -> DiscretizationErrorRecord:
    return DiscretizationErrorRecord(
        reference_status=status,
        element_order=2,
        dof_count=400,
        m1_to_m2_qoi_change=1e-3,
    )


def _surr_rec(verdict: str = "OK") -> SurrogateErrorRecord:
    return SurrogateErrorRecord(
        verdict=verdict,
        eta_energy=1e-3,
        dwr_estimate=2e-3,
        newton_qoi_change=2e-3,
        guards={
            "admissible": verdict != "INADMISSIBLE",
            "positive_definite": verdict != "INDEFINITE",
        },
    )


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"in_parameter_envelope": False}, TrustStatus.OUT_OF_ENVELOPE),
        ({"in_geometry_envelope": False}, TrustStatus.OUT_OF_ENVELOPE),
        ({"solve_failed": True}, TrustStatus.SOLVE_FAILED),
        ({"surrogate": _surr_rec("INADMISSIBLE")}, TrustStatus.INADMISSIBLE),
        ({"surrogate": _surr_rec("INDEFINITE")}, TrustStatus.INDEFINITE),
        (
            {"discretization": _disc_rec("REFERENCE_NOT_CONVERGED")},
            TrustStatus.REFERENCE_NOT_CONVERGED,
        ),
        ({"singular_or_reentrant": True}, TrustStatus.UNCERTIFIED),
        ({"indicators_qualified": False}, TrustStatus.DIAGNOSTIC_ONLY),
        ({"indicators_qualified": True}, TrustStatus.QUALIFIED_INDICATORS),
    ],
)
def test_trust_status_reducer_fails_closed_on_every_guard(
    kwargs: dict[str, object], expected: TrustStatus
) -> None:
    defaults: dict[str, object] = {
        "in_parameter_envelope": True,
        "in_geometry_envelope": True,
        "solve_failed": False,
        "surrogate": _surr_rec("OK"),
        "discretization": _disc_rec("CONVERGED"),
        "singular_or_reentrant": False,
        "indicators_qualified": False,
    }
    defaults.update(kwargs)
    status, reason = reduce_qoi_trust_status(**defaults)  # type: ignore[arg-type]
    assert status is expected
    assert reason


def test_rigorous_continuum_bound_cannot_be_claimed_without_all_proven_components() -> (
    None
):
    with pytest.raises(ValueError, match="rigorous_continuum_bound"):
        QoITrustResult(
            qoi_name="rxn",
            qoi_kind="reaction_force",
            unit="N",
            predicted_value=10.0,
            corrected_value=10.01,
            status=TrustStatus.DIAGNOSTIC_ONLY,
            reason="diagnostic",
            geometry=_geom_rec(),
            discretization=_disc_rec(),
            surrogate=_surr_rec(),
            rigorous_continuum_bound=0.05,
        )


def test_evaluate_cad_trust_end_to_end_on_fem_problem() -> None:
    request = replace(_traction_request(element_order=2), mesh=_refined_unit_square())
    attempt = ScikitFemBackend().solve_many((request,))[0]
    assert attempt.success and attempt.result is not None
    u_fem = np.asarray(attempt.result.displacement, dtype=np.float64)

    bc_name = request.boundary_conditions[0].name
    qois = (
        ReactionForceQoi(name="rxn_left", boundary_condition_name=bc_name),
        RegionalAverageStressQoi(name="avg_p11", component=(0, 0)),
    )
    stat = StatisticalCalibrationRecord(
        nominal_coverage=0.90,
        calibration_sample_count=64,
        conformal_quantile=0.02,
        half_width=0.02,
        exchangeability_scope="fixed-cad-train-envelope",
        in_calibration_envelope=True,
        refused=False,
    )
    report = evaluate_cad_trust(
        request,
        u_fem,
        qois,
        problem_hash="c" * 64,
        geometry_error=_geom_rec(),
        discretization_error=_disc_rec("CONVERGED"),
        claim_level=ClaimLevel.NOT_VERIFIED,
        statistical_by_qoi={"rxn_left": stat},
    )
    assert isinstance(report, CADTrustReport)
    assert report.overall_status is TrustStatus.DIAGNOSTIC_ONLY
    assert report.claim_level == ClaimLevel.NOT_VERIFIED.value
    doc = report.document()
    assert len(report.report_hash) == 64
    assert (
        doc["qoi_results"][0]["error_components"]["statistical_calibration"][
            "is_deterministic_error_term"
        ]
        is False
    )
    # At converged FEM solution, DWR error estimate is essentially zero.
    assert (
        abs(
            doc["qoi_results"][0]["error_components"]["surrogate_discrete"][
                "dwr_estimate"
            ]
        )
        < 1e-8
    )

    # Confirm severity aggregation escalates report overall_status when one QoI fails.
    mixed = build_cad_trust_report(
        problem_hash="c" * 64,
        mesh_id=request.mesh.mesh_id,
        cad_source_hash="a" * 64,
        bundle_digest=None,
        claim_level=ClaimLevel.NOT_VERIFIED,
        qoi_results=(
            report.qoi_results[0],
            replace(
                report.qoi_results[1],
                status=TrustStatus.REFERENCE_NOT_CONVERGED,
                reason="mesh not converged",
            ),
        ),
    )
    assert mixed.overall_status is TrustStatus.REFERENCE_NOT_CONVERGED

    # Evaluate with an invalid region/BC request to confirm SOLVE_FAILED path.
    bad_req = replace(
        request,
        boundary_conditions=(
            replace(request.boundary_conditions[0], boundary_tag="missing-group"),
            *request.boundary_conditions[1:],
        ),
    )
    failed_report = evaluate_cad_trust(
        bad_req,
        u_fem,
        qois,
        problem_hash="c" * 64,
        geometry_error=_geom_rec(),
        discretization_error=_disc_rec("CONVERGED"),
    )
    assert failed_report.overall_status is TrustStatus.SOLVE_FAILED

    # Invalid claim level above HOLDOUT_VERIFIED is refused.
    with pytest.raises(ValueError, match="claim_level"):
        build_cad_trust_report(
            problem_hash="c" * 64,
            mesh_id=request.mesh.mesh_id,
            cad_source_hash="a" * 64,
            bundle_digest=None,
            claim_level="CERTIFIED",
            qoi_results=report.qoi_results,
        )


def test_trust_report_imports_without_fem_or_ml_packages() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc, sys
        blocked = ("scipy", "skfem", "meshio", "sympy", "torch", "numpy", "gmsh")
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in blocked:
                    raise ImportError("blocked " + name)
        sys.meta_path.insert(0, Block())
        from formamech.certify import CADTrustReport, TrustStatus
        assert TrustStatus.UNCERTIFIED.value == "UNCERTIFIED"
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=windows_safe_environment({"PYTHONPATH": str(ROOT / "src")}),
    )
    assert result.returncode == 0, result.stderr
