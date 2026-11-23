"""Unit tests for FEM-guarded differentiable design optimization (P5 / Feature E)."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
from formamech.optimization import (
    DesignConstraint,
    DesignOptimizationStatus,
    DesignParameter,
    DesignProblem,
    FemDesignEvaluation,
    SurrogateDesignEvaluation,
    run_fem_guarded_design_optimization,
)

from tests.platform_support import windows_safe_environment

ROOT = Path(__file__).resolve().parents[2]


def test_optimization_package_imports_without_ml_or_fem() -> None:
    code = (
        "import sys, formamech.optimization; "
        "banned = {'torch', 'numpy', 'scipy', 'skfem', 'gmsh', 'sympy'}; "
        "found = banned.intersection(sys.modules); "
        "assert not found, f'unexpected imports: {found}'"
    )
    env = windows_safe_environment({"PYTHONPATH": str(ROOT / "src")})
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def _sample_problem(*, max_fem: int = 12, grad_interval: int = 3) -> DesignProblem:
    return DesignProblem(
        name="analytic-plate-design",
        geometry_version="cad-v1",
        problem_spec_hash="c" * 64,
        parameters=(
            DesignParameter("radius", lower=0.5, upper=1.5, initial=0.7, unit="mm"),
            DesignParameter("height", lower=2.0, upper=4.0, initial=3.5, unit="mm"),
        ),
        objective_name="area",
        objective_unit="mm^2",
        minimize=True,
        constraints=(
            DesignConstraint(
                name="avg_stress",
                kind="inequality_le",
                limit=14.0,
                tolerance=1e-4,
                unit="MPa",
            ),
        ),
        max_iterations=25,
        max_fem_evaluations=max_fem,
        initial_trust_radius=0.25,
        min_trust_radius=0.01,
        gradient_check_interval=grad_interval,
        gradient_cos_min=0.25,
        accept_eta_tolerance=0.05,
        constraint_penalty_weight=20.0,
    )


def _true_physics(p: Mapping[str, float]) -> tuple[float, float]:
    # Minimize area = 4*height - 0.7854*radius^2 subject to stress = 30 / (height - radius) <= 14.
    r, h = p["radius"], p["height"]
    area = 4.0 * h - 0.78539816339 * (r * r)
    stress = 30.0 / max(h - r, 0.2)
    return area, stress


def test_design_problem_validation_and_hashing() -> None:
    p1 = _sample_problem()
    p2 = _sample_problem()
    assert p1.problem_hash == p2.problem_hash
    assert len(p1.problem_hash) == 64

    with pytest.raises(ValueError, match="initial"):
        DesignParameter("r", lower=0.5, upper=1.5, initial=2.0, unit="mm")
    with pytest.raises(ValueError, match="unique"):
        DesignProblem(
            name="dup",
            geometry_version="v1",
            problem_spec_hash="a" * 64,
            parameters=(
                DesignParameter("x", 0.0, 1.0, 0.5, "mm"),
                DesignParameter("x", 0.0, 2.0, 1.0, "mm"),
            ),
            objective_name="obj",
        )


def test_converges_and_mandates_final_fem_verification() -> None:
    prob = _sample_problem(max_fem=12, grad_interval=4)

    def surr(p: Mapping[str, float]) -> SurrogateDesignEvaluation:
        area, stress = _true_physics(p)
        return SurrogateDesignEvaluation(
            objective=area,
            gradient={"radius": -2.0 * 0.78539816339 * p["radius"], "height": 4.0},
            constraints={"avg_stress": stress * 1.002},
            in_envelope=True,
            admissible=True,
            eta_indicator=0.01,
        )

    def fem(p: Mapping[str, float]) -> FemDesignEvaluation:
        area, stress = _true_physics(p)
        return FemDesignEvaluation(
            succeeded=True,
            objective=area,
            constraints={"avg_stress": stress},
            reference_hash="f" * 64,
        )

    report = run_fem_guarded_design_optimization(
        prob, surrogate_evaluator=surr, fem_evaluator=fem
    )
    assert report.status in (
        DesignOptimizationStatus.CONVERGED_FEM_VERIFIED,
        DesignOptimizationStatus.BUDGET_EXHAUSTED_FEM_VERIFIED,
    )
    assert report.best_design is not None
    assert report.best_design.fem_feasible is True
    assert report.best_design.reference_hash == "f" * 64
    assert report.fem_evaluations >= 2
    assert report.surrogate_evaluations > report.fem_evaluations
    init_area, _ = _true_physics(prob.initial_point())
    assert report.best_design.fem_objective < init_area - 1.5
    assert len(report.gradient_checks) >= 1
    assert all(gc.passed for gc in report.gradient_checks)
    assert report.claim_level == "NOT_VERIFIED"
    assert len(report.report_hash) == 64


def test_wrong_sign_gradient_triggers_trust_radius_shrinkage() -> None:
    prob = DesignProblem(
        name="wrong-sign-test",
        geometry_version="cad-v1",
        problem_spec_hash="d" * 64,
        parameters=(DesignParameter("x", 0.0, 2.0, 1.0, "mm"),),
        objective_name="quad",
        minimize=True,
        max_iterations=8,
        max_fem_evaluations=10,
        gradient_check_interval=2,
        gradient_cos_min=0.25,
    )

    def bad_surr(p: Mapping[str, float]) -> SurrogateDesignEvaluation:
        x = p["x"]
        # True objective is (x - 0.2)^2 (grad = 2*(x - 0.2) > 0 at x=1.0),
        # but surrogate returns flipped sign gradient!
        return SurrogateDesignEvaluation(
            objective=(x - 0.2) ** 2,
            gradient={"x": -2.0 * (x - 0.2)},
            constraints={},
            in_envelope=True,
            admissible=True,
            eta_indicator=0.01,
        )

    def true_fem(p: Mapping[str, float]) -> FemDesignEvaluation:
        x = p["x"]
        return FemDesignEvaluation(
            succeeded=True,
            objective=(x - 0.2) ** 2,
            constraints={},
            reference_hash="b" * 64,
        )

    report = run_fem_guarded_design_optimization(
        prob, surrogate_evaluator=bad_surr, fem_evaluator=true_fem
    )
    assert any(not gc.passed for gc in report.gradient_checks)
    assert any("wrong-sign" in it.reason for it in report.iterations)


def test_trust_gate_failure_escalates_to_fem_and_refusal_paths() -> None:
    prob = _sample_problem(max_fem=6, grad_interval=10)

    refused = run_fem_guarded_design_optimization(
        prob,
        surrogate_evaluator=lambda _: pytest.fail("should not call surrogate"),
        fem_evaluator=lambda _: pytest.fail("should not call fem"),
        geometry_supported=False,
        unsupported_reason="topology change",
    )
    assert refused.status is DesignOptimizationStatus.REFUSED_UNSUPPORTED_GEOMETRY
    assert refused.best_design is None

    # High eta forces FEM escalation.
    def high_eta_surr(p: Mapping[str, float]) -> SurrogateDesignEvaluation:
        area, stress = _true_physics(p)
        return SurrogateDesignEvaluation(
            objective=area,
            gradient={"radius": -1.5, "height": 4.0},
            constraints={"avg_stress": stress},
            in_envelope=True,
            admissible=True,
            eta_indicator=0.9,  # exceeds accept_eta_tolerance=0.05
        )

    def fem_ok(p: Mapping[str, float]) -> FemDesignEvaluation:
        area, stress = _true_physics(p)
        return FemDesignEvaluation(
            succeeded=True, objective=area, constraints={"avg_stress": stress}
        )

    esc_rep = run_fem_guarded_design_optimization(
        prob, surrogate_evaluator=high_eta_surr, fem_evaluator=fem_ok
    )
    assert any(it.escalated_to_fem for it in esc_rep.iterations)
    assert esc_rep.best_design is not None and esc_rep.best_design.fem_feasible

    # Always-infeasible FEM returns FAILED_NO_FEASIBLE_FEM_DESIGN.
    def fem_infeasible(_p: Mapping[str, float]) -> FemDesignEvaluation:
        return FemDesignEvaluation(
            succeeded=True, objective=1.0, constraints={"avg_stress": 999.0}
        )

    fail_rep = run_fem_guarded_design_optimization(
        prob, surrogate_evaluator=high_eta_surr, fem_evaluator=fem_infeasible
    )
    assert fail_rep.status is DesignOptimizationStatus.FAILED_NO_FEASIBLE_FEM_DESIGN
    assert fail_rep.best_design is None

    # Wave 2: DESIGN_OPTIMIZATION_MACHINE, DesignOptimizationReport invariants,
    # and prevention of false stationary CONVERGED_FEM_VERIFIED after inadmissible surrogate step
    from formamech.optimization import (
        DESIGN_OPTIMIZATION_MACHINE,
        DesignOptimizationPhase,
        DesignOptimizationReport,
    )

    assert DESIGN_OPTIMIZATION_MACHINE.can_transition(
        DesignOptimizationPhase.INITIALIZED,
        DesignOptimizationPhase.EXPLORING_TRUST_REGION,
        trigger="start_trust_region_loop",
    )
    assert DESIGN_OPTIMIZATION_MACHINE.is_terminal(
        DesignOptimizationPhase.CONVERGED_FEM_VERIFIED
    )

    with pytest.raises(ValueError, match="fem_feasible best_design"):
        DesignOptimizationReport(
            problem_hash="a" * 64,
            status=DesignOptimizationStatus.CONVERGED_FEM_VERIFIED,
            best_design=None,
            surrogate_evaluations=1,
            fem_evaluations=1,
            failed_designs=0,
            iterations=(),
            gradient_checks=(),
            reason="invalid converged report without best_design",
        )

    # When an inadmissible trial surrogate step (zero gradient) is accepted via FEM escalation,
    # the next iteration must NOT mistake the zero gradient for stationary convergence.
    call_idx = 0

    def surr_inadmissible_on_step(p: Mapping[str, float]) -> SurrogateDesignEvaluation:
        nonlocal call_idx
        call_idx += 1
        area, stress = _true_physics(p)
        if call_idx == 1:
            return SurrogateDesignEvaluation(
                objective=area,
                gradient={"radius": -1.5, "height": 4.0},
                constraints={"avg_stress": stress},
                in_envelope=True,
                admissible=True,
                eta_indicator=0.01,
            )
        return SurrogateDesignEvaluation(
            objective=float("inf"),
            gradient={"radius": 0.0, "height": 0.0},
            constraints={"avg_stress": stress},
            in_envelope=False,
            admissible=False,
            eta_indicator=1.0,
        )

    inadm_rep = run_fem_guarded_design_optimization(
        prob,
        surrogate_evaluator=surr_inadmissible_on_step,
        fem_evaluator=fem_ok,
    )
    assert inadm_rep.best_design is not None and inadm_rep.best_design.fem_feasible
    assert len(inadm_rep.iterations) >= 2
