"""FEM-guarded trust-region differentiable design optimization loop (Feature E / P5).

Workflow: **AI explores -> trust gate decides -> FEM verifies**.

* Explores using differentiable surrogate objectives and gradients inside an
  adaptive trust region.
* Checks geometry/material admissibility and envelope/certificate gates before
  accepting any candidate step.
* Periodically validates surrogate design gradients against FEM finite
  differences; wrong-sign gradients shrink the trust region and escalate to FEM.
* **Never accepts a final design solely on unverified surrogate predictions:**
  every accepted optimal design carries a direct FEM reference evaluation.
"""

from __future__ import annotations

import enum
import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from piginot.contracts import canonical_json_bytes
from piginot.contracts.state_machine import (
    ClaimLevel,
    StateMachineSpec,
    TransitionSpec,
    register_state_machine,
)
from piginot.optimization.problem import DESIGN_REPORT_SCHEMA, DesignProblem


def _sha256_doc(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


class DesignOptimizationStatus(enum.StrEnum):
    CONVERGED_FEM_VERIFIED = "CONVERGED_FEM_VERIFIED"
    BUDGET_EXHAUSTED_FEM_VERIFIED = "BUDGET_EXHAUSTED_FEM_VERIFIED"
    REFUSED_UNSUPPORTED_GEOMETRY = "REFUSED_UNSUPPORTED_GEOMETRY"
    FAILED_NO_FEASIBLE_FEM_DESIGN = "FAILED_NO_FEASIBLE_FEM_DESIGN"


class DesignOptimizationPhase(enum.StrEnum):
    """Explicit stages of the FEM-guarded trust-region design optimization loop."""

    INITIALIZED = "INITIALIZED"
    EXPLORING_TRUST_REGION = "EXPLORING_TRUST_REGION"
    FINAL_FEM_VERIFICATION = "FINAL_FEM_VERIFICATION"
    CONVERGED_FEM_VERIFIED = "CONVERGED_FEM_VERIFIED"
    BUDGET_EXHAUSTED_FEM_VERIFIED = "BUDGET_EXHAUSTED_FEM_VERIFIED"
    REFUSED_UNSUPPORTED_GEOMETRY = "REFUSED_UNSUPPORTED_GEOMETRY"
    FAILED_NO_FEASIBLE_FEM_DESIGN = "FAILED_NO_FEASIBLE_FEM_DESIGN"


DESIGN_OPTIMIZATION_MACHINE: Final[StateMachineSpec[DesignOptimizationPhase]] = (
    register_state_machine(
        StateMachineSpec(
            machine_id="formamech.optimization.trust_region",
            description=(
                "FEM-guarded trust-region design optimization: geometry support check -> "
                "differentiable surrogate exploration with periodic FEM gradient checks "
                "and FEM escalation -> mandatory final FEM verification."
            ),
            states=tuple(DesignOptimizationPhase),
            initial_states=(DesignOptimizationPhase.INITIALIZED,),
            terminal_states=(
                DesignOptimizationPhase.CONVERGED_FEM_VERIFIED,
                DesignOptimizationPhase.BUDGET_EXHAUSTED_FEM_VERIFIED,
                DesignOptimizationPhase.REFUSED_UNSUPPORTED_GEOMETRY,
                DesignOptimizationPhase.FAILED_NO_FEASIBLE_FEM_DESIGN,
            ),
            transitions=(
                TransitionSpec(
                    DesignOptimizationPhase.INITIALIZED,
                    DesignOptimizationPhase.REFUSED_UNSUPPORTED_GEOMETRY,
                    "refuse_unsupported_geometry",
                    "Geometry or topology is unsupported for optimization.",
                ),
                TransitionSpec(
                    DesignOptimizationPhase.INITIALIZED,
                    DesignOptimizationPhase.EXPLORING_TRUST_REGION,
                    "start_trust_region_loop",
                    "Initial surrogate and optional FEM baseline evaluated.",
                ),
                TransitionSpec(
                    DesignOptimizationPhase.EXPLORING_TRUST_REGION,
                    DesignOptimizationPhase.FINAL_FEM_VERIFICATION,
                    "finish_iterations_or_converge",
                    "Trust radius / gradient converged or iteration budget reached.",
                ),
                TransitionSpec(
                    DesignOptimizationPhase.FINAL_FEM_VERIFICATION,
                    DesignOptimizationPhase.CONVERGED_FEM_VERIFIED,
                    "verify_converged_design_with_fem",
                    "Converged candidate verified feasible by converged FEM.",
                ),
                TransitionSpec(
                    DesignOptimizationPhase.FINAL_FEM_VERIFICATION,
                    DesignOptimizationPhase.BUDGET_EXHAUSTED_FEM_VERIFIED,
                    "verify_budget_best_design_with_fem",
                    "Iteration budget reached with at least one FEM-verified feasible design.",
                ),
                TransitionSpec(
                    DesignOptimizationPhase.FINAL_FEM_VERIFICATION,
                    DesignOptimizationPhase.FAILED_NO_FEASIBLE_FEM_DESIGN,
                    "no_feasible_fem_design",
                    "No candidate satisfied all constraints under authoritative FEM.",
                ),
            ),
        )
    )
)


@dataclass(frozen=True, slots=True)
class SurrogateDesignEvaluation:
    """One differentiable surrogate evaluation plus trust-gate metadata."""

    objective: float
    gradient: Mapping[str, float]  # d(merit or objective)/dp
    constraints: Mapping[str, float]
    in_envelope: bool
    admissible: bool
    eta_indicator: float
    reason: str = ""


@dataclass(frozen=True, slots=True)
class FemDesignEvaluation:
    """Authoritative converged FEM evaluation of one candidate design point."""

    succeeded: bool
    objective: float | None
    constraints: Mapping[str, float]
    reference_hash: str | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class GradientConsistencyCheck:
    """Periodic comparison between surrogate gradient and FEM finite differences."""

    iteration: int
    parameters: dict[str, float]
    cosine_similarity: float
    passed: bool
    fem_calls_used: int

    def document(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "parameters": dict(sorted(self.parameters.items())),
            "cosine_similarity": self.cosine_similarity,
            "passed": self.passed,
            "fem_calls_used": self.fem_calls_used,
        }


@dataclass(frozen=True, slots=True)
class DesignIterationRecord:
    """Auditable per-iteration record of the trust-region accept/reject loop."""

    iteration: int
    parameters: dict[str, float]
    trust_radius: float
    surrogate_objective: float
    surrogate_eta: float
    in_envelope: bool
    accepted: bool
    escalated_to_fem: bool
    fem_objective: float | None
    fem_feasible: bool | None
    reason: str

    def document(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "parameters": dict(sorted(self.parameters.items())),
            "trust_radius": self.trust_radius,
            "surrogate_objective": self.surrogate_objective,
            "surrogate_eta": self.surrogate_eta,
            "in_envelope": self.in_envelope,
            "accepted": self.accepted,
            "escalated_to_fem": self.escalated_to_fem,
            "fem_objective": self.fem_objective,
            "fem_feasible": self.fem_feasible,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class VerifiedDesignSolution:
    """Final FEM-verified optimal design point (never surrogate-only)."""

    parameters: dict[str, float]
    fem_objective: float
    fem_constraints: dict[str, float]
    fem_constraint_violations: dict[str, float]
    fem_feasible: bool
    reference_hash: str | None

    def document(self) -> dict[str, Any]:
        return {
            "parameters": dict(sorted(self.parameters.items())),
            "fem_objective": self.fem_objective,
            "fem_constraints": dict(sorted(self.fem_constraints.items())),
            "fem_constraint_violations": dict(
                sorted(self.fem_constraint_violations.items())
            ),
            "fem_feasible": self.fem_feasible,
            "reference_hash": self.reference_hash,
        }


@dataclass(frozen=True, slots=True)
class DesignOptimizationReport:
    """Immutable evidence artifact for one FEM-guarded design campaign."""

    problem_hash: str
    status: DesignOptimizationStatus
    best_design: VerifiedDesignSolution | None
    surrogate_evaluations: int
    fem_evaluations: int
    failed_designs: int
    iterations: tuple[DesignIterationRecord, ...]
    gradient_checks: tuple[GradientConsistencyCheck, ...]
    reason: str
    claim_level: str = "NOT_VERIFIED"

    def __post_init__(self) -> None:
        claim = ClaimLevel(str(self.claim_level)).value
        object.__setattr__(self, "claim_level", claim)
        if self.status in (
            DesignOptimizationStatus.CONVERGED_FEM_VERIFIED,
            DesignOptimizationStatus.BUDGET_EXHAUSTED_FEM_VERIFIED,
        ):
            if self.best_design is None or not self.best_design.fem_feasible:
                raise ValueError(
                    f"{self.status.value} requires a non-None fem_feasible best_design"
                )
            if self.fem_evaluations < 1:
                raise ValueError(
                    f"{self.status.value} requires at least 1 FEM evaluation"
                )
        elif self.status is DesignOptimizationStatus.REFUSED_UNSUPPORTED_GEOMETRY:
            if self.best_design is not None or self.fem_evaluations != 0:
                raise ValueError(
                    "REFUSED_UNSUPPORTED_GEOMETRY must have best_design=None and fem_evaluations=0"
                )

    def document(self) -> dict[str, Any]:
        return {
            "schema": DESIGN_REPORT_SCHEMA,
            "problem_hash": self.problem_hash,
            "status": self.status.value,
            "claim_level": self.claim_level,
            "best_design": (
                None if self.best_design is None else self.best_design.document()
            ),
            "surrogate_evaluations": self.surrogate_evaluations,
            "fem_evaluations": self.fem_evaluations,
            "failed_designs": self.failed_designs,
            "iterations": [it.document() for it in self.iterations],
            "gradient_checks": [g.document() for g in self.gradient_checks],
            "reason": self.reason,
        }

    @property
    def report_hash(self) -> str:
        return _sha256_doc(self.document())


SurrogateEvaluator = Callable[[Mapping[str, float]], SurrogateDesignEvaluation]
FemEvaluator = Callable[[Mapping[str, float]], FemDesignEvaluation]


def _constraint_penalty(
    problem: DesignProblem,
    constraints: Mapping[str, float],
) -> float:
    _, violations = problem.check_constraints(constraints)
    return problem.constraint_penalty_weight * sum(v * v for v in violations.values())


def _merit(
    problem: DesignProblem,
    objective: float,
    constraints: Mapping[str, float],
) -> float:
    signed_obj = objective if problem.minimize else -objective
    return signed_obj + _constraint_penalty(problem, constraints)


def _cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(v * v for v in a))
    nb = math.sqrt(sum(v * v for v in b))
    if na <= 1e-15 or nb <= 1e-15:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)


def run_fem_guarded_design_optimization(
    problem: DesignProblem,
    *,
    surrogate_evaluator: SurrogateEvaluator,
    fem_evaluator: FemEvaluator,
    geometry_supported: bool = True,
    unsupported_reason: str = "",
    fd_step_fraction: float = 1e-3,
) -> DesignOptimizationReport:
    """Run trust-region differentiable optimization with mandatory FEM verification."""
    if not geometry_supported:
        return DesignOptimizationReport(
            problem_hash=problem.problem_hash,
            status=DesignOptimizationStatus.REFUSED_UNSUPPORTED_GEOMETRY,
            best_design=None,
            surrogate_evaluations=0,
            fem_evaluations=0,
            failed_designs=0,
            iterations=(),
            gradient_checks=(),
            reason=unsupported_reason or "unsupported CAD geometry or topology",
        )

    surr_evals = 0
    fem_evals = 0
    failed_designs = 0
    iterations: list[DesignIterationRecord] = []
    grad_checks: list[GradientConsistencyCheck] = []

    best_fem: VerifiedDesignSolution | None = None
    best_fem_merit = float("inf")

    def _call_fem(point: Mapping[str, float]) -> FemDesignEvaluation | None:
        nonlocal fem_evals, failed_designs, best_fem, best_fem_merit
        if fem_evals >= problem.max_fem_evaluations:
            return None
        fem_evals += 1
        res = fem_evaluator(dict(point))
        if not res.succeeded or res.objective is None:
            failed_designs += 1
            return res
        feasible, violations = problem.check_constraints(res.constraints)
        m_val = _merit(problem, res.objective, res.constraints)
        cand_sol = VerifiedDesignSolution(
            parameters=dict(point),
            fem_objective=float(res.objective),
            fem_constraints=dict(res.constraints),
            fem_constraint_violations=violations,
            fem_feasible=feasible,
            reference_hash=res.reference_hash,
        )
        # Prefer feasible solutions; among feasible, minimize merit.
        if best_fem is None:
            if feasible:
                best_fem = cand_sol
                best_fem_merit = m_val
        elif feasible and (not best_fem.fem_feasible or m_val < best_fem_merit):
            best_fem = cand_sol
            best_fem_merit = m_val
        return res

    curr_p = problem.initial_point()
    # Evaluate initial design on surrogate and FEM (reserve 1 FEM call for final check).
    curr_surr = surrogate_evaluator(curr_p)
    surr_evals += 1
    curr_merit = (
        _merit(problem, curr_surr.objective, curr_surr.constraints)
        if curr_surr.admissible
        else float("inf")
    )
    if problem.max_fem_evaluations >= 2:
        _call_fem(curr_p)

    radius = problem.initial_trust_radius
    p_names = [p.name for p in problem.parameters]
    spans = {p.name: p.span for p in problem.parameters}
    converged = False

    for it in range(1, problem.max_iterations + 1):
        if radius < problem.min_trust_radius:
            converged = True
            break

        # Compute normalized steepest-descent step inside trust region.
        # Compute numerical finite-diff merit gradient on surrogate if constraints active.
        grad_vec: list[float] = []
        for p_spec in problem.parameters:
            g_obj = float(curr_surr.gradient.get(p_spec.name, 0.0))
            if not problem.minimize:
                g_obj = -g_obj
            if problem.constraints:
                # Combine objective gradient with central-diff constraint penalty gradient on surrogate.
                h = max(1e-5 * p_spec.span, 1e-7)
                p_plus = dict(curr_p)
                p_minus = dict(curr_p)
                p_plus[p_spec.name] = p_spec.clamp(curr_p[p_spec.name] + h)
                p_minus[p_spec.name] = p_spec.clamp(curr_p[p_spec.name] - h)
                dh = p_plus[p_spec.name] - p_minus[p_spec.name]
                if dh > 0.0:
                    s_plus = surrogate_evaluator(p_plus)
                    s_minus = surrogate_evaluator(p_minus)
                    surr_evals += 2
                    if s_plus.admissible and s_minus.admissible:
                        pen_plus = _constraint_penalty(problem, s_plus.constraints)
                        pen_minus = _constraint_penalty(problem, s_minus.constraints)
                        g_obj = g_obj + (pen_plus - pen_minus) / dh
            grad_vec.append(g_obj * spans[p_spec.name])

        grad_norm = math.sqrt(sum(v * v for v in grad_vec))
        if grad_norm <= 1e-12:
            if curr_surr.admissible and curr_surr.in_envelope:
                converged = True
            break

        # Periodic FEM finite-difference gradient consistency check (reserve 1 FEM call for final check).
        if (
            it % problem.gradient_check_interval == 0
            and fem_evals + len(problem.parameters) + 2 <= problem.max_fem_evaluations
        ):
            base_fem = _call_fem(curr_p)
            fd_fem_grad: list[float] = []
            fem_calls_check = 1 if base_fem is not None else 0
            if (
                base_fem is not None
                and base_fem.succeeded
                and base_fem.objective is not None
            ):
                base_m = _merit(problem, base_fem.objective, base_fem.constraints)
                for p_spec in problem.parameters:
                    h = fd_step_fraction * p_spec.span
                    p_pert = dict(curr_p)
                    sign = 1.0 if curr_p[p_spec.name] + h <= p_spec.upper else -1.0
                    p_pert[p_spec.name] = p_spec.clamp(curr_p[p_spec.name] + sign * h)
                    dh = p_pert[p_spec.name] - curr_p[p_spec.name]
                    f_pert = _call_fem(p_pert)
                    fem_calls_check += 1 if f_pert is not None else 0
                    if (
                        f_pert is not None
                        and f_pert.succeeded
                        and f_pert.objective is not None
                        and abs(dh) > 0.0
                    ):
                        m_pert = _merit(problem, f_pert.objective, f_pert.constraints)
                        fd_fem_grad.append(((m_pert - base_m) / dh) * p_spec.span)
                    else:
                        fd_fem_grad.append(0.0)
                cos_val = _cosine(grad_vec, fd_fem_grad)
                grad_ok = cos_val >= problem.gradient_cos_min
                grad_checks.append(
                    GradientConsistencyCheck(
                        iteration=it,
                        parameters=dict(curr_p),
                        cosine_similarity=cos_val,
                        passed=grad_ok,
                        fem_calls_used=fem_calls_check,
                    )
                )
                if not grad_ok:
                    radius *= 0.5
                    iterations.append(
                        DesignIterationRecord(
                            iteration=it,
                            parameters=dict(curr_p),
                            trust_radius=radius,
                            surrogate_objective=curr_surr.objective,
                            surrogate_eta=curr_surr.eta_indicator,
                            in_envelope=curr_surr.in_envelope,
                            accepted=False,
                            escalated_to_fem=True,
                            fem_objective=base_fem.objective,
                            fem_feasible=problem.check_constraints(
                                base_fem.constraints
                            )[0],
                            reason=f"wrong-sign surrogate gradient (cos={cos_val:.3f}); shrunk trust radius",
                        )
                    )
                    continue

        # Propose projected trust-region step in normalized coordinates.
        trial_p: dict[str, float] = {}
        for p_spec, g_scaled in zip(problem.parameters, grad_vec, strict=True):
            step_norm = -radius * (g_scaled / grad_norm)
            trial_p[p_spec.name] = p_spec.clamp(
                curr_p[p_spec.name] + step_norm * p_spec.span
            )

        if all(abs(trial_p[n] - curr_p[n]) <= 1e-14 for n in p_names):
            converged = True
            break

        trial_surr = surrogate_evaluator(trial_p)
        surr_evals += 1

        gate_ok = (
            trial_surr.admissible
            and trial_surr.in_envelope
            and trial_surr.eta_indicator <= problem.accept_eta_tolerance
        )
        trial_merit = (
            _merit(problem, trial_surr.objective, trial_surr.constraints)
            if trial_surr.admissible
            else float("inf")
        )

        if not gate_ok:
            # Trust gate failed -> shrink region and escalate to FEM if budget permits.
            escalated = False
            fem_obj: float | None = None
            fem_feas: bool | None = None
            if fem_evals + 1 < problem.max_fem_evaluations:
                f_res = _call_fem(trial_p)
                escalated = f_res is not None
                if (
                    f_res is not None
                    and f_res.succeeded
                    and f_res.objective is not None
                ):
                    fem_obj = f_res.objective
                    fem_feas = problem.check_constraints(f_res.constraints)[0]
                    f_merit = _merit(problem, f_res.objective, f_res.constraints)
                    if fem_feas and f_merit < curr_merit:
                        curr_p = trial_p
                        if trial_surr.admissible:
                            curr_surr = trial_surr
                            curr_merit = min(trial_merit, f_merit)
                        else:
                            curr_merit = f_merit
                        radius = max(problem.min_trust_radius, 0.75 * radius)
                        iterations.append(
                            DesignIterationRecord(
                                iteration=it,
                                parameters=dict(trial_p),
                                trust_radius=radius,
                                surrogate_objective=trial_surr.objective,
                                surrogate_eta=trial_surr.eta_indicator,
                                in_envelope=trial_surr.in_envelope,
                                accepted=True,
                                escalated_to_fem=True,
                                fem_objective=fem_obj,
                                fem_feasible=fem_feas,
                                reason="trust gate triggered FEM escalation; FEM verified improvement",
                            )
                        )
                        continue
            radius *= 0.5
            iterations.append(
                DesignIterationRecord(
                    iteration=it,
                    parameters=dict(trial_p),
                    trust_radius=radius,
                    surrogate_objective=trial_surr.objective,
                    surrogate_eta=trial_surr.eta_indicator,
                    in_envelope=trial_surr.in_envelope,
                    accepted=False,
                    escalated_to_fem=escalated,
                    fem_objective=fem_obj,
                    fem_feasible=fem_feas,
                    reason="trust gate failed; rejected step and shrunk trust radius",
                )
            )
            continue

        if trial_merit < curr_merit - 1e-12:
            curr_p = trial_p
            curr_surr = trial_surr
            curr_merit = trial_merit
            radius = min(problem.initial_trust_radius * 1.5, radius * 1.15)
            iterations.append(
                DesignIterationRecord(
                    iteration=it,
                    parameters=dict(trial_p),
                    trust_radius=radius,
                    surrogate_objective=trial_surr.objective,
                    surrogate_eta=trial_surr.eta_indicator,
                    in_envelope=trial_surr.in_envelope,
                    accepted=True,
                    escalated_to_fem=False,
                    fem_objective=None,
                    fem_feasible=None,
                    reason="surrogate step passed trust gate and reduced merit",
                )
            )
        else:
            radius *= 0.5
            iterations.append(
                DesignIterationRecord(
                    iteration=it,
                    parameters=dict(trial_p),
                    trust_radius=radius,
                    surrogate_objective=trial_surr.objective,
                    surrogate_eta=trial_surr.eta_indicator,
                    in_envelope=trial_surr.in_envelope,
                    accepted=False,
                    escalated_to_fem=False,
                    fem_objective=None,
                    fem_feasible=None,
                    reason="no merit reduction; shrunk trust radius",
                )
            )

    # Mandatory final FEM verification of the converged surrogate design point.
    if best_fem is None or best_fem.parameters != curr_p:
        _call_fem(curr_p)

    if best_fem is None or not best_fem.fem_feasible:
        return DesignOptimizationReport(
            problem_hash=problem.problem_hash,
            status=DesignOptimizationStatus.FAILED_NO_FEASIBLE_FEM_DESIGN,
            best_design=best_fem,
            surrogate_evaluations=surr_evals,
            fem_evaluations=fem_evals,
            failed_designs=failed_designs,
            iterations=tuple(iterations),
            gradient_checks=tuple(grad_checks),
            reason="no FEM-verified feasible design found within budget",
        )

    status = (
        DesignOptimizationStatus.CONVERGED_FEM_VERIFIED
        if converged
        else DesignOptimizationStatus.BUDGET_EXHAUSTED_FEM_VERIFIED
    )
    return DesignOptimizationReport(
        problem_hash=problem.problem_hash,
        status=status,
        best_design=best_fem,
        surrogate_evaluations=surr_evals,
        fem_evaluations=fem_evals,
        failed_designs=failed_designs,
        iterations=tuple(iterations),
        gradient_checks=tuple(grad_checks),
        reason="final design independently verified by converged FEM reference",
    )


__all__ = (
    "DESIGN_OPTIMIZATION_MACHINE",
    "DesignIterationRecord",
    "DesignOptimizationPhase",
    "DesignOptimizationReport",
    "DesignOptimizationStatus",
    "FemDesignEvaluation",
    "FemEvaluator",
    "GradientConsistencyCheck",
    "SurrogateDesignEvaluation",
    "SurrogateEvaluator",
    "VerifiedDesignSolution",
    "run_fem_guarded_design_optimization",
)
