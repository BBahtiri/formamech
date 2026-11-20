"""Bundle-to-FEM bridge for FEM-guarded differentiable design optimization (P5).

Connects any verified FormaMech ``Bundle`` and its bound ``FemRecipe`` directly
into :func:`~piginot.optimization.trust_region.run_fem_guarded_design_optimization`:

* Surrogate evaluations use the bundle's neural operator + executable envelope
  gate (``bundle.checker``) and finite-difference surrogate sensitivities.
* FEM evaluations and periodic FD gradient consistency checks execute real
  converged ``ScikitFemBackend`` P2 FEM solves via the bundle's ``FemRecipe``.
* Supported scalar response metrics:
  * ``reaction_force`` (summed nodal reaction force on loaded Dirichlet BC),
  * ``avg_p11`` (area-weighted regional average first-Piola stress ``P_11``),
  * ``peak_p_frobenius`` (maximum Frobenius norm of ``P_served``),
  * ``expr:<formula>`` (any safe ``DerivedExpression`` over design parameters).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from piginot.contracts import CanonicalQuantity, Quantity, Unit
from piginot.contracts.expressions import DerivedExpression, ExpressionTree
from piginot.optimization.problem import (
    DesignConstraint,
    DesignParameter,
    DesignProblem,
)
from piginot.optimization.trust_region import (
    DesignOptimizationReport,
    FemDesignEvaluation,
    SurrogateDesignEvaluation,
    run_fem_guarded_design_optimization,
)


def _eval_expr_ast(tree: ExpressionTree, env: Mapping[str, float]) -> float:
    kind = tree[0]
    if kind == "number":
        return float(cast(float, tree[1]))
    if kind == "name":
        name = cast(str, tree[1])
        if name not in env:
            raise ValueError(f"unknown design symbol {name!r}")
        return float(env[name])
    if kind == "unary":
        op = cast(str, tree[1])
        val = _eval_expr_ast(cast(ExpressionTree, tree[2]), env)
        return -val if op == "-" else val
    if kind == "binary":
        op = cast(str, tree[1])
        lhs = _eval_expr_ast(cast(ExpressionTree, tree[2]), env)
        rhs = _eval_expr_ast(cast(ExpressionTree, tree[3]), env)
        if op == "+":
            return lhs + rhs
        if op == "-":
            return lhs - rhs
        if op == "*":
            return lhs * rhs
        if op == "/":
            return lhs / rhs
    raise ValueError(f"unsupported expression node: {tree!r}")


@dataclass(frozen=True, slots=True)
class BundleDesignSpec:
    """Mapping between a ``DesignProblem`` and a FormaMech bundle's inputs/metrics."""

    problem: DesignProblem
    fixed_inputs: Mapping[str, CanonicalQuantity]
    parameter_units: Mapping[str, Unit]
    objective_metric: str
    constraint_metrics: Mapping[str, str]
    fd_surrogate_step_fraction: float = 1e-3
    fd_fem_step_fraction: float = 2e-2


def parse_bundle_design_document(
    document: Mapping[str, Any],
    *,
    bundle: Any,
) -> BundleDesignSpec:
    """Parse a declarative JSON design-problem document bound to ``bundle``."""
    raw_params = document["parameters"]
    params = tuple(
        DesignParameter(
            name=str(p["name"]),
            lower=float(p["lower"]),
            upper=float(p["upper"]),
            initial=float(p["initial"]),
            unit=str(p["unit"]),
        )
        for p in raw_params
    )
    param_units = {p.name: Unit(p.unit) for p in params}

    raw_cons = document.get("constraints", ())
    constraints = tuple(
        DesignConstraint(
            name=str(c["name"]),
            kind=c["kind"],
            limit=float(c["limit"]),
            tolerance=float(c.get("tolerance", 1e-4)),
            unit=str(c.get("unit", "1")),
        )
        for c in raw_cons
    )
    constraint_metrics = {
        str(c["name"]): str(c.get("metric", c["name"])) for c in raw_cons
    }
    fixed_inputs: dict[str, CanonicalQuantity] = {}
    for k, raw in dict(document.get("fixed_inputs", {})).items():
        fixed_inputs[str(k)] = Quantity(
            float(raw["value"]), Unit(str(raw["unit"]))
        ).to_canonical()

    problem = DesignProblem(
        name=str(document["name"]),
        geometry_version=str(
            document.get("geometry_version", bundle.reference_mesh.mesh_id)
        ),
        problem_spec_hash=str(bundle.manifest.manifest_sha256),
        parameters=params,
        objective_name=str(document["objective_name"]),
        objective_unit=str(document.get("objective_unit", "1")),
        minimize=bool(document.get("minimize", True)),
        constraints=constraints,
        max_iterations=int(document.get("max_iterations", 20)),
        max_fem_evaluations=int(document.get("max_fem_evaluations", 10)),
        initial_trust_radius=float(document.get("initial_trust_radius", 0.25)),
        min_trust_radius=float(document.get("min_trust_radius", 0.01)),
        gradient_check_interval=int(document.get("gradient_check_interval", 5)),
        gradient_cos_min=float(document.get("gradient_cos_min", 0.25)),
        accept_eta_tolerance=float(document.get("accept_eta_tolerance", 0.25)),
        constraint_penalty_weight=float(
            document.get("constraint_penalty_weight", 20.0)
        ),
    )
    return BundleDesignSpec(
        problem=problem,
        fixed_inputs=fixed_inputs,
        parameter_units=param_units,
        objective_metric=str(document.get("objective_metric", problem.objective_name)),
        constraint_metrics=constraint_metrics,
        fd_surrogate_step_fraction=float(
            document.get("fd_surrogate_step_fraction", 1e-3)
        ),
        fd_fem_step_fraction=float(document.get("fd_fem_step_fraction", 2e-2)),
    )


class BundleDesignBridge:
    """Connects a verified FormaMech ``Bundle`` + ``FemRecipe`` to P5 optimization."""

    def __init__(
        self,
        bundle: Any,
        recipe: Any,
        spec: BundleDesignSpec,
        *,
        backend: Any | None = None,
    ) -> None:
        self.bundle = bundle
        self.recipe = recipe
        self.spec = spec
        self._backend = backend
        self._fem_calls = 0

    def _canonical_inputs(
        self, point: Mapping[str, float]
    ) -> dict[str, CanonicalQuantity]:
        inputs = dict(self.spec.fixed_inputs)
        for p in self.spec.problem.parameters:
            inputs[p.name] = Quantity(
                float(point[p.name]), self.spec.parameter_units[p.name]
            ).to_canonical()
        return inputs

    def _eval_metric(
        self,
        metric: str,
        *,
        point: Mapping[str, float],
        linearization: Any,
        state: Any,
        p_served: np.ndarray[Any, Any] | None = None,
    ) -> float:
        from piginot.certify.adjoint import ReactionForceQoi, RegionalAverageStressQoi

        if metric.startswith("expr:"):
            expr = DerivedExpression.parse(metric[5:].strip())
            env = {
                **{k: float(q.value) for k, q in self.spec.fixed_inputs.items()},
                **{k: float(v) for k, v in point.items()},
            }
            return _eval_expr_ast(expr.canonical_ast, env)
        if metric == "reaction_force":
            total = 0.0
            for bc in linearization.request.boundary_conditions:
                if bc.kind == "dirichlet" and any(
                    abs(v.value) > 0.0 for v in bc.values
                ):
                    qoi = ReactionForceQoi(f"rxn_{bc.name}", bc.name)
                    total += abs(qoi.evaluate(linearization, state))
            return total
        if metric == "avg_p11":
            return RegionalAverageStressQoi("avg_p11", component=(0, 0)).evaluate(
                linearization, state
            )
        if metric == "peak_p_frobenius":
            if p_served is not None:
                return float(np.max(np.linalg.norm(p_served, axis=(-2, -1))))
            p_arr = np.asarray(state.stress, dtype=np.float64)[..., :2, :2]
            return float(np.max(np.linalg.norm(p_arr, axis=(-2, -1))))
        raise ValueError(f"unsupported design metric {metric!r}")

    def _surrogate_values(
        self, point: Mapping[str, float]
    ) -> tuple[bool, bool, float, float, dict[str, float], str]:
        from piginot.bundle import predict
        from piginot.fem.linearization import FemLinearization
        from piginot.fem.models import FemSolveFailure
        from piginot.serving.hybrid import default_request_id, operator_displacement

        inputs = self._canonical_inputs(point)
        needed = {self.spec.objective_metric, *self.spec.constraint_metrics.values()}
        needs_field = any(not m.startswith("expr:") for m in needed)

        pred = predict(self.bundle, inputs, queries=None, confidence=False)
        if not pred.in_domain or pred.mesh is None:
            return (
                False,
                False,
                1.0,
                float("inf"),
                {c.name: 0.0 for c in self.spec.problem.constraints},
                "out of bundle envelope",
            )

        if not needs_field:
            obj = self._eval_metric(
                self.spec.objective_metric,
                point=point,
                linearization=None,
                state=None,
            )
            cons = {
                c.name: self._eval_metric(
                    self.spec.constraint_metrics[c.name],
                    point=point,
                    linearization=None,
                    state=None,
                )
                for c in self.spec.problem.constraints
            }
            return True, True, 0.0, obj, cons, ""

        req = self.recipe.build_request(inputs, pred.mesh, default_request_id(inputs))
        lin = FemLinearization(req)
        u_nodal = operator_displacement(
            self.bundle, inputs, pred.mesh, lin.dof_coordinates
        )
        try:
            state = lin.state(lin.with_exact_dirichlet(lin.vector(u_nodal)))
        except FemSolveFailure as exc:
            return (
                True,
                False,
                1.0,
                float("inf"),
                {c.name: 0.0 for c in self.spec.problem.constraints},
                str(exc),
            )
        # Lightweight discrete residual energy indicator (no full FEM solve).
        res_free = state.residual[lin.constraints.free]
        rhs_scale = max(float(np.linalg.norm(state.assembly.internal_force)), 1e-12)
        eta = float(np.linalg.norm(res_free)) / rhs_scale

        obj = self._eval_metric(
            self.spec.objective_metric,
            point=point,
            linearization=lin,
            state=state,
        )
        cons = {
            c.name: self._eval_metric(
                self.spec.constraint_metrics[c.name],
                point=point,
                linearization=lin,
                state=state,
            )
            for c in self.spec.problem.constraints
        }
        return True, True, eta, obj, cons, ""

    def evaluate_surrogate(
        self, point: Mapping[str, float]
    ) -> SurrogateDesignEvaluation:
        in_env, adm, eta, obj, cons, reason = self._surrogate_values(point)
        if not in_env or not adm:
            return SurrogateDesignEvaluation(
                objective=obj,
                gradient={p.name: 0.0 for p in self.spec.problem.parameters},
                constraints=cons,
                in_envelope=in_env,
                admissible=adm,
                eta_indicator=eta,
                reason=reason,
            )
        grad: dict[str, float] = {}
        for p in self.spec.problem.parameters:
            h = max(self.spec.fd_surrogate_step_fraction * p.span, 1e-6)
            p_plus = dict(point)
            p_minus = dict(point)
            p_plus[p.name] = p.clamp(point[p.name] + h)
            p_minus[p.name] = p.clamp(point[p.name] - h)
            dh = p_plus[p.name] - p_minus[p.name]
            if dh <= 0.0:
                grad[p.name] = 0.0
                continue
            ok_p, adm_p, _, o_p, _, _ = self._surrogate_values(p_plus)
            ok_m, adm_m, _, o_m, _, _ = self._surrogate_values(p_minus)
            if ok_p and adm_p and ok_m and adm_m and math.isfinite(o_p - o_m):
                grad[p.name] = (o_p - o_m) / dh
            else:
                grad[p.name] = 0.0
        return SurrogateDesignEvaluation(
            objective=obj,
            gradient=grad,
            constraints=cons,
            in_envelope=True,
            admissible=True,
            eta_indicator=eta,
        )

    def evaluate_fem(self, point: Mapping[str, float]) -> FemDesignEvaluation:
        from piginot.fem import ScikitFemBackend
        from piginot.fem.linearization import FemLinearization

        self._fem_calls += 1
        inputs = self._canonical_inputs(point)
        geom = self.bundle.geometry(inputs)
        if geom.mesh is None:
            return FemDesignEvaluation(
                succeeded=False,
                objective=None,
                constraints={},
                reason="geometry morph refused",
            )
        req = self.recipe.build_request(inputs, geom.mesh, f"opt-fem-{self._fem_calls}")
        backend = self._backend if self._backend is not None else ScikitFemBackend()
        att = backend.solve_many((req,))[0]
        if not att.success or att.result is None:
            return FemDesignEvaluation(
                succeeded=False,
                objective=None,
                constraints={},
                reason=str(att.message),
            )
        lin = FemLinearization(req)
        u_nodal = np.asarray(att.result.displacement, dtype=np.float64)
        state = lin.state(lin.vector(u_nodal))
        obj = self._eval_metric(
            self.spec.objective_metric,
            point=point,
            linearization=lin,
            state=state,
        )
        cons = {
            c.name: self._eval_metric(
                self.spec.constraint_metrics[c.name],
                point=point,
                linearization=lin,
                state=state,
            )
            for c in self.spec.problem.constraints
        }
        return FemDesignEvaluation(
            succeeded=True,
            objective=obj,
            constraints=cons,
            reference_hash=str(att.result_hash),
        )

    def run(self, *, geometry_supported: bool = True) -> DesignOptimizationReport:
        return run_fem_guarded_design_optimization(
            self.spec.problem,
            surrogate_evaluator=self.evaluate_surrogate,
            fem_evaluator=self.evaluate_fem,
            geometry_supported=geometry_supported,
            fd_step_fraction=self.spec.fd_fem_step_fraction,
        )


__all__ = (
    "BundleDesignBridge",
    "BundleDesignSpec",
    "parse_bundle_design_document",
)
