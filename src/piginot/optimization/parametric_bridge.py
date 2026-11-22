"""Parametric CAD bundle bridge for FEM-guarded design optimization (M4 / F3 + P5).

Connects any trained or qualified :class:`~piginot.cad.parametric_adapter.ParametricSurrogateBundle`
directly into :func:`~piginot.optimization.trust_region.run_fem_guarded_design_optimization`:

* **Design variables**: CAD geometry parameters declared in ``ParametricCadSpec``
  (e.g., ``hole_radius``, ``notch_radius``, ``plate_width``).
* **Surrogate exploration**: regenerates OpenCASCADE CAD variants, enforces
  :class:`~piginot.cad.parametric.TopologyStabilityGuard` and
  :class:`~piginot.operators.geometry_tokenizer.GeometryOODDetector` fail-closed,
  and evaluates fast ``BoundaryEnergyOperator`` predictions and sensitivities.
* **Authoritative FEM verification**: periodic FEM finite-difference gradient
  consistency checks and mandatory final verification of the returned optimum
  via converged P2 :class:`~piginot.fem.backend.ScikitFemBackend` solves.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from piginot.cad.parametric import (
    ParametricCadSpec,
    ParametricTopologyError,
    TopologyStabilityGuard,
)
from piginot.cad.parametric_adapter import (
    ParametricFieldMetrics,
    ParametricSurrogateBundle,
    _evaluate_operator_on_mesh,
    evaluate_parametric_fem,
)
from piginot.contracts import CanonicalMesh
from piginot.contracts.expressions import DerivedExpression
from piginot.fem.backend import ScikitFemBackend
from piginot.operators.geometry_tokenizer import (
    GeometryOODStatus,
    GeometryTokenizer,
)
from piginot.optimization.bundle_bridge import _eval_expr_ast
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

SUPPORTED_PARAMETRIC_METRICS = frozenset(
    (
        "area_mm2",
        "peak_von_mises_mpa",
        "reaction_force_n",
        "compliance_n_mm",
        "max_displacement_mm",
        "min_det_f",
    )
)


@dataclass(frozen=True, slots=True)
class ParametricDesignSpec:
    """Declarative mapping between a ``DesignProblem`` and a parametric CAD bundle."""

    problem: DesignProblem
    objective_metric: str
    constraint_metrics: Mapping[str, str]
    fd_surrogate_step_fraction: float = 2.0e-2
    fd_fem_step_fraction: float = 4.0e-2
    remesh_distance_fraction: float = 5.0e-3

    def __post_init__(self) -> None:
        for label, metric in (
            ("objective_metric", self.objective_metric),
            *[(f"constraint.{k}", v) for k, v in self.constraint_metrics.items()],
        ):
            if (
                not metric.startswith("expr:")
                and metric not in SUPPORTED_PARAMETRIC_METRICS
            ):
                raise ValueError(
                    f"unsupported {label} {metric!r}; expected one of "
                    f"{sorted(SUPPORTED_PARAMETRIC_METRICS)} or 'expr:<formula>'"
                )


def parse_parametric_design_document(
    document: Mapping[str, Any],
    *,
    bundle: ParametricSurrogateBundle,
) -> ParametricDesignSpec:
    """Parse a declarative JSON design-problem document bound to a parametric bundle."""
    spec_params = {p.name: p for p in bundle.spec.parameters}
    raw_params = document.get("parameters")
    if raw_params:
        parsed_params: list[DesignParameter] = []
        for p in raw_params:
            p_name = str(p["name"])
            if p_name not in spec_params:
                raise ValueError(
                    f"design parameter {p_name!r} is not declared in ParametricCadSpec "
                    f"({sorted(spec_params)})"
                )
            parsed_params.append(
                DesignParameter(
                    name=p_name,
                    lower=float(p["lower"]),
                    upper=float(p["upper"]),
                    initial=float(p["initial"]),
                    unit=str(p.get("unit", spec_params[p_name].unit)),
                )
            )
        params = tuple(parsed_params)
    else:
        params = bundle.spec.parameters

    raw_cons = document.get("constraints", ())
    constraints = tuple(
        DesignConstraint(
            name=str(c["name"]),
            kind=c["kind"],
            limit=float(c["limit"]),
            tolerance=float(c.get("tolerance", 1.0e-3)),
            unit=str(c.get("unit", "1")),
        )
        for c in raw_cons
    )
    constraint_metrics = {
        str(c["name"]): str(c.get("metric", c["name"])) for c in raw_cons
    }

    problem = DesignProblem(
        name=str(document["name"]),
        geometry_version=str(document.get("geometry_version", bundle.spec.spec_hash)),
        problem_spec_hash=str(bundle.bundle_hash),
        parameters=params,
        objective_name=str(document["objective_name"]),
        objective_unit=str(document.get("objective_unit", "mm2")),
        minimize=bool(document.get("minimize", True)),
        constraints=constraints,
        max_iterations=int(document.get("max_iterations", 18)),
        max_fem_evaluations=int(document.get("max_fem_evaluations", 14)),
        initial_trust_radius=float(document.get("initial_trust_radius", 0.25)),
        min_trust_radius=float(document.get("min_trust_radius", 0.02)),
        gradient_check_interval=int(document.get("gradient_check_interval", 4)),
        gradient_cos_min=float(document.get("gradient_cos_min", 0.15)),
        accept_eta_tolerance=float(document.get("accept_eta_tolerance", 1.0)),
        constraint_penalty_weight=float(
            document.get("constraint_penalty_weight", 25.0)
        ),
    )
    return ParametricDesignSpec(
        problem=problem,
        objective_metric=str(document.get("objective_metric", problem.objective_name)),
        constraint_metrics=constraint_metrics,
        fd_surrogate_step_fraction=float(
            document.get("fd_surrogate_step_fraction", 2.0e-2)
        ),
        fd_fem_step_fraction=float(document.get("fd_fem_step_fraction", 4.0e-2)),
        remesh_distance_fraction=float(
            document.get("remesh_distance_fraction", 5.0e-3)
        ),
    )


def _analytic_template_area_mm2(
    spec: ParametricCadSpec, params: Mapping[str, float]
) -> float | None:
    """Exact smooth area formula for built-in OpenCASCADE templates."""
    env = spec.full_environment(params)
    length = float(env.get("plate_length", 40.0))
    width = float(env.get("plate_width", 20.0))
    hole_r = float(env.get("hole_radius", 0.0))
    if spec.template == "holed-plate":
        return length * width - math.pi * hole_r * hole_r
    if spec.template == "notched-mounting-bracket":
        notch_r = float(env.get("notch_radius", 0.0))
        return length * width - math.pi * hole_r * hole_r - math.pi * notch_r * notch_r
    return None


class ParametricBundleDesignBridge:
    """Connects a ``ParametricSurrogateBundle`` to P5 FEM-guarded optimization."""

    def __init__(
        self,
        bundle: ParametricSurrogateBundle,
        design_spec: ParametricDesignSpec,
        *,
        backend: ScikitFemBackend | None = None,
    ) -> None:
        self.bundle = bundle
        self.design_spec = design_spec
        self._guard = TopologyStabilityGuard(bundle.spec)
        self._tokenizer = GeometryTokenizer(subdivisions_per_facet=1)
        self._backend = ScikitFemBackend() if backend is None else backend
        self._anchor_params: dict[str, float] | None = None
        self._anchor_mesh: CanonicalMesh | None = None
        self._anchor_ood_ok: bool = False
        self._anchor_ood_dist: float = 0.0
        self._anchor_ood_reason: str = ""
        self._fem_calls: int = 0
        self._remesh_count: int = 0

    @property
    def remesh_count(self) -> int:
        return self._remesh_count

    def _full_params(self, point: Mapping[str, float]) -> dict[str, float]:
        full = self.bundle.spec.nominal_parameters()
        for k, v in point.items():
            full[k] = float(v)
        return full

    def _normalized_distance(
        self, a: Mapping[str, float], b: Mapping[str, float]
    ) -> float:
        sq = 0.0
        for p in self.bundle.spec.parameters:
            d = (float(a[p.name]) - float(b[p.name])) / p.span
            sq += d * d
        return math.sqrt(sq)

    def _resolve_mesh(
        self, full_params: dict[str, float], *, force_remesh: bool = False
    ) -> tuple[bool, CanonicalMesh | None, float, str]:
        """Return topology- and OOD-checked mesh (reusing anchor for micro-FD probes)."""
        ok_env, env_msg = self.bundle.spec.check_parameter_envelope(
            full_params, enforce_box=True
        )
        if not ok_env:
            return False, None, 1.0, env_msg

        if (
            not force_remesh
            and self._anchor_mesh is not None
            and self._anchor_params is not None
            and self._normalized_distance(full_params, self._anchor_params)
            < self.design_spec.remesh_distance_fraction
        ):
            return (
                self._anchor_ood_ok,
                self._anchor_mesh,
                self._anchor_ood_dist,
                self._anchor_ood_reason,
            )

        try:
            inst = self._guard.generate(
                full_params, enforce_box=True, enforce_bounds=True
            )
            self._remesh_count += 1
        except ParametricTopologyError as exc:
            return False, None, 1.0, str(exc)

        ood = self.bundle.ood_detector.score(
            inst.mesh, self.bundle.spec.role_map, tokenizer=self._tokenizer
        )
        ood_ok = ood.status is GeometryOODStatus.IN_DISTRIBUTION
        dist_ratio = (
            float(ood.descriptor_distance) / max(ood.descriptor_threshold, 1.0e-12)
            if ood.descriptor_distance is not None
            else 1.0
        )
        self._anchor_params = dict(full_params)
        self._anchor_mesh = inst.mesh
        self._anchor_ood_ok = ood_ok
        self._anchor_ood_dist = dist_ratio
        self._anchor_ood_reason = ood.reason
        return ood_ok, inst.mesh, dist_ratio, ood.reason

    def _eval_metric(
        self,
        metric: str,
        *,
        full_params: Mapping[str, float],
        fields: ParametricFieldMetrics,
        use_smooth_area: bool = False,
    ) -> float:
        if metric.startswith("expr:"):
            expr = DerivedExpression.parse(metric[5:].strip())
            env = self.bundle.spec.full_environment(full_params)
            return _eval_expr_ast(expr.canonical_ast, env)
        if metric == "area_mm2":
            if use_smooth_area:
                exact_a = _analytic_template_area_mm2(self.bundle.spec, full_params)
                if exact_a is not None:
                    return exact_a
            return fields.area_mm2
        if metric == "peak_von_mises_mpa":
            return fields.peak_von_mises_mpa
        if metric == "reaction_force_n":
            return fields.reaction_force_n
        if metric == "compliance_n_mm":
            env = self.bundle.spec.full_environment(full_params)
            lx = float(env.get("plate_length", 40.0))
            ux = self.bundle.physics.nominal_strain * lx
            return fields.reaction_force_n * ux
        if metric == "max_displacement_mm":
            return fields.max_displacement_mm
        if metric == "min_det_f":
            return fields.min_det_f
        raise ValueError(f"unsupported parametric metric {metric!r}")

    def _eval_surrogate_point(
        self, full_params: dict[str, float], *, force_remesh: bool = False
    ) -> tuple[bool, ParametricFieldMetrics | None, float, str]:
        ood_ok, mesh, dist_ratio, reason = self._resolve_mesh(
            full_params, force_remesh=force_remesh
        )
        if not ood_ok or mesh is None:
            return False, None, dist_ratio, reason

        fields, pooled_tokens = _evaluate_operator_on_mesh(
            self.bundle.operator,
            self.bundle.spec,
            self.bundle.physics,
            mesh,
            full_params,
            tokenizer=self._tokenizer,
        )
        full_ood = self.bundle.ood_detector.score(
            mesh,
            self.bundle.spec.role_map,
            pooled_tokens=pooled_tokens,
            tokenizer=self._tokenizer,
        )
        if full_ood.status is not GeometryOODStatus.IN_DISTRIBUTION:
            return False, fields, 1.0, full_ood.reason
        if not fields.converged:
            return False, fields, 1.0, fields.reason
        return True, fields, dist_ratio, "ok"

    def surrogate_evaluator(
        self, point: Mapping[str, float]
    ) -> SurrogateDesignEvaluation:
        """Evaluate parametric CAD surrogate, OOD gate, and smooth objective gradient."""
        full_p = self._full_params(point)
        ok, fields, dist_ratio, reason = self._eval_surrogate_point(full_p)
        zero_cons = {c.name: 0.0 for c in self.design_spec.problem.constraints}
        if not ok or fields is None:
            return SurrogateDesignEvaluation(
                objective=0.0,
                gradient={p.name: 0.0 for p in self.design_spec.problem.parameters},
                constraints=zero_cons,
                in_envelope=False,
                admissible=False,
                eta_indicator=max(dist_ratio, 1.0),
                reason=reason,
            )

        obj_val = self._eval_metric(
            self.design_spec.objective_metric,
            full_params=full_p,
            fields=fields,
            use_smooth_area=True,
        )
        cons_vals = {
            c.name: self._eval_metric(
                self.design_spec.constraint_metrics[c.name],
                full_params=full_p,
                fields=fields,
                use_smooth_area=True,
            )
            for c in self.design_spec.problem.constraints
        }

        # Objective finite-difference sensitivities without remeshing jitter
        grad: dict[str, float] = {}
        is_pure_analytic_obj = self.design_spec.objective_metric.startswith(
            "expr:"
        ) or (
            self.design_spec.objective_metric == "area_mm2"
            and _analytic_template_area_mm2(self.bundle.spec, full_p) is not None
        )
        for p in self.design_spec.problem.parameters:
            if is_pure_analytic_obj:
                h = 1.0e-5 * p.span
                p_plus = dict(full_p)
                p_minus = dict(full_p)
                p_plus[p.name] = p.clamp(full_p[p.name] + h)
                p_minus[p.name] = p.clamp(full_p[p.name] - h)
                dh = p_plus[p.name] - p_minus[p.name]
                if dh > 0.0:
                    f_p = self._eval_metric(
                        self.design_spec.objective_metric,
                        full_params=p_plus,
                        fields=fields,
                        use_smooth_area=True,
                    )
                    f_m = self._eval_metric(
                        self.design_spec.objective_metric,
                        full_params=p_minus,
                        fields=fields,
                        use_smooth_area=True,
                    )
                    grad[p.name] = (f_p - f_m) / dh
                else:
                    grad[p.name] = 0.0
            else:
                h = self.design_spec.fd_surrogate_step_fraction * p.span
                p_pert = dict(full_p)
                sign = 1.0 if full_p[p.name] + h <= p.upper else -1.0
                p_pert[p.name] = p.clamp(full_p[p.name] + sign * h)
                dh = p_pert[p.name] - full_p[p.name]
                if abs(dh) <= 1.0e-12:
                    grad[p.name] = 0.0
                    continue
                ok_p, f_pert, _, _ = self._eval_surrogate_point(p_pert)
                if ok_p and f_pert is not None:
                    v_p = self._eval_metric(
                        self.design_spec.objective_metric,
                        full_params=p_pert,
                        fields=f_pert,
                        use_smooth_area=True,
                    )
                    grad[p.name] = (v_p - obj_val) / dh
                else:
                    grad[p.name] = 0.0

        return SurrogateDesignEvaluation(
            objective=obj_val,
            gradient=grad,
            constraints=cons_vals,
            in_envelope=True,
            admissible=True,
            eta_indicator=min(dist_ratio, 0.99),
            reason="ok",
        )

    def fem_evaluator(self, point: Mapping[str, float]) -> FemDesignEvaluation:
        """Authoritative converged P2 FEM solve on regenerated OpenCASCADE CAD."""
        self._fem_calls += 1
        full_p = self._full_params(point)
        ok_env, env_msg = self.bundle.spec.check_parameter_envelope(
            full_p, enforce_box=True
        )
        if not ok_env:
            return FemDesignEvaluation(
                succeeded=False,
                objective=None,
                constraints={},
                reason=env_msg,
            )
        try:
            inst = self._guard.generate(full_p, enforce_box=True, enforce_bounds=True)
            self._remesh_count += 1
        except ParametricTopologyError as exc:
            return FemDesignEvaluation(
                succeeded=False,
                objective=None,
                constraints={},
                reason=str(exc),
            )

        fem = evaluate_parametric_fem(
            inst.mesh,
            self.bundle.physics,
            backend=self._backend,
            request_id=f"opt-fem-{self._fem_calls}",
        )
        if not fem.converged:
            return FemDesignEvaluation(
                succeeded=False,
                objective=None,
                constraints={},
                reason=fem.reason,
            )
        obj_val = self._eval_metric(
            self.design_spec.objective_metric,
            full_params=full_p,
            fields=fem,
            use_smooth_area=False,
        )
        cons_vals = {
            c.name: self._eval_metric(
                self.design_spec.constraint_metrics[c.name],
                full_params=full_p,
                fields=fem,
                use_smooth_area=False,
            )
            for c in self.design_spec.problem.constraints
        }
        return FemDesignEvaluation(
            succeeded=True,
            objective=obj_val,
            constraints=cons_vals,
            reference_hash=fem.result_hash,
            reason="converged",
        )

    def run(self) -> DesignOptimizationReport:
        """Execute FEM-guarded trust-region optimization over the parametric CAD family."""
        return run_fem_guarded_design_optimization(
            self.design_spec.problem,
            surrogate_evaluator=self.surrogate_evaluator,
            fem_evaluator=self.fem_evaluator,
            geometry_supported=True,
            fd_step_fraction=self.design_spec.fd_fem_step_fraction,
        )


__all__ = (
    "SUPPORTED_PARAMETRIC_METRICS",
    "ParametricBundleDesignBridge",
    "ParametricDesignSpec",
    "parse_parametric_design_document",
)
