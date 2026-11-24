"""FEM-guarded 3D parametric CAD trust-region shape optimization bridge (V3-2 + P5).

Connects any trained :class:`~piginot.training.parametric_3d_engine.Parametric3DBundle`
to :func:`~piginot.optimization.trust_region.run_fem_guarded_design_optimization`:

* Surrogate explores 3D CAD variants in milliseconds guarded by
  :class:`~piginot.cad.parametric_3d.TopologyStabilityGuard3D` and
  :class:`~piginot.operators.geometry_3d.GeometryOODDetector3D`.
* Periodic 3D ``TET10`` FEM finite-difference gradient consistency checks.
* Mandatory final 3D ``TET10`` FEM verification of every accepted optimal design.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from piginot.cad.parametric_3d import (
    ParametricTopology3DError,
    TopologyStabilityGuard3D,
)
from piginot.fem.backend_3d import TorchFemBackend3D
from piginot.fem.linear_solvers import LinearSolverConfig
from piginot.optimization.problem import DesignProblem
from piginot.optimization.trust_region import (
    DesignOptimizationReport,
    FemDesignEvaluation,
    SurrogateDesignEvaluation,
    run_fem_guarded_design_optimization,
)
from piginot.training.parametric_3d_engine import (
    Parametric3DBundle,
    _build_fem_request,
    predict_parametric_3d,
)

SUPPORTED_3D_DESIGN_METRICS = frozenset(
    (
        "volume_mm3",
        "peak_von_mises_mpa",
        "max_displacement_mm",
        "compliance_n_mm",
        "min_det_f",
    )
)


@dataclass(frozen=True, slots=True)
class Parametric3DDesignSpec:
    """Mapping between a ``DesignProblem`` and a 3D parametric CAD bundle."""

    problem: DesignProblem
    objective_metric: str = "volume_mm3"
    constraint_metrics: Mapping[str, str] = ()  # type: ignore[assignment]
    fd_surrogate_step_fraction: float = 3.0e-2
    fd_fem_step_fraction: float = 5.0e-2

    def __post_init__(self) -> None:
        c_map = (
            dict(self.constraint_metrics)
            if isinstance(self.constraint_metrics, Mapping)
            else {}
        )
        object.__setattr__(self, "constraint_metrics", c_map)
        if self.objective_metric not in SUPPORTED_3D_DESIGN_METRICS:
            raise ValueError(
                f"unsupported objective_metric {self.objective_metric!r}; "
                f"expected one of {sorted(SUPPORTED_3D_DESIGN_METRICS)}"
            )
        for name, metric in c_map.items():
            if metric not in SUPPORTED_3D_DESIGN_METRICS:
                raise ValueError(
                    f"unsupported constraint metric {name}={metric!r}; "
                    f"expected one of {sorted(SUPPORTED_3D_DESIGN_METRICS)}"
                )


class Parametric3DDesignBridge:
    """Adapter plugging ``Parametric3DBundle`` into P5 FEM-guarded optimization."""

    def __init__(
        self,
        bundle: Parametric3DBundle,
        design_spec: Parametric3DDesignSpec,
        *,
        backend: TorchFemBackend3D | None = None,
        linear_solver: LinearSolverConfig | None = None,
    ) -> None:
        self.bundle = bundle
        self.design_spec = design_spec
        self.backend = TorchFemBackend3D() if backend is None else backend
        self.linear_solver = linear_solver
        self._guard = TopologyStabilityGuard3D(bundle.spec)

    def _metrics_from_surrogate(
        self, params: Mapping[str, float]
    ) -> tuple[bool, dict[str, float], str]:
        rep = predict_parametric_3d(
            self.bundle, params, mode="fast", backend=self.backend
        )
        if (
            not rep.accepted
            or rep.volume_mm3 is None
            or rep.peak_von_mises_mpa is None
            or rep.max_displacement_mm is None
            or rep.compliance_n_mm is None
            or rep.min_det_f is None
        ):
            return False, {}, rep.reason
        return (
            True,
            {
                "volume_mm3": rep.volume_mm3,
                "peak_von_mises_mpa": rep.peak_von_mises_mpa,
                "max_displacement_mm": rep.max_displacement_mm,
                "compliance_n_mm": rep.compliance_n_mm,
                "min_det_f": rep.min_det_f,
            },
            "ok",
        )

    def _metrics_from_fem(
        self, params: Mapping[str, float]
    ) -> tuple[bool, dict[str, float], str]:
        try:
            inst = self._guard.build_and_verify(params, instance_id="opt-fem-3d")
        except ParametricTopology3DError as exc:
            return False, {}, str(exc)
        req = _build_fem_request(
            inst, self.bundle.physics, linear_solver=self.linear_solver
        )
        res = self.backend.solve(req)
        if not res.converged:
            return False, {}, res.reason
        return (
            True,
            {
                "volume_mm3": float(inst.imported.volume_mm3),
                "peak_von_mises_mpa": float(res.peak_von_mises_mpa),
                "max_displacement_mm": float(res.max_displacement_mm),
                "compliance_n_mm": float(res.external_work_n_mm),
                "min_det_f": float(res.min_det_f),
            },
            res.result_hash,
        )

    def evaluate_surrogate(
        self, design_vector: Mapping[str, float]
    ) -> SurrogateDesignEvaluation:
        ok, m0, reason = self._metrics_from_surrogate(design_vector)
        if not ok:
            return SurrogateDesignEvaluation(
                objective=float("inf"),
                gradient={p.name: 0.0 for p in self.design_spec.problem.parameters},
                constraints={
                    c.name: float("inf") for c in self.design_spec.problem.constraints
                },
                in_envelope=False,
                admissible=False,
                eta_indicator=float("inf"),
                reason=reason,
            )

        obj0 = m0[self.design_spec.objective_metric]
        c_vals = {
            c.name: m0[self.design_spec.constraint_metrics[c.name]]
            for c in self.design_spec.problem.constraints
        }
        obj_grad: dict[str, float] = {}

        for p in self.design_spec.problem.parameters:
            step = self.design_spec.fd_surrogate_step_fraction * max(
                p.upper - p.lower, 1.0e-6
            )
            val = float(design_vector[p.name])
            cand_plus = dict(design_vector)
            cand_plus[p.name] = min(p.upper, val + step)
            delta = cand_plus[p.name] - val
            if abs(delta) < 1.0e-9:
                cand_minus = dict(design_vector)
                cand_minus[p.name] = max(p.lower, val - step)
                delta = cand_minus[p.name] - val
                ok_p, mp, _ = self._metrics_from_surrogate(cand_minus)
            else:
                ok_p, mp, _ = self._metrics_from_surrogate(cand_plus)

            if not ok_p or abs(delta) < 1.0e-12:
                obj_grad[p.name] = 0.0
            else:
                obj_grad[p.name] = (
                    mp[self.design_spec.objective_metric] - obj0
                ) / delta

        return SurrogateDesignEvaluation(
            objective=obj0,
            gradient=obj_grad,
            constraints=c_vals,
            in_envelope=True,
            admissible=True,
            eta_indicator=0.01,
            reason="ok",
        )

    def evaluate_fem(self, design_vector: Mapping[str, float]) -> FemDesignEvaluation:
        ok, m0, ref_or_reason = self._metrics_from_fem(design_vector)
        if not ok:
            return FemDesignEvaluation(
                succeeded=False,
                objective=None,
                constraints={
                    c.name: float("inf") for c in self.design_spec.problem.constraints
                },
                reference_hash=None,
                reason=ref_or_reason,
            )
        admissible = math.isfinite(m0["min_det_f"]) and m0["min_det_f"] > 1.0e-5
        return FemDesignEvaluation(
            succeeded=admissible,
            objective=m0[self.design_spec.objective_metric] if admissible else None,
            constraints={
                c.name: m0[self.design_spec.constraint_metrics[c.name]]
                for c in self.design_spec.problem.constraints
            },
            reference_hash=ref_or_reason,
            reason="converged" if admissible else "inadmissible min det(F)",
        )

    def optimize(self) -> DesignOptimizationReport:
        """Run P5 FEM-guarded trust-region 3D CAD shape optimization."""
        return run_fem_guarded_design_optimization(
            self.design_spec.problem,
            surrogate_evaluator=self.evaluate_surrogate,
            fem_evaluator=self.evaluate_fem,
            fd_step_fraction=self.design_spec.fd_fem_step_fraction,
        )


__all__ = (
    "SUPPORTED_3D_DESIGN_METRICS",
    "Parametric3DDesignBridge",
    "Parametric3DDesignSpec",
)
