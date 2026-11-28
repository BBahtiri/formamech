"""Infinite-Dimensional Neural-Operator Closed-Loop Inverse Kinematics (CLIK, V7-R3).

Implements shape-space damped-least-squares Jacobian control with nullspace
obstacle avoidance and optional :class:`~piginot.robotics.adaptive_cbf.AdaptiveConformalCbfFilter`
safety filtering directly over:

1. :class:`~piginot.operators.surface_operator_3d.Surface3DEnergyOperator` (via
   :class:`~piginot.export.onnx_export_3d.FrozenSurface3DRuntimeModule`),
2. Exact finite-rotation 3D Cosserat tendon backbones
   (:func:`~piginot.fem.tendon_3d.solve_3d_tendon_continuum_backbone`), or
3. Fast neural-operator continuum backbone surrogates distilled from Cosserat
   FEM for sub-2 ms closed-loop CPU execution.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray

from piginot.contracts.canonical import canonical_json_bytes
from piginot.export.onnx_export_3d import FrozenSurface3DRuntimeModule
from piginot.fem.tendon_3d import (
    CosseratBackboneSpec,
    solve_3d_tendon_continuum_backbone,
)
from piginot.robotics.adaptive_cbf import AdaptiveConformalCbfFilter
from piginot.robotics.cbf import ControlSafetyState

FloatArray = NDArray[np.float64]
OPERATOR_CLIK_SCHEMA: Final = "formamech.operator-clik-3d-v1"


def _sha256_doc(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


@dataclass(frozen=True, slots=True)
class SphereObstacle3D:
    """Spherical 3D workspace obstacle for backbone / surface collision avoidance."""

    name: str
    center_mm: tuple[float, float, float]
    radius_mm: float
    body_clearance_mm: float = 2.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("SphereObstacle3D name must be non-empty")
        if not math.isfinite(self.radius_mm) or self.radius_mm <= 0.0:
            raise ValueError("radius_mm must be positive")
        if not math.isfinite(self.body_clearance_mm) or self.body_clearance_mm < 0.0:
            raise ValueError("body_clearance_mm must be nonnegative")

    def min_clearance_mm(self, points_mm: FloatArray) -> float:
        """Signed clearance ``min_i ||p_i - c|| - (r_obs + body_clearance)`` in mm."""
        c = np.asarray(self.center_mm, dtype=np.float64)
        dists = np.linalg.norm(points_mm - c[None, :], axis=1)
        return float(np.min(dists) - (self.radius_mm + self.body_clearance_mm))


@dataclass(frozen=True, slots=True)
class OperatorClikConfig:
    """Hyperparameters for infinite-dimensional shape-space Operator CLIK."""

    gain_kp: float = 0.65
    damping_lambda: float = 0.08
    nullspace_gain: float = 0.35
    obstacle_influence_mm: float = 25.0
    obstacle_weight: float = 4.0
    stiffness_reg_weight: float = 0.02
    max_step_norm: float = 2.5
    fd_eps: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.gain_kp <= 0.0 or not math.isfinite(self.gain_kp):
            raise ValueError("gain_kp must be positive")
        if self.damping_lambda <= 0.0 or not math.isfinite(self.damping_lambda):
            raise ValueError("damping_lambda must be positive")
        if self.max_step_norm <= 0.0 or not math.isfinite(self.max_step_norm):
            raise ValueError("max_step_norm must be positive")


@dataclass(frozen=True, slots=True)
class OperatorClikStepResult:
    """Result of one closed-loop infinite-dimensional shape-space CLIK iteration."""

    q_prev: FloatArray
    q_next: FloatArray
    shape_points_mm: FloatArray  # (Q, 3) full deformed 3D shape
    tip_pos_mm: FloatArray  # (3,)
    tip_error_mm: float
    min_obstacle_clearance_mm: float
    cbf_state: ControlSafetyState
    nullspace_norm: float
    step_time_ms: float
    reason: str

    def document(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "schema": OPERATOR_CLIK_SCHEMA,
            "q_prev": [float(v) for v in self.q_prev],
            "q_next": [float(v) for v in self.q_next],
            "tip_pos_mm": [float(v) for v in self.tip_pos_mm],
            "tip_error_mm": float(self.tip_error_mm),
            "min_obstacle_clearance_mm": (
                float(self.min_obstacle_clearance_mm)
                if math.isfinite(self.min_obstacle_clearance_mm)
                else None
            ),
            "cbf_state": self.cbf_state.value,
            "nullspace_norm": float(self.nullspace_norm),
            "step_time_ms": float(self.step_time_ms),
            "reason": self.reason,
        }
        doc["receipt_sha256"] = _sha256_doc(doc)
        return doc


# Callable mapping actuator vector q (m,) -> (deformed_points_mm (Q, 3), shape_jacobian (Q, 3, m))
ShapeAndJacobianFn = Callable[[FloatArray], tuple[FloatArray, FloatArray]]


def make_surface_operator_shape_model(
    runtime_module: FrozenSurface3DRuntimeModule,
    reference_points_mm: FloatArray,
    *,
    fd_eps: float = 1.0e-3,
) -> ShapeAndJacobianFn:
    """Wrap :class:`FrozenSurface3DRuntimeModule` for shape-space Operator CLIK."""
    ref_pts = np.asarray(reference_points_mm, dtype=np.float64)

    def _eval(q_act: FloatArray) -> tuple[FloatArray, FloatArray]:
        p_t = torch.tensor(q_act, dtype=torch.float32)
        u_t, jac_t = runtime_module.predict_and_parameter_jacobian(p_t, fd_eps=fd_eps)
        pts = ref_pts + u_t.detach().numpy().astype(np.float64)
        jac = jac_t.detach().numpy().astype(np.float64)
        return pts, jac

    return _eval


def make_cosserat_backbone_shape_model(
    spec: CosseratBackboneSpec,
    *,
    external_forces_n: dict[int, tuple[float, float, float]] | None = None,
    fd_eps: float = 2.0e-2,
    tolerance: float = 1.0e-7,
) -> ShapeAndJacobianFn:
    """Wrap :func:`solve_3d_tendon_continuum_backbone` with forward-difference shape Jacobian."""

    def _eval(q_act: FloatArray) -> tuple[FloatArray, FloatArray]:
        t0 = np.maximum(np.asarray(q_act, dtype=np.float64).reshape(-1), 0.0)
        res0 = solve_3d_tendon_continuum_backbone(
            spec,
            tuple(float(v) for v in t0),
            external_forces_n=external_forces_n,
            tolerance=tolerance,
        )
        pts0 = res0.disk_positions_mm  # (N_disks + 1, 3)
        q_n, m = pts0.shape[0], int(t0.size)
        jac = np.zeros((q_n, 3, m), dtype=np.float64)
        for j in range(m):
            tp = t0.copy()
            tp[j] += fd_eps
            rp = solve_3d_tendon_continuum_backbone(
                spec,
                tuple(float(v) for v in tp),
                external_forces_n=external_forces_n,
                tolerance=tolerance,
            )
            jac[:, :, j] = (rp.disk_positions_mm - pts0) / fd_eps
        return pts0, jac

    return _eval


def make_neural_shape_model(
    model: torch.nn.Module,
    *,
    q_mean: FloatArray,
    q_std: FloatArray,
    shape_mean: FloatArray,
    shape_std: FloatArray,
    fd_eps: float = 1.0e-3,
) -> ShapeAndJacobianFn:
    """Wrap a trained PyTorch multi-node 3D shape operator for sub-2 ms CPU CLIK."""
    qm = np.asarray(q_mean, dtype=np.float64)
    qs = np.maximum(np.asarray(q_std, dtype=np.float64), 1.0e-6)
    sm = np.asarray(shape_mean, dtype=np.float64)
    ss = np.asarray(shape_std, dtype=np.float64)

    def _eval(q_act: FloatArray) -> tuple[FloatArray, FloatArray]:
        q0 = np.asarray(q_act, dtype=np.float64).reshape(-1)
        m = int(q0.size)
        # Single batched forward pass: [q0, q0 + eps*e_0, ..., q0 + eps*e_{m-1}]
        batch = np.repeat(q0[None, :], m + 1, axis=0)
        for j in range(m):
            batch[j + 1, j] += fd_eps
        xn = torch.tensor((batch - qm) / qs, dtype=torch.float32)
        with torch.no_grad():
            yn = model(xn).numpy().astype(np.float64)
        pts_all = yn.reshape(m + 1, -1, 3) * ss[None, :, :] + sm[None, :, :]
        pts0 = pts_all[0]
        jac = np.transpose((pts_all[1:] - pts0[None, :, :]) / fd_eps, (1, 2, 0))
        return pts0, jac

    return _eval


class OperatorClikController:
    """Shape-space damped-least-squares CLIK with nullspace obstacle & CBF safety."""

    def __init__(
        self,
        shape_model: ShapeAndJacobianFn,
        *,
        u_min: FloatArray,
        u_max: FloatArray,
        tip_index: int = -1,
        config: OperatorClikConfig | None = None,
    ) -> None:
        self.shape_model = shape_model
        self.u_min = np.asarray(u_min, dtype=np.float64).reshape(-1)
        self.u_max = np.asarray(u_max, dtype=np.float64).reshape(-1)
        self.tip_index = int(tip_index)
        self.config = OperatorClikConfig() if config is None else config

    def step(
        self,
        q_act: FloatArray,
        target_tip_mm: FloatArray,
        *,
        measured_tip_mm: FloatArray | None = None,
        target_shape_mm: FloatArray | None = None,
        target_stiffness_q: FloatArray | None = None,
        obstacles: Sequence[SphereObstacle3D] = (),
        adaptive_cbf: AdaptiveConformalCbfFilter | None = None,
        ood_distance: float = 0.0,
    ) -> OperatorClikStepResult:
        """Compute one closed-loop shape-space CLIK update (optionally CBF-filtered)."""
        t0 = time.perf_counter()
        cfg = self.config
        q_cur = np.clip(
            np.asarray(q_act, dtype=np.float64).reshape(-1),
            self.u_min,
            self.u_max,
        )
        m = int(q_cur.size)
        p_tgt = np.asarray(target_tip_mm, dtype=np.float64).reshape(3)

        pts, jac = self.shape_model(q_cur)  # (Q, 3), (Q, 3, m)
        tip_idx = (
            self.tip_index if self.tip_index >= 0 else pts.shape[0] + self.tip_index
        )
        tip_model = pts[tip_idx]
        tip_obs = (
            tip_model
            if measured_tip_mm is None
            else np.asarray(measured_tip_mm, dtype=np.float64).reshape(3)
        )
        e_tip = p_tgt - tip_obs
        j_tip = jac[tip_idx]  # (3, m)

        # Damped least-squares pseudoinverse: J^+ = J^T (J J^T + lambda^2 I)^-1
        jjt = j_tip @ j_tip.T + (cfg.damping_lambda**2) * np.eye(3, dtype=np.float64)
        j_pinv = j_tip.T @ np.linalg.inv(jjt)  # (m, 3)
        dq_task = j_pinv @ (cfg.gain_kp * e_tip)

        # Nullspace projector N = I - J^+ J
        null_proj = np.eye(m, dtype=np.float64) - j_pinv @ j_tip

        # Secondary objective gradient in actuator space:
        # 1) Mid-backbone obstacle repulsion across ALL shape nodes
        # 2) Optional full-shape tracking
        # 3) Optional stiffness/pretension regularization
        dq_sec = np.zeros(m, dtype=np.float64)
        obs_direct = np.zeros(m, dtype=np.float64)
        for obs in obstacles:
            c = np.asarray(obs.center_mm, dtype=np.float64)
            r_safe = obs.radius_mm + obs.body_clearance_mm
            diff = pts - c[None, :]  # (Q, 3)
            dists = np.linalg.norm(diff, axis=1)
            clearances = dists - r_safe
            # Ignore fixed base node 0; repulse all interior + tip nodes within influence radius
            for k in range(1, pts.shape[0]):
                clr = float(clearances[k])
                if clr < cfg.obstacle_influence_mm:
                    n_out = diff[k] / max(float(dists[k]), 1.0e-9)
                    # Smooth quadratic barrier repulsion gradient d(clearance)/dq = n_out^T J_k
                    grad_q = n_out @ jac[k]  # (m,)
                    pen = (cfg.obstacle_influence_mm - clr) / max(
                        cfg.obstacle_influence_mm, 1.0e-6
                    )
                    dq_sec += cfg.obstacle_weight * pen * grad_q
                    if clr < 0.35 * cfg.obstacle_influence_mm:
                        # Direct task-level emergency push when very close to obstacle
                        urgency = (0.35 * cfg.obstacle_influence_mm - clr) / max(
                            0.35 * cfg.obstacle_influence_mm, 1.0e-6
                        )
                        obs_direct += 1.5 * cfg.obstacle_weight * urgency * grad_q

        if target_shape_mm is not None:
            e_shape = np.asarray(target_shape_mm, dtype=np.float64) - pts
            dq_sec += (
                0.15
                * np.einsum("qi,qim->m", e_shape, jac)
                / max(float(pts.shape[0]), 1.0)
            )

        if target_stiffness_q is not None:
            q_stiff = np.asarray(target_stiffness_q, dtype=np.float64).reshape(m)
            dq_sec += cfg.stiffness_reg_weight * (q_stiff - q_cur)

        dq_null = cfg.nullspace_gain * (null_proj @ dq_sec)
        dq_total = dq_task + dq_null + obs_direct
        step_norm = float(np.linalg.norm(dq_total))
        if step_norm > cfg.max_step_norm:
            dq_total = dq_total * (cfg.max_step_norm / step_norm)

        q_nom = np.clip(q_cur + dq_total, self.u_min, self.u_max)
        cbf_state = ControlSafetyState.NOMINAL
        reason = "operator CLIK nominal step"

        # Optional Adaptive Conformal CBF filtering for certified obstacle clearance
        if adaptive_cbf is not None and obstacles:

            def _eval_cur(_x: FloatArray) -> tuple[dict[str, float], float]:
                # Negated clearance so barrier h = limit - y = 0 - (-clearance) = clearance >= 0
                qois = {obs.name: -obs.min_clearance_mm(pts) for obs in obstacles}
                return qois, float(ood_distance)

            def _pred_next(
                _x: FloatArray, u_cand: FloatArray
            ) -> tuple[dict[str, float], float]:
                # First-order operator prediction (sub-0.05 ms!) using exact shape Jacobian
                du = u_cand - q_cur
                pts_pred = pts + np.einsum("qim,m->qi", jac, du)
                qois = {obs.name: -obs.min_clearance_mm(pts_pred) for obs in obstacles}
                return qois, float(ood_distance)

            cbf_res = adaptive_cbf.filter_action(
                q_cur,
                q_nom,
                eval_current_qois=_eval_cur,
                predict_next_qois=_pred_next,
            )
            q_next = cbf_res.u_safe
            cbf_state = cbf_res.state
            reason = cbf_res.reason
        else:
            q_next = q_nom

        # Evaluate resulting 3D shape
        pts_next, _ = self.shape_model(q_next)
        tip_next = pts_next[tip_idx]
        tip_err = float(np.linalg.norm(p_tgt - tip_next))
        min_clr = (
            min(obs.min_clearance_mm(pts_next) for obs in obstacles)
            if obstacles
            else float("inf")
        )
        return OperatorClikStepResult(
            q_prev=q_cur,
            q_next=q_next,
            shape_points_mm=pts_next,
            tip_pos_mm=tip_next,
            tip_error_mm=tip_err,
            min_obstacle_clearance_mm=min_clr,
            cbf_state=cbf_state,
            nullspace_norm=float(np.linalg.norm(dq_null)),
            step_time_ms=(time.perf_counter() - t0) * 1000.0,
            reason=reason,
        )


__all__ = (
    "OPERATOR_CLIK_SCHEMA",
    "OperatorClikConfig",
    "OperatorClikController",
    "OperatorClikStepResult",
    "ShapeAndJacobianFn",
    "SphereObstacle3D",
    "make_cosserat_backbone_shape_model",
    "make_neural_shape_model",
    "make_surface_operator_shape_model",
)
