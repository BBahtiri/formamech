"""AS6983 Conformal Control Barrier Function (CBF) Real-Time Safety Filter (V6-R4).

Translates FormaMech's one-sided AS6983 conservative shifted safety bounds
(peak stress, tissue strain, contact force, or joint angle limits) plus OOD
distance monitoring into a sub-millisecond Control Barrier Function safety
filter::

    h(x) = limit - y_upper_conformal(x) >= 0
    dh/dx . (f(x) + g(x) u) + gamma_cbf * h(x) >= 0

Governed by the explicit ``CONTROL_SAFETY_MACHINE`` finite state machine
(``NOMINAL -> CBF_MODIFIED | SAFE_HOLD_OOD``), guaranteeing fail-closed safe
retraction/holding without stalling real-time 100+ Hz control loops for online
FEM solves.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from piginot.contracts.canonical import canonical_json_bytes
from piginot.contracts.state_machine import (
    StateMachineSpec,
    TransitionSpec,
    register_state_machine,
)

FloatArray = NDArray[np.float64]
CBF_DECISION_SCHEMA: Final = "formamech.conformal-cbf-decision-v1"


def _sha256_doc(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


class ControlSafetyState(StrEnum):
    """Explicit lifecycle states for one real-time Conformal CBF safety tick."""

    EVALUATING = "EVALUATING"
    NOMINAL = "NOMINAL"
    CBF_MODIFIED = "CBF_MODIFIED"
    SAFE_HOLD_OOD = "SAFE_HOLD_OOD"


CONTROL_SAFETY_MACHINE: Final[StateMachineSpec[ControlSafetyState]] = (
    register_state_machine(
        StateMachineSpec(
            machine_id="formamech.robotics.conformal_cbf",
            description=(
                "Real-time AS6983 Conformal CBF safety filter lifecycle: "
                "EVALUATING -> NOMINAL | CBF_MODIFIED | SAFE_HOLD_OOD."
            ),
            states=tuple(ControlSafetyState),
            initial_states=(ControlSafetyState.EVALUATING,),
            terminal_states=(
                ControlSafetyState.NOMINAL,
                ControlSafetyState.CBF_MODIFIED,
                ControlSafetyState.SAFE_HOLD_OOD,
            ),
            transitions=(
                TransitionSpec(
                    ControlSafetyState.EVALUATING,
                    ControlSafetyState.NOMINAL,
                    "accept_nominal",
                    "Nominal control satisfies all conformal barrier constraints inside OOD envelope.",
                ),
                TransitionSpec(
                    ControlSafetyState.EVALUATING,
                    ControlSafetyState.CBF_MODIFIED,
                    "project_cbf",
                    "Nominal control projected minimally onto the certified conformal barrier half-space.",
                ),
                TransitionSpec(
                    ControlSafetyState.EVALUATING,
                    ControlSafetyState.SAFE_HOLD_OOD,
                    "trigger_safe_hold",
                    "OOD threshold exceeded or infeasible barrier; switch fail-closed to safe fallback action.",
                ),
            ),
        )
    )
)


@dataclass(frozen=True, slots=True)
class ConformalBarrierSpec:
    """One scalar safety constraint enforced using its AS6983 shifted upper bound."""

    name: str
    safety_limit: float
    conformal_shift_margin: float  # >= 0 from AS6983 DEV calibration
    gamma_cbf: float = 5.0  # discrete/continuous barrier decay rate (1/s)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ConformalBarrierSpec name must be non-empty")
        if not math.isfinite(self.safety_limit):
            raise ValueError("safety_limit must be finite")
        if (
            not math.isfinite(self.conformal_shift_margin)
            or self.conformal_shift_margin < 0.0
        ):
            raise ValueError("conformal_shift_margin must be finite and nonnegative")
        if not math.isfinite(self.gamma_cbf) or self.gamma_cbf <= 0.0:
            raise ValueError("gamma_cbf must be positive")


@dataclass(frozen=True, slots=True)
class CbfFilterResult:
    """Auditable real-time safety-filter decision with receipt hash."""

    state: ControlSafetyState
    u_nominal: FloatArray
    u_safe: FloatArray
    barrier_values: dict[str, float]
    min_barrier: float
    ood_distance: float
    modified: bool
    filter_time_ms: float
    reason: str

    def document(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "schema": CBF_DECISION_SCHEMA,
            "state": self.state.value,
            "u_nominal": [float(v) for v in self.u_nominal],
            "u_safe": [float(v) for v in self.u_safe],
            "barrier_values": {
                k: float(v) for k, v in sorted(self.barrier_values.items())
            },
            "min_barrier": float(self.min_barrier),
            "ood_distance": float(self.ood_distance),
            "modified": self.modified,
            "filter_time_ms": float(self.filter_time_ms),
            "reason": self.reason,
        }
        doc["receipt_sha256"] = _sha256_doc(doc)
        return doc


# Predicts (qoi_values dict, next_state(x, u), ood_distance)
SafetyEvaluatorFn = Callable[[FloatArray, FloatArray], tuple[dict[str, float], float]]


class ConformalCbfSafetyFilter:
    """Sub-millisecond AS6983 Conformal Control Barrier Function safety filter."""

    def __init__(
        self,
        barriers: Sequence[ConformalBarrierSpec],
        *,
        u_min: FloatArray,
        u_max: FloatArray,
        safe_hold_action: FloatArray,
        ood_threshold: float = 3.5,
        dt_s: float = 0.01,
    ) -> None:
        if not barriers:
            raise ValueError("at least one ConformalBarrierSpec is required")
        self.barriers = tuple(barriers)
        self.u_min = np.asarray(u_min, dtype=np.float64).reshape(-1)
        self.u_max = np.asarray(u_max, dtype=np.float64).reshape(-1)
        self.safe_hold = np.clip(
            np.asarray(safe_hold_action, dtype=np.float64).reshape(-1),
            self.u_min,
            self.u_max,
        )
        self.ood_threshold = float(ood_threshold)
        self.dt_s = float(dt_s)

    def filter_action(
        self,
        x_current: FloatArray,
        u_nominal: FloatArray,
        *,
        eval_current_qois: Callable[[FloatArray], tuple[dict[str, float], float]],
        predict_next_qois: SafetyEvaluatorFn,
    ) -> CbfFilterResult:
        """Filter ``u_nominal`` to enforce discrete-time Conformal CBF invariance.

        Discrete-time barrier condition per constraint ``b``:
            ``h_next(u) >= (1 - gamma_cbf * dt) * h_cur``
        with ``h(y) = limit - (y_pred + conformal_shift_margin)``.
        """
        t0 = time.perf_counter()
        u_nom = np.clip(
            np.asarray(u_nominal, dtype=np.float64).reshape(-1),
            self.u_min,
            self.u_max,
        )
        qois_cur, ood_cur = eval_current_qois(x_current)
        h_cur: dict[str, float] = {}
        for b in self.barriers:
            y_c = float(qois_cur[b.name])
            h_cur[b.name] = b.safety_limit - (y_c + b.conformal_shift_margin)

        if not math.isfinite(ood_cur) or ood_cur > self.ood_threshold:
            return CbfFilterResult(
                state=ControlSafetyState.SAFE_HOLD_OOD,
                u_nominal=u_nom,
                u_safe=self.safe_hold.copy(),
                barrier_values=h_cur,
                min_barrier=min(h_cur.values()),
                ood_distance=float(ood_cur),
                modified=True,
                filter_time_ms=(time.perf_counter() - t0) * 1000.0,
                reason=f"OOD distance {ood_cur:.3f} exceeds threshold {self.ood_threshold:.3f}",
            )

        qois_next, ood_next = predict_next_qois(x_current, u_nom)
        if not math.isfinite(ood_next) or ood_next > self.ood_threshold:
            return CbfFilterResult(
                state=ControlSafetyState.SAFE_HOLD_OOD,
                u_nominal=u_nom,
                u_safe=self.safe_hold.copy(),
                barrier_values=h_cur,
                min_barrier=min(h_cur.values()),
                ood_distance=float(ood_next),
                modified=True,
                filter_time_ms=(time.perf_counter() - t0) * 1000.0,
                reason=f"predicted next-step OOD distance {ood_next:.3f} exceeds threshold",
            )

        def _margins(q_map: dict[str, float]) -> dict[str, float]:
            out: dict[str, float] = {}
            for b in self.barriers:
                h_n = b.safety_limit - (float(q_map[b.name]) + b.conformal_shift_margin)
                decay = max(0.0, 1.0 - b.gamma_cbf * self.dt_s)
                out[b.name] = h_n - decay * h_cur[b.name]
            return out

        margins_nom = _margins(qois_next)
        if all(m >= 0.0 for m in margins_nom.values()):
            h_next_nom = {
                b.name: b.safety_limit
                - (float(qois_next[b.name]) + b.conformal_shift_margin)
                for b in self.barriers
            }
            return CbfFilterResult(
                state=ControlSafetyState.NOMINAL,
                u_nominal=u_nom,
                u_safe=u_nom.copy(),
                barrier_values=h_next_nom,
                min_barrier=min(h_next_nom.values()),
                ood_distance=float(max(ood_cur, ood_next)),
                modified=False,
                filter_time_ms=(time.perf_counter() - t0) * 1000.0,
                reason="nominal action satisfies all AS6983 conformal barriers",
            )

        # Iterative linearized half-space projection (Dykstra / alternating projection)
        m_dim = int(u_nom.size)
        u_cur = u_nom.copy()
        eps = 1.0e-3
        for _ in range(12):
            q_u, _ = predict_next_qois(x_current, u_cur)
            m_u = _margins(q_u)
            worst_name = min(m_u, key=lambda k: m_u[k])
            if m_u[worst_name] >= -1.0e-7:
                break
            # Compute sensitivity dm / du for active violated barrier
            grad_m = np.zeros(m_dim, dtype=np.float64)
            for j in range(m_dim):
                du = np.zeros(m_dim, dtype=np.float64)
                du[j] = eps
                qp, _ = predict_next_qois(
                    x_current, np.clip(u_cur + du, self.u_min, self.u_max)
                )
                qm, _ = predict_next_qois(
                    x_current, np.clip(u_cur - du, self.u_min, self.u_max)
                )
                grad_m[j] = (_margins(qp)[worst_name] - _margins(qm)[worst_name]) / (
                    2.0 * eps
                )
            gnorm2 = float(np.dot(grad_m, grad_m))
            if gnorm2 <= 1.0e-14:
                break
            # Exact minimum-norm projection onto m(u_cur) + grad_m . delta >= 0
            step = (-m_u[worst_name] + 1.0e-5) / gnorm2
            u_cur = np.clip(u_cur + step * grad_m, self.u_min, self.u_max)

        q_safe, ood_safe = predict_next_qois(x_current, u_cur)
        m_safe = _margins(q_safe)
        h_next_safe = {
            b.name: b.safety_limit - (float(q_safe[b.name]) + b.conformal_shift_margin)
            for b in self.barriers
        }
        if (
            all(mv >= -1.0e-4 for mv in m_safe.values())
            and ood_safe <= self.ood_threshold
        ):
            return CbfFilterResult(
                state=ControlSafetyState.CBF_MODIFIED,
                u_nominal=u_nom,
                u_safe=u_cur,
                barrier_values=h_next_safe,
                min_barrier=min(h_next_safe.values()),
                ood_distance=float(ood_safe),
                modified=True,
                filter_time_ms=(time.perf_counter() - t0) * 1000.0,
                reason="projected nominal action onto certified conformal CBF safe set",
            )

        # Fallback: blend toward safe_hold_action until feasible, else full safe_hold
        for lam in (0.25, 0.5, 0.75, 1.0):
            u_try = (1.0 - lam) * u_cur + lam * self.safe_hold
            q_try, ood_try = predict_next_qois(x_current, u_try)
            m_try = _margins(q_try)
            if (
                all(mv >= -1.0e-4 for mv in m_try.values())
                and ood_try <= self.ood_threshold
            ):
                h_try = {
                    b.name: b.safety_limit
                    - (float(q_try[b.name]) + b.conformal_shift_margin)
                    for b in self.barriers
                }
                return CbfFilterResult(
                    state=ControlSafetyState.CBF_MODIFIED,
                    u_nominal=u_nom,
                    u_safe=u_try,
                    barrier_values=h_try,
                    min_barrier=min(h_try.values()),
                    ood_distance=float(ood_try),
                    modified=True,
                    filter_time_ms=(time.perf_counter() - t0) * 1000.0,
                    reason=f"blended action ({lam:.2f}) toward safe hold to maintain CBF invariance",
                )

        return CbfFilterResult(
            state=ControlSafetyState.SAFE_HOLD_OOD,
            u_nominal=u_nom,
            u_safe=self.safe_hold.copy(),
            barrier_values=h_cur,
            min_barrier=min(h_cur.values()),
            ood_distance=float(ood_safe),
            modified=True,
            filter_time_ms=(time.perf_counter() - t0) * 1000.0,
            reason="barrier infeasible under box actuator bounds; engaged fail-closed safe hold",
        )


__all__ = (
    "CBF_DECISION_SCHEMA",
    "CONTROL_SAFETY_MACHINE",
    "CbfFilterResult",
    "ConformalBarrierSpec",
    "ConformalCbfSafetyFilter",
    "ControlSafetyState",
)
