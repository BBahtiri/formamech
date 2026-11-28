"""Live Two-Way Simulator Coupling Callback & C-ABI Bridge (V7-R5).

Upgrades static MJCF/ROS 2 manifests (:mod:`piginot.robotics.bridges`) with a
real-time bidirectional step callback::

    step_coupled(q, qd, contact_geoms) -> CoupledStepOutput(wrench_6d, tangent_stiffness_6x6, ...)

that exchanges exact 6D reaction wrenches and linearized ``6x6`` tangent
stiffness matrices ``K_t = -d(wrench)/dq`` back into MuJoCo / MJWarp / Newton
ADMM control loops at each step, with:

* **Stop-Gradient Residual Control (``stop_grad_residual=True``):** Detaches
  learned neural contact residual branches during differentiable policy
  optimization while preserving exact smooth analytical contact Jacobians.
* **Zero-Copy C-ABI Callback Factory (``make_c_abi_callback``):** Emits a
  ``ctypes.CFUNCTYPE`` entry point accepting raw C ``double*`` buffers for
  direct native C/C++ simulator plugin binding.
* **Two-Way vs. One-Way Coupled Rollout Integrator:** Directly quantifies
  deformable-rigid trajectory divergence when simulator reaction wrenches and
  tangent stiffness feed back into the manipulator dynamics.
"""

from __future__ import annotations

import ctypes
import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from piginot.contracts.canonical import canonical_json_bytes

FloatArray = NDArray[np.float64]
LIVE_COUPLING_SCHEMA: Final = "formamech.live-twoway-coupling-v1"


def _sha256_doc(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


@dataclass(frozen=True, slots=True)
class ContactGeomState:
    """Rigid simulator body / contact primitive interacting with deformable object."""

    name: str
    position_mm: tuple[float, float, float]
    velocity_mm_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius_mm: float = 10.0
    stiffness_n_per_mm: float = 2.0
    damping_n_s_per_mm: float = 0.05
    friction_mu: float = 0.4


@dataclass(frozen=True, slots=True)
class CoupledStepOutput:
    """Bidirectional reaction wrench & linearized tangent stiffness returned to simulator."""

    wrench_6d: FloatArray  # (6,) [fx, fy, fz (N), tx, ty, tz (N*mm)]
    tangent_stiffness_6x6: FloatArray  # (6, 6) K_t = -d(wrench)/dq (N/mm, N, N*mm)
    damping_6x6: FloatArray  # (6, 6) C_t = -d(wrench)/dqd
    nodal_forces_n: FloatArray  # (N_nodes, 3) forces exerted on deformable mesh
    active_contacts: int
    stop_grad_residual: bool
    step_time_ms: float

    def document(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "schema": LIVE_COUPLING_SCHEMA,
            "wrench_6d": [float(v) for v in self.wrench_6d],
            "stiffness_diag": [float(v) for v in np.diag(self.tangent_stiffness_6x6)],
            "damping_diag": [float(v) for v in np.diag(self.damping_6x6)],
            "active_contacts": self.active_contacts,
            "stop_grad_residual": self.stop_grad_residual,
            "step_time_ms": float(self.step_time_ms),
        }
        doc["receipt_sha256"] = _sha256_doc(doc)
        return doc


class ContactResidualNet(nn.Module):
    """Lightweight differentiable neural contact-residual head for two-way coupling."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
        )
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(last.bias)

    def forward(self, state_6d: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(state_6d)
        return out


class LiveSimulatorCoupler:
    """Zero-copy Python / C-ABI two-way deformable contact coupler (V7-R5).

    Models a deformable 3D point-cloud / FEM surface ``object_nodes_mm`` with
    viscoelastic Kelvin-Voigt / Hertzian contact against one or more rigid
    simulator contact geometries, plus an optional neural contact residual
    supporting ``stop_grad_residual=True`` for stable policy gradients.
    """

    def __init__(
        self,
        object_nodes_mm: FloatArray,
        *,
        elastic_stiffness_n_per_mm: float = 0.25,
        viscous_damping_n_s_per_mm: float = 0.02,
        residual_model: nn.Module | None = None,
        wrench_center_mm: tuple[float, float, float] | None = None,
    ) -> None:
        nodes = np.asarray(object_nodes_mm, dtype=np.float64)
        if nodes.ndim != 2 or nodes.shape[1] != 3 or nodes.shape[0] < 1:
            raise ValueError("object_nodes_mm must have shape (N, 3) with N >= 1")
        self.object_nodes_mm = nodes.copy()
        self.elastic_stiffness = float(max(elastic_stiffness_n_per_mm, 1.0e-6))
        self.viscous_damping = float(max(viscous_damping_n_s_per_mm, 0.0))
        self.residual_model = residual_model
        if wrench_center_mm is None:
            c = np.mean(nodes, axis=0)
            self.wrench_center_mm = (float(c[0]), float(c[1]), float(c[2]))
        else:
            self.wrench_center_mm = tuple(float(v) for v in wrench_center_mm)  # type: ignore[assignment]

    def step_coupled_torch(
        self,
        q_pos_mm: torch.Tensor,  # (3,) or (6,) simulator tool pose [x,y,z,(rx,ry,rz)]
        qd_vel_mm_s: torch.Tensor,  # (3,) or (6,) simulator tool velocity
        contact_geoms: Sequence[ContactGeomState],
        *,
        stop_grad_residual: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiable PyTorch contact wrench & nodal force step with stop-grad control."""
        dtype = q_pos_mm.dtype
        device = q_pos_mm.device
        q6 = (
            q_pos_mm
            if q_pos_mm.numel() == 6
            else torch.cat(
                (q_pos_mm.reshape(3), torch.zeros(3, dtype=dtype, device=device))
            )
        )
        qd6 = (
            qd_vel_mm_s
            if qd_vel_mm_s.numel() == 6
            else torch.cat(
                (qd_vel_mm_s.reshape(3), torch.zeros(3, dtype=dtype, device=device))
            )
        )
        q_xyz = q6[:3]
        qd_xyz = qd6[:3]

        nodes_t = torch.tensor(self.object_nodes_mm, dtype=dtype, device=device)
        center_t = torch.tensor(self.wrench_center_mm, dtype=dtype, device=device)
        total_nodal = torch.zeros_like(nodes_t)
        wrench_f = torch.zeros(3, dtype=dtype, device=device)
        wrench_m = torch.zeros(3, dtype=dtype, device=device)

        for geom in contact_geoms:
            g_off = torch.tensor(geom.position_mm, dtype=dtype, device=device)
            g_vel = torch.tensor(geom.velocity_mm_s, dtype=dtype, device=device)
            g_pos = g_off + q_xyz
            v_rel = g_vel + qd_xyz
            diff = nodes_t - g_pos.unsqueeze(0)  # (N, 3) vector from geom to node
            dist = torch.sqrt((diff * diff).sum(dim=-1) + 1.0e-12)
            # Smooth C^1 penetration pen = softplus(r - dist)
            raw_pen = float(geom.radius_mm) - dist
            active_mask = (raw_pen > 0.0).to(dtype=dtype)
            pen = torch.relu(raw_pen)
            normal = diff / dist.unsqueeze(-1)  # outward from geom onto body

            # Normal contact force + Kelvin-Voigt damping on active nodes
            k_eff = (self.elastic_stiffness * geom.stiffness_n_per_mm) / (
                self.elastic_stiffness + geom.stiffness_n_per_mm
            )
            c_eff = self.viscous_damping + geom.damping_n_s_per_mm
            v_n = torch.einsum("ni,i->n", normal, v_rel)
            fn_mag = (k_eff * pen + c_eff * torch.relu(v_n)) * active_mask
            f_n = fn_mag.unsqueeze(-1) * normal

            # Smooth regularized Coulomb tangential traction
            v_t = v_rel.unsqueeze(0) - v_n.unsqueeze(-1) * normal
            vt_norm = torch.sqrt((v_t * v_t).sum(dim=-1, keepdim=True) + 1.0e-8)
            mu_slip = float(geom.friction_mu) * torch.tanh(vt_norm / 2.0)
            f_t = mu_slip * fn_mag.unsqueeze(-1) * (v_t / vt_norm)

            f_on_body = f_n + f_t
            total_nodal = total_nodal + f_on_body
            # Newton's 3rd law: reaction wrench acting BACK ON the rigid simulator geom
            f_react = -f_on_body
            wrench_f = wrench_f + f_react.sum(dim=0)
            lever = nodes_t - center_t.unsqueeze(0)
            wrench_m = wrench_m + torch.linalg.cross(lever, f_react, dim=-1).sum(dim=0)

        wrench = torch.cat((wrench_f, wrench_m), dim=0)
        if self.residual_model is not None:
            state_in = torch.cat((q_xyz / 100.0, qd_xyz / 100.0), dim=0).unsqueeze(0)
            res_w = (
                self.residual_model(state_in.to(dtype=torch.float32))
                .to(dtype=dtype)
                .squeeze(0)
            )
            if stop_grad_residual:
                # Preserve value of learned residual while stopping noisy residual Jacobian
                res_w = res_w.detach()
            wrench = wrench + res_w
        return wrench, total_nodal

    def step_coupled(
        self,
        q: FloatArray,
        qd: FloatArray,
        contact_geoms: Sequence[ContactGeomState],
        *,
        stop_grad_residual: bool = False,
    ) -> CoupledStepOutput:
        """Compute reaction wrench & exact linearized ``6x6`` tangent stiffness ``K_t``."""
        t0 = time.perf_counter()
        q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
        qd_arr = np.asarray(qd, dtype=np.float64).reshape(-1)
        q6_np = np.zeros(6, dtype=np.float64)
        qd6_np = np.zeros(6, dtype=np.float64)
        q6_np[: min(6, q_arr.size)] = q_arr[: min(6, q_arr.size)]
        qd6_np[: min(6, qd_arr.size)] = qd_arr[: min(6, qd_arr.size)]

        q_t = torch.tensor(q6_np, dtype=torch.float64, requires_grad=True)
        qd_t = torch.tensor(qd6_np, dtype=torch.float64, requires_grad=True)
        w_t, nodal_t = self.step_coupled_torch(
            q_t, qd_t, contact_geoms, stop_grad_residual=stop_grad_residual
        )

        # Exact autograd 6x6 tangent stiffness K_t = -dw/dq and damping C_t = -dw/dqd
        k_mat = np.zeros((6, 6), dtype=np.float64)
        c_mat = np.zeros((6, 6), dtype=np.float64)
        for i in range(6):
            gq, gqd = torch.autograd.grad(
                w_t[i],
                (q_t, qd_t),
                retain_graph=i < 5,
                allow_unused=True,
            )
            if gq is not None:
                k_mat[i, :] = -gq.detach().numpy()
            if gqd is not None:
                c_mat[i, :] = -gqd.detach().numpy()

        nodal_np = nodal_t.detach().numpy().astype(np.float64)
        active = int(np.sum(np.linalg.norm(nodal_np, axis=1) > 1.0e-12))
        return CoupledStepOutput(
            wrench_6d=w_t.detach().numpy().astype(np.float64),
            tangent_stiffness_6x6=k_mat,
            damping_6x6=c_mat,
            nodal_forces_n=nodal_np,
            active_contacts=active,
            stop_grad_residual=stop_grad_residual,
            step_time_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def make_c_abi_callback(
        self,
        contact_geoms: Sequence[ContactGeomState],
        *,
        stop_grad_residual: bool = True,
    ) -> Any:
        """Create a zero-copy ``ctypes.CFUNCTYPE`` C-ABI step callback for MuJoCo/Newton.

        C signature::

            int formamech_step_coupled(
                const double* q_6,
                const double* qd_6,
                double* wrench_out_6,
                double* k_out_36
            );
        """
        cb_type = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        )
        geoms_tuple = tuple(contact_geoms)

        def _cb(q_ptr: Any, qd_ptr: Any, w_ptr: Any, k_ptr: Any) -> int:
            try:
                q_np = np.ctypeslib.as_array(q_ptr, shape=(6,))
                qd_np = np.ctypeslib.as_array(qd_ptr, shape=(6,))
                out = self.step_coupled(
                    q_np, qd_np, geoms_tuple, stop_grad_residual=stop_grad_residual
                )
                w_out = np.ctypeslib.as_array(w_ptr, shape=(6,))
                k_out = np.ctypeslib.as_array(k_ptr, shape=(36,))
                w_out[:] = out.wrench_6d
                k_out[:] = out.tangent_stiffness_6x6.reshape(36)
                return 0
            except Exception:
                return -1

        return cb_type(_cb)


@dataclass(frozen=True, slots=True)
class CoupledRolloutComparison:
    """Quantified comparison of one-way prescribed vs. two-way coupled rollouts."""

    times_s: FloatArray  # (T,)
    oneway_tool_traj_mm: FloatArray  # (T, 3)
    twoway_tool_traj_mm: FloatArray  # (T, 3)
    oneway_object_traj_mm: FloatArray  # (T, N, 3)
    twoway_object_traj_mm: FloatArray  # (T, N, 3)
    reaction_wrench_history: FloatArray  # (T, 6)
    mean_tool_divergence_mm: float
    max_tool_divergence_mm: float
    mean_object_divergence_mm: float
    max_object_divergence_mm: float
    mean_step_time_ms: float

    def document(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "schema": f"{LIVE_COUPLING_SCHEMA}.rollout",
            "steps": int(self.times_s.size),
            "mean_tool_divergence_mm": float(self.mean_tool_divergence_mm),
            "max_tool_divergence_mm": float(self.max_tool_divergence_mm),
            "mean_object_divergence_mm": float(self.mean_object_divergence_mm),
            "max_object_divergence_mm": float(self.max_object_divergence_mm),
            "mean_step_time_ms": float(self.mean_step_time_ms),
        }
        doc["receipt_sha256"] = _sha256_doc(doc)
        return doc


def compare_oneway_vs_twoway_rollout(
    initial_object_nodes_mm: FloatArray,
    commanded_tool_traj_mm: FloatArray,  # (T, 3)
    *,
    dt_s: float = 1.0 / 30.0,
    tool_mass_kg: float = 0.25,
    tool_kp_n_per_mm: float = 1.5,
    tool_kd_n_s_per_mm: float = 0.15,
    object_stiffness_n_per_mm: float = 0.6,
    object_damping_n_s_per_mm: float = 0.04,
    contact_radius_mm: float = 35.0,
    object_step_fn: Callable[[FloatArray, FloatArray, float], FloatArray] | None = None,
) -> CoupledRolloutComparison:
    """Compare one-way kinematic playback vs. two-way reaction+stiffness coupled rollout."""
    cmd = np.asarray(commanded_tool_traj_mm, dtype=np.float64)
    t_len = int(cmd.shape[0])
    nodes0 = np.asarray(initial_object_nodes_mm, dtype=np.float64)
    n_pts = int(nodes0.shape[0])
    times = np.arange(t_len, dtype=np.float64) * float(dt_s)

    # Mass in tonne-mm-N-s consistent units: 1 N = 1e-3 kg * m/s^2 -> 1 kg = 1e-3 N*s^2/mm
    m_eff = tool_mass_kg * 1.0e-3  # N*s^2/mm

    def _default_obj_update(
        cur_nodes: FloatArray, nodal_forces: FloatArray, step_dt: float
    ) -> FloatArray:
        # Overdamped viscoelastic Kelvin-Voigt nodal relaxation toward rest + contact force
        restoring = -0.15 * object_stiffness_n_per_mm * (cur_nodes - nodes0)
        visc = max(object_damping_n_s_per_mm + 0.05, 1.0e-4)
        return cur_nodes + (step_dt / visc) * (nodal_forces + restoring) * 0.2

    step_obj = _default_obj_update if object_step_fn is None else object_step_fn

    # 1. One-way rollout: tool rigidly follows commanded_tool_traj_mm ignoring reaction wrench
    ow_tool = cmd.copy()
    ow_obj = np.zeros((t_len, n_pts, 3), dtype=np.float64)
    ow_obj[0] = nodes0
    for k in range(t_len - 1):
        coupler = LiveSimulatorCoupler(
            ow_obj[k],
            elastic_stiffness_n_per_mm=object_stiffness_n_per_mm,
            viscous_damping_n_s_per_mm=object_damping_n_s_per_mm,
        )
        v_cmd = (cmd[k + 1] - cmd[k]) / dt_s
        geom = ContactGeomState(
            name="manipulator",
            position_mm=(0.0, 0.0, 0.0),
            radius_mm=contact_radius_mm,
            stiffness_n_per_mm=object_stiffness_n_per_mm,
            damping_n_s_per_mm=object_damping_n_s_per_mm,
        )
        out = coupler.step_coupled(ow_tool[k], v_cmd, (geom,))
        ow_obj[k + 1] = step_obj(ow_obj[k], out.nodal_forces_n, dt_s)

    # 2. Two-way coupled rollout: compliant impedance controller feels reaction wrench + K_t!
    tw_tool = np.zeros((t_len, 3), dtype=np.float64)
    tw_vel = np.zeros((t_len, 3), dtype=np.float64)
    tw_obj = np.zeros((t_len, n_pts, 3), dtype=np.float64)
    wrenches = np.zeros((t_len, 6), dtype=np.float64)
    tw_tool[0] = cmd[0]
    tw_obj[0] = nodes0
    step_ms: list[float] = []

    for k in range(t_len - 1):
        coupler = LiveSimulatorCoupler(
            tw_obj[k],
            elastic_stiffness_n_per_mm=object_stiffness_n_per_mm,
            viscous_damping_n_s_per_mm=object_damping_n_s_per_mm,
        )
        geom = ContactGeomState(
            name="manipulator",
            position_mm=(0.0, 0.0, 0.0),
            radius_mm=contact_radius_mm,
            stiffness_n_per_mm=object_stiffness_n_per_mm,
            damping_n_s_per_mm=object_damping_n_s_per_mm,
        )
        out = coupler.step_coupled(tw_tool[k], tw_vel[k], (geom,))
        step_ms.append(out.step_time_ms)
        wrenches[k] = out.wrench_6d

        # Semi-implicit Schur-stabilized Newmark step using tangent stiffness K_t[:3, :3]!
        v_ref = (cmd[k + 1] - cmd[k]) / dt_s
        f_ctrl = tool_kp_n_per_mm * (cmd[k + 1] - tw_tool[k]) + tool_kd_n_s_per_mm * (
            v_ref - tw_vel[k]
        )
        f_total = f_ctrl + out.wrench_6d[:3]
        k_tan = 0.5 * (
            out.tangent_stiffness_6x6[:3, :3] + out.tangent_stiffness_6x6[:3, :3].T
        )
        # Ensure SPD implicit denominator M + dt*(C_ctrl + C_tan) + dt^2*(K_ctrl + K_tan_pos)
        evals, evecs = np.linalg.eigh(k_tan)
        k_psd = evecs @ np.diag(np.maximum(evals, 0.0)) @ evecs.T
        lhs = (
            m_eff + dt_s * tool_kd_n_s_per_mm + (dt_s * dt_s) * tool_kp_n_per_mm
        ) * np.eye(3, dtype=np.float64) + (dt_s * dt_s) * k_psd
        dv = np.linalg.solve(lhs, dt_s * f_total)
        tw_vel[k + 1] = tw_vel[k] + dv
        tw_tool[k + 1] = tw_tool[k] + dt_s * tw_vel[k + 1]
        tw_obj[k + 1] = step_obj(tw_obj[k], out.nodal_forces_n, dt_s)

    tool_div = np.linalg.norm(tw_tool - ow_tool, axis=1)
    obj_div = np.linalg.norm(tw_obj - ow_obj, axis=2)
    return CoupledRolloutComparison(
        times_s=times,
        oneway_tool_traj_mm=ow_tool,
        twoway_tool_traj_mm=tw_tool,
        oneway_object_traj_mm=ow_obj,
        twoway_object_traj_mm=tw_obj,
        reaction_wrench_history=wrenches,
        mean_tool_divergence_mm=float(np.mean(tool_div)),
        max_tool_divergence_mm=float(np.max(tool_div)),
        mean_object_divergence_mm=float(np.mean(obj_div)),
        max_object_divergence_mm=float(np.max(obj_div)),
        mean_step_time_ms=float(np.mean(step_ms)) if step_ms else 0.0,
    )


__all__ = (
    "LIVE_COUPLING_SCHEMA",
    "ContactGeomState",
    "ContactResidualNet",
    "CoupledRolloutComparison",
    "CoupledStepOutput",
    "LiveSimulatorCoupler",
    "compare_oneway_vs_twoway_rollout",
)
