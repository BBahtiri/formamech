"""History-aware neural transition operator for J2 elastoplasticity (M3 / F2).

Learns the incremental displacement field ``u_{n+1}(x)`` conditioned on:

* normalized spatial coordinates ``x``,
* the next applied load vector,
* global area-weighted plastic internal state summary, and
* point-local physical internal state features ``(eps_p11, eps_p22, eps_p12,
  eps_p33, alpha_eq)`` interpolated from cell-owned quadrature/cell states.

Training minimizes the exact **Ortiz-Steinmann incremental variational
pseudo-potential** of :class:`~piginot.stateful.j2_plasticity.J2IsotropicPlasticity`
with exact Dirichlet lifting (label-free by default; optionally augmented with
FEM displacement supervision with explicit ``multi-fidelity-label-efficient``
disclosure). Authoritative plastic internal variables always live at physical
material points and are updated by the exact return-mapping algorithm.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, cast

import torch
from torch import nn

from piginot.lifting.base import AffineDirichletLifting, DirichletLifting
from piginot.objectives.energy import TriangleDomain, stratified_triangle_quadrature
from piginot.operators.energy_operator import QueryFrame
from piginot.stateful.j2_plasticity import J2IsotropicPlasticity
from piginot.stateful.viscoelastic_operator import interpolate_nodal_features_in_cells

PLASTICITY_OPERATOR_ARCH_ID: Final = "pi-ginot-j2-plasticity-transition-v1"
PlasticityKinematics = Literal["plane_strain", "plane_stress"]


@dataclass(frozen=True, slots=True)
class PlasticityOperatorConfig:
    """Architecture configuration for the M3 history-aware plasticity operator."""

    load_dim: int
    kinematics_mode: PlasticityKinematics = "plane_strain"
    hidden_dim: int = 64
    num_layers: int = 4
    positional_frequencies: int = 4
    output_init_std: float = 1.0e-4
    include_local_state: bool = True
    history_scale: float = 100.0

    def __post_init__(self) -> None:
        if self.kinematics_mode not in ("plane_strain", "plane_stress"):
            raise ValueError(
                f"kinematics_mode must be 'plane_strain' or 'plane_stress', "
                f"got {self.kinematics_mode!r}"
            )
        for label in (
            "load_dim",
            "hidden_dim",
            "num_layers",
            "positional_frequencies",
        ):
            val = getattr(self, label)
            if type(val) is not int or val <= 0:
                raise ValueError(f"{label} must be a positive integer")
        for flabel in ("output_init_std", "history_scale"):
            fval = getattr(self, flabel)
            if (
                isinstance(fval, bool)
                or not isinstance(fval, (int, float))
                or not math.isfinite(float(fval))
                or float(fval) <= 0.0
            ):
                raise ValueError(f"{flabel} must be finite and positive")
        if type(self.include_local_state) is not bool:
            raise TypeError("include_local_state must be a boolean")

    @property
    def history_feature_dim(self) -> int:
        # (eps_p11, eps_p22, eps_p12, eps_p33, eq_plastic_strain) -> 5 features
        return 5

    @property
    def positional_dim(self) -> int:
        return 2 + 4 * self.positional_frequencies


def extract_local_plastic_features(
    state: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Extract point-local ``(..., 5)`` physical plastic features."""
    eps_p = state["plastic_strain"]
    alpha = state["eq_plastic_strain"]
    return torch.stack(
        (
            eps_p[..., 0, 0],
            eps_p[..., 1, 1],
            eps_p[..., 0, 1],
            eps_p[..., 2, 2],
            alpha,
        ),
        dim=-1,
    )


def project_plastic_state_to_nodes(
    domain: TriangleDomain,
    cell_state: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Area-weighted projection of cell-owned plastic state onto mesh nodes ``(V, 5)``."""
    local_cell = extract_local_plastic_features(cell_state)  # (C, 5)
    n_verts = int(domain.vertices.shape[0])
    feat_dim = int(local_cell.shape[-1])
    dtype = local_cell.dtype
    device = local_cell.device

    node_accum = torch.zeros((n_verts, feat_dim), dtype=dtype, device=device)
    area_accum = torch.zeros((n_verts,), dtype=dtype, device=device)
    areas = domain.areas.to(dtype=dtype, device=device)
    for local_v in range(3):
        v_idx = domain.cells[:, local_v]
        node_accum.index_add_(0, v_idx, local_cell * areas.unsqueeze(-1))
        area_accum.index_add_(0, v_idx, areas)
    return node_accum / torch.clamp(area_accum.unsqueeze(-1), min=1.0e-15)


def summarize_plastic_state(
    state: Mapping[str, torch.Tensor],
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract area-weighted global summary ``(5,)`` from physical plastic state."""
    local = extract_local_plastic_features(state).reshape(-1, 5)
    if weights is None:
        return local.mean(dim=0)
    w = weights.reshape(-1).to(dtype=local.dtype, device=local.device)
    w = w / torch.clamp(w.sum(), min=1.0e-15)
    return torch.einsum("n,nf->f", w, local)


class HistoryAwarePlasticityOperator(nn.Module):
    """Point-local plastic-history-conditioned displacement operator with exact lifting."""

    def __init__(self, config: PlasticityOperatorConfig) -> None:
        super().__init__()
        if not isinstance(config, PlasticityOperatorConfig):
            raise TypeError("config must be a PlasticityOperatorConfig")
        self.config = config
        local_dim = config.history_feature_dim if config.include_local_state else 0
        in_dim = (
            config.positional_dim
            + config.load_dim
            + config.history_feature_dim
            + local_dim
        )
        layers: list[nn.Module] = [nn.Linear(in_dim, config.hidden_dim), nn.Tanh()]
        for _ in range(config.num_layers - 1):
            layers.extend((nn.Linear(config.hidden_dim, config.hidden_dim), nn.Tanh()))
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(config.hidden_dim, 2)
        nn.init.normal_(self.head.weight, mean=0.0, std=config.output_init_std)
        nn.init.zeros_(self.head.bias)

    @property
    def architecture_id(self) -> str:
        return PLASTICITY_OPERATOR_ARCH_ID

    def _pos_encode(self, coords: torch.Tensor) -> torch.Tensor:
        scales = torch.pow(
            coords.new_tensor(2.0),
            torch.arange(
                self.config.positional_frequencies,
                dtype=coords.dtype,
                device=coords.device,
            ),
        )
        scaled = (coords.unsqueeze(-2) * scales.unsqueeze(-1)).flatten(-2)
        return torch.cat((coords, torch.sin(scaled), torch.cos(scaled)), dim=-1)

    def displacement(
        self,
        *,
        query_xy: torch.Tensor,
        frame: QueryFrame,
        load_vector: torch.Tensor,
        history_features: torch.Tensor,
        lifting: DirichletLifting,
        local_state_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict lifted displacement ``(1, N, 2)`` satisfying essential BCs exactly."""
        norm_xy = frame.normalize(query_xy)
        pos = self._pos_encode(norm_xy)
        n_pts = query_xy.shape[1]
        h_scale = float(self.config.history_scale)
        global_hist = history_features.reshape(-1) * h_scale
        cond = torch.cat((load_vector.reshape(-1), global_hist), dim=0)
        cond_exp = cond.unsqueeze(0).unsqueeze(0).expand(1, n_pts, -1)
        parts: list[torch.Tensor] = [pos, cond_exp]
        if self.config.include_local_state:
            if local_state_features is None:
                local_exp = global_hist.unsqueeze(0).unsqueeze(0).expand(1, n_pts, -1)
            else:
                local_exp = (
                    local_state_features.reshape(
                        1, n_pts, self.config.history_feature_dim
                    )
                    * h_scale
                )
            parts.append(local_exp)
        phi = cast(torch.Tensor, self.head(self.net(torch.cat(parts, dim=-1))))
        affine = lifting(query_xy)
        if type(affine) is not AffineDirichletLifting:
            raise TypeError("lifting must return AffineDirichletLifting")
        return affine.offset + affine.scale * phi

    def displacement_and_gradient(
        self,
        *,
        query_xy: torch.Tensor,
        frame: QueryFrame,
        load_vector: torch.Tensor,
        history_features: torch.Tensor,
        lifting: DirichletLifting,
        create_graph: bool,
        local_state_features: torch.Tensor | None = None,
        local_state_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = (
            query_xy
            if query_xy.requires_grad
            else query_xy.detach().clone().requires_grad_(True)
        )
        with torch.enable_grad():
            local_in = (
                local_state_fn(query)
                if local_state_fn is not None
                else local_state_features
            )
            disp = self.displacement(
                query_xy=query,
                frame=frame,
                load_vector=load_vector,
                history_features=history_features,
                lifting=lifting,
                local_state_features=local_in,
            )
            rows = [
                torch.autograd.grad(
                    disp[..., comp].sum(),
                    query,
                    create_graph=create_graph,
                    retain_graph=True,
                )[0]
                for comp in range(2)
            ]
        gradient = torch.stack(rows, dim=-2)
        if not create_graph:
            return disp.detach(), gradient.detach()
        return disp, gradient


@dataclass(frozen=True, slots=True)
class PlasticityTrainingSummary:
    """Recorded incremental variational training summary for J2 elastoplasticity."""

    epochs: int
    total_steps: int
    initial_loss: float
    final_loss: float
    loss_history: tuple[float, ...]
    max_eq_plastic_strain_seen: float
    min_dissipation_seen: float
    training_mode: str = "variational-energy-label-free"
    claim_level: str = "NOT_VERIFIED"


def _plasticity_incremental_energy_loss(
    model: J2IsotropicPlasticity,
    gradient_2d: torch.Tensor,
    state_old: Mapping[str, torch.Tensor],
    dt: float,
    weights: torch.Tensor,
    area: float,
    *,
    kinematics_mode: PlasticityKinematics,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], float, float]:
    """Differentiable Ortiz-Steinmann incremental variational energy loss."""
    eye2 = torch.eye(2, dtype=gradient_2d.dtype, device=gradient_2d.device)
    f2d = eye2 + gradient_2d
    if kinematics_mode == "plane_strain":
        ps = model.plane_strain_update(eye2.expand_as(f2d), f2d, state_old, dt)
    else:
        ps = model.plane_stress_update(eye2.expand_as(f2d), f2d, state_old, dt)

    w_inc = ps.update_3d.incremental_potential
    yield_energy_density = (
        model.config.yield_stress_mpa * model.config.yield_stress_mpa
    ) / model.config.youngs_modulus_mpa
    energy_scale = max(yield_energy_density * area, 1.0e-12)
    loss = torch.sum(weights * w_inc) / energy_scale
    state_next = {k: v.detach().clone() for k, v in ps.update_3d.state_new.items()}
    max_eq_p = float(state_next["eq_plastic_strain"].max().item())
    min_diss = float(ps.update_3d.dissipation.min().item())
    return loss, state_next, max_eq_p, min_diss


def train_plasticity_operator_variational(
    operator: HistoryAwarePlasticityOperator,
    *,
    domain: TriangleDomain,
    frame: QueryFrame,
    model: J2IsotropicPlasticity,
    trajectories: Sequence[Sequence[tuple[torch.Tensor, float, DirichletLifting]]],
    epochs: int = 8,
    learning_rate: float = 1.0e-3,
    points_per_cell: int = 1,
    seed: int = 0,
    supervised_trajectories: Sequence[Sequence[tuple[torch.Tensor, torch.Tensor]]]
    | None = None,
    supervision_weight: float = 10.0,
) -> PlasticityTrainingSummary:
    """Train ``HistoryAwarePlasticityOperator`` via Ortiz-Steinmann variational energy.

    Each trajectory entry is ``(load_vector, dt, lifting_for_step)``.
    When ``supervised_trajectories`` is ``None``, training uses zero FEM labels
    (``training_mode = "variational-energy-label-free"``). When provided (pairs of
    ``(query_xy, u_fem)`` per step), training explicitly records
    ``training_mode = "multi-fidelity-label-efficient"``.
    """
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if not trajectories:
        raise ValueError("trajectories must be non-empty")

    dtype = next(operator.parameters()).dtype
    device = next(operator.parameters()).device
    kinematics_mode = operator.config.kinematics_mode
    optimizer = torch.optim.Adam(operator.parameters(), lr=learning_rate)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    loss_history: list[float] = []
    max_eq_p_seen = 0.0
    min_diss_seen = float("inf")
    total_steps = 0
    area = float(domain.area)
    n_cells = int(domain.cells.shape[0])
    centroids = domain.vertices[domain.cells].mean(dim=1).to(dtype=dtype, device=device)

    for _ in range(epochs):
        for traj_idx, traj in enumerate(trajectories):
            cell_state = model.initial_state((n_cells,), dtype=dtype, device=device)
            n_traj_steps = max(len(traj), 1)
            traj_u_scale = 1.0e-6
            if supervised_trajectories is not None:
                for _, su in supervised_trajectories[traj_idx]:
                    traj_u_scale = max(
                        traj_u_scale,
                        float(
                            torch.linalg.vector_norm(
                                su.to(dtype=dtype, device=device)
                            ).item()
                        ),
                    )
            optimizer.zero_grad(set_to_none=True)
            for step_idx, (load_vec, dt, lifting) in enumerate(traj):
                q_pts_raw, q_w_raw = stratified_triangle_quadrature(
                    domain,
                    points_per_cell,
                    generator=gen,
                )
                q_pts = q_pts_raw.to(dtype=dtype, device=device)
                q_w = q_w_raw.to(dtype=dtype, device=device)
                q_cells = torch.arange(
                    n_cells, dtype=torch.int64, device=device
                ).repeat_interleave(points_per_cell)
                load_t = load_vec.to(dtype=dtype, device=device)

                hist = summarize_plastic_state(cell_state, weights=domain.areas)
                node_feats = project_plastic_state_to_nodes(domain, cell_state)
                state_at_q = {k: v[q_cells] for k, v in cell_state.items()}

                def _interp(
                    pts: torch.Tensor,
                    _nf: torch.Tensor = node_feats,
                    _qc: torch.Tensor = q_cells,
                ) -> torch.Tensor:
                    return interpolate_nodal_features_in_cells(
                        domain, _nf, pts[0], _qc
                    ).unsqueeze(0)

                _, grad_u = operator.displacement_and_gradient(
                    query_xy=q_pts.unsqueeze(0),
                    frame=frame,
                    load_vector=load_t,
                    history_features=hist,
                    lifting=lifting,
                    create_graph=True,
                    local_state_fn=_interp,
                )
                loss, _, eq_p_max, diss_min = _plasticity_incremental_energy_loss(
                    model,
                    grad_u[0],
                    state_at_q,
                    dt,
                    q_w,
                    area,
                    kinematics_mode=kinematics_mode,
                )

                if supervised_trajectories is not None:
                    sup_pts, sup_u = supervised_trajectories[traj_idx][step_idx]
                    sup_pts_t = sup_pts.to(dtype=dtype, device=device)
                    sup_u_t = sup_u.to(dtype=dtype, device=device)
                    pred_u = operator.displacement(
                        query_xy=sup_pts_t.unsqueeze(0),
                        frame=frame,
                        load_vector=load_t,
                        history_features=hist,
                        lifting=lifting,
                        local_state_features=node_feats.unsqueeze(0)
                        if sup_pts_t.shape[0] == node_feats.shape[0]
                        else None,
                    )[0]
                    sup_loss = (
                        torch.linalg.vector_norm(pred_u - sup_u_t) / traj_u_scale
                    ) ** 2
                    loss = loss + supervision_weight * sup_loss

                (loss / float(n_traj_steps)).backward()  # type: ignore[no-untyped-call]

                # Authoritative physical cell-state update at centroids
                _, grad_c = operator.displacement_and_gradient(
                    query_xy=centroids.unsqueeze(0),
                    frame=frame,
                    load_vector=load_t,
                    history_features=hist,
                    lifting=lifting,
                    create_graph=False,
                    local_state_features=extract_local_plastic_features(
                        cell_state
                    ).unsqueeze(0),
                )
                with torch.no_grad():
                    eye2 = torch.eye(2, dtype=dtype, device=device)
                    f2d_c = eye2 + grad_c[0]
                    if kinematics_mode == "plane_strain":
                        upd_c = model.plane_strain_update(
                            eye2.expand_as(f2d_c), f2d_c, cell_state, dt
                        )
                    else:
                        upd_c = model.plane_stress_update(
                            eye2.expand_as(f2d_c), f2d_c, cell_state, dt
                        )
                    cell_state = {
                        k: v.detach().clone()
                        for k, v in upd_c.update_3d.state_new.items()
                    }

                loss_val = float(loss.detach().item())
                loss_history.append(loss_val)
                max_eq_p_seen = max(max_eq_p_seen, eq_p_max)
                min_diss_seen = min(min_diss_seen, diss_min)
                total_steps += 1

            torch.nn.utils.clip_grad_norm_(operator.parameters(), max_norm=5.0)
            optimizer.step()

    mode_str = (
        "multi-fidelity-label-efficient"
        if supervised_trajectories is not None
        else "variational-energy-label-free"
    )
    return PlasticityTrainingSummary(
        epochs=epochs,
        total_steps=total_steps,
        initial_loss=loss_history[0],
        final_loss=loss_history[-1],
        loss_history=tuple(loss_history),
        max_eq_plastic_strain_seen=max_eq_p_seen,
        min_dissipation_seen=min_diss_seen,
        training_mode=mode_str,
    )


def evaluate_plasticity_discrete_residual(
    *,
    basis: Any,
    free_dofs: Any,
    model: J2IsotropicPlasticity,
    u_vec: Any,
    u_old_vec: Any,
    state_old: Mapping[str, torch.Tensor],
    dt: float,
    kinematics_mode: PlasticityKinematics = "plane_strain",
) -> tuple[float, dict[str, torch.Tensor], float, float, float, bool]:
    """Compute exact discrete FEM residual and authoritative quadrature-point state."""
    import numpy as np
    from skfem import asm  # type: ignore[import-untyped]

    from piginot.stateful.fem_stepper import _internal_form

    grad_old = np.asarray(basis.interpolate(u_old_vec).grad, dtype=np.float64)
    grad_new = np.asarray(basis.interpolate(u_vec).grad, dtype=np.float64)
    f2d_old = torch.from_numpy(
        np.ascontiguousarray(np.transpose(grad_old, (2, 3, 0, 1)) + np.eye(2))
    )
    f2d_new = torch.from_numpy(
        np.ascontiguousarray(np.transpose(grad_new, (2, 3, 0, 1)) + np.eye(2))
    )
    if kinematics_mode == "plane_strain":
        ps = model.plane_strain_update(f2d_old, f2d_new, state_old, dt)
    else:
        ps = model.plane_stress_update(f2d_old, f2d_new, state_old, dt)

    stress_2d = ps.update_3d.stress[..., :2, :2].detach().cpu().numpy()
    stress_field = np.transpose(stress_2d, (2, 3, 0, 1))
    internal = np.asarray(
        asm(_internal_form, basis, stress=stress_field), dtype=np.float64
    )
    res_norm = float(np.linalg.norm(internal[free_dofs]))
    dx = np.asarray(basis.dx, dtype=np.float64)
    diss_np = ps.update_3d.dissipation.detach().cpu().numpy()
    min_diss = float(np.min(diss_np))
    total_diss = float(np.sum(diss_np * dx))
    min_det = float(ps.update_3d.det_f.min().item())
    admissible = bool(ps.closure_converged and ps.update_3d.admissible)
    state_new = {k: v.detach().clone() for k, v in ps.update_3d.state_new.items()}
    return res_norm, state_new, min_det, min_diss, total_diss, admissible


__all__ = (
    "PLASTICITY_OPERATOR_ARCH_ID",
    "HistoryAwarePlasticityOperator",
    "PlasticityKinematics",
    "PlasticityOperatorConfig",
    "PlasticityTrainingSummary",
    "evaluate_plasticity_discrete_residual",
    "extract_local_plastic_features",
    "project_plastic_state_to_nodes",
    "summarize_plastic_state",
    "train_plasticity_operator_variational",
)
