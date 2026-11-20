"""History-aware neural transition operator for finite-strain viscoelasticity (F1).

Learns the incremental displacement field ``u_{n+1}(x)`` conditioned on:
* normalized spatial query coordinates ``x``,
* the next applied load vector and time increment ``dt``,
* the global area-weighted viscous strain summary, and
* optional **point-local physical viscous strain features** ``E_v^(m)(x)``
  interpolated from quadrature-point / cell-owned internal variables.

Training minimizes the exact **incremental variational energy-dissipation
potential** of :class:`~piginot.stateful.viscoelastic.GeneralizedMaxwellViscoelastic`
with exact affine Dirichlet lifting and **zero FEM labels**, redrawing fresh
unbiased stratified triangle quadrature at every step while updating cell-owned
physical internal states.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

import torch
from torch import nn

from piginot.lifting.base import AffineDirichletLifting, DirichletLifting
from piginot.objectives.energy import TriangleDomain, stratified_triangle_quadrature
from piginot.operators.energy_operator import QueryFrame
from piginot.stateful.viscoelastic import GeneralizedMaxwellViscoelastic

VISCOELASTIC_OPERATOR_ARCH_ID: Final = "pi-ginot-viscoelastic-transition-v1"


@dataclass(frozen=True, slots=True)
class ViscoelasticOperatorConfig:
    """Architecture configuration for the F1 history-aware transition operator."""

    load_dim: int
    n_branches: int
    hidden_dim: int = 64
    num_layers: int = 4
    positional_frequencies: int = 4
    output_init_std: float = 1e-4
    include_local_state: bool = True

    def __post_init__(self) -> None:
        for label in (
            "load_dim",
            "n_branches",
            "hidden_dim",
            "num_layers",
            "positional_frequencies",
        ):
            val = getattr(self, label)
            if type(val) is not int or val <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if (
            isinstance(self.output_init_std, bool)
            or not isinstance(self.output_init_std, (int, float))
            or not math.isfinite(float(self.output_init_std))
            or float(self.output_init_std) <= 0.0
        ):
            raise ValueError("output_init_std must be finite and positive")
        if type(self.include_local_state) is not bool:
            raise TypeError("include_local_state must be a boolean")

    @property
    def history_feature_dim(self) -> int:
        # Per branch: (E_v11, E_v22, E_v12, E_v33) -> 4 features per branch.
        return 4 * self.n_branches

    @property
    def positional_dim(self) -> int:
        return 2 + 4 * self.positional_frequencies


def extract_local_viscous_features(viscous_strain: torch.Tensor) -> torch.Tensor:
    """Extract point-local ``(..., 4 * M)`` components ``(E11, E22, E12, E33)``."""
    n_branches = viscous_strain.shape[-3]
    lead = viscous_strain.shape[:-3]
    comps = torch.stack(
        (
            viscous_strain[..., 0, 0],
            viscous_strain[..., 1, 1],
            viscous_strain[..., 0, 1],
            viscous_strain[..., 2, 2],
        ),
        dim=-1,
    )
    return comps.reshape(*lead, 4 * n_branches)


def project_cell_state_to_nodes(
    domain: TriangleDomain,
    cell_viscous_strain: torch.Tensor,
) -> torch.Tensor:
    """Area-weighted projection of cell-owned ``(C, M, 3, 3)`` states onto mesh nodes.

    Returns ``(V, 4 * M)`` point-local viscous strain features for nodal queries.
    """
    local_cell = extract_local_viscous_features(cell_viscous_strain)  # (C, 4*M)
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
    return node_accum / torch.clamp(area_accum.unsqueeze(-1), min=1e-15)


def interpolate_nodal_features_in_cells(
    domain: TriangleDomain,
    node_features: torch.Tensor,
    query_xy: torch.Tensor,
    cell_indices: torch.Tensor,
) -> torch.Tensor:
    """Differentiable P1 barycentric interpolation of ``node_features`` at ``query_xy``.

    Because barycentric coordinates are linear functions of ``query_xy``, passing
    the resulting tensor into ``displacement_and_gradient`` ensures ``autograd``
    computes the exact total spatial gradient ``du/dx = dN/dx + (dN/dEv)(dEv/dx)``.
    """
    verts = domain.vertices[domain.cells[cell_indices]]  # (N, 3, 2)
    v0 = verts[:, 0, :]
    e1 = verts[:, 1, :] - v0
    e2 = verts[:, 2, :] - v0
    rel = query_xy - v0
    two_area = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    lam1 = (rel[:, 0] * e2[:, 1] - rel[:, 1] * e2[:, 0]) / two_area
    lam2 = (e1[:, 0] * rel[:, 1] - e1[:, 1] * rel[:, 0]) / two_area
    lam0 = 1.0 - lam1 - lam2
    cell_feats = node_features[domain.cells[cell_indices]]  # (N, 3, F)
    return (
        lam0.unsqueeze(-1) * cell_feats[:, 0, :]
        + lam1.unsqueeze(-1) * cell_feats[:, 1, :]
        + lam2.unsqueeze(-1) * cell_feats[:, 2, :]
    )


def summarize_viscous_state(
    state: Mapping[str, torch.Tensor],
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract area-weighted global summary ``(4 * M,)`` from ``viscous_strain``."""
    ev = state["viscous_strain"]  # (..., M, 3, 3)
    local = extract_local_viscous_features(ev).reshape(-1, 4 * ev.shape[-3])
    if weights is None:
        return local.mean(dim=0)
    w = weights.reshape(-1).to(dtype=local.dtype, device=local.device)
    w = w / torch.clamp(w.sum(), min=1e-15)
    return torch.einsum("n,nf->f", w, local)


class HistoryAwareViscoelasticOperator(nn.Module):
    """Point-local history-conditioned displacement operator with exact BC lifting."""

    def __init__(self, config: ViscoelasticOperatorConfig) -> None:
        super().__init__()
        if not isinstance(config, ViscoelasticOperatorConfig):
            raise TypeError("config must be a ViscoelasticOperatorConfig")
        self.config = config
        local_dim = config.history_feature_dim if config.include_local_state else 0
        in_dim = (
            config.positional_dim
            + config.load_dim
            + 1  # log1p(dt)
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
        return VISCOELASTIC_OPERATOR_ARCH_ID

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
        return torch.cat(
            (coords, torch.sin(scaled), torch.cos(scaled)),
            dim=-1,
        )

    def displacement(
        self,
        *,
        query_xy: torch.Tensor,
        frame: QueryFrame,
        load_vector: torch.Tensor,
        dt: float,
        history_features: torch.Tensor,
        lifting: DirichletLifting,
        local_state_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict lifted displacement ``(1, N, 2)`` satisfying essential BCs exactly."""
        norm_xy = frame.normalize(query_xy)  # (1, N, 2)
        pos = self._pos_encode(norm_xy)  # (1, N, pos_dim)
        n_pts = query_xy.shape[1]
        global_hist = history_features.reshape(-1)
        cond = torch.cat(
            (
                load_vector.reshape(-1),
                query_xy.new_tensor([math.log1p(dt)]),
                global_hist,
            ),
            dim=0,
        )
        cond_exp = cond.unsqueeze(0).unsqueeze(0).expand(1, n_pts, -1)
        parts: list[torch.Tensor] = [pos, cond_exp]
        if self.config.include_local_state:
            if local_state_features is None:
                local_exp = global_hist.unsqueeze(0).unsqueeze(0).expand(1, n_pts, -1)
            else:
                local_exp = local_state_features.reshape(
                    1, n_pts, self.config.history_feature_dim
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
        dt: float,
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
                dt=dt,
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
class ViscoelasticTrainingSummary:
    """Recorded physics-first incremental variational training summary."""

    epochs: int
    total_steps: int
    initial_loss: float
    final_loss: float
    loss_history: tuple[float, ...]
    min_det_f_seen: float
    claim_level: str = "NOT_VERIFIED"


def _plane_stress_incremental_energy_loss(
    model: GeneralizedMaxwellViscoelastic,
    gradient_2d: torch.Tensor,
    state_old: Mapping[str, torch.Tensor],
    dt: float,
    weights: torch.Tensor,
    area: float,
    *,
    barrier_weight: float = 1e3,
    barrier_threshold: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], float]:
    """Differentiable incremental variational potential energy with implicit F33 closure."""
    eye2 = torch.eye(2, dtype=gradient_2d.dtype, device=gradient_2d.device)
    f2d = eye2 + gradient_2d  # (N, 2, 2)
    det2d = f2d[:, 0, 0] * f2d[:, 1, 1] - f2d[:, 0, 1] * f2d[:, 1, 0]
    admissible = det2d > 1e-6
    safe_f2d = torch.where(admissible[:, None, None], f2d, eye2.expand_as(f2d))

    # Solve exact plane-stress closure F33 without differentiating through Newton
    # iterations (envelope theorem: dW_inc/dF33 == P_33 == 0 at closure!).
    with torch.no_grad():
        ps = model.plane_stress_update(
            eye2.expand_as(safe_f2d), safe_f2d.detach(), state_old, dt
        )
        f33_star = ps.f33.detach()
        state_next = {k: v.detach().clone() for k, v in ps.update_3d.state_new.items()}

    zeros_col = torch.zeros((f2d.shape[0], 2, 1), dtype=f2d.dtype, device=f2d.device)
    zeros_row = torch.zeros((f2d.shape[0], 1, 2), dtype=f2d.dtype, device=f2d.device)
    upper = torch.cat((safe_f2d, zeros_col), dim=-1)
    bottom = torch.cat((zeros_row, f33_star[:, None, None]), dim=-1)
    f3d = torch.cat((upper, bottom), dim=-2)

    upd = model.update(f3d.detach(), f3d, state_old, dt)
    w_inc = torch.where(admissible, upd.incremental_potential, torch.zeros_like(det2d))
    energy = torch.sum(w_inc * weights) / (model.config.equilibrium_modulus_mpa * area)

    guard = torch.where(admissible, torch.minimum(det2d, det2d * f33_star), det2d)
    barrier = torch.mean(torch.relu(barrier_threshold - guard).square())
    loss = energy + barrier_weight * barrier
    return loss, state_next, float(guard.detach().min().item())


def train_viscoelastic_operator_energy(
    operator: HistoryAwareViscoelasticOperator,
    *,
    material: GeneralizedMaxwellViscoelastic,
    domain: TriangleDomain,
    frame: QueryFrame,
    load_trajectories: Sequence[Sequence[tuple[torch.Tensor, float]]],
    lifting_factory: Callable[[torch.Tensor], DirichletLifting],
    epochs: int = 30,
    learning_rate: float = 3e-3,
    points_per_cell: int = 3,
    quadrature_degree: int = 2,
    seed: int = 0,
) -> ViscoelasticTrainingSummary:
    """Train ``operator`` via label-free incremental variational potential energy.

    Uses fresh unbiased ``stratified_triangle_quadrature`` at every step to
    prevent fixed-quadrature hourglass modes while feeding each query point its
    owning cell's physical internal viscous strain tensor ``E_v^(m)[c]``.
    """
    del quadrature_degree
    torch.manual_seed(seed)
    n_cells = int(domain.cells.shape[0])
    dtype = domain.vertices.dtype
    device = domain.vertices.device
    cell_areas = domain.areas
    quad_gen = torch.Generator(device=device).manual_seed(seed + 97)
    optimizer = torch.optim.Adam(
        operator.parameters(), lr=learning_rate, foreach=False, fused=False
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=0.1 * learning_rate
    )

    history: list[float] = []
    total_steps = 0
    min_det_seen = float("inf")
    cell_indices = torch.arange(
        n_cells, dtype=torch.int64, device=device
    ).repeat_interleave(points_per_cell)

    for _epoch in range(epochs):
        epoch_loss = 0.0
        epoch_count = 0
        for traj in load_trajectories:
            cell_state = material.initial_state((n_cells,), dtype=dtype, device=device)
            for load_vec, dt in traj:
                hist_feat = summarize_viscous_state(cell_state, weights=cell_areas)
                lifting = lifting_factory(load_vec)
                pts, wts = stratified_triangle_quadrature(
                    domain, points_per_cell, generator=quad_gen
                )
                q_state = {
                    k: v.unsqueeze(1)
                    .expand(n_cells, points_per_cell, *v.shape[1:])
                    .reshape(n_cells * points_per_cell, *v.shape[1:])
                    for k, v in cell_state.items()
                }
                local_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
                if operator.config.include_local_state:
                    node_ev = project_cell_state_to_nodes(
                        domain, cell_state["viscous_strain"]
                    ).detach()

                    def _interp(
                        q_in: torch.Tensor, nf: torch.Tensor = node_ev
                    ) -> torch.Tensor:
                        return interpolate_nodal_features_in_cells(
                            domain, nf, q_in[0], cell_indices
                        ).unsqueeze(0)

                    local_fn = _interp
                optimizer.zero_grad(set_to_none=True)
                _, grad = operator.displacement_and_gradient(
                    query_xy=pts.unsqueeze(0),
                    frame=frame,
                    load_vector=load_vec,
                    dt=dt,
                    history_features=hist_feat,
                    lifting=lifting,
                    create_graph=True,
                    local_state_fn=local_fn,
                )
                loss, q_state_next, min_det = _plane_stress_incremental_energy_loss(
                    material,
                    grad[0],
                    q_state,
                    dt,
                    wts,
                    domain.area,
                    barrier_threshold=0.20,
                )
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(operator.parameters(), max_norm=1.0)
                optimizer.step()
                cell_state = {
                    k: v.reshape(n_cells, points_per_cell, *v.shape[1:]).mean(dim=1)
                    for k, v in q_state_next.items()
                }
                epoch_loss += float(loss.detach().item())
                epoch_count += 1
                total_steps += 1
                min_det_seen = min(min_det_seen, min_det)
        scheduler.step()
        history.append(epoch_loss / max(epoch_count, 1))

    return ViscoelasticTrainingSummary(
        epochs=epochs,
        total_steps=total_steps,
        initial_loss=history[0],
        final_loss=history[-1],
        loss_history=tuple(history),
        min_det_f_seen=min_det_seen,
    )


__all__ = (
    "VISCOELASTIC_OPERATOR_ARCH_ID",
    "HistoryAwareViscoelasticOperator",
    "ViscoelasticOperatorConfig",
    "ViscoelasticTrainingSummary",
    "extract_local_viscous_features",
    "interpolate_nodal_features_in_cells",
    "project_cell_state_to_nodes",
    "summarize_viscous_state",
    "train_viscoelastic_operator_energy",
)
