"""PyTorch-native vectorized differentiable FEM backend (CPU & GPU, 2D & 2.5D).

Shares FormaMech's exact ``piginot.physics`` constitutive kernels (Neo-Hookean,
Ogden, Mooney-Rivlin, HGO anisotropic, Gent) directly inside vectorized PyTorch
element assembly, implementing the full :class:`SolverBackend` and
:class:`DiscreteLinearization` protocols across:
- ``2d-plane-stress-finite-strain`` (exact parity with ``ScikitFemBackend``)
- ``2d-plane-strain-finite-strain`` (``F_33 == 1``)
- ``2d-axisymmetric-finite-strain`` (``R > 0``, hoop stretch ``1 + u_r/R``,
  circumferential measure ``2 * pi * R dR dZ``)
plus dead tractions, non-conservative follower pressure (exact unsymmetric load
stiffness), and frictionless rigid-indenter Augmented Lagrangian contact.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import sparse  # type: ignore[import-untyped]
from scipy.sparse.linalg import spsolve  # type: ignore[import-untyped]

from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    LoadSpec,
    MaterialSpec,
    hash_spec,
)
from piginot.fem.contact import (
    ContactVerificationReport,
    RigidIndenterSpec,
    assemble_contact_terms,
)
from piginot.fem.models import (
    FemAttempt,
    FemConvergenceEntry,
    FemFailureCode,
    FemRequest,
    FemResult,
    FemSolveFailure,
)
from piginot.fem.protocol import BackendCapabilities
from piginot.physics import (
    consistent_tangent,
    energy_density,
    first_piola,
    plane_stress_closure,
)

FloatArray = NDArray[np.float64]

TORCH_PHYSICS_MODES: Final = (
    "2d-plane-stress-finite-strain",
    "2d-plane-strain-finite-strain",
    "2d-axisymmetric-finite-strain",
)

# 3-point degree-2 Dunavant rule on reference triangle (0,0)-(1,0)-(0,1)
_TRI_Q2_XI = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=np.float64)
_TRI_Q2_ETA = np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0], dtype=np.float64)
_TRI_Q2_W = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=np.float64)

# 6-point degree-4 Dunavant rule on reference triangle (sum(w) = 0.5)
_A1 = 0.445948490915965
_A2 = 0.091576213509771
_W1 = 0.1116907948390055
_W2 = 0.0549758718276610
_TRI_Q4_XI = np.array(
    [_A1, 1.0 - 2.0 * _A1, _A1, _A2, 1.0 - 2.0 * _A2, _A2], dtype=np.float64
)
_TRI_Q4_ETA = np.array(
    [_A1, _A1, 1.0 - 2.0 * _A1, _A2, _A2, 1.0 - 2.0 * _A2], dtype=np.float64
)
_TRI_Q4_W = np.array([_W1, _W1, _W1, _W2, _W2, _W2], dtype=np.float64)

# 1D Gauss-Legendre on [0, 1]
_GL2_T = np.array([0.5 - 0.5 / math.sqrt(3.0), 0.5 + 0.5 / math.sqrt(3.0)])
_GL2_W = np.array([0.5, 0.5])
_GL3_T = np.array([0.5 * (1.0 - math.sqrt(0.6)), 0.5, 0.5 * (1.0 + math.sqrt(0.6))])
_GL3_W = np.array([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0])


def _shape_and_grad_ref(
    order: int, xi: FloatArray, eta: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Return ``N(Q, n_loc)`` and ``dN_dxi(Q, n_loc, 2)`` on the reference triangle."""
    l0 = 1.0 - xi - eta
    l1 = xi
    l2 = eta
    if order == 1:
        n_val = np.stack((l0, l1, l2), axis=-1)
        dn = np.zeros((len(xi), 3, 2), dtype=np.float64)
        dn[:, 0, :] = [-1.0, -1.0]
        dn[:, 1, :] = [1.0, 0.0]
        dn[:, 2, :] = [0.0, 1.0]
        return n_val, dn

    n_val = np.stack(
        (
            l0 * (2.0 * l0 - 1.0),
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            4.0 * l0 * l1,
            4.0 * l1 * l2,
            4.0 * l2 * l0,
        ),
        axis=-1,
    )
    dn = np.zeros((len(xi), 6, 2), dtype=np.float64)
    # dL0/dxi = -1, dL0/deta = -1; dL1/dxi = 1, 0; dL2/dxi = 0, 1
    dn[:, 0, 0] = -(4.0 * l0 - 1.0)
    dn[:, 0, 1] = -(4.0 * l0 - 1.0)
    dn[:, 1, 0] = 4.0 * l1 - 1.0
    dn[:, 1, 1] = 0.0
    dn[:, 2, 0] = 0.0
    dn[:, 2, 1] = 4.0 * l2 - 1.0
    dn[:, 3, 0] = 4.0 * (l0 - l1)
    dn[:, 3, 1] = -4.0 * l1
    dn[:, 4, 0] = 4.0 * l2
    dn[:, 4, 1] = 4.0 * l1
    dn[:, 5, 0] = -4.0 * l2
    dn[:, 5, 1] = 4.0 * (l0 - l2)
    return n_val, dn


def _edge_shape_and_grad_1d(order: int, t: FloatArray) -> tuple[FloatArray, FloatArray]:
    """1D Lagrange basis on ``t in [0, 1]`` for P1 (2 nodes) or P2 (3 nodes: 0, 1, mid)."""
    if order == 1:
        n1d = np.stack((1.0 - t, t), axis=-1)
        dn1d = np.broadcast_to(np.array([-1.0, 1.0]), (len(t), 2)).copy()
        return n1d, dn1d
    n1d = np.stack(
        (
            (1.0 - t) * (1.0 - 2.0 * t),
            t * (2.0 * t - 1.0),
            4.0 * t * (1.0 - t),
        ),
        axis=-1,
    )
    dn1d = np.stack(
        (
            4.0 * t - 3.0,
            4.0 * t - 1.0,
            4.0 - 8.0 * t,
        ),
        axis=-1,
    )
    return n1d, dn1d


@dataclass(frozen=True, slots=True)
class _Discretization:
    order: int
    coords: FloatArray  # (N_nodes, 2)
    conn: NDArray[np.int64]  # (E, n_loc)
    edge_midpoints: Mapping[tuple[int, int], int]
    facet_nodes: dict[str, list[tuple[int, ...]]]  # tag -> list of (a,b) or (a,b,mid)
    facet_normals: dict[str, list[tuple[float, float]]]  # outward unit normal per facet
    facet_lengths: dict[str, list[float]]
    quad_xi: FloatArray
    quad_eta: FloatArray
    quad_w_ref: FloatArray
    n_ref: FloatArray  # (Q, n_loc)
    dn_dx: FloatArray  # (E, Q, n_loc, 2)
    quad_points: FloatArray  # (E, Q, 2)
    quad_area_weights: (
        FloatArray  # (E, Q) = 2*A_e * w_ref (since sum(w_ref)=0.5 -> sum=A_e)
    )


def _build_discretization(mesh: CanonicalMesh, order: int) -> _Discretization:
    base_coords = np.asarray(mesh.nodes, dtype=np.float64)
    base_cells = np.asarray(mesh.cells, dtype=np.int64)
    n_verts = base_coords.shape[0]
    n_elems = base_cells.shape[0]

    edge_mid: dict[tuple[int, int], int] = {}
    if order == 1:
        coords = base_coords
        conn = base_cells
        q_xi, q_eta, q_w = _TRI_Q2_XI, _TRI_Q2_ETA, _TRI_Q2_W
    else:
        extra_coords: list[tuple[float, float]] = []
        conn = np.zeros((n_elems, 6), dtype=np.int64)
        conn[:, :3] = base_cells
        next_idx = n_verts
        for e in range(n_elems):
            v0, v1, v2 = (
                int(base_cells[e, 0]),
                int(base_cells[e, 1]),
                int(base_cells[e, 2]),
            )
            for local_m, (a, b) in enumerate(((v0, v1), (v1, v2), (v2, v0)), start=3):
                key = (a, b) if a < b else (b, a)
                mid_idx = edge_mid.get(key)
                if mid_idx is None:
                    mid_idx = next_idx
                    edge_mid[key] = mid_idx
                    next_idx += 1
                    extra_coords.append(
                        (
                            0.5 * (float(base_coords[a, 0]) + float(base_coords[b, 0])),
                            0.5 * (float(base_coords[a, 1]) + float(base_coords[b, 1])),
                        )
                    )
                conn[e, local_m] = mid_idx
        coords = (
            np.vstack((base_coords, np.asarray(extra_coords, dtype=np.float64)))
            if extra_coords
            else base_coords
        )
        q_xi, q_eta, q_w = _TRI_Q4_XI, _TRI_Q4_ETA, _TRI_Q4_W

    # Compute affine triangle Jacobians J_e = [x1-x0, x2-x0]
    x0 = base_coords[base_cells[:, 0]]
    x1 = base_coords[base_cells[:, 1]]
    x2 = base_coords[base_cells[:, 2]]
    e1 = x1 - x0
    e2 = x2 - x0
    det_j = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    if np.any(det_j <= 0.0):
        raise FemSolveFailure(
            FemFailureCode.NEGATIVE_JACOBIAN,
            "reference mesh contains non-positive triangle Jacobian",
        )
    # J_inv_T = (1/det_j) * [[e2_y, -e1_y], [-e2_x, e1_x]]
    inv_det = 1.0 / det_j
    j_inv_t = np.zeros((n_elems, 2, 2), dtype=np.float64)
    j_inv_t[:, 0, 0] = e2[:, 1] * inv_det
    j_inv_t[:, 0, 1] = -e1[:, 1] * inv_det
    j_inv_t[:, 1, 0] = -e2[:, 0] * inv_det
    j_inv_t[:, 1, 1] = e1[:, 0] * inv_det

    n_ref, dn_ref = _shape_and_grad_ref(order, q_xi, q_eta)
    # dn_dx[e, q, a, K] = sum_r j_inv_t[e, K, r] * dn_ref[q, a, r]
    dn_dx = np.einsum("ekr,qar->eqak", j_inv_t, dn_ref)
    quad_pts = np.einsum("qa,eai->eqi", n_ref, coords[conn])
    # det_j = 2 * area; sum(q_w) = 0.5 -> det_j * q_w sums to area!
    quad_w = det_j[:, None] * q_w[None, :]

    # Outward normals for every boundary facet using owning triangle opposite vertex
    edge_opp: dict[tuple[int, int], int] = {}
    for e in range(n_elems):
        v0, v1, v2 = int(base_cells[e, 0]), int(base_cells[e, 1]), int(base_cells[e, 2])
        for a, b, opp in ((v0, v1, v2), (v1, v2, v0), (v2, v0, v1)):
            key = (a, b) if a < b else (b, a)
            edge_opp[key] = opp

    facet_nodes: dict[str, list[tuple[int, ...]]] = {}
    facet_normals: dict[str, list[tuple[float, float]]] = {}
    facet_lengths: dict[str, list[float]] = {}
    for tag, facet_ids in mesh.semantic_tags.items():
        fn_list: list[tuple[int, ...]] = []
        norm_list: list[tuple[float, float]] = []
        len_list: list[float] = []
        for f_idx in facet_ids:
            a, b = (
                int(mesh.boundary_facets[f_idx][0]),
                int(mesh.boundary_facets[f_idx][1]),
            )
            xa, ya = float(base_coords[a, 0]), float(base_coords[a, 1])
            xb, yb = float(base_coords[b, 0]), float(base_coords[b, 1])
            tx, ty = xb - xa, yb - ya
            seg_len = math.hypot(tx, ty)
            nx, ny = ty / seg_len, -tx / seg_len
            key = (a, b) if a < b else (b, a)
            opp = edge_opp[key]
            xo, yo = float(base_coords[opp, 0]), float(base_coords[opp, 1])
            mx, my = 0.5 * (xa + xb), 0.5 * (ya + yb)
            # Ensure outward normal points away from interior vertex
            if nx * (xo - mx) + ny * (yo - my) > 0.0:
                nx, ny = -nx, -ny
                a, b = b, a
            if order == 1:
                fn_list.append((a, b))
            else:
                fn_list.append((a, b, edge_mid[key]))
            norm_list.append((nx, ny))
            len_list.append(seg_len)
        facet_nodes[tag] = fn_list
        facet_normals[tag] = norm_list
        facet_lengths[tag] = len_list

    return _Discretization(
        order=order,
        coords=coords,
        conn=conn,
        edge_midpoints=edge_mid,
        facet_nodes=facet_nodes,
        facet_normals=facet_normals,
        facet_lengths=facet_lengths,
        quad_xi=q_xi,
        quad_eta=q_eta,
        quad_w_ref=q_w,
        n_ref=n_ref,
        dn_dx=dn_dx,
        quad_points=quad_pts,
        quad_area_weights=quad_w,
    )


@dataclass(frozen=True, slots=True)
class _TorchConstraints:
    dirichlet_dofs: NDArray[np.int64]
    dirichlet_values: FloatArray
    free_dofs: NDArray[np.int64]
    boundary_dofs: Mapping[str, NDArray[np.int64]]
    global_matrix: FloatArray  # empty (0, n_dofs) placeholder for protocol parity

    @property
    def constrained(self) -> NDArray[np.int64]:
        return self.dirichlet_dofs

    @property
    def free(self) -> NDArray[np.int64]:
        return self.free_dofs

    @property
    def target_values(self) -> FloatArray:
        return self.dirichlet_values

    @property
    def by_name(self) -> Mapping[str, NDArray[np.int64]]:
        return self.boundary_dofs


class _TorchBasisShim:
    """Minimal basis shim providing ``interpolate(vec) -> (2, E, Q)`` for estimators."""

    def __init__(self, disc: _Discretization) -> None:
        self._disc = disc

    def interpolate(self, vec: FloatArray) -> FloatArray:
        nodal = np.asarray(vec, dtype=np.float64).reshape(-1, 2)
        u_elem = nodal[self._disc.conn]
        return np.asarray(
            np.einsum("qa,eai->ieq", self._disc.n_ref, u_elem), dtype=np.float64
        )


def _bind_torch_constraints(
    disc: _Discretization,
    bcs: tuple[BoundaryConditionSpec, ...],
    *,
    dirichlet_providers: Mapping[str, Any] | None = None,
    provider_ids: Mapping[str, str] | None = None,
) -> _TorchConstraints:
    n_dofs = disc.coords.shape[0] * 2
    fixed: dict[int, float] = {}
    b_dofs: dict[str, set[int]] = {}
    providers = dirichlet_providers or {}
    pids = provider_ids or {}

    for bc in bcs:
        tag = bc.boundary_tag
        facets = disc.facet_nodes.get(tag, [])
        if not facets:
            continue
        nodes = sorted({n for f in facets for n in f})
        if bc.kind in ("free", "traction_free"):
            continue
        if bc.kind == "symmetry":
            if bc.components:
                comps = tuple(
                    0 if c in ("x", "ux", "r", "ur") else 1 for c in bc.components
                )
            else:
                norms = disc.facet_normals[tag]
                avg_nx = sum(abs(n[0]) for n in norms) / len(norms)
                avg_ny = sum(abs(n[1]) for n in norms) / len(norms)
                comps = (0,) if avg_nx >= avg_ny else (1,)
            for n in nodes:
                for comp in comps:
                    dof = 2 * n + comp
                    fixed[dof] = 0.0
                    b_dofs.setdefault(bc.name, set()).add(dof)
            continue
        if bc.kind == "dirichlet":
            pid = pids.get(bc.name) or pids.get(tag)
            provider = providers.get(pid) if pid is not None else None
            if provider is not None:
                pts = disc.coords[nodes]
                vals = np.asarray(provider.evaluate(pts), dtype=np.float64)
                comps_iter = (
                    tuple(
                        0 if c in ("x", "ux", "r", "ur") else 1 for c in bc.components
                    )
                    if bc.components
                    else (0, 1)
                )
                for idx, n in enumerate(nodes):
                    for comp in comps_iter:
                        dof = 2 * n + comp
                        fixed[dof] = float(vals[idx, comp])
                        b_dofs.setdefault(bc.name, set()).add(dof)
            else:
                comp_vals = (
                    zip(bc.components, bc.values, strict=True)
                    if bc.values
                    else [(c, None) for c in bc.components]
                )
                for comp_name, val in comp_vals:
                    comp = 0 if comp_name in ("x", "ux", "r", "ur") else 1
                    fval = 0.0 if val is None else float(val.value)
                    for n in nodes:
                        dof = 2 * n + comp
                        fixed[dof] = fval
                        b_dofs.setdefault(bc.name, set()).add(dof)

    d_dofs = np.array(sorted(fixed.keys()), dtype=np.int64)
    d_vals = np.array([fixed[d] for d in d_dofs], dtype=np.float64)
    mask = np.ones(n_dofs, dtype=bool)
    mask[d_dofs] = False
    free_dofs = np.nonzero(mask)[0].astype(np.int64)
    return _TorchConstraints(
        dirichlet_dofs=d_dofs,
        dirichlet_values=d_vals,
        free_dofs=free_dofs,
        boundary_dofs={
            k: np.array(sorted(v), dtype=np.int64) for k, v in b_dofs.items()
        },
        global_matrix=np.zeros((0, n_dofs), dtype=np.float64),
    )


def _assemble_dead_external_force(
    disc: _Discretization,
    loads: tuple[LoadSpec, ...],
    *,
    physics_mode: str,
    body_force_providers: Mapping[str, Any] | None = None,
    body_force_provider_ids: Mapping[str, str] | None = None,
    traction_providers: Mapping[str, Any] | None = None,
    traction_provider_ids: Mapping[str, str] | None = None,
) -> FloatArray:
    n_nodes = disc.coords.shape[0]
    f_ext = np.zeros((n_nodes, 2), dtype=np.float64)
    t_pts, w_pts = (_GL2_T, _GL2_W) if disc.order == 1 else (_GL3_T, _GL3_W)
    n1d, _ = _edge_shape_and_grad_1d(disc.order, t_pts)
    bf_provs = body_force_providers or {}
    bf_ids = body_force_provider_ids or {}
    tr_provs = traction_providers or {}
    tr_ids = traction_provider_ids or {}
    is_axi = physics_mode == "2d-axisymmetric-finite-strain"

    for load in loads:
        val_map = {
            c: float(q.value) for c, q in zip(load.components, load.values, strict=True)
        }
        if load.kind == "traction":
            tag = load.boundary_tag or load.name
            facets = disc.facet_nodes.get(tag, [])
            lengths = disc.facet_lengths.get(tag, [])
            pid = tr_ids.get(load.name) or tr_ids.get(tag)
            prov = tr_provs.get(pid) if pid is not None else None
            tx = val_map.get("x", 0.0)
            ty = val_map.get("y", 0.0)
            for fn, seg_len in zip(facets, lengths, strict=True):
                fn_arr = np.array(fn, dtype=np.int64)
                pts = n1d @ disc.coords[fn_arr]  # (Q1, 2)
                if prov is not None:
                    t_vec = np.asarray(prov.evaluate(pts), dtype=np.float64)
                else:
                    t_vec = np.broadcast_to(np.array([tx, ty]), (len(t_pts), 2))
                wq = w_pts * seg_len
                if is_axi:
                    wq = wq * (2.0 * math.pi * pts[:, 0])
                # f_local[a, i] = sum_q wq[q] * n1d[q, a] * t_vec[q, i]
                f_loc = np.einsum("q,qa,qi->ai", wq, n1d, t_vec)
                np.add.at(f_ext, fn_arr, f_loc)
        elif load.kind == "body_force":
            pid = bf_ids.get(load.name)
            prov = bf_provs.get(pid) if pid is not None else None
            bx = val_map.get("x", 0.0)
            by = val_map.get("y", 0.0)
            pts_flat = disc.quad_points.reshape(-1, 2)
            if prov is not None:
                b_vec = np.asarray(prov.evaluate(pts_flat), dtype=np.float64).reshape(
                    disc.quad_points.shape
                )
            else:
                b_vec = np.broadcast_to(
                    np.array([bx, by]), disc.quad_points.shape
                ).copy()
            wq = disc.quad_area_weights
            if is_axi:
                wq = wq * (2.0 * math.pi * disc.quad_points[..., 0])
            f_elem = np.einsum("eq,qa,eqi->eai", wq, disc.n_ref, b_vec)
            np.add.at(f_ext, disc.conn, f_elem)
    return f_ext.reshape(-1)


def _assemble_follower_pressure(
    disc: _Discretization,
    loads: tuple[LoadSpec, ...],
    u_nodal: FloatArray,
    *,
    physics_mode: str,
    load_factor: float = 1.0,
) -> tuple[FloatArray, sparse.csc_matrix]:
    """Exact Total-Lagrangian follower pressure force vector & unsymmetric tangent ``K_L``."""
    n_nodes = disc.coords.shape[0]
    n_dofs = 2 * n_nodes
    f_fol = np.zeros((n_nodes, 2), dtype=np.float64)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    t_pts, w_pts = (_GL2_T, _GL2_W) if disc.order == 1 else (_GL3_T, _GL3_W)
    n1d, dn1d = _edge_shape_and_grad_1d(disc.order, t_pts)
    is_axi = physics_mode == "2d-axisymmetric-finite-strain"

    for load in loads:
        if load.kind != "follower_pressure":
            continue
        p_raw = float(load.values[0].value) if load.values else 0.0
        p_val = p_raw * load_factor
        if abs(p_val) == 0.0:
            continue
        tag = load.boundary_tag or load.name
        facets = disc.facet_nodes.get(tag, [])
        lengths = disc.facet_lengths.get(tag, [])
        for fn, _seg_len in zip(facets, lengths, strict=True):
            fn_arr = np.array(fn, dtype=np.int64)
            x_loc = disc.coords[fn_arr]  # (n_e, 2)
            u_loc = u_nodal[fn_arr]  # (n_e, 2)
            # Current coordinates x = X + u along edge
            x_cur = x_loc + u_loc
            # Tangent dx/dt along edge (oriented so outward normal = [dy/dt, -dx/dt])
            # Check orientation against reference outward normal
            dx_dt_ref = dn1d @ x_loc  # (Q1, 2)
            # Outward normal vector ds = [dy/dt, -dx/dt] * sign
            # For our facet ordering (a->b with outward normal = (ty, -tx)/L), sign = +1
            dx_dt_cur = dn1d @ x_cur  # (Q1, 2)
            n_ds = np.stack((dx_dt_cur[:, 1], -dx_dt_cur[:, 0]), axis=-1)
            # Verify sign matches reference outward normal
            ref_n = np.array([dx_dt_ref[0, 1], -dx_dt_ref[0, 0]])
            xa, ya = float(x_loc[0, 0]), float(x_loc[0, 1])
            xb, yb = float(x_loc[1, 0]), float(x_loc[1, 1])
            sign = 1.0 if (ref_n[0] * (yb - ya) - ref_n[1] * (xb - xa)) >= 0.0 else -1.0
            n_ds = sign * n_ds

            if is_axi:
                r_cur = (n1d @ x_cur)[:, 0]
                factor = 2.0 * math.pi * r_cur
            else:
                factor = np.ones(len(t_pts), dtype=np.float64)

            # External force: f_ext = -p * integral v . (n ds)
            # (positive pressure pushes INWARD along -n!)
            for q in range(len(t_pts)):
                wq = w_pts[q] * p_val * factor[q]
                for a in range(len(fn)):
                    f_fol[fn[a], 0] -= wq * n1d[q, a] * n_ds[q, 0]
                    f_fol[fn[a], 1] -= wq * n1d[q, a] * n_ds[q, 1]
                    # Residual R = f_int - f_ext => dR/du += -df_ext/du
                    # -df_ext_{a,x}/du_{b,y} = +p * w_q * factor * N_a * sign * dN_b/dt
                    # -df_ext_{a,y}/du_{b,x} = -p * w_q * factor * N_a * sign * dN_b/dt
                    for b in range(len(fn)):
                        k_xy = wq * sign * n1d[q, a] * dn1d[q, b]
                        rows.append(2 * fn[a])
                        cols.append(2 * fn[b] + 1)
                        vals.append(k_xy)
                        rows.append(2 * fn[a] + 1)
                        cols.append(2 * fn[b])
                        vals.append(-k_xy)
                        if is_axi:
                            # Extra dr_cur/du_r term in axisymmetric follower pressure
                            k_rr = (
                                w_pts[q]
                                * p_val
                                * (2.0 * math.pi)
                                * n1d[q, a]
                                * n1d[q, b]
                                * n_ds[q, 0]
                            )
                            k_zr = (
                                w_pts[q]
                                * p_val
                                * (2.0 * math.pi)
                                * n1d[q, a]
                                * n1d[q, b]
                                * n_ds[q, 1]
                            )
                            rows.append(2 * fn[a])
                            cols.append(2 * fn[b])
                            vals.append(k_rr)
                            rows.append(2 * fn[a] + 1)
                            cols.append(2 * fn[b])
                            vals.append(k_zr)

    k_fol = sparse.coo_matrix(
        (np.asarray(vals, dtype=np.float64), (rows, cols)),
        shape=(n_dofs, n_dofs),
    ).tocsc()
    return f_fol.reshape(-1), k_fol


@dataclass(frozen=True, slots=True)
class TorchAssemblyState:
    """Assembled internal force, tangent, energy, stress, and deformation gradient."""

    internal_force: FloatArray
    tangent: sparse.csc_matrix
    total_strain_energy: float
    first_piola: FloatArray  # (E, Q, 3, 3)
    deformation_gradient: FloatArray  # (E, Q, 3, 3)
    min_det_f: float

    @property
    def strain_energy(self) -> float:
        return self.total_strain_energy


@dataclass(frozen=True, slots=True)
class TorchLinearizedState:
    """Residual, tangent, and potential energy at a given displacement vector."""

    displacement: FloatArray
    assembly: TorchAssemblyState
    residual: FloatArray
    potential_energy: float
    external_force: FloatArray


def _normalize_material(material: MaterialSpec) -> MaterialSpec:
    if material.model == "finite_strain_plane_stress_neo_hookean":
        return replace(material, model="compressible_neo_hookean")
    return material


def _assemble_torch_internal(
    disc: _Discretization,
    request: FemRequest,
    u_vec: FloatArray,
    *,
    physics_mode: str,
    device: torch.device,
) -> TorchAssemblyState:
    """Vectorized PyTorch assembly of internal force, tangent, and strain energy."""
    material = _normalize_material(request.material)
    n_nodes = disc.coords.shape[0]
    n_elems, n_loc = disc.conn.shape
    n_q = disc.quad_xi.shape[0]
    dtype = torch.float64

    u_t = torch.tensor(u_vec.reshape(n_nodes, 2), dtype=dtype, device=device)
    conn_t = torch.tensor(disc.conn, dtype=torch.int64, device=device)
    dn_dx_t = torch.tensor(disc.dn_dx, dtype=dtype, device=device)
    n_ref_t = torch.tensor(disc.n_ref, dtype=dtype, device=device)
    w_area_t = torch.tensor(disc.quad_area_weights, dtype=dtype, device=device)

    u_elem = u_t[conn_t]  # (E, n_loc, 2)
    grad_u = torch.einsum("eaj,eqak->eqjk", u_elem, dn_dx_t)  # (E, Q, 2, 2)
    eye2 = torch.eye(2, dtype=dtype, device=device)
    f2d = eye2 + grad_u
    det_2d = f2d[..., 0, 0] * f2d[..., 1, 1] - f2d[..., 0, 1] * f2d[..., 1, 0]

    is_axi = physics_mode == "2d-axisymmetric-finite-strain"
    is_pstrain = physics_mode == "2d-plane-strain-finite-strain"

    if is_axi:
        r_t = torch.tensor(disc.quad_points[..., 0], dtype=dtype, device=device)
        if bool((r_t <= 0.0).any().item()):
            raise FemSolveFailure(
                FemFailureCode.UNSUPPORTED_REQUEST,
                "axisymmetric quadrature requires strictly positive radius R > 0",
            )
        u_r = torch.einsum("qa,ea->eq", n_ref_t, u_elem[..., 0])
        f33 = 1.0 + u_r / r_t
        det_3d = det_2d * f33
        w_vol_t = (2.0 * math.pi) * r_t * w_area_t
    elif is_pstrain:
        f33 = torch.ones_like(det_2d)
        det_3d = det_2d
        w_vol_t = w_area_t
    else:
        if bool((det_2d <= 1.0e-12).any().item()):
            raise FemSolveFailure(
                FemFailureCode.NEGATIVE_JACOBIAN,
                f"inadmissible 2D Jacobian min det(F_2D)={float(det_2d.min().item()):.3e}",
            )
        f33 = plane_stress_closure(
            f2d.reshape(-1, 2, 2),
            material,
            max_iterations=request.config.plane_stress_max_iterations,
        ).reshape(n_elems, n_q)
        det_3d = det_2d * f33
        w_vol_t = w_area_t

    min_det = float(det_3d.min().item())
    if not math.isfinite(min_det) or min_det <= 1.0e-12:
        raise FemSolveFailure(
            FemFailureCode.NEGATIVE_JACOBIAN,
            f"inadmissible 3D Jacobian min det(F)={min_det:.3e}",
        )

    f3d = torch.zeros((n_elems, n_q, 3, 3), dtype=dtype, device=device)
    f3d[..., :2, :2] = f2d
    f3d[..., 2, 2] = f33
    f3d_flat = f3d.reshape(-1, 3, 3)

    with torch.no_grad():
        psi_t = energy_density(f3d_flat, material).reshape(n_elems, n_q)
        p3d_t = first_piola(f3d_flat, material).reshape(n_elems, n_q, 3, 3)
        a4d_t = consistent_tangent(f3d_flat, material).reshape(n_elems, n_q, 3, 3, 3, 3)

    strain_energy = float((w_vol_t * psi_t).sum().item())
    p2d_t = p3d_t[..., :2, :2]
    f_elem = torch.einsum("eq,eqij,eqaj->eai", w_vol_t, p2d_t, dn_dx_t)

    if physics_mode == "2d-plane-stress-finite-strain":
        a3333 = a4d_t[..., 2, 2, 2, 2]
        a_ij33 = a4d_t[..., :2, :2, 2, 2]
        a_33kl = a4d_t[..., 2, 2, :2, :2]
        a_eff = (
            a4d_t[..., :2, :2, :2, :2]
            - (a_ij33[..., :, :, None, None] * a_33kl[..., None, None, :, :])
            / a3333[..., None, None, None, None]
        )
        k_elem = torch.einsum(
            "eq,eqaj,eqijkl,eqbl->eaibk", w_vol_t, dn_dx_t, a_eff, dn_dx_t
        )
    elif is_pstrain:
        a_eff = a4d_t[..., :2, :2, :2, :2]
        k_elem = torch.einsum(
            "eq,eqaj,eqijkl,eqbl->eaibk", w_vol_t, dn_dx_t, a_eff, dn_dx_t
        )
    else:
        # Axisymmetric internal force hoop term + exact hoop tangent couplings
        inv_r = 1.0 / r_t
        nr_inv = torch.einsum("qa,eq->eqa", n_ref_t, inv_r)  # N_a / R
        f_elem[..., 0] = f_elem[..., 0] + torch.einsum(
            "eq,eq,eqa->ea", w_vol_t, p3d_t[..., 2, 2], nr_inv
        )
        a_2d = a4d_t[..., :2, :2, :2, :2]
        k_elem = torch.einsum(
            "eq,eqaj,eqijkl,eqbl->eaibk", w_vol_t, dn_dx_t, a_2d, dn_dx_t
        )
        # + delta_{i,0} * (N_a/R) * A_{22,kL} * dN_b/dX_L
        k_elem[:, :, 0, :, :] = k_elem[:, :, 0, :, :] + torch.einsum(
            "eq,eqa,eqkl,eqbl->eabk",
            w_vol_t,
            nr_inv,
            a4d_t[..., 2, 2, :2, :2],
            dn_dx_t,
        )
        # + delta_{k,0} * dN_a/dX_J * A_{iJ,22} * (N_b/R)
        k_elem[:, :, :, :, 0] = k_elem[:, :, :, :, 0] + torch.einsum(
            "eq,eqaj,eqij,eqb->eaib",
            w_vol_t,
            dn_dx_t,
            a4d_t[..., :2, :2, 2, 2],
            nr_inv,
        )
        # + delta_{i,0} * delta_{k,0} * A_{2222} * (N_a/R) * (N_b/R)
        k_elem[:, :, 0, :, 0] = k_elem[:, :, 0, :, 0] + torch.einsum(
            "eq,eq,eqa,eqb->eab",
            w_vol_t,
            a4d_t[..., 2, 2, 2, 2],
            nr_inv,
            nr_inv,
        )

    f_int_nodal = np.zeros((n_nodes, 2), dtype=np.float64)
    np.add.at(f_int_nodal, disc.conn, f_elem.detach().cpu().numpy())

    # Assemble sparse CSC global tangent
    dofs_elem = np.empty((n_elems, n_loc * 2), dtype=np.int64)
    dofs_elem[:, 0::2] = 2 * disc.conn
    dofs_elem[:, 1::2] = 2 * disc.conn + 1
    k_np = k_elem.detach().cpu().numpy().reshape(n_elems, n_loc * 2, n_loc * 2)
    row_idx = np.broadcast_to(dofs_elem[:, :, None], k_np.shape).ravel()
    col_idx = np.broadcast_to(dofs_elem[:, None, :], k_np.shape).ravel()
    n_dofs = 2 * n_nodes
    tangent = sparse.coo_matrix(
        (k_np.ravel(), (row_idx, col_idx)), shape=(n_dofs, n_dofs)
    ).tocsc()

    return TorchAssemblyState(
        internal_force=f_int_nodal.reshape(-1),
        tangent=tangent,
        total_strain_energy=strain_energy,
        first_piola=p3d_t.detach().cpu().numpy(),
        deformation_gradient=f3d.detach().cpu().numpy(),
        min_det_f=min_det,
    )


def _recover_nodal_stress(disc: _Discretization, p3d: FloatArray) -> FloatArray:
    """Area-weighted nodal patch recovery of 3x3 First Piola stress."""
    n_nodes = disc.coords.shape[0]
    numer = np.zeros((n_nodes, 3, 3), dtype=np.float64)
    denom = np.zeros(n_nodes, dtype=np.float64)
    # Element-averaged stress weighted by element area
    elem_area = disc.quad_area_weights.sum(axis=1)  # (E,)
    elem_p = (
        np.einsum("eq,eqij->eij", disc.quad_area_weights, p3d)
        / elem_area[:, None, None]
    )
    for loc in range(disc.conn.shape[1]):
        nodes = disc.conn[:, loc]
        np.add.at(numer, nodes, elem_area[:, None, None] * elem_p)
        np.add.at(denom, nodes, elem_area)
    return np.asarray(
        numer / np.maximum(denom, 1.0e-30)[:, None, None], dtype=np.float64
    )


class TorchFemLinearization:
    """``DiscreteLinearization`` implementation backed by PyTorch vectorized assembly."""

    def __init__(
        self,
        request: FemRequest,
        *,
        physics_mode: str = "2d-plane-stress-finite-strain",
        device: str = "cpu",
        backend: TorchFemBackend | None = None,
    ) -> None:
        self.request = request
        self.material = _normalize_material(request.material)
        self.physics_mode = physics_mode
        self.device = torch.device(device)
        self.disc = _build_discretization(request.mesh, request.config.element_order)
        self.basis = _TorchBasisShim(self.disc)
        self.constraints = _bind_torch_constraints(
            self.disc,
            request.boundary_conditions,
            dirichlet_providers=(
                backend._dirichlet_field_providers if backend is not None else None
            ),
            provider_ids=request.boundary_field_provider_ids,
        )
        self.dead_external_force = _assemble_dead_external_force(
            self.disc,
            request.loads,
            physics_mode=physics_mode,
            body_force_providers=(
                backend._body_force_providers if backend is not None else None
            ),
            body_force_provider_ids=request.body_force_provider_ids,
            traction_providers=(
                backend._traction_field_providers if backend is not None else None
            ),
            traction_provider_ids=request.traction_field_provider_ids,
        )

    @property
    def dof_coordinates(self) -> FloatArray:
        return self.disc.coords.copy()

    @property
    def dof_count(self) -> int:
        return int(self.disc.coords.shape[0] * 2)

    @property
    def has_global_constraints(self) -> bool:
        return False

    @property
    def is_tangent_symmetric(self) -> bool:
        return not any(load.kind == "follower_pressure" for load in self.request.loads)

    def vector(self, nodal: FloatArray) -> FloatArray:
        arr = np.asarray(nodal, dtype=np.float64)
        if arr.shape != (self.disc.coords.shape[0], 2):
            raise ValueError(
                f"expected nodal shape {(self.disc.coords.shape[0], 2)}, got {arr.shape}"
            )
        return arr.reshape(-1).copy()

    def nodal(self, vector: FloatArray) -> FloatArray:
        arr = np.asarray(vector, dtype=np.float64)
        return arr.reshape(self.disc.coords.shape[0], 2).copy()

    def dirichlet_deviation(self, vector: FloatArray) -> float:
        if self.constraints.dirichlet_dofs.size == 0:
            return 0.0
        diff = (
            vector[self.constraints.dirichlet_dofs] - self.constraints.dirichlet_values
        )
        return float(np.linalg.norm(diff, ord=np.inf))

    def with_exact_dirichlet(self, vector: FloatArray) -> FloatArray:
        out = np.asarray(vector, dtype=np.float64).copy()
        if self.constraints.dirichlet_dofs.size:
            out[self.constraints.dirichlet_dofs] = self.constraints.dirichlet_values
        return out

    def state(
        self, vector: FloatArray, *, load_factor: float = 1.0
    ) -> TorchLinearizedState:
        vec = np.asarray(vector, dtype=np.float64)
        asm = _assemble_torch_internal(
            self.disc,
            self.request,
            vec,
            physics_mode=self.physics_mode,
            device=self.device,
        )
        f_ext = load_factor * self.dead_external_force
        tangent = asm.tangent
        if not self.is_tangent_symmetric:
            f_fol, k_fol = _assemble_follower_pressure(
                self.disc,
                self.request.loads,
                vec.reshape(-1, 2),
                physics_mode=self.physics_mode,
                load_factor=load_factor,
            )
            f_ext = f_ext + f_fol
            tangent = (tangent + k_fol).tocsc()
            asm = TorchAssemblyState(
                internal_force=asm.internal_force,
                tangent=tangent,
                total_strain_energy=asm.total_strain_energy,
                first_piola=asm.first_piola,
                deformation_gradient=asm.deformation_gradient,
                min_det_f=asm.min_det_f,
            )
        residual = asm.internal_force - f_ext
        ext_work = float(np.dot(f_ext, vec))
        return TorchLinearizedState(
            displacement=vec.copy(),
            assembly=asm,
            residual=residual,
            potential_energy=asm.total_strain_energy - ext_work,
            external_force=f_ext,
        )

    def free_tangent(self, state: TorchLinearizedState) -> sparse.csc_matrix:
        free = self.constraints.free_dofs
        return state.assembly.tangent[free, :][:, free].tocsc()

    def newton_increment(self, state: TorchLinearizedState) -> FloatArray:
        free = self.constraints.free_dofs
        k_ff = self.free_tangent(state)
        rhs = -state.residual[free]
        du_free = np.asarray(spsolve(k_ff, rhs), dtype=np.float64)
        du = np.zeros(self.dof_count, dtype=np.float64)
        du[free] = du_free
        return du

    def solve_adjoint(
        self, state: TorchLinearizedState, rhs_free: FloatArray
    ) -> FloatArray:
        k_ff_t = self.free_tangent(state).T.tocsc()
        return np.asarray(
            spsolve(k_ff_t, np.asarray(rhs_free, dtype=np.float64)),
            dtype=np.float64,
        )

    def energy_norm(self, difference: FloatArray, state: TorchLinearizedState) -> float:
        diff = np.asarray(difference, dtype=np.float64)
        k_sym = 0.5 * (state.assembly.tangent + state.assembly.tangent.T)
        val = float(diff @ (k_sym @ diff))
        return math.sqrt(max(val, 0.0))

    def recovered_stress(self, state: TorchLinearizedState) -> FloatArray:
        return _recover_nodal_stress(self.disc, state.assembly.first_piola)

    def qois(self, state: TorchLinearizedState) -> Mapping[str, float]:
        rec = self.recovered_stress(state)
        out: dict[str, float] = {}
        for qoi in self.request.qois:
            if qoi.field in ("u", "displacement"):
                out[qoi.name] = float(
                    np.linalg.norm(self.nodal(state.displacement), axis=1).max()
                )
            elif qoi.field == "total_strain_energy":
                out[qoi.name] = float(state.assembly.total_strain_energy)
            else:
                frob = np.linalg.norm(rec[:, :2, :2], axis=(-2, -1))
                out[qoi.name] = float(frob.max())
        return out

    def quadrature_weights(self) -> FloatArray:
        if self.physics_mode == "2d-axisymmetric-finite-strain":
            return (
                (2.0 * math.pi)
                * self.disc.quad_points[..., 0]
                * self.disc.quad_area_weights
            ).copy()
        return self.disc.quad_area_weights.copy()

    def assemble_grad_weighted_linear_form(self, dq_df: FloatArray) -> FloatArray:
        """Assemble ``int dq_df_{iJ} * grad(v)_{iJ} dOmega`` for adjoint DWR QoIs."""
        wq = self.quadrature_weights()
        f_elem = np.einsum("eq,eqij,eqaj->eai", wq, dq_df, self.disc.dn_dx)
        out = np.zeros((self.disc.coords.shape[0], 2), dtype=np.float64)
        np.add.at(out, self.disc.conn, f_elem)
        return out.reshape(-1)


class TorchFemBackend:
    """PyTorch-native vectorized differentiable FEM solver satisfying ``SolverBackend``."""

    backend_name: str = "torch-fem"
    backend_id: str = "torch-fem-v1"
    capabilities: BackendCapabilities = BackendCapabilities(
        spatial_dimensions=(2, 3),
        physics_modes=TORCH_PHYSICS_MODES,
        element_orders=(1, 2),
        supports_warm_start=True,
        supports_linearization=True,
        supports_dead_tractions=True,
        supports_follower_pressure=True,
        supports_contact=True,
        supports_gpu=True,
        supports_mpi=False,
        license_spdx="MIT",
    )

    def __init__(
        self,
        *,
        physics_mode: str = "2d-plane-stress-finite-strain",
        device: str = "cpu",
        body_force_providers: Mapping[str, Any] | None = None,
        dirichlet_field_providers: Mapping[str, Any] | None = None,
        traction_field_providers: Mapping[str, Any] | None = None,
    ) -> None:
        if physics_mode not in TORCH_PHYSICS_MODES:
            raise ValueError(
                f"physics_mode must be one of {TORCH_PHYSICS_MODES}, got {physics_mode!r}"
            )
        self.physics_mode = physics_mode
        self.device = device
        self._body_force_providers = dict(body_force_providers or {})
        self._dirichlet_field_providers = dict(dirichlet_field_providers or {})
        self._traction_field_providers = dict(traction_field_providers or {})

    def linearize(self, request: FemRequest) -> TorchFemLinearization:
        return TorchFemLinearization(
            request,
            physics_mode=self.physics_mode,
            device=self.device,
            backend=self,
        )

    def _solve_newton_continuation(
        self,
        request: FemRequest,
        initial: FloatArray | None,
        load_factors: Sequence[float],
    ) -> FemAttempt:
        try:
            lin = self.linearize(request)
            if lin.dof_count > request.config.max_dofs:
                raise FemSolveFailure(
                    FemFailureCode.RESOURCE_EXHAUSTION,
                    f"DOF count {lin.dof_count} exceeds max_dofs={request.config.max_dofs}",
                )
            u = (
                np.zeros(lin.dof_count, dtype=np.float64)
                if initial is None
                else np.asarray(initial, dtype=np.float64).copy()
            )
            history: list[FemConvergenceEntry] = []
            free = lin.constraints.free_dofs
            d_dofs = lin.constraints.dirichlet_dofs
            d_vals = lin.constraints.dirichlet_values
            tol = request.config.solver_tolerance
            max_it = request.config.max_newton_iterations
            final_state: TorchLinearizedState | None = None

            for step_idx, lam in enumerate(load_factors, start=1):
                if d_dofs.size:
                    u[d_dofs] = float(lam) * d_vals
                state = lin.state(u, load_factor=float(lam))
                res_norm = float(np.linalg.norm(state.residual[free]))
                ref_scale = max(float(np.linalg.norm(state.external_force[free])), 1.0)
                converged = res_norm <= tol * ref_scale
                for it in range(1, max_it + 1):
                    if converged:
                        break
                    du = lin.newton_increment(state)
                    if not np.all(np.isfinite(du)):
                        raise FemSolveFailure(
                            FemFailureCode.NEWTON_NON_CONVERGENCE,
                            "non-finite Newton increment",
                        )
                    alpha = 1.0
                    accepted = False
                    next_state: TorchLinearizedState | None = None
                    while alpha >= 2.0**-10:
                        cand = u + alpha * du
                        try:
                            cand_state = lin.state(cand, load_factor=float(lam))
                        except FemSolveFailure:
                            alpha *= 0.5
                            continue
                        cand_res = float(np.linalg.norm(cand_state.residual[free]))
                        if (
                            cand_res < res_norm * (1.0 - 1.0e-4 * alpha)
                            or cand_res <= tol * ref_scale
                        ):
                            u = cand
                            state = cand_state
                            res_norm = cand_res
                            next_state = cand_state
                            accepted = True
                            break
                        alpha *= 0.5
                    if not accepted or next_state is None:
                        # Take smallest admissible damped step if line search stalls
                        cand = u + 0.25 * du
                        state = lin.state(cand, load_factor=float(lam))
                        u = cand
                        res_norm = float(np.linalg.norm(state.residual[free]))
                    inc_norm = float(np.linalg.norm(alpha * du[free]))
                    history.append(
                        FemConvergenceEntry(
                            load_step=step_idx,
                            iteration=it,
                            residual_norm=res_norm,
                            increment_norm=inc_norm,
                        )
                    )
                    if res_norm <= tol * ref_scale and inc_norm <= tol * max(
                        float(np.linalg.norm(u[free])), 1.0
                    ):
                        converged = True
                        break
                if not converged:
                    raise FemSolveFailure(
                        FemFailureCode.NEWTON_NON_CONVERGENCE,
                        f"Newton failed to converge at step {step_idx} (res={res_norm:.3e})",
                    )
                final_state = state

            assert final_state is not None
            rec = lin.recovered_stress(final_state)
            reactions: dict[str, tuple[float, float]] = {}
            for tag, tdofs in lin.constraints.boundary_dofs.items():
                rx = float(final_state.residual[tdofs[tdofs % 2 == 0]].sum())
                ry = float(final_state.residual[tdofs[tdofs % 2 == 1]].sum())
                reactions[tag] = (rx, ry)

            q_pts = tuple(
                tuple((float(p[0]), float(p[1])) for p in elem)
                for elem in lin.disc.quad_points
            )
            q_w = tuple(
                tuple(float(w) for w in elem) for elem in lin.quadrature_weights()
            )
            p_out = tuple(
                tuple(
                    (
                        (float(m[0, 0]), float(m[0, 1]), float(m[0, 2])),
                        (float(m[1, 0]), float(m[1, 1]), float(m[1, 2])),
                        (float(m[2, 0]), float(m[2, 1]), float(m[2, 2])),
                    )
                    for m in elem
                )
                for elem in final_state.assembly.first_piola
            )
            f_out = tuple(
                tuple(
                    (
                        (float(m[0, 0]), float(m[0, 1]), float(m[0, 2])),
                        (float(m[1, 0]), float(m[1, 1]), float(m[1, 2])),
                        (float(m[2, 0]), float(m[2, 1]), float(m[2, 2])),
                    )
                    for m in elem
                )
                for elem in final_state.assembly.deformation_gradient
            )
            rec_out = tuple(
                (
                    (float(m[0, 0]), float(m[0, 1]), float(m[0, 2])),
                    (float(m[1, 0]), float(m[1, 1]), float(m[1, 2])),
                    (float(m[2, 0]), float(m[2, 1]), float(m[2, 2])),
                )
                for m in rec
            )
            ext_work = float(
                np.dot(final_state.external_force, final_state.displacement)
            )
            result = FemResult(
                request_id=request.request_id,
                mesh_id=request.mesh.mesh_id,
                element_order=request.config.element_order,
                displacement_dof_coordinates=tuple(
                    (float(c[0]), float(c[1])) for c in lin.disc.coords
                ),
                displacement=tuple(
                    (float(r[0]), float(r[1]))
                    for r in lin.nodal(final_state.displacement)
                ),
                quadrature_points=q_pts,
                quadrature_weights=q_w,
                first_piola_stress=p_out,
                deformation_gradient=f_out,
                recovered_first_piola_stress=rec_out,
                reactions=reactions,
                rigid_mode_multipliers=(),
                total_strain_energy=float(final_state.assembly.total_strain_energy),
                external_work=ext_work,
                total_potential_energy=float(
                    final_state.assembly.total_strain_energy - ext_work
                ),
                convergence_history=tuple(history),
                qois=dict(lin.qois(final_state)),
            )
            return FemAttempt(
                request=request,
                recipe_hash=request.recipe_hash,
                success=True,
                result=result,
                result_hash=hash_spec(result, "content"),
                store_content_hash=None,
                failure_code=None,
                message="",
            )
        except FemSolveFailure as exc:
            return FemAttempt(
                request=request,
                recipe_hash=request.recipe_hash,
                success=False,
                result=None,
                result_hash=None,
                store_content_hash=None,
                failure_code=exc.code,
                message=str(exc),
            )

    def solve_many(self, requests: tuple[FemRequest, ...]) -> tuple[FemAttempt, ...]:
        results: list[FemAttempt] = []
        for req in requests:
            steps = req.config.load_steps
            factors = tuple((i + 1) / steps for i in range(steps))
            results.append(self._solve_newton_continuation(req, None, factors))
        return tuple(results)

    def solve_from(
        self,
        request: FemRequest,
        initial: FloatArray,
        load_factors: Sequence[float] = (1.0,),
    ) -> FemAttempt:
        return self._solve_newton_continuation(request, initial, tuple(load_factors))

    def solve_contact(
        self,
        request: FemRequest,
        indenter: RigidIndenterSpec,
        *,
        max_auglag_iterations: int = 8,
        gap_tolerance_mm: float = 1.0e-5,
    ) -> tuple[FloatArray, ContactVerificationReport]:
        """Solve a frictionless rigid-indenter contact problem via Augmented Lagrangian."""
        lin = self.linearize(request)
        if indenter.contact_boundary_tag is not None:
            facets = lin.disc.facet_nodes.get(indenter.contact_boundary_tag, [])
            cand = np.array(sorted({n for f in facets for n in f}), dtype=np.int64)
        else:
            cand = np.array(
                sorted(
                    {
                        n
                        for flist in lin.disc.facet_nodes.values()
                        for f in flist
                        for n in f
                    }
                ),
                dtype=np.int64,
            )
        if cand.size == 0:
            raise ValueError("contact candidate node set is empty")

        u: FloatArray = np.zeros(lin.dof_count, dtype=np.float64)
        free = lin.constraints.free_dofs
        d_dofs = lin.constraints.dirichlet_dofs
        d_vals = lin.constraints.dirichlet_values
        if d_dofs.size:
            u[d_dofs] = d_vals

        multipliers: FloatArray = np.zeros(cand.size, dtype=np.float64)
        tol = request.config.solver_tolerance
        auglag_used = 0
        inner_converged = False

        for al_it in range(1, max_auglag_iterations + 1):
            auglag_used = al_it
            inner_converged = False
            for _ in range(request.config.max_newton_iterations):
                state = lin.state(u, load_factor=1.0)
                e_c, f_c, k_c_blocks, _ = assemble_contact_terms(
                    indenter,
                    lin.disc.coords,
                    u.reshape(-1, 2),
                    cand,
                    multipliers=multipliers,
                )
                res = state.residual + f_c.reshape(-1)
                res_norm = float(np.linalg.norm(res[free]))
                if res_norm <= tol * max(
                    float(np.linalg.norm(state.external_force[free])), 1.0
                ):
                    inner_converged = True
                    break
                e_curr = (
                    state.assembly.total_strain_energy
                    - float(np.dot(u, state.external_force))
                    + e_c
                )
                # Add nodal 2x2 contact Hessian blocks into global sparse tangent
                rows: list[int] = []
                cols: list[int] = []
                vals: list[float] = []
                for idx, node in enumerate(cand):
                    blk = k_c_blocks[idx]
                    if np.any(blk != 0.0):
                        for r_i in range(2):
                            for c_j in range(2):
                                rows.append(2 * int(node) + r_i)
                                cols.append(2 * int(node) + c_j)
                                vals.append(float(blk[r_i, c_j]))
                k_total = state.assembly.tangent
                if vals:
                    k_total = (
                        k_total
                        + sparse.coo_matrix(
                            (vals, (rows, cols)),
                            shape=(lin.dof_count, lin.dof_count),
                        ).tocsc()
                    )
                k_ff = k_total[free, :][:, free].tocsc()
                du_free = np.asarray(spsolve(k_ff, -res[free]), dtype=np.float64)
                du = np.zeros(lin.dof_count, dtype=np.float64)
                du[free] = du_free
                slope = float(np.dot(res[free], du_free))
                alpha = 1.0
                best_u = u
                while alpha >= 2.0**-12:
                    u_trial = u + alpha * du
                    if d_dofs.size:
                        u_trial[d_dofs] = d_vals
                    try:
                        st_tr = lin.state(u_trial, load_factor=1.0)
                        e_c_tr, _, _, _ = assemble_contact_terms(
                            indenter,
                            lin.disc.coords,
                            u_trial.reshape(-1, 2),
                            cand,
                            multipliers=multipliers,
                        )
                        e_tr = (
                            st_tr.assembly.total_strain_energy
                            - float(np.dot(u_trial, st_tr.external_force))
                            + e_c_tr
                        )
                        best_u = u_trial
                        if slope >= 0.0 or e_tr <= e_curr + 1.0e-4 * alpha * slope:
                            break
                    except FemSolveFailure:
                        pass
                    alpha *= 0.5
                u = best_u
                if d_dofs.size:
                    u[d_dofs] = d_vals

            _, f_c_final, _, updated_press = assemble_contact_terms(
                indenter,
                lin.disc.coords,
                u.reshape(-1, 2),
                cand,
                multipliers=multipliers,
            )
            multipliers = updated_press
            deformed = lin.disc.coords[cand] + u.reshape(-1, 2)[cand]
            gap, _, _ = indenter.gap_normal_curvature(deformed)
            min_gap = float(gap.min())
            max_pen = float(max(0.0, -min_gap))
            if inner_converged and max_pen <= gap_tolerance_mm:
                break

        deformed = lin.disc.coords[cand] + u.reshape(-1, 2)[cand]
        gap, _, _ = indenter.gap_normal_curvature(deformed)
        min_gap = float(gap.min())
        max_pen = float(max(0.0, -min_gap))
        _, f_c_end, _, end_press = assemble_contact_terms(
            indenter,
            lin.disc.coords,
            u.reshape(-1, 2),
            cand,
            multipliers=multipliers,
        )
        # Contact force applied BY indenter ON body = -f_c_end
        total_f = (-f_c_end).sum(axis=0)
        report = ContactVerificationReport(
            min_gap_mm=min_gap,
            max_penetration_mm=max_pen,
            active_node_count=int(np.sum(end_press > 0.0)),
            total_contact_force_n=(float(total_f[0]), float(total_f[1])),
            auglag_iterations=auglag_used,
            admissible=inner_converged
            and max_pen <= max(gap_tolerance_mm * 10.0, 1.0e-3),
        )
        return u.reshape(-1, 2), report


__all__ = (
    "TORCH_PHYSICS_MODES",
    "TorchAssemblyState",
    "TorchFemBackend",
    "TorchFemLinearization",
    "TorchLinearizedState",
)
