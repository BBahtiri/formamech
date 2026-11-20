"""Permutation-invariant 2D CAD boundary tokenization and geometry-level OOD scoring.

Extracts canonical, measure-weighted boundary point clouds with outward unit
normals, normalized discrete curvature, essential-boundary distance, and closed
loop topology markers directly from ``CanonicalMesh`` data for
:class:`~piginot.operators.boundary_energy_operator.BoundaryEnergyOperator`.

Geometry-level OOD scoring operates independently of parameter-envelope gates
and explicitly refuses unsupported boundary topology (open curves or
non-manifold boundary vertices).
"""

from __future__ import annotations

import enum
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import torch

from piginot.contracts import CanonicalMesh, canonical_json_bytes
from piginot.operators.energy_operator import QueryFrame

GEOMETRY_OOD_SCHEMA: Final = "piginot.geometry-ood-report-v1"
GEOMETRY_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "curvature_norm",
    "is_hole_loop",
    "bc_dirichlet_x",
    "bc_dirichlet_y",
    "bc_symmetry",
    "bc_traction_free",
    "dist_to_essential_norm",
)
SUPPORTED_BC_ROLES: Final = frozenset(
    ("dirichlet_xy", "dirichlet_x", "dirichlet_y", "symmetry", "traction_free")
)


class UnsupportedTopologyError(ValueError):
    """Raised when a 2D boundary fails closed-manifold loop admissibility."""


@dataclass(frozen=True, slots=True)
class BoundaryRoleMap:
    """Explicit mapping from every semantic boundary group to a supported BC role."""

    roles: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.roles, Mapping) or not self.roles:
            raise ValueError("roles must be a non-empty mapping of boundary groups")
        cleaned: dict[str, str] = {}
        for group, role in self.roles.items():
            if not isinstance(group, str) or not group.strip():
                raise ValueError("boundary group names must be non-empty text")
            if role not in SUPPORTED_BC_ROLES:
                raise ValueError(
                    f"unsupported boundary role {role!r} for group {group!r}; "
                    f"expected one of {sorted(SUPPORTED_BC_ROLES)}"
                )
            cleaned[group] = role
        object.__setattr__(self, "roles", cleaned)

    def role_flags(self, role: str) -> tuple[float, float, float, float]:
        """Return ``(dirichlet_x, dirichlet_y, symmetry, traction_free)``."""
        if role == "dirichlet_xy":
            return (1.0, 1.0, 0.0, 0.0)
        if role == "dirichlet_x":
            return (1.0, 0.0, 0.0, 0.0)
        if role == "dirichlet_y":
            return (0.0, 1.0, 0.0, 0.0)
        if role == "symmetry":
            return (0.0, 0.0, 1.0, 0.0)
        return (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class GeometryTopologySummary:
    """Closed-loop topology invariants extracted from one planar mesh."""

    loop_count: int
    hole_count: int
    outer_loop_area_mm2: float
    domain_area_mm2: float
    perimeter_mm: float
    aspect_ratio: float
    extent_x_mm: float
    extent_y_mm: float
    origin_x_mm: float
    origin_y_mm: float


@dataclass(frozen=True, slots=True)
class BoundaryTokenBatch:
    """Padded batch of measure-weighted boundary tensors for ``BoundaryEnergyOperator``."""

    boundary_xy: torch.Tensor
    boundary_normals: torch.Tensor
    boundary_features: torch.Tensor
    weights: torch.Tensor
    mask: torch.Tensor
    frame: QueryFrame
    mesh_ids: tuple[str, ...]
    topologies: tuple[GeometryTopologySummary, ...]


def _mesh_area(mesh: CanonicalMesh) -> float:
    total = 0.0
    for a, b, c in mesh.cells:
        ax, ay = mesh.nodes[a]
        bx, by = mesh.nodes[b]
        cx, cy = mesh.nodes[c]
        area = 0.5 * ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
        if area <= 0.0:
            raise UnsupportedTopologyError("mesh contains non-positive triangle area")
        total += area
    return total


def _facet_owner_interior_point(
    mesh: CanonicalMesh,
) -> dict[tuple[int, int], tuple[float, float]]:
    """Map each undirected boundary edge to its owning triangle's opposite vertex."""
    owners: dict[tuple[int, int], list[int]] = {}
    for a, b, c in mesh.cells:
        for u, v, opp in ((a, b, c), (b, c, a), (c, a, b)):
            key = (u, v) if u < v else (v, u)
            owners.setdefault(key, []).append(opp)
    interior: dict[tuple[int, int], tuple[float, float]] = {}
    for a, b in mesh.boundary_facets:
        key = (a, b) if a < b else (b, a)
        opps = owners.get(key, [])
        if len(opps) != 1:
            raise UnsupportedTopologyError(
                f"boundary facet {key} has {len(opps)} adjacent cells (expected 1)"
            )
        interior[key] = mesh.nodes[opps[0]]
    return interior


def _extract_closed_loops(
    mesh: CanonicalMesh,
) -> tuple[list[list[int]], dict[int, int], dict[int, bool]]:
    """Decompose boundary facets into closed manifold loops and tag hole loops.

    Returns ``(loops_of_vertices, facet_to_loop, loop_is_hole)``.
    """
    adj: dict[int, list[tuple[int, int]]] = {}
    for f_idx, (a, b) in enumerate(mesh.boundary_facets):
        if a == b:
            raise UnsupportedTopologyError("degenerate zero-length boundary facet")
        adj.setdefault(a, []).append((b, f_idx))
        adj.setdefault(b, []).append((a, f_idx))
    for node, nbrs in adj.items():
        if len(nbrs) != 2:
            raise UnsupportedTopologyError(
                f"boundary vertex {node} has degree {len(nbrs)} (expected 2)"
            )

    visited_facets: set[int] = set()
    loops: list[list[int]] = []
    facet_to_loop: dict[int, int] = {}

    for start_f_idx, (start_u, start_v) in enumerate(mesh.boundary_facets):
        if start_f_idx in visited_facets:
            continue
        loop_id = len(loops)
        visited_facets.add(start_f_idx)
        facet_to_loop[start_f_idx] = loop_id
        loop_verts = [start_u, start_v]
        curr = start_v
        prev_facet = start_f_idx
        while curr != start_u:
            candidates = [(nxt, f_id) for nxt, f_id in adj[curr] if f_id != prev_facet]
            if len(candidates) != 1:  # pragma: no cover - guarded by degree==2
                raise UnsupportedTopologyError("non-manifold boundary loop traversal")
            nxt, f_id = candidates[0]
            if f_id in visited_facets:
                raise UnsupportedTopologyError("unexpected cycle before closing loop")
            visited_facets.add(f_id)
            facet_to_loop[f_id] = loop_id
            prev_facet = f_id
            curr = nxt
            if curr != start_u:
                loop_verts.append(curr)
        loops.append(loop_verts)

    if not loops:
        raise UnsupportedTopologyError("mesh has no boundary loops")

    def polygon_abs_area(verts: list[int]) -> float:
        s = 0.0
        n = len(verts)
        for i in range(n):
            x0, y0 = mesh.nodes[verts[i]]
            x1, y1 = mesh.nodes[verts[(i + 1) % n]]
            s += x0 * y1 - x1 * y0
        return 0.5 * abs(s)

    loop_areas = [polygon_abs_area(loop) for loop in loops]
    outer_idx = max(range(len(loops)), key=lambda i: loop_areas[i])
    loop_is_hole = {i: (i != outer_idx) for i in range(len(loops))}
    return loops, facet_to_loop, loop_is_hole


def _vertex_curvatures(
    mesh: CanonicalMesh, loops: Sequence[Sequence[int]], char_length: float
) -> dict[int, float]:
    """Scale-normalized discrete unsigned turning curvature ``|dtheta| / ds * L``."""
    curvatures: dict[int, float] = {}
    for loop in loops:
        n = len(loop)
        for i in range(n):
            p_prev = mesh.nodes[loop[(i - 1) % n]]
            p_curr = mesh.nodes[loop[i]]
            p_next = mesh.nodes[loop[(i + 1) % n]]
            v1x, v1y = p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]
            v2x, v2y = p_next[0] - p_curr[0], p_next[1] - p_curr[1]
            l1 = math.hypot(v1x, v1y)
            l2 = math.hypot(v2x, v2y)
            if l1 <= 0.0 or l2 <= 0.0:
                curvatures[loop[i]] = 0.0
                continue
            cos_t = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (l1 * l2)))
            angle = math.acos(cos_t)
            ds = 0.5 * (l1 + l2)
            # Bounded normalized curvature in [0, inf) scaled by characteristic extent.
            curvatures[loop[i]] = (angle / ds) * char_length
    return curvatures


def _facet_roles(mesh: CanonicalMesh, role_map: BoundaryRoleMap) -> list[str]:
    mesh_groups = set(mesh.semantic_tags)
    declared_groups = set(role_map.roles)
    missing = sorted(mesh_groups - declared_groups)
    if missing:
        raise ValueError(
            f"every boundary group must have a declared role; missing: {missing}"
        )
    assigned: list[str | None] = [None] * len(mesh.boundary_facets)
    for group, facet_indices in mesh.semantic_tags.items():
        role = role_map.roles[group]
        for idx in facet_indices:
            if assigned[idx] is not None and assigned[idx] != role:
                raise ValueError(f"boundary facet {idx} has conflicting roles")
            assigned[idx] = role
    if any(r is None for r in assigned):
        raise ValueError("unclassified boundary facet found in mesh")
    return [r for r in assigned if r is not None]


@dataclass(frozen=True, slots=True)
class GeometryTokenizer:
    """Converts planar ``CanonicalMesh`` objects into canonical boundary token batches."""

    subdivisions_per_facet: int = 1

    def __post_init__(self) -> None:
        if (
            type(self.subdivisions_per_facet) is not int
            or self.subdivisions_per_facet < 1
        ):
            raise ValueError("subdivisions_per_facet must be a positive integer")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return GEOMETRY_FEATURE_NAMES

    def tokenize_one(
        self,
        mesh: CanonicalMesh,
        role_map: BoundaryRoleMap,
        *,
        dtype: torch.dtype = torch.float64,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        QueryFrame,
        GeometryTopologySummary,
    ]:
        """Extract ``(xy, normals, features, weights, frame, topology)`` for one mesh."""
        domain_area = _mesh_area(mesh)
        interior = _facet_owner_interior_point(mesh)
        loops, facet_to_loop, loop_is_hole = _extract_closed_loops(mesh)
        roles = _facet_roles(mesh, role_map)

        xs = [p[0] for p in mesh.nodes]
        ys = [p[1] for p in mesh.nodes]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        ext_x = max_x - min_x
        ext_y = max_y - min_y
        if ext_x <= 0.0 or ext_y <= 0.0:
            raise UnsupportedTopologyError("mesh bounding box has zero extent")
        char_len = math.hypot(ext_x, ext_y)
        curvatures = _vertex_curvatures(mesh, loops, char_len)

        # Essential-boundary segment midpoints for normalized distance feature.
        essential_pts: list[tuple[float, float]] = []
        for f_idx, (a, b) in enumerate(mesh.boundary_facets):
            if roles[f_idx] != "traction_free":
                ax, ay = mesh.nodes[a]
                bx, by = mesh.nodes[b]
                essential_pts.append((0.5 * (ax + bx), 0.5 * (ay + by)))

        pts_list: list[tuple[float, float]] = []
        nrm_list: list[tuple[float, float]] = []
        feat_list: list[tuple[float, ...]] = []
        wt_list: list[float] = []
        perimeter = 0.0
        k_sub = self.subdivisions_per_facet

        for f_idx, (a, b) in enumerate(mesh.boundary_facets):
            ax, ay = mesh.nodes[a]
            bx, by = mesh.nodes[b]
            dx, dy = bx - ax, by - ay
            length = math.hypot(dx, dy)
            if length <= 0.0:
                raise UnsupportedTopologyError("zero-length boundary facet")
            perimeter += length
            # Candidate unit normal, oriented strictly away from owning triangle interior.
            nx, ny = dy / length, -dx / length
            key = (a, b) if a < b else (b, a)
            ox, oy = interior[key]
            mx, my = 0.5 * (ax + bx), 0.5 * (ay + by)
            if (ox - mx) * nx + (oy - my) * ny > 0.0:
                nx, ny = -nx, -ny

            is_hole = 1.0 if loop_is_hole[facet_to_loop[f_idx]] else 0.0
            d_x, d_y, sym, t_free = role_map.role_flags(roles[f_idx])
            ka, kb = curvatures[a], curvatures[b]

            for s in range(k_sub):
                frac = (s + 0.5) / k_sub
                px = ax + frac * dx
                py = ay + frac * dy
                curv = (1.0 - frac) * ka + frac * kb
                if essential_pts:
                    dist_ess = (
                        min(math.hypot(px - ex, py - ey) for ex, ey in essential_pts)
                        / char_len
                    )
                else:
                    dist_ess = 1.0
                pts_list.append((px, py))
                nrm_list.append((nx, ny))
                feat_list.append(
                    (math.log1p(curv), is_hole, d_x, d_y, sym, t_free, dist_ess)
                )
                wt_list.append(length / k_sub)

        outer_id = next(i for i, hole in loop_is_hole.items() if not hole)
        outer_loop = loops[outer_id]
        outer_area = 0.5 * abs(
            sum(
                mesh.nodes[outer_loop[i]][0]
                * mesh.nodes[outer_loop[(i + 1) % len(outer_loop)]][1]
                - mesh.nodes[outer_loop[(i + 1) % len(outer_loop)]][0]
                * mesh.nodes[outer_loop[i]][1]
                for i in range(len(outer_loop))
            )
        )
        topology = GeometryTopologySummary(
            loop_count=len(loops),
            hole_count=len(loops) - 1,
            outer_loop_area_mm2=outer_area,
            domain_area_mm2=domain_area,
            perimeter_mm=perimeter,
            aspect_ratio=max(ext_x, ext_y) / min(ext_x, ext_y),
            extent_x_mm=ext_x,
            extent_y_mm=ext_y,
            origin_x_mm=min_x,
            origin_y_mm=min_y,
        )
        frame = QueryFrame(
            origin=torch.tensor([[min_x, min_y]], dtype=dtype),
            extent=torch.tensor([[ext_x, ext_y]], dtype=dtype),
        )
        return (
            torch.tensor(pts_list, dtype=dtype),
            torch.tensor(nrm_list, dtype=dtype),
            torch.tensor(feat_list, dtype=dtype),
            torch.tensor(wt_list, dtype=dtype),
            frame,
            topology,
        )

    def tokenize_batch(
        self,
        items: Sequence[tuple[CanonicalMesh, BoundaryRoleMap]],
        *,
        dtype: torch.dtype = torch.float64,
        pad_to: int | None = None,
    ) -> BoundaryTokenBatch:
        """Tokenize and deterministically pad a batch of distinct 2D geometries."""
        if not items:
            raise ValueError("items must contain at least one (mesh, role_map) pair")
        per_sample = [
            self.tokenize_one(mesh, roles, dtype=dtype) for mesh, roles in items
        ]
        max_n = max(sample[0].shape[0] for sample in per_sample)
        if pad_to is not None:
            if pad_to < max_n:
                raise ValueError(
                    f"pad_to={pad_to} is smaller than max point count {max_n}"
                )
            max_n = pad_to

        batch_size = len(per_sample)
        feat_dim = len(GEOMETRY_FEATURE_NAMES)
        xy = torch.zeros((batch_size, max_n, 2), dtype=dtype)
        normals = torch.zeros((batch_size, max_n, 2), dtype=dtype)
        features = torch.zeros((batch_size, max_n, feat_dim), dtype=dtype)
        weights = torch.zeros((batch_size, max_n), dtype=dtype)
        mask = torch.zeros((batch_size, max_n), dtype=torch.bool)
        origins = torch.cat([s[4].origin for s in per_sample], dim=0)
        extents = torch.cat([s[4].extent for s in per_sample], dim=0)

        for i, (s_xy, s_nrm, s_feat, s_wt, _, _) in enumerate(per_sample):
            n = s_xy.shape[0]
            xy[i, :n] = s_xy
            normals[i, :n] = s_nrm
            features[i, :n] = s_feat
            weights[i, :n] = s_wt
            mask[i, :n] = True

        return BoundaryTokenBatch(
            boundary_xy=xy,
            boundary_normals=normals,
            boundary_features=features,
            weights=weights,
            mask=mask,
            frame=QueryFrame(origin=origins, extent=extents),
            mesh_ids=tuple(mesh.mesh_id for mesh, _ in items),
            topologies=tuple(s[5] for s in per_sample),
        )


class GeometryOODStatus(enum.StrEnum):
    IN_DISTRIBUTION = "IN_DISTRIBUTION"
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"
    UNSUPPORTED_TOPOLOGY = "UNSUPPORTED_TOPOLOGY"


@dataclass(frozen=True, slots=True)
class GeometryDescriptor:
    """Scale-invariant geometric descriptor vector for OOD scoring."""

    mesh_id: str
    hole_count: int
    aspect_ratio: float
    isoperimetric_ratio: float
    curvature_p90: float
    curvature_max: float
    dirichlet_fraction: float
    symmetry_fraction: float
    traction_free_fraction: float

    def vector(self) -> tuple[float, ...]:
        return (
            float(self.hole_count),
            math.log(self.aspect_ratio),
            self.isoperimetric_ratio,
            self.curvature_p90,
            self.curvature_max,
            self.dirichlet_fraction,
            self.symmetry_fraction,
            self.traction_free_fraction,
        )

    def document(self) -> dict[str, Any]:
        return {
            "mesh_id": self.mesh_id,
            "hole_count": self.hole_count,
            "aspect_ratio": self.aspect_ratio,
            "isoperimetric_ratio": self.isoperimetric_ratio,
            "curvature_p90": self.curvature_p90,
            "curvature_max": self.curvature_max,
            "dirichlet_fraction": self.dirichlet_fraction,
            "symmetry_fraction": self.symmetry_fraction,
            "traction_free_fraction": self.traction_free_fraction,
        }


def compute_geometry_descriptor(
    mesh: CanonicalMesh,
    role_map: BoundaryRoleMap,
    *,
    tokenizer: GeometryTokenizer | None = None,
) -> GeometryDescriptor:
    """Compute a scale-invariant geometric descriptor from one mesh."""
    tok = GeometryTokenizer() if tokenizer is None else tokenizer
    _, _, features, weights, _, topo = tok.tokenize_one(mesh, role_map)
    total_w = float(weights.sum().item())
    curv = features[:, 0]
    curv_sorted = torch.sort(curv).values
    p90_idx = min(int(math.ceil(0.9 * len(curv_sorted))) - 1, len(curv_sorted) - 1)
    dirichlet_mask = (features[:, 2] > 0.5) | (features[:, 3] > 0.5)
    sym_mask = features[:, 4] > 0.5
    free_mask = features[:, 5] > 0.5
    return GeometryDescriptor(
        mesh_id=mesh.mesh_id,
        hole_count=topo.hole_count,
        aspect_ratio=topo.aspect_ratio,
        isoperimetric_ratio=topo.perimeter_mm / math.sqrt(topo.domain_area_mm2),
        curvature_p90=float(curv_sorted[max(p90_idx, 0)].item()),
        curvature_max=float(curv.max().item()),
        dirichlet_fraction=float(weights[dirichlet_mask].sum().item()) / total_w,
        symmetry_fraction=float(weights[sym_mask].sum().item()) / total_w,
        traction_free_fraction=float(weights[free_mask].sum().item()) / total_w,
    )


@dataclass(frozen=True, slots=True)
class GeometryOODReport:
    """Canonical geometry-level OOD assessment independent of parameter envelopes."""

    mesh_id: str
    status: GeometryOODStatus
    descriptor_distance: float | None
    descriptor_threshold: float
    token_distance: float | None
    token_threshold: float | None
    supported_hole_counts: tuple[int, ...]
    observed_hole_count: int | None
    reason: str
    descriptor: GeometryDescriptor | None = None

    def document(self) -> dict[str, Any]:
        return {
            "schema": GEOMETRY_OOD_SCHEMA,
            "mesh_id": self.mesh_id,
            "status": self.status.value,
            "descriptor_distance": self.descriptor_distance,
            "descriptor_threshold": self.descriptor_threshold,
            "token_distance": self.token_distance,
            "token_threshold": self.token_threshold,
            "supported_hole_counts": list(self.supported_hole_counts),
            "observed_hole_count": self.observed_hole_count,
            "reason": self.reason,
            "descriptor": None
            if self.descriptor is None
            else self.descriptor.document(),
        }

    @property
    def report_hash(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.document())).hexdigest()


@dataclass(frozen=True, slots=True)
class GeometryOODDetector:
    """TRAIN-fitted regularized Mahalanobis detector on geometry descriptors/tokens."""

    descriptor_mean: tuple[float, ...]
    descriptor_inv_cov: tuple[tuple[float, ...], ...]
    descriptor_threshold: float
    supported_hole_counts: tuple[int, ...]
    token_mean: tuple[float, ...] | None = None
    token_inv_var: tuple[float, ...] | None = None
    token_threshold: float | None = None

    @classmethod
    def fit(
        cls,
        train_descriptors: Sequence[GeometryDescriptor],
        *,
        descriptor_threshold: float = 3.5,
        regularization: float = 1e-3,
        train_token_means: Sequence[torch.Tensor] | None = None,
        token_threshold: float | None = None,
    ) -> GeometryOODDetector:
        if len(train_descriptors) < 2:
            raise ValueError(
                "fitting GeometryOODDetector requires >= 2 TRAIN geometries"
            )
        if not math.isfinite(descriptor_threshold) or descriptor_threshold <= 0.0:
            raise ValueError("descriptor_threshold must be finite and positive")
        if not math.isfinite(regularization) or regularization <= 0.0:
            raise ValueError("regularization must be finite and positive")

        mat = torch.tensor([d.vector() for d in train_descriptors], dtype=torch.float64)
        mean = mat.mean(dim=0)
        centered = mat - mean
        cov = (centered.T @ centered) / max(len(train_descriptors) - 1, 1)
        cov = cov + regularization * torch.eye(mat.shape[1], dtype=torch.float64)
        inv_cov = torch.linalg.inv(cov)

        t_mean_tuple: tuple[float, ...] | None = None
        t_inv_var_tuple: tuple[float, ...] | None = None
        if train_token_means is not None:
            if len(train_token_means) != len(train_descriptors):
                raise ValueError(
                    "train_token_means length must match train_descriptors"
                )
            if token_threshold is None or token_threshold <= 0.0:
                raise ValueError(
                    "token_threshold must be positive when tokens are given"
                )
            stacked = torch.stack(
                [
                    t.detach().to(dtype=torch.float64).reshape(-1)
                    for t in train_token_means
                ],
                dim=0,
            )
            t_mean = stacked.mean(dim=0)
            t_var = stacked.var(dim=0, unbiased=True) + regularization
            t_mean_tuple = tuple(float(v) for v in t_mean.tolist())
            t_inv_var_tuple = tuple(float(1.0 / v) for v in t_var.tolist())

        hole_counts = tuple(sorted({d.hole_count for d in train_descriptors}))
        return cls(
            descriptor_mean=tuple(float(v) for v in mean.tolist()),
            descriptor_inv_cov=tuple(
                tuple(float(v) for v in row) for row in inv_cov.tolist()
            ),
            descriptor_threshold=float(descriptor_threshold),
            supported_hole_counts=hole_counts,
            token_mean=t_mean_tuple,
            token_inv_var=t_inv_var_tuple,
            token_threshold=None if token_threshold is None else float(token_threshold),
        )

    def score(
        self,
        mesh: CanonicalMesh,
        role_map: BoundaryRoleMap,
        *,
        pooled_tokens: torch.Tensor | None = None,
        tokenizer: GeometryTokenizer | None = None,
    ) -> GeometryOODReport:
        """Assess a query geometry; explicitly refuse unsupported topology or unseen holes."""
        try:
            descriptor = compute_geometry_descriptor(
                mesh, role_map, tokenizer=tokenizer
            )
        except (UnsupportedTopologyError, ValueError) as exc:
            return GeometryOODReport(
                mesh_id=mesh.mesh_id,
                status=GeometryOODStatus.UNSUPPORTED_TOPOLOGY,
                descriptor_distance=None,
                descriptor_threshold=self.descriptor_threshold,
                token_distance=None,
                token_threshold=self.token_threshold,
                supported_hole_counts=self.supported_hole_counts,
                observed_hole_count=None,
                reason=str(exc),
                descriptor=None,
            )

        vec = torch.tensor(descriptor.vector(), dtype=torch.float64)
        mean = torch.tensor(self.descriptor_mean, dtype=torch.float64)
        inv_cov = torch.tensor(self.descriptor_inv_cov, dtype=torch.float64)
        diff = vec - mean
        dist_sq = float((diff @ (inv_cov @ diff)).item())
        dist = math.sqrt(max(dist_sq, 0.0))

        tok_dist: float | None = None
        if (
            self.token_mean is not None
            and self.token_inv_var is not None
            and pooled_tokens is not None
        ):
            t_vec = pooled_tokens.detach().to(dtype=torch.float64).reshape(-1)
            t_mean = torch.tensor(self.token_mean, dtype=torch.float64)
            t_inv = torch.tensor(self.token_inv_var, dtype=torch.float64)
            if t_vec.shape != t_mean.shape:
                raise ValueError("pooled_tokens shape does not match fitted token_mean")
            tok_dist = math.sqrt(
                max(float(torch.mean(((t_vec - t_mean) ** 2) * t_inv).item()), 0.0)
            )

        if descriptor.hole_count not in self.supported_hole_counts:
            return GeometryOODReport(
                mesh_id=mesh.mesh_id,
                status=GeometryOODStatus.UNSUPPORTED_TOPOLOGY,
                descriptor_distance=dist,
                descriptor_threshold=self.descriptor_threshold,
                token_distance=tok_dist,
                token_threshold=self.token_threshold,
                supported_hole_counts=self.supported_hole_counts,
                observed_hole_count=descriptor.hole_count,
                reason=(
                    f"observed hole_count={descriptor.hole_count} is outside "
                    f"supported classes {self.supported_hole_counts}"
                ),
                descriptor=descriptor,
            )

        if dist > self.descriptor_threshold:
            return GeometryOODReport(
                mesh_id=mesh.mesh_id,
                status=GeometryOODStatus.OUT_OF_DISTRIBUTION,
                descriptor_distance=dist,
                descriptor_threshold=self.descriptor_threshold,
                token_distance=tok_dist,
                token_threshold=self.token_threshold,
                supported_hole_counts=self.supported_hole_counts,
                observed_hole_count=descriptor.hole_count,
                reason="geometry descriptor Mahalanobis distance exceeds threshold",
                descriptor=descriptor,
            )

        if (
            tok_dist is not None
            and self.token_threshold is not None
            and tok_dist > self.token_threshold
        ):
            return GeometryOODReport(
                mesh_id=mesh.mesh_id,
                status=GeometryOODStatus.OUT_OF_DISTRIBUTION,
                descriptor_distance=dist,
                descriptor_threshold=self.descriptor_threshold,
                token_distance=tok_dist,
                token_threshold=self.token_threshold,
                supported_hole_counts=self.supported_hole_counts,
                observed_hole_count=descriptor.hole_count,
                reason="geometry latent token distance exceeds threshold",
                descriptor=descriptor,
            )

        return GeometryOODReport(
            mesh_id=mesh.mesh_id,
            status=GeometryOODStatus.IN_DISTRIBUTION,
            descriptor_distance=dist,
            descriptor_threshold=self.descriptor_threshold,
            token_distance=tok_dist,
            token_threshold=self.token_threshold,
            supported_hole_counts=self.supported_hole_counts,
            observed_hole_count=descriptor.hole_count,
            reason="geometry descriptor and topology within fitted envelope",
            descriptor=descriptor,
        )


__all__ = (
    "GEOMETRY_FEATURE_NAMES",
    "GEOMETRY_OOD_SCHEMA",
    "SUPPORTED_BC_ROLES",
    "BoundaryRoleMap",
    "BoundaryTokenBatch",
    "GeometryDescriptor",
    "GeometryOODDetector",
    "GeometryOODReport",
    "GeometryOODStatus",
    "GeometryTokenizer",
    "GeometryTopologySummary",
    "UnsupportedTopologyError",
    "compute_geometry_descriptor",
)
