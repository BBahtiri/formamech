"""Total-Lagrangian 2D follower-pressure load vector and exact unsymmetric tangent.

On a reference boundary curve ``Gamma_{0,p}`` traversed counter-clockwise (so
the material domain lies on its left, with outward reference unit normal ``N``
and unit tangent ``T = (-N_y, N_x)``), the deformed outward area-normal vector
per unit reference arc length and unit thickness is given by 2D Nanson's law:

    n ds = cof(F_2D) N dS = (N + [du_y/dS, -du_x/dS]^T) dS.

Applying a uniform surface pressure ``t = -p n`` yields the configuration-
dependent external virtual work

    W_ext(u; v) = int_{Gamma_{0,p}} (-p) v . (N + [u_{y,S}, -u_{x,S}]^T) h dS

and the exact (generally unsymmetric on open boundaries) load-stiffness matrix

    (K_L)_{ab} = - d(f_ext)_a / d(u)_b
               = int_{Gamma_{0,p}} p (v_1 du_{2,S} - v_2 du_{1,S}) h dS.

On a closed loop (or when both endpoints are clamped), integrating by parts
shows ``K_L`` is symmetric (conservative enclosed-cavity pressure); on an open
boundary with free endpoints, ``K_L`` is unsymmetric because follower pressure
possesses no scalar potential energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]
from skfem import BilinearForm, FacetBasis, LinearForm, asm  # type: ignore[import-untyped]
from skfem.helpers import grad  # type: ignore[import-untyped]

from piginot.contracts import CanonicalMesh, LoadSpec, Unit
from piginot.fem.binding import canonical_facet_map, facets_for_tag
from piginot.fem.models import FemFailureCode, FemSolveFailure

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FollowerLoadAssembly:
    """Configuration-dependent follower force vector and exact load stiffness."""

    external_force: FloatArray
    load_stiffness: csr_matrix
    has_follower_loads: bool


@LinearForm  # type: ignore[untyped-decorator]
def _follower_force_form(test: Any, data: Any) -> Any:
    # data.n has shape (2, n_facets, n_qpts) pointing outward from the domain.
    # Counter-clockwise unit tangent T = (-N_y, N_x) keeps the domain on the left.
    nx = data.n[0]
    ny = data.n[1]
    tx = -ny
    ty = nx
    # Tangential derivative du_i / dS = grad(u_i) . T
    du = data.u_grad  # shape (2, 2, n_facets, n_qpts): du[comp, coord, ...]
    dux_ds = du[0, 0] * tx + du[0, 1] * ty
    duy_ds = du[1, 0] * tx + du[1, 1] * ty
    # n ds / dS = (N_x + duy_ds, N_y - dux_ds)
    traction_x = -data.pressure * (nx + duy_ds)
    traction_y = -data.pressure * (ny - dux_ds)
    return traction_x * test[0] + traction_y * test[1]


@BilinearForm  # type: ignore[untyped-decorator]
def _follower_stiffness_form(increment: Any, test: Any, data: Any) -> Any:
    nx = data.n[0]
    ny = data.n[1]
    tx = -ny
    ty = nx
    g_inc = grad(increment)  # shape (2, 2, n_facets, n_qpts)
    dux_ds = g_inc[0, 0] * tx + g_inc[0, 1] * ty
    duy_ds = g_inc[1, 0] * tx + g_inc[1, 1] * ty
    # K_L = -d(f_ext)/du = +p * (v_x * duy_ds - v_y * dux_ds)
    return data.pressure * (test[0] * duy_ds - test[1] * dux_ds)


def is_closed_boundary_loop(mesh: CanonicalMesh, boundary_tag: str) -> bool:
    """Return ``True`` iff the facets in ``boundary_tag`` form closed 2-regular loops."""
    if boundary_tag not in mesh.semantic_tags:
        return False
    facet_indices = mesh.semantic_tags[boundary_tag]
    if not facet_indices:
        return False
    degree: dict[int, int] = {}
    for idx in facet_indices:
        a, b = mesh.boundary_facets[idx]
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    return bool(degree) and all(deg == 2 for deg in degree.values())


def follower_pressure_value(load: LoadSpec) -> float:
    """Validate and extract the scalar pressure magnitude (MPa) of a follower load."""
    if load.boundary_tag is None:
        raise FemSolveFailure(
            FemFailureCode.UNSUPPORTED_REQUEST,
            f"follower pressure load {load.name!r} requires a boundary_tag",
        )
    if len(load.values) != 1 or load.values[0].unit is not Unit.MPA:
        raise FemSolveFailure(
            FemFailureCode.UNSUPPORTED_REQUEST,
            f"follower pressure load {load.name!r} requires one scalar pressure in MPa",
        )
    if load.kind == "cavity_pressure":
        pass
    elif load.kind != "follower_pressure":
        raise FemSolveFailure(
            FemFailureCode.UNSUPPORTED_REQUEST,
            f"unexpected follower load kind {load.kind!r}",
        )
    return float(load.values[0].value)


def assemble_follower_loads(
    basis: Any,
    skmesh: Any,
    mesh: CanonicalMesh,
    loads: tuple[LoadSpec, ...],
    displacement: FloatArray,
    *,
    thickness: float = 1.0,
) -> FollowerLoadAssembly:
    """Assemble configuration-dependent follower pressure force and load stiffness."""
    follower_loads = tuple(
        load for load in loads if load.kind in ("follower_pressure", "cavity_pressure")
    )
    if not follower_loads:
        return FollowerLoadAssembly(
            external_force=np.zeros(basis.N, dtype=np.float64),
            load_stiffness=csr_matrix((basis.N, basis.N), dtype=np.float64),
            has_follower_loads=False,
        )

    facet_map = canonical_facet_map(skmesh)
    total_force = np.zeros(basis.N, dtype=np.float64)
    total_stiffness = csr_matrix((basis.N, basis.N), dtype=np.float64)

    for load in follower_loads:
        pressure = follower_pressure_value(load)
        assert load.boundary_tag is not None
        if load.kind == "cavity_pressure" and not is_closed_boundary_loop(
            mesh, load.boundary_tag
        ):
            raise FemSolveFailure(
                FemFailureCode.UNSUPPORTED_REQUEST,
                f"cavity_pressure {load.name!r} requires a closed boundary loop",
            )
        facets = facets_for_tag(mesh, facet_map, load.boundary_tag)
        facet_basis = FacetBasis(
            skmesh,
            basis.elem,
            facets=facets,
            dofs=basis.dofs,
            intorder=5,
        )
        u_grad = np.asarray(
            facet_basis.interpolate(displacement).grad, dtype=np.float64
        )
        f_part = np.asarray(
            asm(
                _follower_force_form,
                facet_basis,
                u_grad=u_grad,
                pressure=pressure,
            ),
            dtype=np.float64,
        )
        k_part = asm(
            _follower_stiffness_form,
            facet_basis,
            pressure=pressure,
        ).tocsr()
        total_force = total_force + f_part * thickness
        total_stiffness = total_stiffness + k_part * thickness

    return FollowerLoadAssembly(
        external_force=total_force,
        load_stiffness=total_stiffness,
        has_follower_loads=True,
    )


__all__ = (
    "FollowerLoadAssembly",
    "assemble_follower_loads",
    "follower_pressure_value",
    "is_closed_boundary_loop",
)
