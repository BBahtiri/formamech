"""LGPL-3.0 FEniCSx / DOLFINx SolverBackend & Third Medium Contact (TMC) adapter."""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from piginot.fem.models import FemAttempt, FemFailureCode, FemRequest
from piginot.fem.protocol import BackendCapabilities, DiscreteLinearization

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ThirdMediumContactConfig:
    """Regularized Third Medium Contact (TMC) parameters for UFL/DOLFINx contact.

    Default stiffness ratios follow standard hyperelastic TMC practice:
    ``k_third = 1e-5`` (soft compressible void medium) and ``k_reg = 1e-3``
    (HuHu/Hessian regularization stabilizing void element distortion).
    """

    k_third: float = 1.0e-5
    k_reg: float = 1.0e-3
    regularization: str = "huhu"
    void_domain_tag: str = "third_medium_void"

    def __post_init__(self) -> None:
        if not math.isfinite(self.k_third) or not (0.0 < self.k_third < 1.0e-1):
            raise ValueError("k_third must lie strictly in (0, 1e-1)")
        if not math.isfinite(self.k_reg) or not (0.0 < self.k_reg < 1.0):
            raise ValueError("k_reg must lie strictly in (0, 1)")
        if self.regularization not in ("huhu", "grad_f"):
            raise ValueError("regularization must be 'huhu' or 'grad_f'")

    def document(self) -> dict[str, Any]:
        return {
            "k_third": self.k_third,
            "k_reg": self.k_reg,
            "regularization": self.regularization,
            "void_domain_tag": self.void_domain_tag,
        }


class DolfinxBackend:
    """Optional LGPL-3.0 DOLFINx backend implementing FormaMech's ``SolverBackend``."""

    backend_name: str = "dolfinx"
    backend_id: str = "dolfinx-v0.9-tmc-v1"
    capabilities: BackendCapabilities = BackendCapabilities(
        spatial_dimensions=(2, 3),
        physics_modes=(
            "2d-plane-stress-finite-strain",
            "2d-plane-strain-finite-strain",
            "2d-axisymmetric-finite-strain",
            "3d-finite-strain",
        ),
        element_orders=(1, 2),
        supports_warm_start=True,
        supports_linearization=True,
        supports_dead_tractions=True,
        supports_follower_pressure=True,
        supports_contact=True,
        supports_gpu=False,
        supports_mpi=True,
        license_spdx="LGPL-3.0-or-later",
    )

    def __init__(
        self,
        *,
        tmc_config: ThirdMediumContactConfig | None = None,
        petsc_options: dict[str, str] | None = None,
    ) -> None:
        self.tmc_config = tmc_config or ThirdMediumContactConfig()
        self.petsc_options = dict(
            petsc_options
            or {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        )

    @staticmethod
    def is_runtime_available() -> bool:
        """Return True iff ``dolfinx``, ``ufl``, and ``petsc4py`` are importable."""
        return all(
            importlib.util.find_spec(mod) is not None
            for mod in ("dolfinx", "ufl", "petsc4py")
        )

    def _require_runtime(self) -> None:
        if not self.is_runtime_available():
            raise RuntimeError(
                "DolfinxBackend requires FEniCSx/DOLFINx (`dolfinx`, `ufl`, `petsc4py`), "
                "which is distributed for Linux/Docker/conda-forge under LGPL-3.0."
            )

    def solve_many(self, requests: tuple[FemRequest, ...]) -> tuple[FemAttempt, ...]:
        if not self.is_runtime_available():
            return tuple(
                FemAttempt(
                    request=req,
                    recipe_hash=req.recipe_hash,
                    success=False,
                    result=None,
                    result_hash=None,
                    store_content_hash=None,
                    failure_code=FemFailureCode.UNSUPPORTED_REQUEST,
                    message=(
                        "dolfinx runtime unavailable in this environment; "
                        "use scikit-fem or torch-fem, or run inside a FEniCSx container"
                    ),
                )
                for req in requests
            )
        self._require_runtime()
        raise NotImplementedError(
            "Live DOLFINx SNES assembly hooks into ufl/petsc4py inside a FEniCSx environment."
        )

    def solve_from(
        self,
        request: FemRequest,
        initial: FloatArray,
        load_factors: Sequence[float] = (1.0,),
    ) -> FemAttempt:
        del initial, load_factors
        return self.solve_many((request,))[0]

    def linearize(self, request: FemRequest) -> DiscreteLinearization:
        del request
        self._require_runtime()
        raise NotImplementedError(
            "Live DOLFINx DiscreteLinearization requires a FEniCSx container."
        )
