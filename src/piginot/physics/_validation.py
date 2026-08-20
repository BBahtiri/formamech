"""Validation shared by the constitutive and plane-stress kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from piginot.contracts import Dimension, MaterialSpec, Unit
from piginot.physics.errors import (
    InadmissibleDeformationError,
    InvalidMaterialError,
    InvalidTensorError,
    InvalidTensorShapeError,
    NonFiniteTensorError,
)

DETERMINANT_MARGIN = 1.0e-12
_MODEL = "compressible_neo_hookean"
_MATERIAL_KEYS = frozenset({"elastic_modulus", "poisson_ratio"})
_MIN_ELASTIC_MODULUS_MPA = 1.0
_MAX_ELASTIC_MODULUS_MPA = 10_000.0
_MAX_POISSON_RATIO = 0.49


@dataclass(frozen=True, slots=True)
class LameParameters:
    """Validated Lamé parameters in the canonical MPa stress unit."""

    mu: float
    lam: float


def validate_material(material: object) -> LameParameters:
    """Validate and convert the frozen material contract to Lamé parameters."""

    if not isinstance(material, MaterialSpec):
        raise InvalidMaterialError("material must be a MaterialSpec")
    if material.model != _MODEL:
        raise InvalidMaterialError(f"material model must be {_MODEL!r}")
    if frozenset(material.properties) != _MATERIAL_KEYS:
        raise InvalidMaterialError(
            "material properties must contain exactly elastic_modulus and poisson_ratio"
        )

    elastic_modulus = material.properties["elastic_modulus"]
    poisson_ratio = material.properties["poisson_ratio"]
    if (
        elastic_modulus.unit is not Unit.MPA
        or elastic_modulus.dimension is not Dimension.STRESS
    ):
        raise InvalidMaterialError("elastic_modulus must use canonical MPa stress")
    if (
        poisson_ratio.unit is not Unit.ONE
        or poisson_ratio.dimension is not Dimension.DIMENSIONLESS
    ):
        raise InvalidMaterialError("poisson_ratio must be dimensionless")

    young = elastic_modulus.value
    nu = poisson_ratio.value
    if (
        not math.isfinite(young)
        or not _MIN_ELASTIC_MODULUS_MPA <= young <= _MAX_ELASTIC_MODULUS_MPA
    ):
        raise InvalidMaterialError("elastic_modulus must satisfy 1 <= E <= 10000 MPa")
    if not math.isfinite(nu) or not 0.0 <= nu <= _MAX_POISSON_RATIO:
        raise InvalidMaterialError("poisson_ratio must satisfy 0 <= nu <= 0.49")

    mu = young / (2.0 * (1.0 + nu))
    lam = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return LameParameters(mu=mu, lam=lam)


def validate_tensor(value: object, trailing_shape: tuple[int, int]) -> torch.Tensor:
    """Return a validated dense real floating tensor with a matrix tail."""

    if not isinstance(value, torch.Tensor):
        raise InvalidTensorError("deformation must be a torch.Tensor")
    if value.layout is not torch.strided:
        raise InvalidTensorError("deformation tensor must use strided layout")
    if value.dtype not in (torch.float32, torch.float64):
        raise InvalidTensorError("deformation tensor dtype must be float32 or float64")
    if value.ndim < 2 or tuple(value.shape[-2:]) != trailing_shape:
        raise InvalidTensorShapeError(
            f"deformation tensor must have trailing shape {trailing_shape}"
        )
    if value.numel() == 0:
        raise InvalidTensorError("deformation tensor must be non-empty")
    if not bool(torch.isfinite(value).all().item()):
        raise NonFiniteTensorError("deformation tensor must contain only finite values")
    return value


def determinant_is_admissible(determinant: torch.Tensor) -> torch.Tensor:
    """Return the elementwise strict finite-determinant predicate."""

    return torch.isfinite(determinant) & (determinant > DETERMINANT_MARGIN)


def require_admissible(determinant: torch.Tensor) -> None:
    """Fail closed unless every deformation determinant is admitted."""

    if not bool(determinant_is_admissible(determinant).all().item()):
        raise InadmissibleDeformationError(
            f"deformation determinant must be finite and > {DETERMINANT_MARGIN}"
        )
