"""Independent test-only reference operations for the S4 scientific gate."""

from __future__ import annotations

import math

import torch

from piginot.contracts import CanonicalQuantity, MaterialSpec, Unit


def make_material(
    elastic_modulus: float = 760.0,
    poisson_ratio: float = 0.3,
    *,
    model: str = "compressible_neo_hookean",
) -> MaterialSpec:
    """Build a canonical Neo-Hookean material for a test case."""
    return MaterialSpec(
        name="neo_hookean",
        model=model,
        properties={
            "elastic_modulus": CanonicalQuantity(elastic_modulus, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(poisson_ratio, Unit.ONE),
        },
    )


def lame_parameters(material: MaterialSpec) -> tuple[float, float]:
    """Derive Lamé parameters independently from the production package."""
    elastic_modulus = material.properties["elastic_modulus"].value
    poisson_ratio = material.properties["poisson_ratio"].value
    shear_modulus = elastic_modulus / (2.0 * (1.0 + poisson_ratio))
    first_lame = (
        elastic_modulus
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    return shear_modulus, first_lame


def random_admissible_deformations(
    count: int,
    dimension: int,
    *,
    seed: int,
    scale: float = 0.2,
) -> torch.Tensor:
    """Generate general matrices with positive determinant by exponentiation."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    generators = torch.randn(
        count,
        dimension,
        dimension,
        generator=generator,
        dtype=torch.float64,
    )
    return torch.matrix_exp(scale * generators)


def _closure_unrolled(
    deformation_2d: torch.Tensor,
    material: MaterialSpec,
    *,
    initial_f33: float | torch.Tensor = 1.0,
    iterations: int = 32,
) -> torch.Tensor:
    """Autodiff-through-Newton oracle kept deliberately outside production code."""
    shear_modulus, first_lame = lame_parameters(material)
    determinant_2d = torch.linalg.det(deformation_2d)
    initial = torch.as_tensor(
        initial_f33,
        dtype=deformation_2d.dtype,
        device=deformation_2d.device,
    )
    log_f33 = torch.log(initial).expand(determinant_2d.shape)
    log_determinant_2d = torch.log(determinant_2d)

    for _ in range(iterations):
        squared_f33 = torch.exp(2.0 * log_f33)
        residual = shear_modulus * (squared_f33 - 1.0) + first_lame * (
            log_determinant_2d + log_f33
        )
        derivative = 2.0 * shear_modulus * squared_f33 + first_lame
        log_f33 = log_f33 - residual / derivative

    return torch.exp(log_f33)


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return a scalar relative L2 discrepancy in float64."""
    difference = torch.linalg.vector_norm(actual - expected)
    reference = torch.linalg.vector_norm(expected)
    return (difference / reference).item()


def max_relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return a stable elementwise maximum relative discrepancy."""
    floor = torch.full_like(expected, 10.0 * torch.finfo(expected.dtype).eps)
    denominator = torch.maximum(torch.abs(expected), floor)
    return torch.max(torch.abs(actual - expected) / denominator).item()


def logarithmic_material_grid(count: int) -> tuple[tuple[float, float], ...]:
    """Cover soft-to-stiff moduli and the full admitted Poisson-ratio interval."""
    if count < 2:
        raise ValueError("material grid requires at least two points")
    log_upper = math.log10(10_000.0)
    moduli = torch.logspace(0.0, log_upper, count, dtype=torch.float64)
    poisson_ratios = torch.linspace(0.0, 0.49, count, dtype=torch.float64)
    return tuple(zip(moduli.tolist(), poisson_ratios.tolist(), strict=True))
