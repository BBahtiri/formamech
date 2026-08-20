"""Performance gate for the custom plane-stress backward."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import torch
from piginot.physics import plane_stress_closure

from .reference import _closure_unrolled, make_material, random_admissible_deformations


def _backward_duration(
    base: torch.Tensor,
    operation: Callable[[torch.Tensor], torch.Tensor],
) -> float:
    deformation_2d = base.detach().requires_grad_(True)
    output = operation(deformation_2d)
    start = time.perf_counter()
    torch.autograd.grad(output.sum(), deformation_2d)
    return time.perf_counter() - start


def test_custom_backward_is_at_least_three_times_faster_on_4000_points() -> None:
    deformation_2d = random_admissible_deformations(4_000, 2, seed=4301, scale=0.15)
    material = make_material(elastic_modulus=10_000.0, poisson_ratio=0.3)

    def custom(value: torch.Tensor) -> torch.Tensor:
        return plane_stress_closure(value, material)

    def oracle(value: torch.Tensor) -> torch.Tensor:
        return _closure_unrolled(value, material)

    _backward_duration(deformation_2d, custom)
    _backward_duration(deformation_2d, oracle)
    custom_times = tuple(_backward_duration(deformation_2d, custom) for _ in range(7))
    oracle_times = tuple(_backward_duration(deformation_2d, oracle) for _ in range(7))
    speedup = statistics.median(oracle_times) / statistics.median(custom_times)

    assert speedup >= 3.0, f"custom backward speedup was only {speedup:.2f}x"
