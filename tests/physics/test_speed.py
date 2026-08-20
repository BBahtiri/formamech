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
    cotangent: torch.Tensor,
    operation: Callable[[torch.Tensor], torch.Tensor],
) -> float:
    deformation_2d = base.detach().clone().requires_grad_(True)
    output = operation(deformation_2d)
    start = time.perf_counter()
    torch.autograd.grad(output, deformation_2d, grad_outputs=cotangent)
    return time.perf_counter() - start


def test_custom_backward_is_at_least_three_times_faster_on_4000_points() -> None:
    deformation_2d = random_admissible_deformations(4_000, 2, seed=4301, scale=0.15)
    cotangent = torch.linspace(0.5, 1.5, 4_000, dtype=torch.float64, device="cpu")
    material = make_material(elastic_modulus=10_000.0, poisson_ratio=0.3)

    assert deformation_2d.shape == (4_000, 2, 2)
    assert deformation_2d.device.type == "cpu"
    assert deformation_2d.dtype is torch.float64
    assert cotangent.device.type == "cpu"
    assert cotangent.dtype is torch.float64

    def custom(value: torch.Tensor) -> torch.Tensor:
        return plane_stress_closure(value, material)

    def oracle(value: torch.Tensor) -> torch.Tensor:
        return _closure_unrolled(value, material)

    previous_threads = torch.get_num_threads()
    custom_times: list[float] = []
    oracle_times: list[float] = []
    try:
        torch.set_num_threads(1)
        assert torch.get_num_threads() == 1

        for index in range(5):
            operations = (custom, oracle) if index % 2 == 0 else (oracle, custom)
            for operation in operations:
                _backward_duration(deformation_2d, cotangent, operation)

        for index in range(21):
            if index % 2 == 0:
                custom_times.append(
                    _backward_duration(deformation_2d, cotangent, custom)
                )
                oracle_times.append(
                    _backward_duration(deformation_2d, cotangent, oracle)
                )
            else:
                oracle_times.append(
                    _backward_duration(deformation_2d, cotangent, oracle)
                )
                custom_times.append(
                    _backward_duration(deformation_2d, cotangent, custom)
                )
    finally:
        torch.set_num_threads(previous_threads)

    custom_median = statistics.median(custom_times)
    oracle_median = statistics.median(oracle_times)
    speedup = oracle_median / custom_median

    print(  # noqa: T201 - gate evidence records the authoritative timing metric
        "G4.speed "
        f"custom_median={custom_median * 1_000:.6f}ms "
        f"unrolled_median={oracle_median * 1_000:.6f}ms "
        f"speedup={speedup:.3f}x"
    )

    assert speedup >= 3.0, (
        f"custom backward speedup was only {speedup:.3f}x "
        f"({custom_median * 1_000:.6f}ms custom vs "
        f"{oracle_median * 1_000:.6f}ms unrolled)"
    )
