"""Differentiable plane-stress closure for the Neo-Hookean kernel."""

from __future__ import annotations

import math
from typing import Any

import torch

from piginot.contracts import MaterialSpec
from piginot.physics._validation import (
    require_admissible,
    validate_material,
    validate_tensor,
)
from piginot.physics.errors import (
    NewtonNonConvergenceError,
    PlaneStressClosureError,
)


def _closure_function(
    log_f33: torch.Tensor,
    log_j2: torch.Tensor,
    mu: float,
    lam: float,
) -> torch.Tensor:
    return mu * (torch.exp(2.0 * log_f33) - 1.0) + lam * (log_j2 + log_f33)


def _initial_log_guess(
    initial_f33: float | torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if isinstance(initial_f33, torch.Tensor):
        if initial_f33.layout is not torch.strided:
            raise PlaneStressClosureError("initial_f33 tensor must use strided layout")
        if initial_f33.dtype not in (torch.float32, torch.float64):
            raise PlaneStressClosureError(
                "initial_f33 tensor dtype must be float32 or float64"
            )
        if initial_f33.device != reference.device:
            raise PlaneStressClosureError(
                "initial_f33 tensor must be on the deformation device"
            )
        try:
            guess = torch.broadcast_to(
                initial_f33.to(dtype=reference.dtype), reference.shape
            )
        except RuntimeError as exc:
            raise PlaneStressClosureError(
                "initial_f33 tensor must broadcast to the deformation batch shape"
            ) from exc
    else:
        if isinstance(initial_f33, bool) or not isinstance(initial_f33, (int, float)):
            raise PlaneStressClosureError("initial_f33 must be a real scalar or tensor")
        if not math.isfinite(float(initial_f33)):
            raise PlaneStressClosureError("initial_f33 must be finite")
        guess = torch.full_like(reference, float(initial_f33))

    if guess.numel() == 0 or not bool(torch.isfinite(guess).all().item()):
        raise PlaneStressClosureError("initial_f33 must be finite and non-empty")
    if not bool((guess > 0.0).all().item()):
        raise PlaneStressClosureError("initial_f33 must be strictly positive")
    return torch.log(guess)


def _solve_log_f33(
    log_j2: torch.Tensor,
    mu: float,
    lam: float,
    initial_f33: float | torch.Tensor,
    max_iterations: int,
) -> torch.Tensor:
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise PlaneStressClosureError("max_iterations must be an integer")
    if max_iterations <= 0:
        raise NewtonNonConvergenceError(
            "plane-stress Newton solve requires at least one iteration"
        )

    lower = torch.where(log_j2 >= 0.0, -log_j2 - 1.0, torch.zeros_like(log_j2))
    upper = torch.where(log_j2 >= 0.0, torch.zeros_like(log_j2), -log_j2 + 1.0)
    log_f33 = _initial_log_guess(initial_f33, log_j2).clamp(lower, upper)
    converged = torch.zeros_like(log_j2, dtype=torch.bool)
    epsilon = torch.finfo(log_j2.dtype).eps

    for _ in range(max_iterations):
        residual = _closure_function(log_f33, log_j2, mu, lam)
        derivative = 2.0 * mu * torch.exp(2.0 * log_f33) + lam
        correction = residual / derivative
        newton = log_f33 - correction
        tolerance = (
            0.5 * epsilon * torch.maximum(torch.ones_like(log_f33), log_f33.abs())
        )
        newly_converged = (residual == 0.0) | (correction.abs() <= tolerance)
        converged = converged | newly_converged
        if bool(converged.all().item()):
            break

        midpoint = 0.5 * (lower + upper)
        inside = torch.isfinite(newton) & (newton > lower) & (newton < upper)
        candidate = torch.where(inside, newton, midpoint)
        candidate_residual = _closure_function(candidate, log_j2, mu, lam)
        lower = torch.where(candidate_residual <= 0.0, candidate, lower)
        upper = torch.where(candidate_residual > 0.0, candidate, upper)
        log_f33 = torch.where(converged, log_f33, candidate)

    final_residual = _closure_function(log_f33, log_j2, mu, lam)
    final_derivative = 2.0 * mu * torch.exp(2.0 * log_f33) + lam
    final_correction = final_residual / final_derivative
    final_tolerance = (
        0.5 * epsilon * torch.maximum(torch.ones_like(log_f33), log_f33.abs())
    )
    final_converged = (final_residual == 0.0) | (
        final_correction.abs() <= final_tolerance
    )
    if not bool(final_converged.all().item()):
        raise NewtonNonConvergenceError(
            f"plane-stress Newton solve did not converge in {max_iterations} iterations"
        )
    return log_f33


def _nearest_representable_root(
    f33: torch.Tensor,
    determinant_2d: torch.Tensor,
    mu: float,
    lam: float,
) -> torch.Tensor:
    """Select the adjacent float with the smallest physical closure residual."""

    negative_infinity = torch.full_like(f33, -torch.inf)
    positive_infinity = torch.full_like(f33, torch.inf)
    below = torch.nextafter(f33, negative_infinity)
    above = torch.nextafter(f33, positive_infinity)
    candidates = torch.stack((below, f33, above), dim=0)
    residual = mu * (candidates.square() - 1.0) + lam * torch.log(
        determinant_2d.unsqueeze(0) * candidates
    )
    best = torch.argmin(residual.abs(), dim=0, keepdim=True)
    return torch.gather(candidates, 0, best).squeeze(0)


class _PlaneStressClosure(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        deformation_2d: torch.Tensor,
        mu: float,
        lam: float,
        initial_f33: float | torch.Tensor,
        max_iterations: int,
    ) -> torch.Tensor:
        determinant_2d = torch.linalg.det(deformation_2d)
        log_f33 = _solve_log_f33(
            torch.log(determinant_2d),
            mu,
            lam,
            initial_f33,
            max_iterations,
        )
        f33 = _nearest_representable_root(torch.exp(log_f33), determinant_2d, mu, lam)
        ctx.save_for_backward(deformation_2d, f33)
        ctx.mu = mu
        ctx.lam = lam
        return f33

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        deformation_2d, f33 = ctx.saved_tensors
        inverse_transpose = torch.linalg.inv(deformation_2d).transpose(-2, -1)
        denominator = 2.0 * ctx.mu * f33 + ctx.lam / f33
        derivative = -ctx.lam * inverse_transpose / denominator[..., None, None]
        return grad_output[..., None, None] * derivative, None, None, None, None


def plane_stress_closure(
    deformation_2d: torch.Tensor,
    material: MaterialSpec,
    *,
    initial_f33: float | torch.Tensor = 1.0,
    max_iterations: int = 32,
) -> torch.Tensor:
    """Solve ``P33 = 0`` and differentiate with the implicit-function theorem.

    Material properties are immutable scalar contract data and are consequently
    treated as constants by autograd; only ``deformation_2d`` receives gradients.
    """

    parameters = validate_material(material)
    deformation_2d = validate_tensor(deformation_2d, (2, 2))
    determinant_2d = torch.linalg.det(deformation_2d)
    require_admissible(determinant_2d)

    if parameters.lam == 0.0:
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            raise PlaneStressClosureError("max_iterations must be an integer")
        if max_iterations <= 0:
            raise NewtonNonConvergenceError(
                "plane-stress Newton solve requires at least one iteration"
            )
        _initial_log_guess(initial_f33, determinant_2d)
        return deformation_2d[..., 0, 0] * 0.0 + 1.0

    closure: torch.Tensor = _PlaneStressClosure.apply(  # type: ignore[no-untyped-call]
        deformation_2d,
        parameters.mu,
        parameters.lam,
        initial_f33,
        max_iterations,
    )
    return closure


__all__ = ("plane_stress_closure",)
