"""Tensor argument checks shared by the operators and the liftings."""

from __future__ import annotations

import torch


def validate_field(
    value: torch.Tensor,
    name: str,
    *,
    trailing: int,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.layout is not torch.strided:
        raise ValueError(f"{name} must be a dense strided tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must use float32 or float64")
    if value.ndim != 3 or value.shape[-1] != trailing or value.numel() == 0:
        raise ValueError(f"{name} must have shape (batch, count, {trailing})")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def validate_matrix(value: torch.Tensor, name: str, *, trailing: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.layout is not torch.strided:
        raise ValueError(f"{name} must be a dense strided tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must use float32 or float64")
    if value.ndim != 2 or value.shape[-1] != trailing or value.numel() == 0:
        raise ValueError(f"{name} must have shape (batch, {trailing})")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def require_same_context(
    left: torch.Tensor,
    right: torch.Tensor,
    left_name: str,
    right_name: str,
) -> None:
    if left.dtype != right.dtype or left.device != right.device:
        raise ValueError(f"{left_name} and {right_name} must share dtype and device")


__all__ = ("require_same_context", "validate_field", "validate_matrix")
