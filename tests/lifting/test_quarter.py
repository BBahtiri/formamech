"""The quarter lifting moved to piginot.lifting without numerical change."""

from __future__ import annotations

import torch

import piginot.operators as operators
from piginot.lifting import AffineDirichletLifting, AxisAlignedTensionLifting
from piginot.operators import energy_operator


def test_the_operators_package_reexports_the_same_objects() -> None:
    assert energy_operator.AxisAlignedTensionLifting is AxisAlignedTensionLifting
    assert energy_operator.AffineDirichletLifting is AffineDirichletLifting
    assert operators.AxisAlignedTensionLifting is AxisAlignedTensionLifting


def test_the_lifting_is_the_adr_0018_formula() -> None:
    generator = torch.Generator().manual_seed(0)
    extent = torch.tensor([[27.5, 10.25], [20.0, 8.0]], dtype=torch.float64)
    displacement = torch.tensor([4.125, 2.0], dtype=torch.float64)
    query = torch.rand((2, 64, 2), generator=generator, dtype=torch.float64)
    query = query * extent[:, None, :]
    lifted = AxisAlignedTensionLifting(extent, displacement)(query)
    xi = query[..., 0] / extent[:, 0:1]
    eta = query[..., 1] / extent[:, 1:2]
    d = displacement[:, None]
    assert torch.equal(lifted.offset[..., 0], d * xi)
    assert torch.equal(lifted.offset[..., 1], torch.zeros_like(eta))
    assert torch.equal(lifted.scale[..., 0], d * xi * (1.0 - xi))
    assert torch.equal(lifted.scale[..., 1], d * eta)
