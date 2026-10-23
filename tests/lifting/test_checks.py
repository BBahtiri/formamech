"""Argument checks of the lifting interface and the quarter lifting."""

from __future__ import annotations

import math

import pytest
import torch

from piginot.lifting import AffineDirichletLifting, AxisAlignedTensionLifting

F64 = torch.float64


def _field(
    shape: tuple[int, ...] = (1, 3, 2), dtype: torch.dtype = F64
) -> torch.Tensor:
    return torch.ones(shape, dtype=dtype)


@pytest.mark.parametrize(
    ("offset", "error", "match"),
    [
        ([[[0.0, 0.0]]], TypeError, "torch.Tensor"),
        (torch.ones((1, 3, 2), dtype=F64).to_sparse(), ValueError, "dense"),
        (torch.ones((1, 3, 2), dtype=torch.int64), ValueError, "float32 or float64"),
        (torch.ones((3, 2), dtype=F64), ValueError, "shape"),
        (torch.full((1, 3, 2), math.inf, dtype=F64), ValueError, "finite"),
    ],
)
def test_affine_lifting_fields_are_checked(
    offset: object, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        AffineDirichletLifting(offset, _field())  # type: ignore[arg-type]


def test_affine_lifting_shapes_and_contexts_must_agree() -> None:
    with pytest.raises(ValueError, match="same shape"):
        AffineDirichletLifting(_field(), _field((1, 4, 2)))
    with pytest.raises(ValueError, match="dtype and device"):
        AffineDirichletLifting(_field(), _field(dtype=torch.float32))


@pytest.mark.parametrize(
    ("extent", "displacement", "error", "match"),
    [
        ([[1.0, 1.0]], torch.ones(1, dtype=F64), TypeError, "torch.Tensor"),
        (torch.ones((1, 2), dtype=F64).to_sparse(), torch.ones(1), ValueError, "dense"),
        (torch.ones((1, 2), dtype=torch.int32), torch.ones(1), ValueError, "float32"),
        (torch.ones((1, 3), dtype=F64), torch.ones(1, dtype=F64), ValueError, "shape"),
        (torch.full((1, 2), math.nan, dtype=F64), torch.ones(1), ValueError, "finite"),
        (
            -torch.ones((1, 2), dtype=F64),
            torch.ones(1, dtype=F64),
            ValueError,
            "positive",
        ),
        (torch.ones((1, 2), dtype=F64), 1.0, TypeError, "torch.Tensor"),
        (torch.ones((1, 2), dtype=F64), torch.ones(2, dtype=F64), ValueError, "shape"),
        (
            torch.ones((1, 2), dtype=F64),
            torch.ones(1, dtype=torch.float32),
            ValueError,
            "dtype",
        ),
        (
            torch.ones((1, 2), dtype=F64),
            torch.tensor([math.inf], dtype=F64),
            ValueError,
            "finite",
        ),
    ],
)
def test_quarter_lifting_arguments_are_checked(
    extent: object, displacement: object, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        AxisAlignedTensionLifting(extent, displacement)  # type: ignore[arg-type]


def test_quarter_lifting_queries_are_checked() -> None:
    lifting = AxisAlignedTensionLifting(
        torch.ones((1, 2), dtype=F64), torch.ones(1, dtype=F64)
    )
    with pytest.raises(ValueError, match="batch"):
        lifting(_field((2, 3, 2)))
    with pytest.raises(ValueError, match="dtype and device"):
        lifting(_field(dtype=torch.float32))
