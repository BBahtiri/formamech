"""Property tests of the ADF lifting (ADR 0106, G24.lifting-exact)."""

from __future__ import annotations

import math

import pytest
import torch

from piginot.lifting import (
    AdfLifting,
    AxisAlignedTensionLifting,
    DirichletPiece,
    boundary_points,
    boundary_residual,
)

BATCH = 1000
DTYPE = torch.float64


def _random(generator: torch.Generator, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.rand(BATCH, generator=generator, dtype=DTYPE)


def _point(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.stack((x, y), dim=-1)


def _plate(
    seed: int, trimmed: bool = True, left_fraction: float = 1.0
) -> tuple[AdfLifting, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    a, b = _random(generator, 10.0, 40.0), _random(generator, 4.0, 15.0)
    d = _random(generator, 0.5, 8.0)
    zero = torch.zeros(BATCH, dtype=DTYPE)
    left = DirichletPiece(
        _point(zero, left_fraction * b), _point(zero, zero), zero, trimmed=trimmed
    )
    right = DirichletPiece(_point(a, zero), _point(a, b), d, trimmed=trimmed)
    bottom = DirichletPiece(_point(zero, zero), _point(a, zero), zero, trimmed=trimmed)
    lifting = AdfLifting(
        torch.zeros((BATCH, 2), dtype=DTYPE),
        _point(a, b),
        d,
        ((left, right), (bottom,)),
    )
    return lifting, a, b, d


def _inside(a: torch.Tensor, b: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    unit = torch.rand((BATCH, count, 2), generator=generator, dtype=DTYPE)
    unit = 0.001 + 0.998 * unit
    return unit * torch.stack((a, b), dim=-1)[:, None, :]


def test_untrimmed_lines_reduce_to_the_quarter_lifting() -> None:
    lifting, a, b, d = _plate(seed=1, trimmed=False)
    query = _inside(a, b, 32, seed=2)
    adf = lifting(query)
    quarter = AxisAlignedTensionLifting(_point(a, b), d)(query)
    scale = d[:, None, None]
    assert float(((adf.offset - quarter.offset).abs() / scale).max()) <= 1e-12
    assert float(((adf.scale - quarter.scale).abs() / scale).max()) <= 1e-12


@pytest.mark.parametrize("trimmed", [True, False])
def test_the_data_are_met_on_every_piece(trimmed: bool) -> None:
    lifting, *_ = _plate(seed=3, trimmed=trimmed, left_fraction=0.6)
    generator = torch.Generator().manual_seed(4)
    for count in (2, 17):
        assert boundary_residual(lifting, count, generator=generator) <= 1e-12
    assert (
        boundary_residual(lifting, 9, generator=generator, correction_scale=1e3)
        <= 1e-12
    )


def _arc_lifting(seed: int) -> AdfLifting:
    """A disc sector: u_x fixed on the arc, u_y on the two radii."""

    generator = torch.Generator().manual_seed(seed)
    radius = _random(generator, 5.0, 20.0)
    angle = _random(generator, 0.3, 2.8)
    centre = _point(_random(generator, -3.0, 3.0), _random(generator, -3.0, 3.0))
    start = centre + _point(radius, torch.zeros_like(radius))
    end = centre + radius[:, None] * _point(torch.cos(angle), torch.sin(angle))
    value = _random(generator, -2.0, 2.0)
    arc = DirichletPiece(start, end, value, center=centre)
    lower = DirichletPiece(centre, start, torch.zeros_like(value))
    upper = DirichletPiece(end, centre, torch.zeros_like(value))
    origin = centre - radius[:, None]
    extent = 2.0 * radius[:, None].expand(-1, 2)
    return AdfLifting(
        origin, extent.contiguous(), value.abs() + 1.0, ((arc,), (lower, upper))
    )


def test_arcs_meet_their_data() -> None:
    lifting = _arc_lifting(seed=5)
    generator = torch.Generator().manual_seed(6)
    assert boundary_residual(lifting, 33, generator=generator) <= 1e-12


def test_the_normal_derivative_is_one_on_every_piece() -> None:
    lifting, a, b, _ = _plate(seed=7, left_fraction=0.5)
    fraction = torch.rand((BATCH, 5), generator=torch.Generator().manual_seed(8))
    fraction = (0.05 + 0.9 * fraction).to(DTYPE)
    left = torch.stack((torch.zeros_like(fraction), 0.5 * b[:, None] * fraction), -1)
    bottom = torch.stack((a[:, None] * fraction, torch.zeros_like(fraction)), -1)
    for points, component, normal in ((left, 0, (1.0, 0.0)), (bottom, 1, (0.0, 1.0))):
        query = points.clone().requires_grad_(True)
        phi = lifting.components(query)[component][0]
        (gradient,) = torch.autograd.grad(phi.sum(), query)
        expected = torch.tensor(normal, dtype=DTYPE) / lifting.length[:, None, None]
        assert torch.allclose(gradient, expected.expand_as(gradient), rtol=1e-9)


def test_the_arc_normal_derivative_points_to_the_centre() -> None:
    lifting = _arc_lifting(seed=9)
    arc = lifting.pieces[0][0]
    points = boundary_points(arc, 9)[:, 1:-1].clone().requires_grad_(True)
    phi = lifting.components(points)[0][0]
    (gradient,) = torch.autograd.grad(phi.sum(), points)
    assert arc.center is not None
    inward = arc.center[:, None, :] - points.detach()
    inward = inward / torch.linalg.vector_norm(inward, dim=-1, keepdim=True)
    expected = inward / lifting.length[:, None, None]
    assert torch.allclose(gradient, expected, rtol=1e-7, atol=1e-12)


def test_phi_is_positive_inside_the_domain() -> None:
    lifting, a, b, _ = _plate(seed=10, left_fraction=0.7)
    (phi_x, _), (phi_y, _) = lifting.components(_inside(a, b, 16, seed=11))
    assert bool((phi_x > 0).all()) and bool((phi_y > 0).all())


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_values_and_gradients_are_finite_at_end_points_and_joints(
    dtype: torch.dtype,
) -> None:
    a, b, d = 20.0, 8.0, 3.0

    def piece(
        start: tuple[float, float], end: tuple[float, float], value: float
    ) -> DirichletPiece:
        return DirichletPiece(
            torch.tensor([start], dtype=dtype),
            torch.tensor([end], dtype=dtype),
            torch.tensor([value], dtype=dtype),
        )

    lifting = AdfLifting(
        torch.zeros((1, 2), dtype=dtype),
        torch.tensor([[a, b]], dtype=dtype),
        torch.tensor([d], dtype=dtype),
        (
            (piece((0.0, 0.5 * b), (0.0, 0.0), 0.0), piece((a, 0.0), (a, b), d)),
            (
                piece((0.0, 0.0), (0.5 * a, 0.0), 0.0),
                piece((0.5 * a, 0.0), (a, 0.0), 0.0),
            ),
        ),
    )
    points = torch.tensor(
        [
            [
                [0.0, 0.5 * b],
                [0.0, 0.0],
                [a, b],
                [a, 0.0],
                [0.5 * a, 0.0],
                [0.0, 0.25 * b],
                [a, 0.5 * b],
                [0.3 * a, 0.6 * b],
                [3.0 * a, -2.0 * b],
                [-a, b],
            ]
        ],
        dtype=dtype,
        requires_grad=True,
    )
    lifted = lifting(points)
    assert bool(
        torch.isfinite(lifted.offset).all() and torch.isfinite(lifted.scale).all()
    )
    (gradient,) = torch.autograd.grad((lifted.offset + lifted.scale).sum(), points)
    assert bool(torch.isfinite(gradient).all())
    assert float(lifted.offset.detach()[0, 4, 1]) == 0.0
    assert float(lifted.scale.detach()[0, 4, 1]) == 0.0


def _pieces(batch: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start = torch.zeros((batch, 2), dtype=DTYPE)
    end = torch.ones((batch, 2), dtype=DTYPE)
    return start, end, torch.zeros(batch, dtype=DTYPE)


def test_invalid_pieces_are_refused() -> None:
    start, end, value = _pieces()
    with pytest.raises(ValueError, match="distinct"):
        DirichletPiece(start, start.clone(), value)
    with pytest.raises(ValueError, match="same shape"):
        DirichletPiece(start, torch.ones((2, 2), dtype=DTYPE), value)
    with pytest.raises(ValueError, match="value must have shape"):
        DirichletPiece(start, end, torch.zeros(2, dtype=DTYPE))
    with pytest.raises(ValueError, match="value must be finite"):
        DirichletPiece(start, end, torch.tensor([math.inf], dtype=DTYPE))
    with pytest.raises(TypeError, match="bool"):
        DirichletPiece(start, end, value, trimmed=1)  # type: ignore[arg-type]
    centre = torch.tensor([[1.0, 0.0]], dtype=DTYPE)
    with pytest.raises(ValueError, match="always trimmed"):
        DirichletPiece(
            start, torch.tensor([[2.0, 0.0]], dtype=DTYPE), value, centre, False
        )
    with pytest.raises(ValueError, match="one circle"):
        DirichletPiece(start, torch.tensor([[1.0, 2.0]], dtype=DTYPE), value, centre)
    with pytest.raises(ValueError, match="half circle"):
        DirichletPiece(start, torch.tensor([[2.0, 0.0]], dtype=DTYPE), value, centre)
    with pytest.raises(ValueError, match="shape of start"):
        DirichletPiece(start, end, value, torch.zeros((2, 2), dtype=DTYPE))


def test_invalid_liftings_are_refused() -> None:
    start, end, value = _pieces()
    piece = DirichletPiece(start, end, value)
    origin, extent = torch.zeros((1, 2), dtype=DTYPE), torch.ones((1, 2), dtype=DTYPE)
    d = torch.ones(1, dtype=DTYPE)
    with pytest.raises(ValueError, match="at least one"):
        AdfLifting(origin, extent, d, ((piece,), ()))
    with pytest.raises(TypeError, match="DirichletPiece"):
        AdfLifting(origin, extent, d, ((piece,), ("piece",)))  # type: ignore[arg-type]
    other = DirichletPiece(*_pieces(batch=2))
    with pytest.raises(ValueError, match="batch"):
        AdfLifting(origin, extent, d, ((piece,), (other,)))
    with pytest.raises(ValueError, match="strictly positive"):
        AdfLifting(origin, -extent, d, ((piece,), (piece,)))
    with pytest.raises(ValueError, match="same shape"):
        AdfLifting(torch.zeros((2, 2), dtype=DTYPE), extent, d, ((piece,), (piece,)))
    with pytest.raises(TypeError, match="torch.Tensor"):
        AdfLifting(origin, extent, 1.0, ((piece,), (piece,)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        AdfLifting(origin, extent, torch.ones(2, dtype=DTYPE), ((piece,), (piece,)))
    with pytest.raises(ValueError, match="finite"):
        AdfLifting(
            origin, extent, torch.tensor([math.nan], dtype=DTYPE), ((piece,), (piece,))
        )
    lifting = AdfLifting(origin, extent, d, ((piece,), (piece,)))
    with pytest.raises(ValueError, match="batch"):
        lifting(torch.zeros((2, 3, 2), dtype=DTYPE))


def test_boundary_points_span_each_piece() -> None:
    start, end, value = _pieces()
    line = boundary_points(DirichletPiece(start, end, value), 5)
    assert torch.equal(line[:, 0], start) and torch.equal(line[:, -1], end)
    centre = torch.tensor([[0.0, 1.0]], dtype=DTYPE)
    arc = DirichletPiece(
        torch.tensor([[1.0, 1.0]], dtype=DTYPE),
        torch.tensor([[0.0, 2.0]], dtype=DTYPE),
        value,
        centre,
    )
    points = boundary_points(arc, 7)
    radius = torch.linalg.vector_norm(points - centre[:, None, :], dim=-1)
    assert torch.allclose(radius, torch.ones_like(radius))
    with pytest.raises(ValueError, match="two end points"):
        boundary_points(arc, 1)


def test_a_lifting_that_ignores_a_piece_misses_its_data() -> None:
    lifting, *_ = _plate(seed=12)
    broken = AdfLifting(
        lifting.origin,
        lifting.extent,
        lifting.displacement,
        ((lifting.pieces[0][0],), lifting.pieces[1]),
    )
    probe = AdfLifting(
        broken.origin,
        broken.extent,
        broken.displacement,
        (lifting.pieces[0], lifting.pieces[1]),
    )
    generator = torch.Generator().manual_seed(13)
    assert boundary_residual(probe, 5, generator=generator) <= 1e-12
    right = lifting.pieces[0][1]
    points = boundary_points(right, 5)
    lifted = broken(points)
    phi = torch.ones_like(lifted.offset)
    u = lifted.offset + lifted.scale * phi
    miss = (u[..., 0] - right.value[:, None]).abs() / lifting.displacement[:, None]
    assert float(miss.max()) > 1e-3


def test_a_point_a_rounding_error_outside_keeps_the_inward_derivative() -> None:
    lifting, a, b, _ = _plate(seed=14, left_fraction=0.5)
    y = 0.25 * b[:, None].expand(-1, 3)
    x = torch.tensor([[-1e-15, 0.0, 1e-15]], dtype=DTYPE).expand(BATCH, -1)
    points = torch.stack((x * a[:, None], y), dim=-1).requires_grad_(True)
    phi = lifting.components(points)[0][0]
    (gradient,) = torch.autograd.grad(phi.sum(), points)
    expected = 1.0 / lifting.length[:, None]
    assert torch.allclose(gradient[..., 0], expected.expand(-1, 3), rtol=1e-9)
    assert bool((phi.detach() >= 0).all())
    far = torch.stack((-1e-3 * a[:, None], y[:, :1]), dim=-1).requires_grad_(True)
    phi_far = lifting.components(far)[0][0]
    (outside,) = torch.autograd.grad(phi_far.sum(), far)
    assert bool((outside[..., 0] < 0).all())
