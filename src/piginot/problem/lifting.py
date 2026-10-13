"""The operator's query frame and exact lifting, built from a ProblemSpec.

Expressions are evaluated per case in float64, left to right, exactly as the
bundles' scaled parameters are, so the ``quarter`` lifting built here equals a
shipped bundle's lifting bit for bit (ADR 0106 decision 6).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from piginot.contracts.problem_spec import (
    COMPONENTS,
    BoundaryCondition,
    Expression,
    Piece,
    ProblemSpec,
)
from piginot.lifting import (
    AdfLifting,
    AxisAlignedTensionLifting,
    DirichletLifting,
    DirichletPiece,
)
from piginot.operators.energy_operator import QueryFrame

Values = Sequence[Mapping[str, float]]


def _column(expressions: Sequence[Expression], values: Values) -> list[list[float]]:
    return [[e.evaluate(case) for e in expressions] for case in values]


def query_frame(
    spec: ProblemSpec, values: Values, dtype: torch.dtype = torch.float64
) -> QueryFrame:
    """The frame of each case: ``origin + [0, extent]`` per axis."""

    values = [spec.scope(case) for case in values]
    geometry = spec.geometry
    return QueryFrame(
        torch.tensor(_column(geometry.frame_origin, values), dtype=dtype),
        torch.tensor(_column(geometry.frame_extent, values), dtype=dtype),
    )


def _piece_value(
    condition: BoundaryCondition,
    component: str,
    start: Sequence[float],
    case: Mapping[str, float],
) -> float:
    value = condition.value(component)
    if value is None:
        return 0.0
    if value.nominal_strain is not None:
        return case[value.nominal_strain] * start[COMPONENTS.index(component)]
    assert value.expression is not None
    return value.expression.evaluate(case)


def _point(
    pair: tuple[Expression, Expression], case: Mapping[str, float]
) -> list[float]:
    return [pair[0].evaluate(case), pair[1].evaluate(case)]


def dirichlet_piece(
    piece: Piece,
    condition: BoundaryCondition,
    component: str,
    values: Values,
    dtype: torch.dtype,
) -> DirichletPiece:
    """The tensors of one declared piece of ``condition`` for ``component``."""

    starts = [_point(piece.start, case) for case in values]
    return DirichletPiece(
        start=torch.tensor(starts, dtype=dtype),
        end=torch.tensor([_point(piece.end, case) for case in values], dtype=dtype),
        value=torch.tensor(
            [
                _piece_value(condition, component, start, case)
                for start, case in zip(starts, values, strict=True)
            ],
            dtype=dtype,
        ),
        center=None
        if piece.center is None
        else torch.tensor([_point(piece.center, case) for case in values], dtype=dtype),
        trimmed=piece.trimmed,
    )


def adf_pieces(
    spec: ProblemSpec,
    values: Values,
    dtype: torch.dtype = torch.float64,
    *,
    drop: tuple[str, int] | None = None,
) -> tuple[tuple[DirichletPiece, ...], tuple[DirichletPiece, ...]]:
    """Each component's Dirichlet pieces; ``drop`` removes one ``(condition,
    index)`` piece (the verification mutant)."""

    if spec.lifting.kind != "adf":
        raise ValueError("the spec's lifting is not an adf lifting")
    values = [spec.scope(case) for case in values]
    result: dict[str, list[DirichletPiece]] = {c: [] for c in COMPONENTS}
    for condition in spec.boundary_conditions:
        if not condition.is_dirichlet:
            continue
        for index, piece in enumerate(spec.lifting.pieces[condition.name]):
            if drop == (condition.name, index):
                continue
            for component in condition.components:
                result[component].append(
                    dirichlet_piece(piece, condition, component, values, dtype)
                )
    return tuple(result["x"]), tuple(result["y"])


def build_lifting(
    spec: ProblemSpec,
    values: Values,
    dtype: torch.dtype = torch.float64,
    *,
    drop: tuple[str, int] | None = None,
) -> DirichletLifting:
    """The spec's lifting for a batch of cases."""

    values = [spec.scope(case) for case in values]
    frame = query_frame(spec, values, dtype)
    displacement = torch.tensor(
        [spec.lifting.displacement_scale.evaluate(case) for case in values],
        dtype=dtype,
    )
    if spec.lifting.kind == "quarter":
        return AxisAlignedTensionLifting(frame.extent, displacement)
    return AdfLifting(
        frame.origin,
        frame.extent,
        displacement,
        adf_pieces(spec, values, dtype, drop=drop),
    )


__all__ = ("adf_pieces", "build_lifting", "dirichlet_piece", "query_frame")
