"""Topology invariants for canonical planar triangle complexes."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass

from piginot.contracts import CanonicalMesh


@dataclass(frozen=True, slots=True)
class TopologySummary:
    """Refinement-independent topology of a planar cell complex."""

    component_count: int
    hole_count: int
    signature: str


def topology_summary(mesh: CanonicalMesh) -> TopologySummary:
    """Summarize connected components and holes using Euler characteristic."""

    return summarize_cells(mesh.cells)


def summarize_cells(cells: Sequence[Sequence[int]]) -> TopologySummary:
    """Return topology for a validated planar triangle cell complex."""

    frozen_cells = tuple(tuple(cell) for cell in cells)
    components = _components(frozen_cells)
    component_count = len(components)
    hole_count = sum(
        _component_holes(frozen_cells, component) for component in components
    )
    signature = (
        f"planar-cell-complex-v1:components={component_count};holes={hole_count}"
    )
    return TopologySummary(component_count, hole_count, signature)


def _components(cells: tuple[tuple[int, ...], ...]) -> tuple[frozenset[int], ...]:
    edge_to_cells: dict[tuple[int, int], set[int]] = defaultdict(set)
    for cell_id, cell in enumerate(cells):
        for left, right in _cell_edges(cell):
            edge_to_cells[_edge(left, right)].add(cell_id)

    remaining = set(range(len(cells)))
    components: list[frozenset[int]] = []
    while remaining:
        component = {remaining.pop()}
        queue = deque(component)
        while queue:
            cell_id = queue.popleft()
            for left, right in _cell_edges(cells[cell_id]):
                neighbors = edge_to_cells[_edge(left, right)] & remaining
                remaining.difference_update(neighbors)
                component.update(neighbors)
                queue.extend(neighbors)
        components.append(frozenset(component))
    return tuple(components)


def _component_holes(
    cells: tuple[tuple[int, ...], ...], component: frozenset[int]
) -> int:
    vertices = {node for cell_id in component for node in cells[cell_id]}
    edges = {
        _edge(left, right)
        for cell_id in component
        for left, right in _cell_edges(cells[cell_id])
    }
    euler_characteristic = len(vertices) - len(edges) + len(component)
    return 1 - euler_characteristic


def _cell_edges(cell: Sequence[int]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (cell[index], cell[(index + 1) % len(cell)]) for index in range(len(cell))
    )


def _edge(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)
