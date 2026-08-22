"""Immutable typed input contracts for geometry ingestion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from piginot.contracts import Unit


class CellType(StrEnum):
    """Cell types admitted by the S5 canonical geometry contract."""

    TRIANGLE = "triangle"


class MeshFamilyKind(StrEnum):
    """Whether a collection represents geometry variation."""

    FIXED_DEGENERATE = "fixed_degenerate"
    VARIABLE = "variable"


class PointLocation(StrEnum):
    """Containment classification for a physical query point."""

    INTERIOR = "INTERIOR"
    BOUNDARY = "BOUNDARY"
    EXTERIOR = "EXTERIOR"


@dataclass(frozen=True, slots=True)
class MeshPayload:
    """A defensive snapshot at the typed mesh-adapter boundary.

    Validation deliberately belongs to :func:`canonicalize_mesh`: retaining the
    submitted values here lets that boundary report one stable reason code rather
    than leaking incidental ``TypeError`` or mutation from an adapter.
    """

    nodes: tuple[tuple[float, float], ...]
    cells: tuple[tuple[int, ...], ...]
    cell_type: CellType
    boundary_facets: tuple[tuple[int, int], ...]
    boundary_tags: Mapping[str, tuple[int, ...]]
    region_ids: tuple[int, ...]
    unit: Unit
    source_provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        nodes = tuple(tuple(node) for node in self.nodes)
        cells = tuple(tuple(cell) for cell in self.cells)
        facets = tuple(tuple(facet) for facet in self.boundary_facets)
        tags = _snapshot_index_mapping(self.boundary_tags)
        provenance = _snapshot_text_mapping(self.source_provenance)

        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "boundary_facets", facets)
        object.__setattr__(self, "boundary_tags", tags)
        object.__setattr__(self, "region_ids", tuple(self.region_ids))
        object.__setattr__(self, "source_provenance", provenance)


def _snapshot_index_mapping(
    values: Mapping[str, Sequence[int]],
) -> Mapping[str, tuple[int, ...]]:
    snapshot = {key: tuple(indices) for key, indices in values.items()}
    return MappingProxyType(dict(sorted(snapshot.items())))


def _snapshot_text_mapping(values: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(dict(sorted(values.items())))
