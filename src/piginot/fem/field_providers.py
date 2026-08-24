"""Versioned spatial boundary-field providers for FEM verification."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@runtime_checkable
class DirichletFieldProvider(Protocol):
    """Evaluate canonical displacement in mm at reference points."""

    @property
    def provider_id(self) -> str: ...

    def evaluate(self, reference_points: FloatArray) -> FloatArray: ...


@runtime_checkable
class TractionFieldProvider(Protocol):
    """Evaluate canonical nominal traction in MPa on a reference boundary."""

    @property
    def provider_id(self) -> str: ...

    def evaluate(
        self,
        reference_points: FloatArray,
        outward_reference_normals: FloatArray,
    ) -> FloatArray: ...


class _IdentifiedProvider(Protocol):
    @property
    def provider_id(self) -> str: ...


ProviderT = TypeVar("ProviderT", bound=_IdentifiedProvider)


def immutable_provider_registry(
    providers: Iterable[ProviderT],
    *,
    label: str,
) -> Mapping[str, ProviderT]:
    """Build a deterministic read-only registry indexed by stable provider ID."""

    entries: list[tuple[str, ProviderT]] = []
    for provider in providers:
        provider_id = provider.provider_id
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise ValueError(f"{label} provider IDs must be non-empty strings")
        entries.append((provider_id, provider))
    entries.sort(key=lambda item: item[0])
    if len({provider_id for provider_id, _ in entries}) != len(entries):
        raise ValueError(f"{label} provider IDs must be unique")
    return MappingProxyType(dict(entries))


__all__ = (
    "DirichletFieldProvider",
    "FloatArray",
    "TractionFieldProvider",
    "immutable_provider_registry",
)
