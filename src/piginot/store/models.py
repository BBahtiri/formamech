"""Immutable public models for content retention."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RetentionTier(StrEnum):
    """Retention policy for stored object bytes."""

    PERMANENT = "PERMANENT"
    RETAINED = "RETAINED"
    PRUNABLE = "PRUNABLE"


@dataclass(frozen=True, slots=True)
class ContentInfo:
    """Committed identity and materialization state for one content hash."""

    content_hash: str
    size_bytes: int
    retention: RetentionTier
    materialized: bool


@dataclass(frozen=True, slots=True)
class PruneResult:
    """Deterministic summary of one prune operation."""

    pruned_hashes: tuple[str, ...]
    pruned_bytes: int
    dry_run: bool

    @property
    def object_count(self) -> int:
        """Return the number of selected content objects."""

        return len(self.pruned_hashes)
