"""Burned HOLDOUT cases (S20, ADR 0101 decision 4).

A HOLDOUT that has been revealed is spent. Its case IDs may be analysed
descriptively, but no selection, tuning, calibration, acquisition or compiler
path may use them. Code on those paths calls :meth:`BurnedCaseRegistry.guard`
before it touches a case.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

GUARDED_PURPOSES = frozenset(
    {"selection", "tuning", "calibration", "acquisition", "compiler"}
)
SCHEMA_VERSION = "1.0"


class BurnedCaseError(PermissionError):
    """A burned HOLDOUT case reached a selection, tuning or calibration path."""


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be non-empty canonical text")
    return value


@dataclass(frozen=True, slots=True)
class BurnedCaseRegistry:
    """The burned case IDs of one campaign and the records they come from."""

    campaign_id: str
    case_ids: frozenset[str]
    sources: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.campaign_id, "campaign_id")
        if type(self.case_ids) is not frozenset or not self.case_ids:
            raise ValueError("case_ids must be a non-empty frozenset")
        for case_id in self.case_ids:
            _text(case_id, "case id")
        if type(self.sources) is not tuple or not self.sources:
            raise ValueError("sources must be a non-empty tuple")
        for source in self.sources:
            _text(source, "source")

    @classmethod
    def from_holdout_records(
        cls, campaign_id: str, records: Mapping[str, Mapping[str, Any]]
    ) -> BurnedCaseRegistry:
        """Burn every case of the given revealed HOLDOUT records."""

        case_ids: set[str] = set()
        for source, record in records.items():
            if record.get("evidence_role") != "HOLDOUT":
                raise ValueError(f"{source} is not a HOLDOUT record")
            case_ids.update(
                _text(case["case_id"], "case id") for case in record["cases"]
            )
        return cls(campaign_id, frozenset(case_ids), tuple(sorted(records)))

    def guard(self, case_ids: Iterable[str], *, purpose: str) -> None:
        """Raise if any of ``case_ids`` is burned and ``purpose`` is guarded."""

        if purpose not in GUARDED_PURPOSES:
            raise ValueError(
                f"purpose must be one of {sorted(GUARDED_PURPOSES)}; descriptive "
                "use goes through descriptive()"
            )
        burned = sorted(self.case_ids.intersection(case_ids))
        if burned:
            shown = ", ".join(burned[:5]) + (" ..." if len(burned) > 5 else "")
            raise BurnedCaseError(
                f"{len(burned)} burned HOLDOUT case(s) of {self.campaign_id} "
                f"reached a {purpose} path: {shown}"
            )

    def descriptive(self, case_ids: Iterable[str]) -> tuple[str, ...]:
        """The burned cases among ``case_ids``, for descriptive analysis only."""

        return tuple(sorted(self.case_ids.intersection(case_ids)))

    def to_document(self) -> dict[str, Any]:
        ids = sorted(self.case_ids)
        digest = hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "case_ids": ids,
            "case_ids_sha256": digest,
            "sources": list(self.sources),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> BurnedCaseRegistry:
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported burned-case registry schema")
        registry = cls(
            document["campaign_id"],
            frozenset(document["case_ids"]),
            tuple(document["sources"]),
        )
        if registry.to_document() != dict(document):
            raise ValueError("burned-case registry is not canonical or was altered")
        return registry


def guard_all(
    registries: Sequence[BurnedCaseRegistry], case_ids: Iterable[str], *, purpose: str
) -> None:
    """Apply every campaign's registry to the same case IDs."""

    ids = tuple(case_ids)
    for registry in registries:
        registry.guard(ids, purpose=purpose)
