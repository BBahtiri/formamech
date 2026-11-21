"""Tier 2 fallback-to-active-learning return loop ledger (pure standard library).

Captures converged Tier 2 warm-start FEM solves strictly from ``TRAIN``-scoped
workloads so every escalated certification query simultaneously serves the user
and feeds high-value physics-residual labels back into active retraining—while
hard-refusing any ``DEVELOPMENT`` or ``HOLDOUT`` partition to prevent leakage.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from piginot.active.acquisition import AcquisitionCandidate, FEMCostLedgerEntry
from piginot.contracts import canonical_json_bytes

FALLBACK_LEDGER_SCHEMA: Final = "formamech.tier2-fallback-ledger-v1"


@dataclass(frozen=True, slots=True)
class FallbackReturnRecord:
    """One TRAIN-scoped Tier 2 escalated FEM solve captured for active retraining."""

    case_id: str
    partition: str
    request_hash: str
    result_hash: str
    serving_path: str
    feature_vector: tuple[float, ...]
    eta_energy: float
    dwr_sensitivity: float = 0.0
    ood_distance: float = 0.0
    geometry_token_vector: tuple[float, ...] = ()
    newton_iterations: int = 0
    dof_count: int = 0
    wall_seconds: float = 0.0
    parameters: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id must be non-empty text")
        if self.partition != "TRAIN":
            raise ValueError(
                f"fallback active-learning return loop strictly refuses non-TRAIN "
                f"partition {self.partition!r}"
            )
        if not self.feature_vector:
            raise ValueError("feature_vector must be non-empty")
        for label in ("eta_energy", "dwr_sensitivity", "ood_distance", "wall_seconds"):
            val = float(getattr(self, label))
            if not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{label} must be finite and non-negative")
            object.__setattr__(self, label, val)
        object.__setattr__(
            self,
            "parameters",
            {str(k): float(v) for k, v in sorted(self.parameters.items())},
        )

    def to_candidate(self) -> AcquisitionCandidate:
        return AcquisitionCandidate(
            case_id=self.case_id,
            partition="TRAIN",
            feature_vector=self.feature_vector,
            eta_energy=self.eta_energy,
            ood_distance=self.ood_distance,
            ensemble_disagreement=0.0,
            dwr_sensitivity=self.dwr_sensitivity,
            geometry_token_vector=self.geometry_token_vector,
            parameters=self.parameters,
        )

    def to_cost_entry(self, attempt_index: int) -> FEMCostLedgerEntry:
        return FEMCostLedgerEntry(
            case_id=self.case_id,
            partition="TRAIN",
            attempt_index=attempt_index,
            succeeded=True,
            wall_seconds=self.wall_seconds,
            dof_count=self.dof_count,
            result_hash=self.result_hash,
            failure_reason="",
        )

    def document(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "partition": self.partition,
            "request_hash": self.request_hash,
            "result_hash": self.result_hash,
            "serving_path": self.serving_path,
            "feature_vector": list(self.feature_vector),
            "eta_energy": self.eta_energy,
            "dwr_sensitivity": self.dwr_sensitivity,
            "ood_distance": self.ood_distance,
            "geometry_token_vector": list(self.geometry_token_vector),
            "newton_iterations": self.newton_iterations,
            "dof_count": self.dof_count,
            "wall_seconds": self.wall_seconds,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class FallbackFeedbackLedger:
    """Immutable RFC 8785 ledger of captured Tier 2 fallback solves."""

    campaign_id: str
    records: tuple[FallbackReturnRecord, ...] = ()

    def append(self, record: FallbackReturnRecord) -> FallbackFeedbackLedger:
        if any(existing.case_id == record.case_id for existing in self.records):
            raise ValueError(f"duplicate fallback case_id {record.case_id!r}")
        return FallbackFeedbackLedger(
            campaign_id=self.campaign_id,
            records=(*self.records, record),
        )

    def to_acquisition_pool(self) -> tuple[AcquisitionCandidate, ...]:
        return tuple(rec.to_candidate() for rec in self.records)

    def document(self) -> dict[str, Any]:
        return {
            "schema": FALLBACK_LEDGER_SCHEMA,
            "campaign_id": self.campaign_id,
            "training_mode_disclosure": "multi-fidelity-label-efficient-not-label-free",
            "count": len(self.records),
            "records": [r.document() for r in self.records],
        }

    @property
    def ledger_hash(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.document())).hexdigest()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**self.document(), "ledger_hash": self.ledger_hash}
        path.write_bytes(canonical_json_bytes(payload) + b"\n")


__all__ = (
    "FALLBACK_LEDGER_SCHEMA",
    "FallbackFeedbackLedger",
    "FallbackReturnRecord",
)
