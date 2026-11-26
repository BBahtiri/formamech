"""NAFEMS EDSWG-aligned SimOps / SPDM canonical archive metadata schema (V5-3).

Zero-heavy-dependency contracts capturing solver provenance, explicit physical
units (never inferred), boundary-condition summaries, and strict separation
between **valid physical failures** (buckling, collapse, fracture — preserved
for training failure-regime boundaries) and **solver numerical defects**
(under-converged, diverged, non-finite, inverted elements — quarantined).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from piginot.contracts import Unit, canonical_json_bytes
from piginot.ingest.contracts import CaeSourceFormat

SIMOPS_RECORD_SCHEMA: Final = "formamech.simops-record-v1"
SIMOPS_CURATION_REPORT_SCHEMA: Final = "formamech.simops-curation-report-v1"


def _sha256_doc(doc: object) -> str:
    return hashlib.sha256(canonical_json_bytes(doc)).hexdigest()


class FailureDisposition(StrEnum):
    """Explicit separation of converged runs, physical failures, and solver defects."""

    VALID_CONVERGED = "VALID_CONVERGED"
    VALID_PHYSICAL_FAILURE = "VALID_PHYSICAL_FAILURE"
    QUARANTINED_SOLVER_DEFECT = "QUARANTINED_SOLVER_DEFECT"


@dataclass(frozen=True, slots=True)
class SolverProvenance:
    """Auditable solver convergence and execution provenance from deck/logs."""

    solver_name: str
    solver_version: str = "unknown"
    convergence_status: str = "converged"
    final_residual: float | None = None
    cutbacks: int = 0
    termination_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.solver_name, str) or not self.solver_name.strip():
            raise ValueError("solver_name must be non-empty text")
        if self.final_residual is not None and not math.isfinite(
            float(self.final_residual)
        ):
            raise ValueError("final_residual must be finite when provided")
        if type(self.cutbacks) is not int or self.cutbacks < 0:
            raise ValueError("cutbacks must be a non-negative integer")

    def document(self) -> dict[str, Any]:
        return {
            "solver_name": self.solver_name,
            "solver_version": self.solver_version,
            "convergence_status": self.convergence_status,
            "final_residual": (
                None if self.final_residual is None else float(self.final_residual)
            ),
            "cutbacks": self.cutbacks,
            "termination_reason": self.termination_reason,
        }


@dataclass(frozen=True, slots=True)
class BcDescriptorSummary:
    """Extracted semantic boundary-condition summary for BC-shift tracking."""

    load_case_id: str = "default"
    clamped_node_count: int = 0
    loaded_node_count: int = 0
    net_force_n: tuple[float, float, float] = (0.0, 0.0, 0.0)
    thermal_bc_present: bool = False
    bc_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.net_force_n) != 3 or any(
            not math.isfinite(float(v)) for v in self.net_force_n
        ):
            raise ValueError("net_force_n must be 3 finite floats")

    def document(self) -> dict[str, Any]:
        return {
            "load_case_id": self.load_case_id,
            "clamped_node_count": self.clamped_node_count,
            "loaded_node_count": self.loaded_node_count,
            "net_force_n": [float(v) for v in self.net_force_n],
            "thermal_bc_present": self.thermal_bc_present,
            "bc_tags": list(self.bc_tags),
        }


@dataclass(frozen=True, slots=True)
class SimOpsRecord:
    """NAFEMS EDSWG-aligned canonical SimOps record for one external CAE run."""

    sample_id: str
    geometry_group_id: str
    source_format: CaeSourceFormat
    source_sha256: str
    length_unit: Unit
    stress_unit: Unit
    provenance: SolverProvenance
    disposition: FailureDisposition
    bc_summary: BcDescriptorSummary = field(default_factory=BcDescriptorSummary)
    material_summary: Mapping[str, float] = field(default_factory=dict)
    physical_failure_label: str | None = None
    quarantine_reason: str | None = None
    min_det_f: float | None = None
    qois: Mapping[str, float] = field(default_factory=dict)
    split: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty text")
        if (
            not isinstance(self.geometry_group_id, str)
            or not self.geometry_group_id.strip()
        ):
            raise ValueError("geometry_group_id must be non-empty text")
        object.__setattr__(self, "source_format", CaeSourceFormat(self.source_format))
        object.__setattr__(self, "length_unit", Unit(self.length_unit))
        object.__setattr__(self, "stress_unit", Unit(self.stress_unit))
        object.__setattr__(self, "disposition", FailureDisposition(self.disposition))
        if (
            self.disposition is FailureDisposition.VALID_PHYSICAL_FAILURE
            and not (self.physical_failure_label or "").strip()
        ):
            raise ValueError(
                "VALID_PHYSICAL_FAILURE requires a non-empty physical_failure_label"
            )
        if (
            self.disposition is FailureDisposition.QUARANTINED_SOLVER_DEFECT
            and not (self.quarantine_reason or "").strip()
        ):
            raise ValueError(
                "QUARANTINED_SOLVER_DEFECT requires a non-empty quarantine_reason"
            )

    @property
    def accepted(self) -> bool:
        return self.disposition is not FailureDisposition.QUARANTINED_SOLVER_DEFECT

    def document(self) -> dict[str, Any]:
        return {
            "schema": SIMOPS_RECORD_SCHEMA,
            "sample_id": self.sample_id,
            "geometry_group_id": self.geometry_group_id,
            "source_format": self.source_format.value,
            "source_sha256": self.source_sha256,
            "length_unit": self.length_unit.value,
            "stress_unit": self.stress_unit.value,
            "provenance": self.provenance.document(),
            "disposition": self.disposition.value,
            "bc_summary": self.bc_summary.document(),
            "material_summary": {
                str(k): float(v) for k, v in sorted(self.material_summary.items())
            },
            "physical_failure_label": self.physical_failure_label,
            "quarantine_reason": self.quarantine_reason,
            "min_det_f": None if self.min_det_f is None else float(self.min_det_f),
            "qois": {str(k): float(v) for k, v in sorted(self.qois.items())},
            "split": self.split,
        }


@dataclass(frozen=True, slots=True)
class SimOpsCurationReport:
    """Auditable summary of multi-solver archive screening & sharded dataset curation."""

    total_samples: int
    valid_converged_count: int
    valid_physical_failure_count: int
    quarantined_solver_defect_count: int
    geometry_group_splits: Mapping[str, str]
    records: tuple[SimOpsRecord, ...]
    dataset_root: str | None = None
    dataset_manifest_sha256: str | None = None

    @property
    def accepted_count(self) -> int:
        return self.valid_converged_count + self.valid_physical_failure_count

    @property
    def report_hash(self) -> str:
        return _sha256_doc(self.document())

    def document(self) -> dict[str, Any]:
        return {
            "schema": SIMOPS_CURATION_REPORT_SCHEMA,
            "total_samples": self.total_samples,
            "accepted_count": self.accepted_count,
            "valid_converged_count": self.valid_converged_count,
            "valid_physical_failure_count": self.valid_physical_failure_count,
            "quarantined_solver_defect_count": self.quarantined_solver_defect_count,
            "geometry_group_splits": dict(sorted(self.geometry_group_splits.items())),
            "dataset_root": self.dataset_root,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "records": [r.document() for r in self.records],
        }


__all__ = (
    "SIMOPS_CURATION_REPORT_SCHEMA",
    "SIMOPS_RECORD_SCHEMA",
    "BcDescriptorSummary",
    "FailureDisposition",
    "SimOpsCurationReport",
    "SimOpsRecord",
    "SolverProvenance",
)
