"""Zero-heavy-dependency contracts & FSM for solver-agnostic CAE ingestion (V3-3)."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from piginot.contracts.canonical import canonical_json_bytes
from piginot.contracts.state_machine import (
    StateMachineSpec,
    StateSnapshot,
    TransitionSpec,
    register_state_machine,
)

CAE_ARCHIVE_SCHEMA: Final = "formamech.cae-archive-v1"
CAE_OUTLIER_SCREEN_SCHEMA: Final = "formamech.cae-outlier-screen-v1"


class CaeSourceFormat(StrEnum):
    """Supported external CAE simulation archive formats."""

    VTU = "vtu"
    XDMF = "xdmf"
    CALCULIX_FRD = "calculix-frd"
    ABAQUS_INP = "abaqus-inp"
    NASTRAN_BDF = "nastran-bdf"
    ANSYS_CDB = "ansys-cdb"
    LSDYNA_K = "lsdyna-k"
    SIMJEB_NPZ = "simjeb-npz"
    CGNS_HDF5 = "cgns-hdf5"
    CUSTOM_PLUGIN = "custom-plugin"


class IngestPipelineState(StrEnum):
    """Explicit lifecycle state machine for external CAE archive ingestion."""

    DISCOVERED = "DISCOVERED"
    PARSED = "PARSED"
    SCREENED = "SCREENED"
    PARTITIONED = "PARTITIONED"
    QUARANTINED = "QUARANTINED"
    FAILED = "FAILED"


INGEST_PIPELINE_MACHINE: Final[StateMachineSpec[IngestPipelineState]] = (
    register_state_machine(
        StateMachineSpec(
            machine_id="formamech.ingest.cae_pipeline",
            description=(
                "Solver-agnostic CAE archive ETL lifecycle: DISCOVERED -> PARSED -> "
                "SCREENED -> PARTITIONED | QUARANTINED | FAILED."
            ),
            states=tuple(IngestPipelineState),
            initial_states=(IngestPipelineState.DISCOVERED,),
            terminal_states=(
                IngestPipelineState.PARTITIONED,
                IngestPipelineState.QUARANTINED,
                IngestPipelineState.FAILED,
            ),
            transitions=(
                TransitionSpec(
                    IngestPipelineState.DISCOVERED,
                    IngestPipelineState.PARSED,
                    "parse_samples",
                    "All discovered CAE files parsed into canonical mesh/field buffers.",
                ),
                TransitionSpec(
                    IngestPipelineState.DISCOVERED,
                    IngestPipelineState.FAILED,
                    "parse_error",
                    "No valid samples discovered or parser rejected corrupted input.",
                ),
                TransitionSpec(
                    IngestPipelineState.PARSED,
                    IngestPipelineState.SCREENED,
                    "screen_outliers",
                    "Physical admissibility, deduplication, and 3-sigma QoI checks finished.",
                ),
                TransitionSpec(
                    IngestPipelineState.PARSED,
                    IngestPipelineState.FAILED,
                    "screen_error",
                    "Screening aborted due to malformed field dimensions.",
                ),
                TransitionSpec(
                    IngestPipelineState.SCREENED,
                    IngestPipelineState.PARTITIONED,
                    "assign_partitions",
                    "Accepted samples partitioned into leak-free TRAIN/DEV/HOLDOUT.",
                ),
                TransitionSpec(
                    IngestPipelineState.SCREENED,
                    IngestPipelineState.QUARANTINED,
                    "quarantine_all",
                    "All samples failed admissibility or outlier screening.",
                ),
            ),
        )
    )
)


class OutlierReasonCode(StrEnum):
    """Explicit fail-closed reasons for quarantining an ingested CAE sample."""

    NONFINITE_FIELD = "NONFINITE_FIELD"
    NEGATIVE_JACOBIAN = "NEGATIVE_JACOBIAN"
    STATISTICAL_ZSCORE = "STATISTICAL_ZSCORE"
    DUPLICATE_HASH = "DUPLICATE_HASH"


@dataclass(frozen=True, slots=True)
class OutlierFlag:
    """One recorded outlier or physical-admissibility violation on an ingested run."""

    sample_id: str
    reason_code: OutlierReasonCode
    metric_name: str
    value: float | None
    threshold: float | None
    detail: str

    def document(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "reason_code": self.reason_code.value,
            "metric_name": self.metric_name,
            "value": (
                None
                if self.value is None or not math.isfinite(self.value)
                else float(self.value)
            ),
            "threshold": (
                None
                if self.threshold is None or not math.isfinite(self.threshold)
                else float(self.threshold)
            ),
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class CaeSampleSummary:
    """Immutable metadata record for one ingested external CAE run."""

    sample_id: str
    source_format: CaeSourceFormat
    source_sha256: str
    spatial_dim: int
    node_count: int
    cell_count: int
    cell_type: str
    field_names: tuple[str, ...]
    parameters: Mapping[str, float]
    global_qois: Mapping[str, float]
    min_det_f: float | None
    admissible: bool
    partition_role: str = "UNASSIGNED"

    def document(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_format": self.source_format.value,
            "source_sha256": self.source_sha256,
            "spatial_dim": self.spatial_dim,
            "node_count": self.node_count,
            "cell_count": self.cell_count,
            "cell_type": self.cell_type,
            "field_names": list(self.field_names),
            "parameters": {k: float(v) for k, v in sorted(self.parameters.items())},
            "global_qois": {
                k: float(v)
                for k, v in sorted(self.global_qois.items())
                if math.isfinite(v)
            },
            "min_det_f": (
                None
                if self.min_det_f is None or not math.isfinite(self.min_det_f)
                else float(self.min_det_f)
            ),
            "admissible": self.admissible,
            "partition_role": self.partition_role,
        }


@dataclass(frozen=True, slots=True)
class OutlierScreeningReport:
    """Auditable hygiene report produced before any surrogate touches external CAE data."""

    state: IngestPipelineState
    total_samples: int
    accepted_count: int
    quarantined_count: int
    zscore_threshold: float
    det_f_floor: float
    flags: tuple[OutlierFlag, ...]
    samples: tuple[CaeSampleSummary, ...]

    def snapshot(self) -> StateSnapshot[IngestPipelineState]:
        return INGEST_PIPELINE_MACHINE.snapshot(
            self.state,
            blockers=(
                ()
                if self.state is IngestPipelineState.PARTITIONED
                else ("all samples quarantined or pipeline incomplete",)
            ),
            metadata={
                "accepted_count": self.accepted_count,
                "quarantined_count": self.quarantined_count,
            },
        )

    @property
    def report_hash(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.document())).hexdigest()

    def document(self) -> dict[str, Any]:
        return {
            "schema": CAE_OUTLIER_SCREEN_SCHEMA,
            "state": self.state.value,
            "total_samples": self.total_samples,
            "accepted_count": self.accepted_count,
            "quarantined_count": self.quarantined_count,
            "zscore_threshold": float(self.zscore_threshold),
            "det_f_floor": float(self.det_f_floor),
            "flags": [f.document() for f in self.flags],
            "samples": [s.document() for s in self.samples],
        }


__all__ = (
    "CAE_ARCHIVE_SCHEMA",
    "CAE_OUTLIER_SCREEN_SCHEMA",
    "INGEST_PIPELINE_MACHINE",
    "CaeSampleSummary",
    "CaeSourceFormat",
    "IngestPipelineState",
    "OutlierFlag",
    "OutlierReasonCode",
    "OutlierScreeningReport",
)
