"""Solver-agnostic CAE archive ingestion and physics-aware outlier screening (V3-3)."""

from piginot.ingest.contracts import (
    CAE_ARCHIVE_SCHEMA,
    CAE_OUTLIER_SCREEN_SCHEMA,
    INGEST_PIPELINE_MACHINE,
    CaeSampleSummary,
    CaeSourceFormat,
    IngestPipelineState,
    OutlierFlag,
    OutlierReasonCode,
    OutlierScreeningReport,
)
from piginot.ingest.readers import (
    READER_ENTRY_POINT_GROUP,
    ParsedCaeSample,
    read_abaqus_inp_and_fields,
    read_calculix_inp_frd,
    read_nastran_bdf_and_fields,
    read_vtu_or_xdmf,
    register_cae_reader,
    resolve_cae_reader,
)
from piginot.ingest.screening import screen_cae_archive

__all__ = (
    "CAE_ARCHIVE_SCHEMA",
    "CAE_OUTLIER_SCREEN_SCHEMA",
    "INGEST_PIPELINE_MACHINE",
    "READER_ENTRY_POINT_GROUP",
    "CaeSampleSummary",
    "CaeSourceFormat",
    "IngestPipelineState",
    "OutlierFlag",
    "OutlierReasonCode",
    "OutlierScreeningReport",
    "ParsedCaeSample",
    "read_abaqus_inp_and_fields",
    "read_calculix_inp_frd",
    "read_nastran_bdf_and_fields",
    "read_vtu_or_xdmf",
    "register_cae_reader",
    "resolve_cae_reader",
    "screen_cae_archive",
)
