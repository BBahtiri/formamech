"""FormaMech reproducible 3D scaling, external V&V datasets & industry benchmarks (V4-2 / V5-3)."""

from piginot.benchmarks.external_data import (
    DATASET_REGISTRY,
    EXTERNAL_DATASET_MANIFEST_SCHEMA,
    ExternalDatasetSpec,
    LicenseTier,
    PreparedDatasetCache,
    list_external_datasets,
    prepare_dataset_cache,
    resolve_dataset_spec,
)
from piginot.benchmarks.industry_suite import (
    INDUSTRY_BENCHMARK_SCHEMA,
    BcShiftStressResult,
    IndustryBenchmarkConfig,
    IndustryBenchmarkReport,
    ScaleBenchmarkPoint,
    run_industry_benchmark_suite,
)
from piginot.benchmarks.suite_3d import (
    BENCHMARK_REPORT_SCHEMA,
    BenchmarkProfile,
    BenchmarkReport3D,
    run_benchmark_suite_3d,
)

__all__ = (
    "BENCHMARK_REPORT_SCHEMA",
    "DATASET_REGISTRY",
    "EXTERNAL_DATASET_MANIFEST_SCHEMA",
    "INDUSTRY_BENCHMARK_SCHEMA",
    "BcShiftStressResult",
    "BenchmarkProfile",
    "BenchmarkReport3D",
    "ExternalDatasetSpec",
    "IndustryBenchmarkConfig",
    "IndustryBenchmarkReport",
    "LicenseTier",
    "PreparedDatasetCache",
    "ScaleBenchmarkPoint",
    "list_external_datasets",
    "prepare_dataset_cache",
    "resolve_dataset_spec",
    "run_benchmark_suite_3d",
    "run_industry_benchmark_suite",
)
