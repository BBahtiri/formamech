"""Certificate-driven active FEM acquisition and multi-fidelity learning (Feature C)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from piginot.active.acquisition import (
    ACQUISITION_ARMS,
    ACQUISITION_PLAN_SCHEMA,
    FEM_COST_LEDGER_SCHEMA,
    SELECTION_EVIDENCE_SCHEMA,
    TRAINING_MODES,
    AcquisitionCandidate,
    AcquisitionPlan,
    AcquisitionWeights,
    FEMCostLedger,
    FEMCostLedgerEntry,
    SelectionEvidence,
    SelectionScoreBreakdown,
    select_acquisition_batch,
)
from piginot.active.campaign import (
    ACTIVE_CAMPAIGN_SUMMARY_SCHEMA,
    ActiveFemCampaignResult,
    ActiveFemSolveOutcome,
    run_active_fem_campaign,
)
from piginot.active.feedback import (
    FALLBACK_LEDGER_SCHEMA,
    FallbackFeedbackLedger,
    FallbackReturnRecord,
)

if TYPE_CHECKING:
    from piginot.active.test_planner import (
        PHYSICAL_TEST_PLAN_SCHEMA,
        RETROSPECTIVE_BENCH_SCHEMA,
        BayBeAcquisitionBackend,
        BoTorchAcquisitionBackend,
        PhysicalTestCandidate,
        PhysicalTestPlanConfig,
        PhysicalTestPlanResult,
        RbfGpInformationGainBackend,
        RecommendedPhysicalTest,
        RetrospectiveTestReductionReport,
        TestAcquisitionBackend,
        recommend_next_physical_tests,
        resolve_test_acquisition_backend,
        run_retrospective_test_reduction_benchmark,
    )

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "MULTIFIDELITY_DISCLOSURE": (
        "piginot.active.multifidelity",
        "MULTIFIDELITY_DISCLOSURE",
    ),
    "MultiFidelityConfig": ("piginot.active.multifidelity", "MultiFidelityConfig"),
    "MultiFidelityTerms": ("piginot.active.multifidelity", "MultiFidelityTerms"),
    "multifidelity_energy_loss": (
        "piginot.active.multifidelity",
        "multifidelity_energy_loss",
    ),
    "PHYSICAL_TEST_PLAN_SCHEMA": (
        "piginot.active.test_planner",
        "PHYSICAL_TEST_PLAN_SCHEMA",
    ),
    "RETROSPECTIVE_BENCH_SCHEMA": (
        "piginot.active.test_planner",
        "RETROSPECTIVE_BENCH_SCHEMA",
    ),
    "BayBeAcquisitionBackend": (
        "piginot.active.test_planner",
        "BayBeAcquisitionBackend",
    ),
    "BoTorchAcquisitionBackend": (
        "piginot.active.test_planner",
        "BoTorchAcquisitionBackend",
    ),
    "PhysicalTestCandidate": (
        "piginot.active.test_planner",
        "PhysicalTestCandidate",
    ),
    "PhysicalTestPlanConfig": (
        "piginot.active.test_planner",
        "PhysicalTestPlanConfig",
    ),
    "PhysicalTestPlanResult": (
        "piginot.active.test_planner",
        "PhysicalTestPlanResult",
    ),
    "RbfGpInformationGainBackend": (
        "piginot.active.test_planner",
        "RbfGpInformationGainBackend",
    ),
    "RecommendedPhysicalTest": (
        "piginot.active.test_planner",
        "RecommendedPhysicalTest",
    ),
    "RetrospectiveTestReductionReport": (
        "piginot.active.test_planner",
        "RetrospectiveTestReductionReport",
    ),
    "TestAcquisitionBackend": (
        "piginot.active.test_planner",
        "TestAcquisitionBackend",
    ),
    "recommend_next_physical_tests": (
        "piginot.active.test_planner",
        "recommend_next_physical_tests",
    ),
    "resolve_test_acquisition_backend": (
        "piginot.active.test_planner",
        "resolve_test_acquisition_backend",
    ),
    "run_retrospective_test_reduction_benchmark": (
        "piginot.active.test_planner",
        "run_retrospective_test_reduction_benchmark",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    mod_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(mod_name), attr_name)
    globals()[name] = value
    return value


__all__ = (
    "ACQUISITION_ARMS",
    "ACQUISITION_PLAN_SCHEMA",
    "ACTIVE_CAMPAIGN_SUMMARY_SCHEMA",
    "FALLBACK_LEDGER_SCHEMA",
    "FEM_COST_LEDGER_SCHEMA",
    "MULTIFIDELITY_DISCLOSURE",
    "PHYSICAL_TEST_PLAN_SCHEMA",
    "RETROSPECTIVE_BENCH_SCHEMA",
    "SELECTION_EVIDENCE_SCHEMA",
    "TRAINING_MODES",
    "AcquisitionCandidate",
    "AcquisitionPlan",
    "AcquisitionWeights",
    "ActiveFemCampaignResult",
    "ActiveFemSolveOutcome",
    "BayBeAcquisitionBackend",
    "BoTorchAcquisitionBackend",
    "FEMCostLedger",
    "FEMCostLedgerEntry",
    "FallbackFeedbackLedger",
    "FallbackReturnRecord",
    "MultiFidelityConfig",
    "MultiFidelityTerms",
    "PhysicalTestCandidate",
    "PhysicalTestPlanConfig",
    "PhysicalTestPlanResult",
    "RbfGpInformationGainBackend",
    "RecommendedPhysicalTest",
    "RetrospectiveTestReductionReport",
    "SelectionEvidence",
    "SelectionScoreBreakdown",
    "TestAcquisitionBackend",
    "multifidelity_energy_loss",
    "recommend_next_physical_tests",
    "resolve_test_acquisition_backend",
    "run_active_fem_campaign",
    "run_retrospective_test_reduction_benchmark",
    "select_acquisition_batch",
)
